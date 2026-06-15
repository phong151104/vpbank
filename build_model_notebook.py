# -*- coding: utf-8 -*-
"""Generator: builds model.ipynb (modeling + tuning) using nbformat."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

# ----------------------------------------------------------------------------
md('''# VPINS — AIA Challenge: Xây dựng, thử nghiệm & tuning mô hình

**Nhiệm vụ 1:** từ 4.000 khách trong `test_data.txt`, chọn **top 800 ID** có khả năng mua AIA cao nhất. Chấm điểm = số người mua thật trong 800 ID → đây là **bài toán xếp hạng top-20%**.

**Cách làm trong notebook này** (gọi các hàm trong `src/`):
1. Nạp dữ liệu + feature engineering
2. Model zoo — chạy CV baseline (OOF) cho 10 mô hình
3. So sánh cách xử lý mất cân bằng (class weight / SMOTE / undersample)
4. Feature selection ("less is more")
5. **Tuning Optuna** cho 3 mô hình boosting mạnh nhất
6. **Ensemble** (rank-averaging + stacking)
7. Mô hình cuối → sinh `submission_800.txt`
8. SHAP — bắc cầu sang phần giải thích (Nhiệm vụ 2)

> Metric: **hits@20%** (số dương trong top 20% theo điểm) là chính; phụ là ROC-AUC & PR-AUC. Đánh giá bằng `RepeatedStratifiedKFold` (OOF) để trung thực; TargetEncoder fit theo từng fold để tránh rò rỉ.''')

# ----------------------------------------------------------------------------
md('''## 1. Nạp dữ liệu, feature engineering & cấu hình''')

code('''import warnings, os, sys, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
sys.path.insert(0, os.getcwd())

from src.data import load_data, TARGET, FEATURES, disp, NAME, A3
from src.features import build_features, feature_sets, make_preprocessor
from src.models import get_models
from src.cv import run_zoo, oof_proba, build_pipeline
from src.tune import tune_model, build_tuned_estimator
from src.ensemble import compare_ensembles, rank_average, to_rank, stack_oof
from src.predict import fit_full, score_test, make_submission, verify_submission
from src.metrics import summarize, topk_hits, lift_at_k

plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "Tahoma", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
def savefig(n): plt.savefig(FIG / (n + ".png"), dpi=150, bbox_inches="tight")

# ---- Cấu hình ngân sách tuning (tăng để tối ưu sâu hơn) ----
N_TRIALS = {"LightGBM": 40, "XGBoost": 40, "CatBoost": 30}
N_REPEATS = 3      # số lần lặp CV cho OOF
SEED = 42
print("Cấu hình:", N_TRIALS, "| n_repeats =", N_REPEATS)''')

code('''train, test = load_data()
Xtr, Xte, ENG = build_features(train, test)
y = train[TARGET].astype(int)
pos_rate = y.mean()
test_ids = test["ID"].values

SETS = feature_sets(ENG)
COLS = SETS["eng"]            # 85 gốc + đặc trưng tổng hợp
print(f"Train {Xtr.shape} | Test {Xte.shape} | tỉ lệ dương {pos_rate:.4f}")
print(f"Đặc trưng kỹ thuật thêm ({len(ENG)}):", ENG)
print(f"Bộ đặc trưng dùng chính 'eng': {len(COLS)} cột")''')

# ----------------------------------------------------------------------------
md('''## 2. Model zoo — chạy CV baseline (OOF)

Chạy 10 mô hình với tham số mặc định hợp lý + xử lý mất cân bằng (class weight / scale_pos_weight). Xếp hạng theo **hits@20%** trên OOF.''')

code('''models = get_models(pos_rate)
print("Đang chạy", len(models), "mô hình (mỗi mô hình", N_REPEATS, "x 5-fold)...\\n")
t0 = time.time()
base_res, base_oof = run_zoo(models, Xtr, y, COLS, n_repeats=N_REPEATS, seed=SEED)
print(f"\\nXong trong {time.time()-t0:.0f}s")
base_res[["AUC", "AP", "hits@20%", "hits@20%_std", "recall@20%", "lift@20%"]].round(4)''')

code('''# Biểu đồ so sánh model zoo theo hits@20% và AUC
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
r = base_res.sort_values("hits@20%")
axes[0].barh(r.index, r["hits@20%"], color="#d1495b")
axes[0].set_title("Model zoo — hits@20% (số người mua bắt được trong top 20% OOF)")
axes[0].set_xlabel("hits@20%")
for i, v in enumerate(r["hits@20%"]): axes[0].text(v+0.5, i, int(v), va="center", fontsize=8)
r2 = base_res.sort_values("AUC")
axes[1].barh(r2.index, r2["AUC"], color="#4c72b0"); axes[1].set_xlim(0.6, 0.82)
axes[1].set_title("Model zoo — ROC-AUC"); axes[1].set_xlabel("AUC")
for i, v in enumerate(r2["AUC"]): axes[1].text(v+0.002, i, f"{v:.3f}", va="center", fontsize=8)
plt.tight_layout(); savefig("M01_model_zoo"); plt.show()
print("Top 3 theo hits@20%:", list(base_res.head(3).index))''')

# ----------------------------------------------------------------------------
md('''## 3. So sánh cách xử lý mất cân bằng

Với cùng một mô hình (LightGBM), so sánh: (a) **class weight** (`scale_pos_weight`), (b) **SMOTE** (oversampling), (c) **RandomUnderSampler**, (d) **không xử lý**. Đo trên cùng OOF.''')

code('''from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from lightgbm import LGBMClassifier
from sklearn.base import clone

spw = (1 - pos_rate) / pos_rate
def lgbm(**kw): return LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                                      subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                                      n_jobs=-1, verbose=-1, **kw)
def oof_imb(sampler, clf):
    pre = make_preprocessor(COLS, scale=False)
    steps = [("pre", pre)]
    if sampler is not None: steps.append(("samp", sampler))
    steps.append(("clf", clone(clf)))
    pipe = ImbPipeline(steps)
    oof = np.zeros(len(y))
    for r in range(N_REPEATS):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED+r)
        oof += cross_val_predict(pipe, Xtr, y, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
    return oof / N_REPEATS

scen = {
 "Không xử lý":        (None, lgbm()),
 "scale_pos_weight":   (None, lgbm(scale_pos_weight=spw)),
 "SMOTE":              (SMOTE(random_state=SEED), lgbm()),
 "Undersample":        (RandomUnderSampler(random_state=SEED), lgbm()),
}
rows = []
for nm, (s, c) in scen.items():
    o = oof_imb(s, c); d = summarize(y, o); d["Cách xử lý"] = nm; rows.append(d)
imb_res = pd.DataFrame(rows).set_index("Cách xử lý").sort_values("hits@20%", ascending=False)
print(imb_res[["AUC", "AP", "hits@20%", "lift@20%"]].round(4))
print("\\n=> Chọn cách xử lý tốt nhất cho boosting:", imb_res.index[0])''')

# ----------------------------------------------------------------------------
md('''## 4. Feature selection — kiểm chứng "less is more"

CoIL 2000 cho thấy tập đặc trưng gọn nhiều khi cho kết quả tốt hơn. Ta xếp hạng bằng mutual information, rồi so OOF của mô hình tốt nhất trên: **raw (85)** vs **eng (đầy đủ)** vs **subset gọn**.''')

code('''from sklearn.feature_selection import mutual_info_classif
Xenc = make_preprocessor(COLS, scale=False).fit_transform(Xtr, y)
mi = pd.Series(mutual_info_classif(Xenc, y, discrete_features=False, random_state=SEED),
               index=COLS).sort_values(ascending=False)

best_name = base_res.index[0]              # mô hình baseline tốt nhất
best_spec = models[best_name]
def oof_hits(cols):
    o, _ = oof_proba(best_spec, Xtr, y, cols, n_repeats=N_REPEATS, seed=SEED)
    return summarize(y, o)

cand = {"raw (85)": SETS["raw"], "eng (đầy đủ)": COLS}
for k in [20, 30, 40]:
    cand[f"top{k} MI"] = mi.head(k).index.tolist()
fs_rows = []
for nm, cs in cand.items():
    d = oof_hits(cs); d["Bộ đặc trưng"] = nm; d["n_cột"] = len(cs); fs_rows.append(d)
fs_res = pd.DataFrame(fs_rows).set_index("Bộ đặc trưng").sort_values("hits@20%", ascending=False)
print(f"Mô hình dùng để so: {best_name}")
print(fs_res[["n_cột", "AUC", "hits@20%", "lift@20%"]].round(4))
FINAL_COLS = SETS["eng"] if fs_res.index[0] == "eng (đầy đủ)" else cand[fs_res.index[0]]
print("\\n=> Bộ đặc trưng chọn để tuning:", fs_res.index[0], f"({len(FINAL_COLS)} cột)")''')

code('''# Top đặc trưng theo mutual information (biểu đồ)
fig, ax = plt.subplots(figsize=(8, 7))
t = mi.head(18)[::-1]
ax.barh([disp(c) for c in t.index], t.values, color="#55a868")
ax.set_title("Top 18 đặc trưng theo Mutual Information (sau feature engineering)")
ax.set_xlabel("Mutual information"); ax.tick_params(labelsize=8)
savefig("M02_mutual_information"); plt.show()''')

# ----------------------------------------------------------------------------
md('''## 5. Tuning siêu tham số bằng Optuna

Tuning 3 mô hình boosting (**LightGBM, XGBoost, CatBoost**) với objective = OOF `hits@20%`, dùng TPE sampler + MedianPruner và **early stopping** trong từng fold. Sau đó tính OOF của mô hình đã tune để so với baseline.''')

for mname in ["LightGBM", "XGBoost", "CatBoost"]:
    code(f'''# ===== Tuning {mname} =====
t0 = time.time()
bp_{mname}, val_{mname}, iter_{mname}, study_{mname} = tune_model(
    "{mname}", Xtr, y, FINAL_COLS, pos_rate, n_trials=N_TRIALS["{mname}"], n_splits=5, seed=SEED)
print(f"{mname}: best OOF hits@20% = {{val_{mname}:.0f}} | n_iter≈{{iter_{mname}}} | {{time.time()-t0:.0f}}s")
print("Tham số tốt nhất:")
for k, v in bp_{mname}.items():
    print(f"   {{k}}: {{v}}")''')

code('''# Tính OOF cho 3 mô hình đã tune (n_estimators cố định) -> thêm vào kho OOF
tuned_specs = {}
oof_dict = dict(base_oof)   # gồm cả baseline
for nm, bp, mi_ in [("LightGBM", bp_LightGBM, iter_LightGBM),
                    ("XGBoost",  bp_XGBoost,  iter_XGBoost),
                    ("CatBoost", bp_CatBoost, iter_CatBoost)]:
    est = build_tuned_estimator(nm, bp, mi_, pos_rate, seed=SEED)
    spec = {"est": est, "scale": False, "te": True}
    tuned_specs[nm + "*"] = spec
    o, _ = oof_proba(spec, Xtr, y, FINAL_COLS, n_repeats=N_REPEATS, seed=SEED)
    oof_dict[nm + "*"] = o
    s = summarize(y, o)
    print(f"{nm+'*':12s} (tuned)  AUC={s['AUC']:.4f}  hits@20%={s['hits@20%']:>3d}  lift={s['lift@20%']:.2f}")

# So sánh baseline vs tuned
cmp = []
for nm in ["LightGBM", "XGBoost", "CatBoost"]:
    cmp.append({"model": nm, "baseline hits@20%": summarize(y, oof_dict[nm])["hits@20%"],
                "tuned hits@20%": summarize(y, oof_dict[nm+"*"])["hits@20%"]})
pd.DataFrame(cmp).set_index("model")''')

code('''# Lịch sử tối ưu Optuna (LightGBM) + tầm quan trọng tham số
import optuna.visualization.matplotlib as ovm
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
vals = [t.value for t in study_LightGBM.trials if t.value is not None]
best_so_far = np.maximum.accumulate(vals)
axes[0].plot(vals, "o", alpha=.4, label="từng trial"); axes[0].plot(best_so_far, "-", color="red", label="tốt nhất")
axes[0].set_title("Optuna LightGBM — hits@20% qua các trial"); axes[0].set_xlabel("trial"); axes[0].legend()
try:
    imp = optuna.importance.get_param_importances(study_LightGBM)
    axes[1].barh(list(imp.keys())[::-1], list(imp.values())[::-1], color="#8172b3")
    axes[1].set_title("Tầm quan trọng tham số (LightGBM)")
except Exception as e:
    axes[1].text(0.1, 0.5, "Không tính được importance:\\n"+str(e)[:60]); axes[1].axis("off")
plt.tight_layout(); savefig("M03_optuna_lightgbm"); plt.show()''')

# ----------------------------------------------------------------------------
md('''## 6. Ensemble — rank-averaging & stacking

Kết hợp top mô hình (sau tune) để tăng độ ổn định cho xếp hạng. So sánh từng mô hình đơn vs **rank-average** vs **stacking (LR meta-learner)** trên OOF.''')

code('''# Chọn 3-4 ứng viên tốt nhất trong toàn bộ kho OOF (baseline + tuned)
ranking = pd.Series({k: summarize(y, v)["hits@20%"] for k, v in oof_dict.items()}).sort_values(ascending=False)
print("Xếp hạng toàn bộ theo hits@20%:"); print(ranking.head(8))
TOP = ranking.head(3).index.tolist()
print("\\nỨng viên ensemble:", TOP)

ens_res, ra_oof, st_oof = compare_ensembles(oof_dict, TOP, y)
print(); print(ens_res[["AUC", "AP", "hits@20%", "lift@20%"]].round(4))
FINAL = ens_res.index[0]
print("\\n=> Phương án cuối (OOF tốt nhất):", FINAL)''')

# ----------------------------------------------------------------------------
md('''## 7. Mô hình cuối → sinh `submission_800.txt`

Refit phương án cuối trên **toàn bộ** 5.822 khách train, chấm điểm 4.000 khách test, lấy **top 800 ID**.''')

code('''all_specs = {**models, **tuned_specs}
def refit_score(name):
    pipe = fit_full(all_specs[name], Xtr, y, FINAL_COLS)
    return score_test(pipe, Xte)

# Tạo điểm test theo phương án cuối
if FINAL.startswith("Rank-Avg"):
    member_scores = {n: refit_score(n) for n in TOP}
    test_score = np.mean([to_rank(member_scores[n]) for n in TOP], axis=0)
    print("Phương án: rank-average của", TOP)
elif FINAL == "Stacking-LR":
    from sklearn.linear_model import LogisticRegression
    M_oof = np.column_stack([oof_dict[n] for n in TOP])
    meta = LogisticRegression(max_iter=2000, class_weight="balanced").fit(M_oof, y)
    M_test = np.column_stack([refit_score(n) for n in TOP])
    test_score = meta.predict_proba(M_test)[:, 1]
    print("Phương án: stacking-LR trên", TOP)
else:
    test_score = refit_score(FINAL)
    print("Phương án: mô hình đơn", FINAL)

top800, full = make_submission(test_score, test_ids, k=800,
                               out_path="submission_800.txt", full_csv="outputs/test_scores.csv")
ok, ids = verify_submission("submission_800.txt")
print("Kiểm tra file nộp:", ok)
print("5 ID điểm cao nhất:", top800["ID"].head().tolist())''')

code('''# Phân bố điểm test + ngưỡng top-800
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
axes[0].hist(full["score"], bins=50, color="#4c72b0", edgecolor="white")
thr = full.iloc[799]["score"]
axes[0].axvline(thr, color="red", ls="--", label=f"ngưỡng top-800 = {thr:.3f}")
axes[0].set_title("Phân bố điểm dự đoán trên 4.000 khách test"); axes[0].set_xlabel("Điểm"); axes[0].legend()
axes[1].plot(full["rank"], np.cumsum(np.ones(len(full)))/np.arange(1, len(full)+1), alpha=0)  # placeholder
sorted_score = full["score"].values
axes[1].plot(full["rank"], sorted_score, color="#d1495b")
axes[1].axvline(800, color="red", ls="--", label="top 800")
axes[1].set_title("Điểm theo thứ hạng"); axes[1].set_xlabel("Thứ hạng"); axes[1].set_ylabel("Điểm"); axes[1].legend()
plt.tight_layout(); savefig("M04_test_score_dist"); plt.show()
exp_oof = summarize(y, oof_dict[TOP[0]] if FINAL not in ("Stacking-LR",) and not FINAL.startswith("Rank-Avg") else ra_oof)
print(f"Kỳ vọng (theo OOF): lift@20% ≈ {ens_res.iloc[0]['lift@20%']:.2f}x, "
      f"tức trong 800 khách chọn có ~{ens_res.iloc[0]['lift@20%']*pos_rate*800:.0f} người mua "
      f"(so với ~{pos_rate*800:.0f} nếu chọn ngẫu nhiên).")''')

# ----------------------------------------------------------------------------
md('''## 8. Giải thích mô hình (SHAP) — bắc cầu Nhiệm vụ 2

Dùng SHAP trên mô hình boosting tốt nhất để chỉ ra đặc trưng nào đẩy xác suất mua AIA — đầu vào cho phần giải thích & khuyến nghị chiến dịch.''')

code('''# Chọn 1 mô hình cây tốt nhất đã refit để giải thích bằng SHAP
tree_candidates = [n for n in TOP if n.rstrip("*") in ("LightGBM", "XGBoost", "CatBoost", "RandomForest", "ExtraTrees", "HistGB")]
expl_name = tree_candidates[0] if tree_candidates else "LightGBM*"
pre = make_preprocessor(FINAL_COLS, scale=False).fit(Xtr, y)
Xtr_enc = pre.transform(Xtr)
feat_names = [disp(c) for c in FINAL_COLS]
from sklearn.base import clone
clf = clone(all_specs[expl_name]["est"]).fit(Xtr_enc, y)
try:
    import shap
    samp_idx = np.random.RandomState(0).choice(len(Xtr), 1000, replace=False)
    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(Xtr_enc[samp_idx])
    sv1 = sv[1] if isinstance(sv, list) else sv
    shap.summary_plot(sv1, Xtr_enc[samp_idx], feature_names=feat_names, show=False, max_display=15)
    plt.title(f"SHAP — ảnh hưởng đặc trưng tới xác suất mua AIA ({expl_name})")
    savefig("M05_shap_summary"); plt.show()
    print("Đã vẽ SHAP cho", expl_name)
except Exception as e:
    print("SHAP lỗi/không sẵn:", type(e).__name__, str(e)[:120])''')

# ----------------------------------------------------------------------------
md('''## 9. Tổng kết

Bảng so sánh tổng hợp + danh sách output. Các con số dưới đây là **OOF trên train** (proxy trung thực cho hiệu quả trên test).''')

code('''print("="*66); print("TỔNG KẾT MÔ HÌNH — VPINS AIA CHALLENGE"); print("="*66)
print(f"Train {len(y)} KH | Test {len(test_ids)} KH | tỉ lệ mua {pos_rate*100:.2f}%")
print(f"Bộ đặc trưng: {len(FINAL_COLS)} cột | Phương án cuối: {FINAL}")
print(f"OOF phương án cuối: AUC={ens_res.iloc[0]['AUC']:.4f}, "
      f"hits@20%={int(ens_res.iloc[0]['hits@20%'])}/{int(y.sum())}, "
      f"lift@20%={ens_res.iloc[0]['lift@20%']:.2f}x")
print("\\nFile kết quả:")
print("  - submission_800.txt        (800 ID nộp bài)")
print("  - outputs/test_scores.csv   (điểm + thứ hạng toàn bộ 4.000 khách)")
print("\\nBiểu đồ (figures/): M01..M05 + EDA cũ. Dùng cho slide trình bày.")
base_res[["AUC", "hits@20%", "lift@20%"]].round(4)''')

# ----------------------------------------------------------------------------
nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
with open("model.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("WROTE model.ipynb with", len(cells), "cells")
