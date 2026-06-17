# -*- coding: utf-8 -*-
"""Generator: builds training.ipynb (tuning + chọn K=15-30, KHÔNG ensemble)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

# ----------------------------------------------------------------------------
md('''# VPINS — AIA Challenge: Training cuối (tuning + chọn số feature K=15–30)

Notebook training cuối: với pool đặc trưng đã xếp hạng (`consensus_rank`), **quét số feature K∈[15,30] như một hyperparameter trong Optuna** cho từng mô hình, rồi **chọn 1 mô hình đơn tốt nhất** (không ensemble) để sinh file nộp.

**Lá chắn chống chọn-lệch:**
- Đánh giá **grouped OOF** (StratifiedGroupKFold theo profile) — không leakage.
- Metric chính **`mean_hits_over_k`** (trung bình hits ở top 15–25%), bền hơn hits@20% đơn lẻ.
- Chọn cuối theo **1-SE rule**: ưu tiên mô hình **đơn giản hơn** (ít feature / tuyến tính) nếu nằm trong 1 sai số chuẩn của bộ tốt nhất.
- Báo cáo **meanK ± SE** (lặp nhiều seed) + ghi chú optimism.

**Mô hình:** LightGBM, XGBoost, CatBoost, LogReg, LDA. **~40 trials/model**.''')

# ----------------------------------------------------------------------------
md('''## 0. Setup & cấu hình''')
code('''import warnings, os, sys, time, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
sys.path.insert(0, os.getcwd())

from src.data import load_data, TARGET, disp, NAME
from src.features import build_features_v2, load_feature_set, make_preprocessor, encoded_layout
from src.cv import profile_groups, oof_proba, build_pipeline
from src.tune import tune_model_k, build_final_spec
from src.predict import fit_full, score_test, make_submission, verify_submission
from src.metrics import summarize, mean_hits_over_k, topk_hits

plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "Tahoma", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")
FIG = Path("figures"); FIG.mkdir(exist_ok=True); Path("outputs").mkdir(exist_ok=True)
def savefig(n): plt.savefig(FIG / (n + ".png"), dpi=150, bbox_inches="tight")

K_RANGE = (15, 30)
N_TRIALS = 40
N_REPEATS_FINAL = 5
SEED = 42
MODELS = ["LightGBM", "XGBoost", "CatBoost", "LogReg", "LDA"]
print("Config:", dict(K_RANGE=K_RANGE, N_TRIALS=N_TRIALS, N_REPEATS_FINAL=N_REPEATS_FINAL))''')

code('''train, test = load_data()
Xtr, Xte, ENG = build_features_v2(train, test)
y = train[TARGET].astype(int); pos_rate = y.mean()
GROUPS = profile_groups(Xtr)
POOL = load_feature_set("consensus_rank")     # 125 feature xếp theo importance đồng thuận
test_ids = test["ID"].values
print(f"Train {Xtr.shape} | Test {Xte.shape} | dương {pos_rate:.4f} | profiles {len(np.unique(GROUPS))}")
print(f"Pool đặc trưng: {len(POOL)} | top10: {[disp(c) for c in POOL[:10]]}")
print(f"Quét K trong {K_RANGE} (top-K theo consensus).")''')

# ----------------------------------------------------------------------------
md('''## 2. Tuning từng mô hình — K là hyperparameter

Mỗi mô hình một study Optuna; mỗi trial chọn **K∈[15,30]** + tham số mô hình, chấm bằng grouped OOF `mean_hits_over_k`. Boosting dùng early stopping trong fold.''')
code('''tuned = {}   # name -> dict(bp, K, val, mean_iter, spec)''')

for m in ["LightGBM", "XGBoost", "CatBoost", "LogReg", "LDA"]:
    code(f'''# ===== Tuning {m} (K là hyperparameter) =====
t0 = time.time()
bp, K, val, mi, study = tune_model_k("{m}", Xtr, y, POOL, pos_rate, k_range=K_RANGE,
                                     n_trials=N_TRIALS, n_splits=5, seed=SEED, groups=GROUPS,
                                     objective="hits_meanK")
spec = build_final_spec("{m}", bp, K, mi, pos_rate, seed=SEED)
tuned["{m}"] = dict(bp=bp, K=K, val=val, mean_iter=mi, spec=spec, study=study)
print(f"{m}: bestK={{K}} | meanK(obj)={{val:.1f}} | {{time.time()-t0:.0f}}s")
print("   tham số:", {{k: (round(v,4) if isinstance(v,float) else v) for k,v in bp.items()}})''')

# ----------------------------------------------------------------------------
md('''## 3. Đánh giá trung thực + chọn mô hình cuối (1-SE rule)

Với mỗi mô hình đã tune: grouped OOF lặp `N_REPEATS_FINAL` seed → `mean_hits_over_k` ± **SE**, kèm hits@20% & AUC. Áp **1-SE rule**: trong các mô hình nằm trong 1 SE của bộ tốt nhất, ưu tiên **ít feature hơn**, rồi **tuyến tính**.''')
code('''def eval_repeats(spec, cols, n_rep=N_REPEATS_FINAL):
    mks = []
    for r in range(n_rep):
        o, _ = oof_proba(spec, Xtr, y, cols, n_repeats=1, seed=SEED + r, groups=GROUPS)
        mks.append(mean_hits_over_k(y, o))
    o_avg, _ = oof_proba(spec, Xtr, y, cols, n_repeats=n_rep, seed=SEED, groups=GROUPS)
    s = summarize(y, o_avg)
    return float(np.mean(mks)), float(np.std(mks) / np.sqrt(n_rep)), s

rows = []
for nm in MODELS:
    d = tuned[nm]; mk, se, s = eval_repeats(d["spec"], POOL[:d["K"]])
    rows.append({"model": nm, "K": d["K"], "meanK": mk, "SE": se,
                 "hits@20%": s["hits@20%"], "AUC": s["AUC"],
                 "linear": nm in ("LogReg", "LDA")})
res = pd.DataFrame(rows).sort_values("meanK", ascending=False).reset_index(drop=True)
print(res.round(4).to_string(index=False))

# 1-SE rule
best = res.iloc[0]; thr = best["meanK"] - best["SE"]
cand = res[res["meanK"] >= thr].copy()
cand = cand.sort_values(["K", "linear", "meanK"], ascending=[True, False, False])   # ít feature -> tuyến tính -> meanK
FINAL = cand.iloc[0]["model"]
res["selected"] = res["model"] == FINAL
res.to_csv("outputs/training_results.csv", index=False, encoding="utf-8-sig")
print(f"\\nBest meanK={best['meanK']:.1f} (±{best['SE']:.1f}) -> ngưỡng 1-SE={thr:.1f}")
print(f"Ứng viên trong 1-SE: {list(cand['model'])}")
print(f"=> CHỌN (1-SE rule, ưu tiên gọn): {FINAL}  (K={int(tuned[FINAL]['K'])})")''')

code('''# TR01 — meanK ± SE theo mô hình
fig, ax = plt.subplots(figsize=(9, 5))
r = res.sort_values("meanK")
colors = ["#d1495b" if m == FINAL else "#9aa7b5" for m in r["model"]]
ax.barh(r["model"], r["meanK"], xerr=r["SE"], color=colors, error_kw=dict(alpha=.5))
ax.axvline(thr, color="black", ls="--", lw=1, label=f"ngưỡng 1-SE = {thr:.1f}")
for i, (mk, k_) in enumerate(zip(r["meanK"], r["K"])):
    ax.text(mk + 0.3, i, f"{mk:.1f} (K={int(k_)})", va="center", fontsize=8)
ax.set_title("Mô hình đã tune — mean_hits_over_k ± SE (grouped OOF)")
ax.set_xlabel("mean hits @ K=15..25%"); ax.legend()
plt.tight_layout(); savefig("TR01_model_select"); plt.show()''')

# ----------------------------------------------------------------------------
md('''## 4. Minh bạch ảnh hưởng của K

Dù K được tune ẩn trong Optuna, trích các trial để xem **objective tốt nhất theo từng K** — cho thấy vùng K hiệu quả (câu chuyện 15–30).''')
code('''fig, ax = plt.subplots(figsize=(11, 5.5))
for nm in MODELS:
    st = tuned[nm]["study"]
    df = pd.DataFrame([{"K": t.params.get("K"), "val": t.value}
                       for t in st.trials if t.value is not None and "K" in t.params])
    if len(df):
        g = df.groupby("K")["val"].max()
        ax.plot(g.index, g.values, marker="o", ms=4, label=f"{nm} (bestK={tuned[nm]['K']})")
ax.set_title("Objective tốt nhất theo K (trích từ các trial Optuna)")
ax.set_xlabel("K = số feature giữ lại"); ax.set_ylabel("best mean_hits_over_k")
ax.legend(fontsize=8)
plt.tight_layout(); savefig("TR02_K_effect"); plt.show()''')

# ----------------------------------------------------------------------------
md('''## 5. Mô hình cuối → `submission_800.txt`

Refit mô hình đã chọn trên toàn train → chấm điểm 4.000 test → top 800 ID.''')
code('''d = tuned[FINAL]; FK = d["K"]; COLS = POOL[:FK]
pipe = fit_full(d["spec"], Xtr, y, COLS)
test_score = score_test(pipe, Xte)
top800, full = make_submission(test_score, test_ids, k=800,
                               out_path="submission_800.txt", full_csv="outputs/test_scores.csv")
ok, ids = verify_submission("submission_800.txt")
print(f"Mô hình cuối: {FINAL} | K={FK} | tham số: {d['bp']}")
print("Kiểm tra file nộp:", ok)
print("5 ID điểm cao nhất:", top800["ID"].head().tolist())

fig, ax = plt.subplots(figsize=(8, 4.4))
ax.hist(full["score"], bins=50, color="#4c72b0", edgecolor="white")
thr = full.iloc[799]["score"]
ax.axvline(thr, color="red", ls="--", label=f"ngưỡng top-800 = {thr:.3f}")
ax.set_title(f"Phân bố điểm test — mô hình cuối {FINAL} (K={FK})"); ax.set_xlabel("Điểm"); ax.legend()
plt.tight_layout(); savefig("TR03_final_score_dist"); plt.show()
lift = summarize(y, oof_proba(d["spec"], Xtr, y, COLS, n_repeats=N_REPEATS_FINAL, groups=GROUPS)[0])["lift@20%"]
print(f"Kỳ vọng (grouped OOF): lift@20% ~ {lift:.2f}x -> ~{lift*pos_rate*800:.0f} người mua trong 800 chọn "
      f"(so ~{pos_rate*800:.0f} nếu ngẫu nhiên).")''')

# ----------------------------------------------------------------------------
md('''## 6. Giải thích mô hình cuối (đầu vào Nhiệm vụ 2)''')
code('''pre = pipe.named_steps["pre"]; clf = pipe.named_steps["clf"]
Xenc = pre.transform(Xtr)
feat_names = [disp(c) for c in encoded_layout(COLS)[0]]
if FINAL in ("LightGBM", "XGBoost", "CatBoost"):
    try:
        import shap
        si = np.random.RandomState(0).choice(len(Xenc), min(1000, len(Xenc)), replace=False)
        sv = shap.TreeExplainer(clf).shap_values(Xenc[si])
        sv1 = sv[1] if isinstance(sv, list) else sv
        shap.summary_plot(sv1, Xenc[si], feature_names=feat_names, show=False, max_display=15)
        plt.title(f"SHAP — {FINAL} (K={FK})"); savefig("TR04_explain"); plt.show()
    except Exception as e:
        print("SHAP lỗi:", str(e)[:100])
else:
    coef = clf.coef_[0]
    imp = pd.Series(coef, index=feat_names).sort_values()
    sel = pd.concat([imp.head(8), imp.tail(8)])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(sel.index, sel.values, color=["#d1495b" if v > 0 else "#9aa7b5" for v in sel.values])
    ax.axvline(0, color="black", lw=.8)
    ax.set_title(f"Hệ số {FINAL} (K={FK}) — dương = tăng khả năng mua AIA"); ax.tick_params(labelsize=8)
    plt.tight_layout(); savefig("TR04_explain"); plt.show()
    print("Top yếu tố tăng khả năng mua:", list(imp.tail(5).index[::-1]))''')

# ----------------------------------------------------------------------------
md('''## 7. Tổng kết''')
code('''print("="*64); print("TỔNG KẾT TRAINING — VPINS AIA"); print("="*64)
print(f"Mô hình cuối: {FINAL} | K={FK} feature (consensus top-{FK})")
print(f"grouped OOF: meanK={res[res.model==FINAL]['meanK'].iloc[0]:.1f} ± {res[res.model==FINAL]['SE'].iloc[0]:.1f}"
      f" | hits@20%={int(res[res.model==FINAL]['hits@20%'].iloc[0])} | AUC={res[res.model==FINAL]['AUC'].iloc[0]:.4f}")
print("\\nBảng so sánh các mô hình (outputs/training_results.csv):")
print(res.round(4).to_string(index=False))
print("\\nFile: submission_800.txt | outputs/test_scores.csv | figures/TR01..TR04")
print("Lưu ý: meanK là OOF trung thực; vẫn có optimism nhẹ do chọn max trên 5 mô hình -> coi như cận trên.")''')

# ----------------------------------------------------------------------------
nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
with open("training.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("WROTE training.ipynb with", len(cells), "cells")
