# -*- coding: utf-8 -*-
"""Báo cáo importance đầy đủ: bảng full (tất cả feature × 7 phương pháp + consensus)
+ biểu đồ giúp chốt số lượng feature.
Xuất: outputs/feature_importance_full.csv, figures/FE07_consensus_full.png,
      figures/FE08_perf_vs_K.png
"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path

from src.data import load_data, TARGET, FEATURES, disp
from src.features import build_features_v2, make_preprocessor, encoded_layout, feature_groups_v2
from src.cv import profile_groups, oof_proba
from src.models import get_models
from src.metrics import summarize, mean_hits_over_k
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import pointbiserialr
from lightgbm import LGBMClassifier

plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "Tahoma", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")
FIG = Path("figures"); FIG.mkdir(exist_ok=True); Path("outputs").mkdir(exist_ok=True)
SEED = 42

train, test = load_data()
Xtr, Xte, ENG = build_features_v2(train, test)
y = train[TARGET].astype(int); pos_rate = y.mean()
GROUPS = profile_groups(Xtr)
FULL = list(FEATURES) + list(ENG)
ordered, dmask = encoded_layout(FULL)
Xenc = make_preprocessor(FULL, scale=False).fit_transform(Xtr, y)
print(f"FULL = {len(FULL)} đặc trưng | train {Xtr.shape}")

# ---- 7 phương pháp importance ----
imp = pd.DataFrame(index=ordered)
imp["MI"] = mutual_info_classif(Xenc, y, discrete_features=dmask, random_state=SEED)
imp["ANOVA_F"] = f_classif(Xenc, y)[0]
imp["point_biserial"] = [abs(pointbiserialr(y, Xenc[:, j]).statistic) for j in range(Xenc.shape[1])]
l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, class_weight="balanced", max_iter=3000)
l1.fit(StandardScaler().fit_transform(Xenc), y); imp["L1_coef"] = np.abs(l1.coef_[0])
lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED, n_jobs=-1, verbose=-1)
lgb.fit(Xenc, y); imp["LGB_gain"] = lgb.booster_.feature_importance(importance_type="gain")
tr_idx, va_idx = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xenc, y, GROUPS))
lgb2 = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED, n_jobs=-1, verbose=-1).fit(Xenc[tr_idx], y.iloc[tr_idx])
imp["perm"] = permutation_importance(lgb2, Xenc[va_idx], y.iloc[va_idx], scoring="roc_auc",
                                     n_repeats=5, random_state=SEED, n_jobs=1).importances_mean
try:
    import shap
    si = np.random.RandomState(0).choice(len(Xenc), 800, replace=False)
    sv = shap.TreeExplainer(lgb).shap_values(Xenc[si])
    sv1 = sv[1] if isinstance(sv, list) else sv
    imp["SHAP"] = np.abs(sv1).mean(axis=0)
    METHODS = ["MI", "ANOVA_F", "point_biserial", "L1_coef", "LGB_gain", "perm", "SHAP"]
except Exception as e:
    print("SHAP bỏ qua:", str(e)[:80]); METHODS = ["MI", "ANOVA_F", "point_biserial", "L1_coef", "LGB_gain", "perm"]

# ---- consensus = trung bình thứ hạng + độ ổn định ----
ranks = imp[METHODS].rank(ascending=True)
imp["consensus"] = ranks.mean(axis=1)
imp["rank_std"] = ranks.std(axis=1)          # thấp = các phương pháp nhất trí
for m in METHODS:
    imp["rk_" + m] = ranks[m].astype(int)    # thứ hạng từng phương pháp (1..N)
imp = imp.sort_values("consensus", ascending=False)
imp.insert(0, "feature", [disp(c) for c in imp.index])
imp.insert(1, "col", imp.index)
imp.insert(2, "rank", np.arange(1, len(imp) + 1))

out_csv = "outputs/feature_importance_full.csv"
imp.round(5).to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"Đã ghi bảng FULL ({len(imp)} feature × {len(METHODS)} phương pháp): {out_csv}")
print("\nTOÀN BỘ cột:", list(imp.columns))
print("\n--- 25 feature đầu (đủ thứ hạng 7 phương pháp) ---")
show = ["rank", "feature"] + ["rk_" + m for m in METHODS] + ["consensus", "rank_std"]
print(imp[show].head(25).to_string(index=False))

# ---- Biểu đồ 1: điểm consensus toàn bộ feature (đường cong elbow) ----
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(imp["rank"], imp["consensus"], marker="o", ms=3, color="#4c72b0")
for k in [20, 30, 40, 50, 60]:
    ax.axvline(k, color="grey", ls=":", lw=.8)
    ax.text(k, imp["consensus"].max(), f"K={k}", rotation=90, va="top", ha="right", fontsize=7, color="grey")
ax.set_title("Điểm consensus theo thứ hạng feature (cao = quan trọng) — toàn bộ %d feature" % len(imp))
ax.set_xlabel("Thứ hạng feature"); ax.set_ylabel("Consensus (trung bình rank 7 phương pháp)")
plt.tight_layout(); plt.savefig(FIG / "FE07_consensus_full.png", dpi=150, bbox_inches="tight")
print("\nĐã lưu figures/FE07_consensus_full.png")

# ---- Biểu đồ 2: hiệu năng grouped OOF theo số feature top-K (để CHỐT số lượng) ----
CONS_ORDER = imp["col"].tolist()
REF = {"LogReg": {"est": LogisticRegression(max_iter=2000, class_weight="balanced"), "scale": True, "te": True},
       "LightGBM": get_models(pos_rate)["LightGBM"]}
Ks = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, len(FULL)]
rows = []
for K in Ks:
    cols = CONS_ORDER[:K]
    rec = {"K": K}
    mks = []
    for nm, sp in REF.items():
        o, _ = oof_proba(sp, Xtr, y, cols, n_repeats=2, groups=GROUPS)
        rec[f"meanK_{nm}"] = mean_hits_over_k(y, o)
        rec[f"hits_{nm}"] = summarize(y, o)["hits@20%"]
        mks.append(rec[f"meanK_{nm}"])
    rec["meanK_avg"] = float(np.mean(mks))
    rows.append(rec); print(f"  K={K:3d}: meanK_avg={rec['meanK_avg']:.1f}  hits_LGB={rec['hits_LightGBM']}")
perf = pd.DataFrame(rows)
perf.to_csv("outputs/perf_vs_K.csv", index=False, encoding="utf-8-sig")

best = perf.loc[perf["meanK_avg"].idxmax()]
# K "tiết kiệm": nhỏ nhất đạt >= best - 1.0 (trong khoảng nhiễu)
thr = best["meanK_avg"] - 1.0
parsi = perf[perf["meanK_avg"] >= thr]["K"].min()

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(perf["K"], perf["meanK_LogReg"], marker="o", label="LogReg meanK", color="#dd8452")
ax.plot(perf["K"], perf["meanK_LightGBM"], marker="s", label="LightGBM meanK", color="#55a868")
ax.plot(perf["K"], perf["meanK_avg"], marker="D", label="Trung bình 2 mô hình", color="#d1495b", lw=2.2)
ax.axvline(best["K"], color="black", ls="--", lw=1, label=f"tốt nhất K={int(best['K'])}")
ax.axvline(parsi, color="blue", ls=":", lw=1.4, label=f"tiết kiệm K={int(parsi)} (trong ±1)")
ax.set_title("Hiệu năng grouped OOF theo số feature (consensus top-K) — để chốt số lượng")
ax.set_xlabel("Số feature giữ lại (K, theo consensus)"); ax.set_ylabel("mean hits @ K=15..25%")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "FE08_perf_vs_K.png", dpi=150, bbox_inches="tight")
print("\nĐã lưu figures/FE08_perf_vs_K.png")
print("\n==== GỢI Ý CHỐT SỐ FEATURE ====")
print(perf[["K", "meanK_avg", "hits_LightGBM"]].to_string(index=False))
print(f"\n>> Tốt nhất: K={int(best['K'])} (meanK_avg={best['meanK_avg']:.1f}). "
      f"Tiết kiệm (≈ tốt nhất, ít feature hơn): K={int(parsi)}.")
print("Bạn xem outputs/feature_importance_full.csv + figures/FE08 để tự chốt K.")
