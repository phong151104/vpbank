# -*- coding: utf-8 -*-
"""Smoke test nhanh: kiểm tra toàn bộ pipeline chạy được (cấu hình nhỏ)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.data import load_data, TARGET, FEATURES
from src.features import build_features, feature_sets
from src.models import get_models
from src.cv import run_zoo, oof_proba
from src.tune import tune_model, build_tuned_estimator
from src.ensemble import compare_ensembles
from src.predict import fit_full, score_test, make_submission, verify_submission

train, test = load_data()
print("loaded", train.shape, test.shape)
Xtr, Xte, eng = build_features(train, test)
y = train[TARGET]
sets = feature_sets(eng)
cols = sets["eng"]
pos_rate = y.mean()
print("pos_rate", round(pos_rate, 4), "| eng cols", len(cols))

# 2 mô hình, n_repeats=1 cho nhanh
models = get_models(pos_rate)
mini = {k: models[k] for k in ["LogReg", "LightGBM"]}
res, oof = run_zoo(mini, Xtr, y, cols, n_repeats=1)
print(res[["AUC", "hits@20%", "lift@20%"]])

# tune nhanh 3 trial
bp, val, mi, study = tune_model("LightGBM", Xtr, y, cols, pos_rate, n_trials=3, n_splits=3)
print("tuned LightGBM hits@20% (oof):", val, "| mean_iter", mi)
est = build_tuned_estimator("LightGBM", bp, mi, pos_rate)
print("tuned estimator OK:", type(est).__name__)

# ensemble nhanh
res_e, ra, st = compare_ensembles(oof, ["LogReg", "LightGBM"], y)
print(res_e[["AUC", "hits@20%"]])

# submission
spec = {"est": est, "scale": False, "te": True}
pipe = fit_full(spec, Xtr, y, cols)
sc = score_test(pipe, Xte)
top, full = make_submission(sc, test["ID"].values, k=800,
                            out_path="outputs/_smoke_sub.txt",
                            full_csv="outputs/_smoke_scores.csv")
ok, ids = verify_submission("outputs/_smoke_sub.txt")
print("submission checks:", ok)
print("SMOKE OK")
