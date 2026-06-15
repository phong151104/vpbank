# -*- coding: utf-8 -*-
"""Tái lập kết quả Nhiệm vụ 1 KHÔNG cần notebook.

Quy trình: nạp dữ liệu -> feature engineering -> tuning 3 booster (Optuna)
-> rank-average -> ghi submission_800.txt + outputs/test_scores.csv.

Cách dùng:
    python run_pipeline.py                 # mặc định (n_trials=40/40/30)
    python run_pipeline.py --trials 20     # nhanh hơn
    python run_pipeline.py --no-tune       # bỏ tuning, dùng tham số mặc định
"""
import argparse, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np

from src.data import load_data, TARGET
from src.features import build_features, feature_sets
from src.models import get_models
from src.cv import oof_proba
from src.tune import tune_model, build_tuned_estimator
from src.ensemble import to_rank
from src.predict import fit_full, score_test, make_submission, verify_submission
from src.metrics import summarize

BOOSTERS = ["LightGBM", "XGBoost", "CatBoost"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=None, help="số trial Optuna mỗi booster")
    ap.add_argument("--no-tune", action="store_true", help="bỏ tuning, dùng tham số mặc định")
    ap.add_argument("--repeats", type=int, default=3, help="số lần lặp CV cho OOF")
    args = ap.parse_args()
    default_trials = {"LightGBM": 40, "XGBoost": 40, "CatBoost": 30}

    train, test = load_data()
    Xtr, Xte, ENG = build_features(train, test)
    y = train[TARGET].astype(int); pos_rate = y.mean()
    cols = feature_sets(ENG)["eng"]
    test_ids = test["ID"].values
    print(f"Train {Xtr.shape} | Test {Xte.shape} | tỉ lệ mua {pos_rate:.4f} | {len(cols)} đặc trưng")

    specs, oof = {}, {}
    for nm in BOOSTERS:
        t0 = time.time()
        if args.no_tune:
            spec = {"est": get_models(pos_rate)[nm]["est"], "scale": False, "te": True}
        else:
            nt = args.trials or default_trials[nm]
            bp, val, mi, _ = tune_model(nm, Xtr, y, cols, pos_rate, n_trials=nt)
            est = build_tuned_estimator(nm, bp, mi, pos_rate)
            spec = {"est": est, "scale": False, "te": True}
        specs[nm] = spec
        o, _ = oof_proba(spec, Xtr, y, cols, n_repeats=args.repeats)
        oof[nm] = o
        print(f"  {nm:10s} OOF hits@20%={summarize(y,o)['hits@20%']:>3d}  ({time.time()-t0:.0f}s)")

    # Rank-average OOF -> báo cáo
    ra_oof = np.mean([to_rank(oof[n]) for n in BOOSTERS], axis=0)
    s = summarize(y, ra_oof)
    print(f"\nRank-Avg OOF: AUC={s['AUC']:.4f} hits@20%={s['hits@20%']} lift@20%={s['lift@20%']:.2f}")

    # Refit toàn train -> điểm test -> rank-average -> submission
    test_scores = {n: score_test(fit_full(specs[n], Xtr, y, cols), Xte) for n in BOOSTERS}
    test_ra = np.mean([to_rank(test_scores[n]) for n in BOOSTERS], axis=0)
    make_submission(test_ra, test_ids, k=800)
    ok, _ = verify_submission("submission_800.txt")
    print("Đã ghi submission_800.txt | kiểm tra:", ok)


if __name__ == "__main__":
    main()
