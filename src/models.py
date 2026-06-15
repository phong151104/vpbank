# -*- coding: utf-8 -*-
"""Model zoo: trả về dict {tên: spec}. Mỗi spec gồm estimator + cờ scale/te."""
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

SEED = 42


def get_models(pos_rate):
    """pos_rate = tỉ lệ dương (để tính scale_pos_weight cho boosting)."""
    spw = (1 - pos_rate) / pos_rate  # ~15.7
    M = {}
    # ----- Tuyến tính / xác suất (cần scale) -----
    M["LogReg"]     = dict(est=LogisticRegression(max_iter=3000, class_weight="balanced",
                                                  C=1.0, solver="liblinear"),
                           scale=True, te=True)
    M["GaussianNB"] = dict(est=GaussianNB(), scale=True, te=True)
    M["BernoulliNB"]= dict(est=BernoulliNB(binarize=0.0), scale=False, te=False)  # sở hữu>0 = 1
    M["LDA"]        = dict(est=LinearDiscriminantAnalysis(), scale=True, te=True)
    # ----- Cây / ensemble -----
    M["RandomForest"] = dict(est=RandomForestClassifier(
        n_estimators=400, class_weight="balanced_subsample", min_samples_leaf=2,
        n_jobs=-1, random_state=SEED), scale=False, te=True)
    M["ExtraTrees"]   = dict(est=ExtraTreesClassifier(
        n_estimators=400, class_weight="balanced_subsample", min_samples_leaf=2,
        n_jobs=-1, random_state=SEED), scale=False, te=True)
    M["HistGB"]       = dict(est=HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, class_weight="balanced",
        random_state=SEED), scale=False, te=True)
    # ----- Boosting ngoài -----
    M["LightGBM"] = dict(est=LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=spw, random_state=SEED,
        n_jobs=-1, verbose=-1), scale=False, te=True)
    M["XGBoost"] = dict(est=XGBClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
        eval_metric="auc", random_state=SEED, n_jobs=-1), scale=False, te=True)
    M["CatBoost"] = dict(est=CatBoostClassifier(
        iterations=500, learning_rate=0.03, depth=5, auto_class_weights="Balanced",
        random_seed=SEED, verbose=0), scale=False, te=True)
    return M
