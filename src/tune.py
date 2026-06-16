# -*- coding: utf-8 -*-
"""Tuning siêu tham số bằng Optuna (objective = OOF hits@top20%).

Boosting dùng early stopping trong từng fold; tiền xử lý (TargetEncoder) fit
riêng từng fold để không rò rỉ. Trả về (best_params, best_value, mean_best_iter).
"""
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from .features import make_preprocessor
from .metrics import topk_hits, mean_hits_over_k

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42


def _cv_oof(make_model, fit_kwargs_fn, X, y, cols, n_splits, seed, groups=None, inner_es=False):
    """OOF + best_iteration trung bình, có early stopping, tiền xử lý theo fold.

    - groups != None: StratifiedGroupKFold (cùng profile -> cùng fold), tránh rò rỉ.
    - inner_es=True: early-stopping trên inner split TÁCH từ train-fold (tiền xử lý cũng
      fit trên inner-train), chấm điểm trên val-fold chưa đụng -> bỏ leak "early-stop trên
      chính fold được chấm". Mặc định False giữ nguyên hành vi cũ."""
    if groups is None:
        split_iter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y)
    else:
        split_iter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                          random_state=seed).split(X, y, groups)
    oof = np.zeros(len(y)); best_iters = []
    for tr, va in split_iter:
        if inner_es:
            tr2, es = train_test_split(tr, test_size=0.15, random_state=seed, stratify=y.iloc[tr])
            pre = make_preprocessor(cols, scale=False)
            Xtr = pre.fit_transform(X.iloc[tr2], y.iloc[tr2])
            Xes = pre.transform(X.iloc[es]); Xva = pre.transform(X.iloc[va])
            model = make_model()
            model.fit(Xtr, y.iloc[tr2], **fit_kwargs_fn(Xes, y.iloc[es]))
        else:
            pre = make_preprocessor(cols, scale=False)
            Xtr = pre.fit_transform(X.iloc[tr], y.iloc[tr])
            Xva = pre.transform(X.iloc[va])
            model = make_model()
            model.fit(Xtr, y.iloc[tr], **fit_kwargs_fn(Xva, y.iloc[va]))
        oof[va] = model.predict_proba(Xva)[:, 1]
        bi = getattr(model, "best_iteration_", None)
        if bi is None:
            bi = getattr(model, "best_iteration", None)
        if bi:
            best_iters.append(int(bi))
    return oof, (int(np.mean(best_iters)) if best_iters else None)


def _objective_factory(name, X, y, cols, spw, n_splits, seed,
                       groups=None, objective="hits", inner_es=False):
    def _obj(trial):   # tên KHÁC tham số `objective` để không che mất chuỗi metric
        if name == "LightGBM":
            params = dict(
                n_estimators=2000, learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                num_leaves=trial.suggest_int("num_leaves", 15, 127),
                max_depth=trial.suggest_int("max_depth", 3, 12),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                scale_pos_weight=trial.suggest_categorical("scale_pos_weight", [1.0, spw, np.sqrt(spw)]),
                subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1)
            mk = lambda: LGBMClassifier(**params)
            fk = lambda Xv, yv: dict(eval_set=[(Xv, yv)], eval_metric="auc",
                                     callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])
        elif name == "XGBoost":
            params = dict(
                n_estimators=2000, learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                gamma=trial.suggest_float("gamma", 1e-3, 5, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                scale_pos_weight=trial.suggest_categorical("scale_pos_weight", [1.0, spw, np.sqrt(spw)]),
                tree_method="hist", eval_metric="auc", early_stopping_rounds=60,
                random_state=seed, n_jobs=-1)
            mk = lambda: XGBClassifier(**params)
            fk = lambda Xv, yv: dict(eval_set=[(Xv, yv)], verbose=False)
        elif name == "CatBoost":
            params = dict(
                iterations=2000, learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                depth=trial.suggest_int("depth", 4, 10),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                random_strength=trial.suggest_float("random_strength", 1e-3, 10, log=True),
                auto_class_weights="Balanced", random_seed=seed, verbose=0)
            mk = lambda: CatBoostClassifier(**params)
            fk = lambda Xv, yv: dict(eval_set=(Xv, yv), early_stopping_rounds=60, verbose=0)
        else:
            raise ValueError(name)
        oof, _ = _cv_oof(mk, fk, X, y, cols, n_splits, seed, groups=groups, inner_es=inner_es)
        if objective == "auc":
            return roc_auc_score(y, oof)
        if objective == "ap":
            return average_precision_score(y, oof)
        if objective == "hits_meanK":
            return mean_hits_over_k(y, oof)
        return topk_hits(y, oof)   # mặc định "hits"
    return _obj


def tune_model(name, X, y, cols, pos_rate, n_trials=40, n_splits=5, seed=SEED,
               groups=None, objective="hits", inner_es=False):
    """objective: 'hits' (cũ) | 'auc' | 'ap' | 'hits_meanK'. groups != None -> GroupKFold.
    inner_es=True -> early-stopping trên inner split (bỏ leak). Mặc định = hành vi cũ."""
    spw = (1 - pos_rate) / pos_rate
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(_objective_factory(name, X, y, cols, spw, n_splits, seed,
                                      groups=groups, objective=objective, inner_es=inner_es),
                   n_trials=n_trials, show_progress_bar=False)
    # tính best_iteration trung bình với bộ tham số tốt nhất (để refit không cần early stop)
    bp = study.best_params
    mean_iter = _best_iter_for(name, bp, X, y, cols, spw, n_splits, seed,
                               groups=groups, inner_es=inner_es)
    return bp, study.best_value, mean_iter, study


def _best_iter_for(name, bp, X, y, cols, spw, n_splits, seed, groups=None, inner_es=False):
    if name == "LightGBM":
        mk = lambda: LGBMClassifier(n_estimators=2000, subsample_freq=1, random_state=seed,
                                    n_jobs=-1, verbose=-1, **bp)
        fk = lambda Xv, yv: dict(eval_set=[(Xv, yv)], eval_metric="auc",
                                 callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])
    elif name == "XGBoost":
        mk = lambda: XGBClassifier(n_estimators=2000, tree_method="hist", eval_metric="auc",
                                   early_stopping_rounds=60, random_state=seed, n_jobs=-1, **bp)
        fk = lambda Xv, yv: dict(eval_set=[(Xv, yv)], verbose=False)
    elif name == "CatBoost":
        mk = lambda: CatBoostClassifier(iterations=2000, auto_class_weights="Balanced",
                                        random_seed=seed, verbose=0, **bp)
        fk = lambda Xv, yv: dict(eval_set=(Xv, yv), early_stopping_rounds=60, verbose=0)
    _, mi = _cv_oof(mk, fk, X, y, cols, n_splits, seed, groups=groups, inner_es=inner_es)
    return mi


def build_tuned_estimator(name, best_params, mean_iter, pos_rate, seed=SEED):
    """Tạo estimator với tham số đã tune + số vòng cố định (refit toàn train)."""
    n_iter = int(mean_iter * 1.1) if mean_iter else 600  # +10% vì train đầy đủ hơn fold
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=n_iter, subsample_freq=1, random_state=seed,
                              n_jobs=-1, verbose=-1, **best_params)
    if name == "XGBoost":
        bp = {k: v for k, v in best_params.items()}
        return XGBClassifier(n_estimators=n_iter, tree_method="hist", eval_metric="auc",
                             random_state=seed, n_jobs=-1, **bp)
    if name == "CatBoost":
        return CatBoostClassifier(iterations=n_iter, auto_class_weights="Balanced",
                                  random_seed=seed, verbose=0, **best_params)
    raise ValueError(name)


# ============================================================================
#  Tuning với K (số feature top-consensus) là HYPERPARAMETER
#  + hỗ trợ mô hình tuyến tính (LogReg, LDA). KHÔNG ensemble.
# ============================================================================
_BOOSTERS = ("LightGBM", "XGBoost", "CatBoost")


def _suggest_boost(name, trial, spw, seed):
    """(params, make_model, fit_kwargs_fn) cho boosting — KHÔNG gồm 'K'."""
    if name == "LightGBM":
        params = dict(
            n_estimators=2000, learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            scale_pos_weight=trial.suggest_categorical("scale_pos_weight", [1.0, spw, float(np.sqrt(spw))]),
            subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1)
        mk = lambda: LGBMClassifier(**params)
        fk = lambda Xv, yv: dict(eval_set=[(Xv, yv)], eval_metric="auc",
                                 callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])
    elif name == "XGBoost":
        params = dict(
            n_estimators=2000, learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            gamma=trial.suggest_float("gamma", 1e-3, 5, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            scale_pos_weight=trial.suggest_categorical("scale_pos_weight", [1.0, spw, float(np.sqrt(spw))]),
            tree_method="hist", eval_metric="auc", early_stopping_rounds=60, random_state=seed, n_jobs=-1)
        mk = lambda: XGBClassifier(**params)
        fk = lambda Xv, yv: dict(eval_set=[(Xv, yv)], verbose=False)
    elif name == "CatBoost":
        params = dict(
            iterations=2000, learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            random_strength=trial.suggest_float("random_strength", 1e-3, 10, log=True),
            auto_class_weights="Balanced", random_seed=seed, verbose=0)
        mk = lambda: CatBoostClassifier(**params)
        fk = lambda Xv, yv: dict(eval_set=(Xv, yv), early_stopping_rounds=60, verbose=0)
    else:
        raise ValueError(name)
    return params, mk, fk


def _linear_spec(name, p):
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    if name == "LogReg":
        return {"est": LogisticRegression(max_iter=3000, solver="liblinear", **p), "scale": True, "te": True}
    if name == "LDA":
        return {"est": LinearDiscriminantAnalysis(solver="lsqr", **p), "scale": True, "te": True}
    raise ValueError(name)


def _suggest_linear(name, trial):
    if name == "LogReg":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", None])}
    if name == "LDA":
        return {"shrinkage": trial.suggest_float("shrinkage", 0.0, 1.0)}
    raise ValueError(name)


def _grouped_oof(spec, X, y, cols, n_splits, seed, groups, n_repeats=1):
    """OOF grouped cho mô hình tuyến tính (không early stopping)."""
    from .cv import build_pipeline
    from sklearn.model_selection import cross_val_predict
    pipe = build_pipeline(spec, cols)
    oof = np.zeros(len(y))
    for r in range(n_repeats):
        if groups is None:
            cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
            p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        else:
            cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed + r)
            p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", groups=groups, n_jobs=1)[:, 1]
        oof += p
    return oof / n_repeats


def _score_obj(objective, y, oof):
    if objective == "auc": return roc_auc_score(y, oof)
    if objective == "ap":  return average_precision_score(y, oof)
    if objective == "hits_meanK": return mean_hits_over_k(y, oof)
    return topk_hits(y, oof)


def tune_model_k(name, X, y, pool, pos_rate, k_range=(15, 30), n_trials=40,
                 n_splits=5, seed=SEED, groups=None, objective="hits_meanK"):
    """Tune model với K (số feature top-consensus) là hyperparameter.

    pool = danh sách feature đã xếp hạng (consensus_rank); mỗi trial dùng pool[:K].
    Trả (best_params_không_K, best_K, best_value, mean_iter|None, study)."""
    spw = (1 - pos_rate) / pos_rate
    is_boost = name in _BOOSTERS

    def _obj(trial):
        K = trial.suggest_int("K", k_range[0], k_range[1])
        cols = pool[:K]
        if is_boost:
            _, mk, fk = _suggest_boost(name, trial, spw, seed)
            oof, _ = _cv_oof(mk, fk, X, y, cols, n_splits, seed, groups=groups, inner_es=True)
        else:
            spec = _linear_spec(name, _suggest_linear(name, trial))
            oof = _grouped_oof(spec, X, y, cols, n_splits, seed, groups)
        return _score_obj(objective, y, oof)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(_obj, n_trials=n_trials, show_progress_bar=False)
    bp = dict(study.best_params); K = int(bp.pop("K"))
    cols = pool[:K]
    mean_iter = (_best_iter_for(name, bp, X, y, cols, spw, n_splits, seed, groups=groups, inner_es=True)
                 if is_boost else None)
    return bp, K, study.best_value, mean_iter, study


def build_final_spec(name, best_params, K, mean_iter, pos_rate, seed=SEED):
    """spec (cho fit_full/oof_proba) từ kết quả tune_model_k."""
    if name in _BOOSTERS:
        est = build_tuned_estimator(name, best_params, mean_iter, pos_rate, seed)
        return {"est": est, "scale": False, "te": True}
    return _linear_spec(name, best_params)
