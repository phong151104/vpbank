# -*- coding: utf-8 -*-
"""Cross-validation: dựng pipeline, sinh OOF, chạy cả model zoo."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     cross_val_predict)

from .features import make_preprocessor
from .metrics import summarize
from .data import FEATURES

SEED = 42


def profile_groups(X, cols=FEATURES):
    """Mã nhóm (int) theo 'profile' = bộ giá trị giống nhau trên `cols` (mặc định 85 đặc trưng).

    Dùng làm `groups` cho StratifiedGroupKFold để các dòng trùng đặc trưng không
    nằm cả ở train lẫn validation (tránh rò rỉ — xem EDA mục dòng trùng)."""
    return X.groupby(list(cols)).ngroup().values


def build_pipeline(spec, cols):
    """spec: dict từ get_models(); cols: danh sách cột đầu vào."""
    pre = make_preprocessor(cols, use_target_encoding=spec.get("te", True),
                            scale=spec.get("scale", False))
    return Pipeline([("pre", pre), ("clf", clone(spec["est"]))])


def oof_proba(spec, X, y, cols, n_splits=5, n_repeats=3, seed=SEED, n_jobs=1, groups=None):
    """OOF probability (trung bình qua n_repeats lần chia fold khác nhau).

    Nếu `groups` != None: dùng StratifiedGroupKFold (cùng profile -> cùng fold) để
    đánh giá trung thực. Mặc định None giữ nguyên StratifiedKFold như trước."""
    pipe = build_pipeline(spec, cols)
    oof = np.zeros(len(y), dtype=float)
    per_repeat = []
    for r in range(n_repeats):
        if groups is None:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        else:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba",
                              n_jobs=n_jobs, groups=groups)[:, 1]
        oof += p
        per_repeat.append(summarize(y, p)["hits@20%"])
    oof /= n_repeats
    return oof, per_repeat


def run_zoo(models, X, y, cols, n_splits=5, n_repeats=3, seed=SEED, verbose=True, groups=None):
    """Chạy toàn bộ model zoo, trả (bảng kết quả, dict OOF proba)."""
    rows, oof_dict = [], {}
    for name, spec in models.items():
        oof, per_rep = oof_proba(spec, X, y, cols, n_splits, n_repeats, seed, groups=groups)
        oof_dict[name] = oof
        s = summarize(y, oof)
        s["hits@20%_std"] = float(np.std(per_rep))
        s["model"] = name
        rows.append(s)
        if verbose:
            print(f"{name:14s} AUC={s['AUC']:.4f}  AP={s['AP']:.4f}  "
                  f"hits@20%={s['hits@20%']:>3d} (±{s['hits@20%_std']:.1f})  lift={s['lift@20%']:.2f}")
    res = pd.DataFrame(rows).set_index("model").sort_values("hits@20%", ascending=False)
    return res, oof_dict
