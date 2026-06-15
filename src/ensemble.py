# -*- coding: utf-8 -*-
"""Kết hợp mô hình: rank-averaging (bền cho ranking) + stacking LR."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from .metrics import summarize


def to_rank(score):
    """Đổi điểm -> thứ hạng chuẩn hoá [0,1] (1 = cao nhất)."""
    r = rankdata(score, method="average")
    return r / len(r)


def rank_average(oof_dict, names, weights=None):
    """Trung bình thứ hạng của các mô hình được chọn."""
    weights = weights or [1.0] * len(names)
    agg = np.zeros(len(next(iter(oof_dict.values()))))
    for w, n in zip(weights, names):
        agg += w * to_rank(oof_dict[n])
    return agg / sum(weights)


def stack_oof(oof_dict, names, y, seed=42):
    """Stacking: LR (meta) học trên ma trận OOF proba. Trả OOF của stack."""
    M = np.column_stack([oof_dict[n] for n in names])
    meta = LogisticRegression(max_iter=2000, class_weight="balanced")
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    stack = cross_val_predict(meta, M, y, cv=skf, method="predict_proba")[:, 1]
    return stack, meta


def compare_ensembles(oof_dict, top_names, y):
    """So sánh: từng mô hình đơn, rank-average, stacking."""
    rows = []
    for n in top_names:
        s = summarize(y, oof_dict[n]); s["combo"] = n; rows.append(s)
    ra = rank_average(oof_dict, top_names)
    s = summarize(y, ra); s["combo"] = "Rank-Avg (" + "+".join(top_names) + ")"; rows.append(s)
    st, _ = stack_oof(oof_dict, top_names, y)
    s = summarize(y, st); s["combo"] = "Stacking-LR"; rows.append(s)
    res = pd.DataFrame(rows).set_index("combo").sort_values("hits@20%", ascending=False)
    return res, ra, st
