# -*- coding: utf-8 -*-
"""Metric cho bài toán xếp hạng top-K (top 800/4000 = top 20%)."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

TOPK_FRAC = 0.20  # 800/4000


def _topk_idx(y_score, frac=TOPK_FRAC, k=None):
    n = len(y_score)
    k = k if k is not None else int(np.ceil(frac * n))
    return np.argsort(-np.asarray(y_score))[:k], k


def topk_hits(y_true, y_score, frac=TOPK_FRAC, k=None):
    """Số dương nằm trong top-K theo điểm dự đoán."""
    idx, _ = _topk_idx(y_score, frac, k)
    return int(np.asarray(y_true)[idx].sum())


def lift_at_k(y_true, y_score, frac=TOPK_FRAC, k=None):
    """Lift = (precision trong top-K) / (tỉ lệ dương nền)."""
    y_true = np.asarray(y_true)
    idx, kk = _topk_idx(y_score, frac, k)
    base = y_true.mean()
    if base == 0:
        return 0.0
    return (y_true[idx].mean()) / base


def recall_at_k(y_true, y_score, frac=TOPK_FRAC, k=None):
    y_true = np.asarray(y_true)
    idx, _ = _topk_idx(y_score, frac, k)
    tot = y_true.sum()
    return float(y_true[idx].sum() / tot) if tot else 0.0


def mean_hits_over_k(y_true, y_score, fracs=(0.15, 0.175, 0.20, 0.225, 0.25)):
    """Trung bình hits@K qua một dải K (mặc định 15%..25%).

    Tiêu chí chọn model/ensemble ỔN ĐỊNH hơn hits@20% đơn lẻ (vốn nhiễu ±std)."""
    return float(np.mean([topk_hits(y_true, y_score, f) for f in fracs]))


def summarize(y_true, y_score, frac=TOPK_FRAC):
    """Bảng tóm tắt: AUC, AP(PR-AUC), hits@K, recall@K, lift@K."""
    return {
        "AUC":        roc_auc_score(y_true, y_score),
        "AP":         average_precision_score(y_true, y_score),
        "hits@20%":   topk_hits(y_true, y_score, frac),
        "recall@20%": recall_at_k(y_true, y_score, frac),
        "lift@20%":   lift_at_k(y_true, y_score, frac),
    }
