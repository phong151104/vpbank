# -*- coding: utf-8 -*-
"""Sinh dự đoán cuối: refit toàn train -> chấm điểm 4.000 test -> top 800 ID."""
import numpy as np
import pandas as pd
from pathlib import Path

from .cv import build_pipeline
from .data import TARGET


def fit_full(spec, Xtr, y, cols):
    pipe = build_pipeline(spec, cols)
    pipe.fit(Xtr, y)
    return pipe


def score_test(pipe, Xte):
    return pipe.predict_proba(Xte)[:, 1]


def make_submission(test_scores, test_ids, k=800, out_path="submission_800.txt",
                    full_csv="outputs/test_scores.csv"):
    """test_scores: mảng điểm cho 4.000 khách; test_ids: cột ID tương ứng.
    Ghi file top-k ID (1 ID/dòng) + CSV toàn bộ điểm đã xếp hạng."""
    df = pd.DataFrame({"ID": np.asarray(test_ids), "score": np.asarray(test_scores)})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    top = df.head(k)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True) if Path(out_path).parent != Path("") else None
    Path(full_csv).parent.mkdir(parents=True, exist_ok=True)
    top["ID"].to_csv(out_path, index=False, header=False)
    df.to_csv(full_csv, index=False)
    return top, df


def verify_submission(out_path="submission_800.txt", k=800, id_min=1, id_max=4000):
    ids = pd.read_csv(out_path, header=None)[0].tolist()
    ok = {
        "đủ k dòng": len(ids) == k,
        "ID duy nhất": len(set(ids)) == len(ids),
        "trong [1,4000]": all(id_min <= i <= id_max for i in ids),
    }
    return ok, ids
