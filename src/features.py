# -*- coding: utf-8 -*-
"""Feature engineering + bộ tiền xử lý (TargetEncoder cho cột cardinal cao)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder, StandardScaler

from .data import FEATURES, CONTRIB, NUMBER, HICARD, NAME

# Cột bảo hiểm "lõi" hay liên quan tới việc mua chéo (theo EDA/nghiệp vụ)
_CAR_LIKE   = ["47", "48", "49"]        # ô tô, xe tải nhẹ, mô tô
_KEY_FLAGS  = {
    "flag_co_xe":       "47",   # đóng phí BH ô tô > 0
    "flag_chay_no":     "59",   # cháy nổ
    "flag_nhan_tho":    "55",   # nhân thọ
    "flag_tai_san":     "63",   # tài sản
    "flag_tn_ca_nhan":  "44",   # trách nhiệm bên thứ ba (cá nhân)
    "flag_thuyen":      "61",   # thuyền (hiếm, nhưng phân biệt)
}


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    """Thêm đặc trưng tổng hợp row-wise (KHÔNG dùng target -> không leakage).

    Trả về (Xtr, Xte, eng_cols) — eng_cols là danh sách cột kỹ thuật mới.
    """
    eng_cols = []
    out = {}
    for name, df in [("tr", train), ("te", test)]:
        d = df.copy()
        d["agg_total_contrib"] = d[CONTRIB].sum(axis=1)
        d["agg_total_number"]  = d[NUMBER].sum(axis=1)
        d["agg_n_contrib_types"] = (d[CONTRIB] > 0).sum(axis=1)
        d["agg_n_number_types"]  = (d[NUMBER]  > 0).sum(axis=1)
        d["agg_car_related"]     = d[_CAR_LIKE].sum(axis=1)
        for fname, col in _KEY_FLAGS.items():
            d[fname] = (d[col] > 0).astype(int)
        # tương tác: sức mua x sở hữu BH xe ; thu nhập x sức mua
        d["ix_pp_x_car"]     = d["43"] * d["flag_co_xe"]
        d["ix_income_x_pp"]  = d["42"] * d["43"]
        out[name] = d
    eng_cols = (["agg_total_contrib", "agg_total_number", "agg_n_contrib_types",
                 "agg_n_number_types", "agg_car_related"]
                + list(_KEY_FLAGS.keys()) + ["ix_pp_x_car", "ix_income_x_pp"])
    # tên hiển thị cho đặc trưng mới
    NAME.update({
        "agg_total_contrib": "Tổng mức đóng phí BH khác",
        "agg_total_number": "Tổng số HĐ BH khác",
        "agg_n_contrib_types": "Số loại BH có đóng phí",
        "agg_n_number_types": "Số loại BH đang sở hữu",
        "agg_car_related": "Mức BH liên quan xe",
        "flag_co_xe": "Có BH ô tô", "flag_chay_no": "Có BH cháy nổ",
        "flag_nhan_tho": "Có BH nhân thọ", "flag_tai_san": "Có BH tài sản",
        "flag_tn_ca_nhan": "Có BH trách nhiệm cá nhân", "flag_thuyen": "Có BH thuyền",
        "ix_pp_x_car": "Sức mua × có BH xe", "ix_income_x_pp": "Thu nhập × sức mua",
    })
    return out["tr"], out["te"], eng_cols


# ---- Các bộ đặc trưng để thử nghiệm ----------------------------------------
def feature_sets(eng_cols):
    """Định nghĩa các 'feature set' để so sánh."""
    return {
        "raw":     FEATURES,                 # 85 gốc
        "eng":     FEATURES + eng_cols,      # 85 + tổng hợp
    }


def make_preprocessor(cols, use_target_encoding=True, scale=False):
    """ColumnTransformer: TargetEncoder cho cột cardinal cao (subtype/main type),
    còn lại passthrough; tuỳ chọn StandardScaler (cho mô hình tuyến tính/NB)."""
    te_cols = [c for c in HICARD if c in cols]
    rest    = [c for c in cols if c not in te_cols]
    transformers = []
    if te_cols:
        transformers.append(("te", TargetEncoder(target_type="binary", random_state=42), te_cols))
    transformers.append(("pass", "passthrough", rest))
    ct = ColumnTransformer(transformers, remainder="drop")
    if scale:
        from sklearn.pipeline import Pipeline
        return Pipeline([("ct", ct), ("scale", StandardScaler())])
    return ct
