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


def encoded_layout(cols):
    """Khớp THỨ TỰ output của make_preprocessor: [TE(HICARD) liên tục] + [phần còn lại rời rạc].

    Trả (ordered_cols, discrete_mask). Dùng cho mutual_info_classif: vừa gán đúng tên cột
    (ColumnTransformer ĐẢO thứ tự, đặt cột TargetEncoder lên đầu) vừa đánh dấu cột rời rạc."""
    te_cols = [c for c in HICARD if c in cols]
    rest = [c for c in cols if c not in te_cols]
    ordered = te_cols + rest
    mask = [False] * len(te_cols) + [True] * len(rest)
    return ordered, mask


def extra_flags(df):
    """Đặc trưng flag thử nghiệm (P5, opt-in) — trả DataFrame cùng index với df.

    Notebook chỉ ghép thêm nếu OOF (grouped) cải thiện."""
    out = pd.DataFrame(index=df.index)
    out["flag_so_hd_oto"]    = (df["68"] > 0).astype(int)                 # có >=1 HĐ ô tô
    out["flag_da_dang_bh"]   = ((df[NUMBER] > 0).sum(axis=1) >= 3).astype(int)  # sở hữu >=3 loại BH
    NAME.update({"flag_so_hd_oto": "Có HĐ ô tô (≥1)", "flag_da_dang_bh": "Sở hữu ≥3 loại BH"})
    return out


# ============================================================================
#  FEATURE ENGINEERING v2 (nghiệp vụ) + tiện ích chọn lọc / khử đa cộng tuyến
#  Mọi đặc trưng v2 là ROW-WISE, KHÔNG dùng target -> an toàn để ghi parquet.
#  (Encoder phụ thuộc target: TargetEncoder/WOE chỉ fit trong pipeline theo fold.)
# ============================================================================

# Nhóm bảo hiểm theo lĩnh vực (chỉ số CỘT đóng phí; số HĐ = +21)
INS_DOMAINS = {
    "vehicle":     [47, 48, 49, 50, 51, 54],   # ô tô, tải nhẹ, mô tô, tải lớn, rơ-moóc, moped
    "agriculture": [46, 52, 53],               # TN nông nghiệp, máy kéo, máy nông nghiệp
    "property":    [59, 63],                    # cháy nổ, tài sản
    "life_health": [55, 56, 57, 58, 64],        # nhân thọ, TN cá nhân, TN gia đình, khuyết tật, an sinh
    "recreation":  [60, 61, 62],                # ván lướt sóng, thuyền, xe đạp
    "liability":   [44, 45],                    # TN bên thứ ba (cá nhân / DN)
}
_DOMAIN_VN = {"vehicle": "xe cộ", "agriculture": "nông nghiệp", "property": "nhà/tài sản",
              "life_health": "nhân thọ-sức khoẻ", "recreation": "giải trí", "liability": "trách nhiệm"}


def frequency_encode(train, test, cols=HICARD):
    """Frequency encoding (TARGET-FREE): tần suất giá trị tính trên train, áp cho test."""
    tr, te = train.copy(), test.copy()
    for c in cols:
        freq = tr[c].value_counts(normalize=True)
        tr["freq_" + c] = tr[c].map(freq).astype(float)
        te["freq_" + c] = te[c].map(freq).fillna(0.0).astype(float)
        NAME["freq_" + c] = "Tần suất " + NAME.get(c, c)
    return tr, te


def build_features_v2(train, test):
    """Tạo toàn bộ đặc trưng nghiệp vụ (v1 + v2). Trả (Xtr, Xte, eng_cols).

    eng_cols = danh sách cột kỹ thuật (không gồm 85 cột gốc). An toàn leakage:
    chỉ phép biến đổi row-wise + frequency-encode (tính trên train, không dùng target)."""
    Xtr, Xte, eng1 = build_features(train, test)
    EXtr, EXte = extra_flags(Xtr), extra_flags(Xte)
    Xtr = pd.concat([Xtr, EXtr], axis=1); Xte = pd.concat([Xte, EXte], axis=1)
    eng1 = eng1 + list(EXtr.columns)

    eps = 1e-6
    new_cols = []
    for df in (Xtr, Xte):
        # --- Gộp theo lĩnh vực bảo hiểm ---
        for dom, cidx in INS_DOMAINS.items():
            ccols = [str(c) for c in cidx]; ncols = [str(c + 21) for c in cidx]
            df[f"dom_{dom}_contrib"] = df[ccols].sum(axis=1)
            df[f"dom_{dom}_ntypes"]  = (df[ncols] > 0).sum(axis=1)
        tot_c = df[CONTRIB].sum(axis=1); tot_n = df[NUMBER].sum(axis=1)
        # --- Tỉ lệ / chuẩn hoá ---
        df["ratio_vehicle_share"]     = df["dom_vehicle_contrib"] / (tot_c + eps)
        df["ratio_lifehealth_share"]  = df["dom_life_health_contrib"] / (tot_c + eps)
        df["ratio_contrib_per_policy"]= tot_c / (tot_n + eps)
        df["ratio_ntypes_balance"]    = (df[NUMBER] > 0).sum(axis=1) / ((df[CONTRIB] > 0).sum(axis=1) + eps)
        # --- Chỉ số socio tổng hợp ---
        df["idx_income_level"]  = 1*df["37"] + 2*df["38"] + 3*df["39"] + 4*df["40"] + 5*df["41"]
        df["idx_affluence"]     = df["42"] + df["43"] + df["idx_income_level"]/5.0 + df["25"] + df["19"] + df["31"] + df["33"]
        df["idx_education"]      = 1*df["18"] + 2*df["17"] + 3*df["16"]
        df["idx_family"]        = df["10"] + df["15"] - df["13"]
        df["idx_religiosity"]   = df["6"] + df["7"] + df["8"] - df["9"]
        # --- Tương tác có ý nghĩa ---
        df["ix_affluence_x_owns"] = df["idx_affluence"] * (tot_n > 0).astype(int)
        df["ix_age_x_maintype"]   = df["4"] * df["5"]

    new_cols = ([f"dom_{d}_contrib" for d in INS_DOMAINS] + [f"dom_{d}_ntypes" for d in INS_DOMAINS]
                + ["ratio_vehicle_share", "ratio_lifehealth_share", "ratio_contrib_per_policy",
                   "ratio_ntypes_balance", "idx_income_level", "idx_affluence", "idx_education",
                   "idx_family", "idx_religiosity", "ix_affluence_x_owns", "ix_age_x_maintype"])
    # tên tiếng Việt cho biểu đồ
    for d in INS_DOMAINS:
        NAME[f"dom_{d}_contrib"] = f"Tổng phí BH {_DOMAIN_VN[d]}"
        NAME[f"dom_{d}_ntypes"]  = f"Số loại BH {_DOMAIN_VN[d]}"
    NAME.update({
        "ratio_vehicle_share": "Tỉ trọng phí BH xe", "ratio_lifehealth_share": "Tỉ trọng phí BH nhân thọ-SK",
        "ratio_contrib_per_policy": "Phí/HĐ trung bình", "ratio_ntypes_balance": "Cân bằng #loại số HĐ/đóng phí",
        "idx_income_level": "Chỉ số mức thu nhập", "idx_affluence": "Chỉ số sung túc (affluence)",
        "idx_education": "Chỉ số học vấn", "idx_family": "Chỉ số gia đình", "idx_religiosity": "Chỉ số tôn giáo",
        "ix_affluence_x_owns": "Sung túc × đã có BH", "ix_age_x_maintype": "Tuổi × nhóm KH chính",
    })
    # frequency encoding (target-free) cho subtype/main type
    Xtr, Xte = frequency_encode(Xtr, Xte, HICARD)
    new_cols += ["freq_" + c for c in HICARD]
    return Xtr, Xte, eng1 + new_cols


def feature_groups_v2(eng_cols):
    """Gom đặc trưng v2 theo nhóm để vẽ/diễn giải."""
    g = {"Đóng phí lĩnh vực": [c for c in eng_cols if c.startswith("dom_") and c.endswith("contrib")],
         "Số loại lĩnh vực":  [c for c in eng_cols if c.startswith("dom_") and c.endswith("ntypes")],
         "Tỉ lệ":             [c for c in eng_cols if c.startswith("ratio_")],
         "Chỉ số socio":      [c for c in eng_cols if c.startswith("idx_")],
         "Tương tác":         [c for c in eng_cols if c.startswith("ix_")],
         "Cờ (flag)":         [c for c in eng_cols if c.startswith("flag_")],
         "Tổng hợp gốc":      [c for c in eng_cols if c.startswith("agg_")],
         "Frequency-encode":  [c for c in eng_cols if c.startswith("freq_")]}
    return {k: v for k, v in g.items() if v}


# ---- Khử đa cộng tuyến --------------------------------------------------------
def cluster_representatives(X, cols, score, threshold=0.90, method="spearman"):
    """Gom cụm theo |corr|>threshold (hierarchical, average-linkage trên 1-|corr|).

    Mỗi cụm giữ 1 đại diện = biến có `score` (vd MI với target) cao nhất.
    Trả (kept_cols, clusters: rep -> [members])."""
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import squareform
    cols = list(cols)
    if len(cols) < 2:
        return cols, {c: [c] for c in cols}
    corr = X[cols].corr(method=method).abs().fillna(0.0)
    dist = 1.0 - corr.values
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    Z = sch.linkage(squareform(dist, checks=False), method="average")
    labels = sch.fcluster(Z, t=1.0 - threshold, criterion="distance")
    kept, clusters = [], {}
    for lab in np.unique(labels):
        members = [cols[i] for i in range(len(cols)) if labels[i] == lab]
        rep = max(members, key=lambda c: score.get(c, 0.0))
        kept.append(rep); clusters[rep] = members
    return kept, clusters


def vif_prune(X, cols, thresh=10.0, max_remove=None):
    """Loại lặp biến có VIF cao nhất tới khi mọi VIF <= thresh (cho khối tuyến tính)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    cur = list(cols); removed = []
    while len(cur) > 2:
        M = X[cur].astype(float).values
        M = np.column_stack([np.ones(len(M)), M])
        vifs = np.array([variance_inflation_factor(M, i + 1) for i in range(len(cur))])
        j = int(np.nanargmax(vifs))
        if vifs[j] > thresh and (max_remove is None or len(removed) < max_remove):
            removed.append((cur.pop(j), float(vifs[j])))
        else:
            break
    return cur, removed


def load_feature_set(key="final_selected", path=None):
    """Đọc danh sách đặc trưng đã chốt từ outputs/feature_set.json."""
    import json
    from .data import ROOT
    p = path or (ROOT / "outputs" / "feature_set.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)[key]


# ---- WOE encoder CV-safe (dùng trong pipeline, KHÔNG bake vào parquet) --------
from sklearn.base import BaseEstimator, TransformerMixin


class WOEEncoder(BaseEstimator, TransformerMixin):
    """Weight-of-Evidence encoder cho cột rời rạc (fit cần y -> CV-safe trong pipeline)."""
    def __init__(self, smoothing=0.5):
        self.smoothing = smoothing

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y).astype(float)
        pos, neg = y.sum(), len(y) - y.sum()
        self.maps_ = []
        for j in range(X.shape[1]):
            col = X[:, j]; m = {}
            for v in np.unique(col):
                mask = col == v
                p = (y[mask].sum() + self.smoothing) / (pos + 2 * self.smoothing)
                n = ((~y[mask].astype(bool)).sum() + self.smoothing) / (neg + 2 * self.smoothing)
                m[v] = float(np.log(p / n))
            self.maps_.append(m)
        return self

    def transform(self, X):
        X = np.asarray(X); out = np.zeros(X.shape, dtype=float)
        for j in range(X.shape[1]):
            m = self.maps_[j]
            out[:, j] = [m.get(v, 0.0) for v in X[:, j]]
        return out
