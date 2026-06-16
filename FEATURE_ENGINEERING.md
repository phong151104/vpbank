# Feature Engineering — VPINS AIA Challenge

Tài liệu mô tả **chi tiết** quy trình feature engineering trong [feature_engineering.ipynb](feature_engineering.ipynb) và [src/features.py](src/features.py). Mục tiêu: từ 85 đặc trưng gốc, tạo thêm đặc trưng nghiệp vụ, phân tích tương quan & importance, **khử đa cộng tuyến**, rồi chọn ra bộ đặc trưng tối ưu bàn giao cho phần train — tất cả **không gây rò rỉ dữ liệu (leakage)** và được đánh giá **trung thực** bằng grouped cross-validation.

## Nguyên tắc xuyên suốt

1. **Không leakage:** mọi đặc trưng mới là phép biến đổi **row-wise** (theo từng dòng), **không dùng nhãn**. Vì vậy tính một lần trên cả train + test, an toàn để lưu ra file. Các encoder *phụ thuộc nhãn* (TargetEncoder, WOE) **chỉ fit bên trong pipeline theo từng fold** của CV.
2. **Đánh giá trung thực:** mọi quyết định giữ/loại đặc trưng được chấm bằng **grouped OOF** — `StratifiedGroupKFold` chia theo "profile" (các dòng trùng 85 đặc trưng nằm trọn một phía), tránh thổi phồng điểm do 651 dòng trùng (11,2%). Dùng **2 mô hình tham chiếu**: LogReg (tuyến tính) + LightGBM (cây) để đặc trưng bền cho cả hai họ.
3. **Metric:** `mean_hits_over_k` (trung bình số người mua bắt được ở top 15–25%) làm chính; phụ là `hits@20%` và ROC-AUC.

## Luồng tổng quát

```
85 đặc trưng gốc
   │  build_features_v2()  (+40 đặc trưng nghiệp vụ, row-wise)
   ▼
125 đặc trưng (RAW + v2)
   │  khử đa cộng tuyến:  gom cụm |corr|>0.9  →  VIF lặp
   ▼
78 đặc trưng (after multicollinearity)
   │  importance 7 phương pháp → consensus rank
   │  chọn lọc: consensus top-K / RFECV / L1 / stability  (validate grouped OOF)
   ▼
60 đặc trưng FINAL  ('stability core ≥0.6')  →  feature_set.json + train_fe/test_fe.parquet
```

Số cột thực tế: **RAW 85 → +40 = 125 → 78 (sau khử đa cộng tuyến) → 60 (FINAL)**.

---

## 1. Đặc trưng gốc (85)

| Nhóm | Cột | Ý nghĩa |
|---|---|---|
| Nhân khẩu học (socio) | 1–42 | Suy ra từ mã bưu chính: phân nhóm KH, tuổi, học vấn, tôn giáo, hôn nhân, nghề nghiệp, tầng lớp, nhà/xe, thu nhập |
| Sức mua | 43 | Hạng purchasing power |
| Đóng phí BH khác (contribution) | 44–64 | Mức đóng phí 21 loại bảo hiểm (thang 0–9) |
| Số HĐ BH khác (number) | 65–85 | Số hợp đồng 21 loại bảo hiểm |

**Cardinal cao (HICARD):** cột **1** (phân nhóm KH, 41 mức) và **5** (nhóm KH chính, 10 mức) — cần mã hoá đặc biệt.

---

## 2. Đặc trưng kỹ thuật tạo mới (40) — `build_features_v2()`

### 2.1 Tổng hợp gốc (`agg_*`, 5 cột)
| Tên | Công thức |
|---|---|
| `agg_total_contrib` | Σ đóng phí (cột 44–64) |
| `agg_total_number` | Σ số HĐ (cột 65–85) |
| `agg_n_contrib_types` | số loại có đóng phí ( #{cột 44–64 > 0} ) |
| `agg_n_number_types` | số loại đang sở hữu ( #{cột 65–85 > 0} ) |
| `agg_car_related` | Σ mức BH liên quan xe (47,48,49) |

### 2.2 Cờ sở hữu (`flag_*`, 8 cột)
Nhị phân "có/không" cho từng loại BH then chốt: `flag_co_xe` (47), `flag_chay_no` (59), `flag_nhan_tho` (55), `flag_tai_san` (63), `flag_tn_ca_nhan` (44), `flag_thuyen` (61), `flag_so_hd_oto` (số HĐ ô tô 68 ≥1), `flag_da_dang_bh` (sở hữu ≥3 loại BH).

### 2.3 Gộp theo lĩnh vực bảo hiểm (`dom_*`, 12 cột)
Gom 21 loại BH thành **6 lĩnh vực**; mỗi lĩnh vực có **tổng đóng phí** (`dom_<lĩnh vực>_contrib`) và **số loại sở hữu** (`dom_<lĩnh vực>_ntypes`):

| Lĩnh vực | Cột đóng phí (số HĐ = +21) |
|---|---|
| `vehicle` (xe cộ) | 47, 48, 49, 50, 51, 54 |
| `agriculture` (nông nghiệp) | 46, 52, 53 |
| `property` (nhà/tài sản) | 59, 63 |
| `life_health` (nhân thọ–sức khoẻ) | 55, 56, 57, 58, 64 |
| `recreation` (giải trí) | 60, 61, 62 |
| `liability` (trách nhiệm) | 44, 45 |

### 2.4 Tỉ lệ / chuẩn hoá (`ratio_*`, 4 cột)
| Tên | Công thức | Ý nghĩa |
|---|---|---|
| `ratio_vehicle_share` | dom_vehicle_contrib / tổng đóng phí | tỉ trọng chi cho BH xe |
| `ratio_lifehealth_share` | dom_life_health_contrib / tổng đóng phí | tỉ trọng chi cho BH nhân thọ–SK |
| `ratio_contrib_per_policy` | tổng đóng phí / tổng số HĐ | mức phí trung bình mỗi HĐ |
| `ratio_ntypes_balance` | #loại sở hữu / #loại đóng phí | cân bằng giữa số HĐ và đóng phí |

### 2.5 Chỉ số socio tổng hợp (`idx_*`, 5 cột)
| Tên | Công thức | Ý nghĩa |
|---|---|---|
| `idx_income_level` | 1·col37 + 2·col38 + 3·col39 + 4·col40 + 5·col41 | điểm mức thu nhập (gộp 5 dải %) |
| `idx_affluence` | col42 + col43 + idx_income_level/5 + col25 + col19 + col31 + col33 | **chỉ số sung túc** (thu nhập + sức mua + tầng lớp A + địa vị cao + sở hữu nhà + 2 ô tô) |
| `idx_education` | 1·col18 + 2·col17 + 3·col16 | điểm học vấn (thấp→cao) |
| `idx_family` | col10 + col15 − col13 | cấu trúc gia đình (kết hôn + có con − độc thân) |
| `idx_religiosity` | col6 + col7 + col8 − col9 | mức độ tôn giáo (có đạo − không đạo) |

### 2.6 Tương tác (`ix_*`, 4 cột)
`ix_pp_x_car` (sức mua × có BH xe), `ix_income_x_pp` (thu nhập × sức mua), `ix_affluence_x_owns` (sung túc × đã sở hữu BH), `ix_age_x_maintype` (tuổi × nhóm KH chính).

### 2.7 Frequency-encode (`freq_*`, 2 cột)
`freq_1`, `freq_5` = **tần suất** xuất hiện của mỗi mức subtype/main type (tính trên train, áp cho test). **Target-free** → an toàn lưu ra file.

> **Tại sao các đặc trưng này?** Nhóm `dom_*`/`flag_*`/`agg_*` nắm bắt **mức độ đã sở hữu bảo hiểm khác** — tín hiệu mua chéo mạnh nhất theo EDA. Nhóm `idx_*` nén các khối socio tương quan (thu nhập, tầng lớp) thành chỉ số gọn, dễ diễn giải cho phần trình bày.

---

## 3. Mã hoá biến phân loại cardinal cao (cột 1 & 5)

So sánh 3 cách (chấm bằng grouped OOF trong notebook):
- **TargetEncoder** (mặc định, CV-safe): thay mỗi mức bằng tỉ lệ mua trung bình, **cross-fit nội bộ + fit theo fold** → không leakage.
- **WOE** (`WOEEncoder`, CV-safe): log(odds) của mỗi mức, có smoothing; cũng chỉ fit trong pipeline.
- **Frequency** (target-free): dùng `freq_1`, `freq_5` đã tạo ở 2.7.

> TargetEncoder/WOE **không** được "nướng" sẵn vào file parquet (sẽ leakage); chúng nằm trong `make_preprocessor()` và được fit lại mỗi fold khi train.

---

## 4. Phân tích tương quan (đa phương pháp)

| Phương pháp | Dùng cho |
|---|---|
| **Spearman** | tương quan hạng (đặc trưng ordinal) với nhãn & giữa các đặc trưng |
| **Pearson** | tương quan tuyến tính (đặc trưng liên tục/chỉ số) |
| **Point-biserial** | liên hệ giữa biến liên tục và nhãn nhị phân |
| **Mutual Information** | quan hệ phi tuyến với nhãn (dùng `encoded_layout` để đúng thứ tự cột + đánh dấu cột rời rạc) |
| **Cramér's V** | liên hệ giữa hai biến phân loại nhân khẩu học |

Kèm heatmap toàn cục + clustermap để lộ các **cụm tương quan cao** (ví dụ điển hình: cặp Đóng phí ↔ Số HĐ cùng loại, rho ≈ 0,993).

---

## 5. Khử đa cộng tuyến (multicollinearity)

Hai bước, có so grouped OOF trước/sau để chắc **không tụt điểm**:

1. **Gom cụm theo tương quan** — `cluster_representatives()`: phân cụm phân cấp trên khoảng cách `1 − |corr|` (average linkage), gộp các đặc trưng có `|corr| > 0,9` vào một cụm, **mỗi cụm chỉ giữ 1 đại diện** = biến có Mutual Information với nhãn cao nhất. → giảm **125 → 80** cột (gộp **45** cột trùng: các cặp Đóng phí↔Số HĐ, khối thu nhập/sức mua/tầng lớp...).
2. **VIF lặp** — `vif_prune()`: với khối liên tục/chỉ số, loại dần biến có Variance Inflation Factor > 10 (cộng tuyến tuyến tính, ví dụ `agg_total_contrib` VIF≈29, `dom_vehicle_ntypes` VIF≈24, hay các dải thu nhập 37–41 cộng với `idx_income_level`). → còn **78** cột.

---

## 6. Importance đa phương pháp → xếp hạng đồng thuận

Tính importance bằng **7 phương pháp** rồi lấy **trung bình thứ hạng** (consensus rank) để giảm thiên lệch của bất kỳ phương pháp đơn lẻ:

| Loại | Phương pháp |
|---|---|
| Filter | Mutual Information, ANOVA F-test, \|point-biserial\| |
| Embedded | hệ số \|L1-logistic\|, LightGBM **gain** |
| Wrapper | **permutation importance** (grouped, scoring=AUC) |
| Giải thích | **SHAP** (mean \|SHAP value\|) |

Kết quả: bảng + biểu đồ top-20 (`figures/FE05_consensus_importance.png`) và thứ tự `consensus_rank` lưu trong `feature_set.json`.

---

## 7. Chọn lọc đặc trưng

So nhiều **bộ ứng viên**, chấm bằng harness grouped OOF của **cả 2 mô hình** tham chiếu:
- `full v2` (125), `after corr-cluster` (78), `after VIF`
- `consensus top-K` với K ∈ {20, 30, 40, 50}
- **RFECV** (đệ quy loại biến, CV theo nhóm)
- **L1-selected** (hệ số khác 0)
- **stability core** — chạy L1-logistic trên 20 mẫu con (70% dữ liệu), giữ biến **được chọn ≥ 60% số lần** → bộ lõi ổn định

**Quy tắc chọn** (sửa lỗi "thiên bộ to" của v1): tối đa `mean_hits_over_k` (2 mô hình) → hoà thì **ưu tiên ít cột hơn** → rồi AUC.

> **Kết quả thực tế:** bộ thắng là **`stability core (≥0.6)` với 60 cột** — `meanK` (2 mô hình) = **173,7**, cao nhất trong các bộ (so `full v2` 166,9). So mốc đầu: meanK **LogReg 165 → 185**, **LightGBM 158 → 162** — gọn hơn 125 cột mà **tăng** hiệu năng grouped OOF (đúng tinh thần *"less is more"* của bộ dữ liệu này).

---

## 8. Chống leakage & đánh giá trung thực

- **Row-wise:** notebook tái tính `build_features_v2` độc lập và so khớp để khẳng định đặc trưng không phụ thuộc thứ tự/nhãn.
- **Encoder trong pipeline:** TargetEncoder/WOE chỉ fit trên train-fold.
- **Grouped CV:** `StratifiedGroupKFold` theo profile (xem [src/cv.py](src/cv.py) `profile_groups`).
- **Drift train↔test:** tính **PSI** cho từng đặc trưng mới (PSI < 0,1 = ổn định) — cảnh báo nếu phân phối lệch.

---

## 9. Bàn giao (artifact)

Ghi ra để phần train dùng lại mà không cần chạy lại notebook:

| File | Nội dung |
|---|---|
| `outputs/feature_set.json` | các danh sách: `raw`, `all_engineered`, `full_v2`, `after_multicollinearity`, `consensus_rank`, **`final_selected`** (60 cột), `final_name` |
| `outputs/train_fe.parquet` | train đã gắn đặc trưng (ID + FINAL + nhãn) |
| `outputs/test_fe.parquet` | test đã gắn đặc trưng (ID + FINAL) |

**Cách dùng trong `model_v2`:**
```python
from src.features import build_features_v2, load_feature_set
Xtr, Xte, _ = build_features_v2(train, test)
COLS = load_feature_set("final_selected")   # 60 cột đã chốt
# rồi truyền COLS vào oof_proba / build_pipeline như cũ
```

---

## 10. Bảng tổng hợp số liệu

| Mốc | Số cột | grouped OOF (meanK LogReg / LightGBM) |
|---|---|---|
| Đặc trưng gốc | 85 | 165,2 / 158,4 |
| + đặc trưng nghiệp vụ v2 | +40 → **125** | — |
| sau gom cụm tương quan (\|corr\|>0.9) | 80 | — |
| sau VIF | 78 | — |
| **FINAL (`stability core ≥0.6`)** | **60** | **185,0 / 162,4** |

## Tham chiếu code

- Hàm: [src/features.py](src/features.py) — `build_features_v2`, `frequency_encode`, `cluster_representatives`, `vif_prune`, `WOEEncoder`, `make_preprocessor`, `encoded_layout`, `load_feature_set`, `INS_DOMAINS`.
- Đánh giá: [src/cv.py](src/cv.py) (`profile_groups`, `oof_proba`), [src/metrics.py](src/metrics.py) (`summarize`, `mean_hits_over_k`).
- Notebook: [feature_engineering.ipynb](feature_engineering.ipynb) — 10 mục.
- Biểu đồ: `figures/FE01..FE06` (audit, tương quan, Cramér's V, heatmap, consensus importance, so sánh chọn lọc).
