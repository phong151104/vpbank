# VPINS — AIA Challenge: Báo cáo kỹ thuật chi tiết

> Tài liệu mô tả **đề bài** và **toàn bộ cách giải** theo từng bước: EDA → Feature
> Engineering → Training. Dành cho data scientist đọc hiểu và tái lập. Phần cuối có
> **đánh giá trung thực & hạn chế**.

---

## Mục lục
1. [Bài toán](#1-bài-toán)
2. [Dữ liệu](#2-dữ-liệu)
3. [Tổng quan giải pháp](#3-tổng-quan-giải-pháp)
4. [Giai đoạn 1 — EDA](#4-giai-đoạn-1--eda-edaipynb)
5. [Giai đoạn 2 — Feature Engineering](#5-giai-đoạn-2--feature-engineering-feature_engineeringipynb)
6. [Giai đoạn 3 — Training & chọn mô hình](#6-giai-đoạn-3--training--chọn-mô-hình-trainingipynb)
7. [Kết quả](#7-kết-quả)
8. [Nhiệm vụ 2 — Giải thích cho chiến dịch](#8-nhiệm-vụ-2--giải-thích-cho-chiến-dịch)
9. [Đánh giá trung thực & hạn chế](#9-đánh-giá-trung-thực--hạn-chế)
10. [Cách tái lập](#10-cách-tái-lập)

---

## 1. Bài toán

**Bối cảnh.** VPINS (bộ phận bảo hiểm của VPB) muốn đẩy mạnh một sản phẩm bảo hiểm sức khoẻ
("AIA policy"). Cần giúp giám đốc **xác định khách hàng nào sẵn sàng mua** để triển khai chiến
dịch hiệu quả.

**Hai nhiệm vụ:**
- **Nhiệm vụ 1 — Dự đoán/xếp hạng:** từ **4.000 khách** trong `test_data.txt`, chọn ra **đúng
  800 ID** triển vọng nhất. Chấm điểm = số người **thực sự mua** nằm trong 800 ID đó (đối chiếu
  với tập nhãn ẩn). → Đây là **bài toán xếp hạng top-K (top 800/4000 = top 20%)**, KHÔNG phải
  phân loại theo ngưỡng 0.5.
- **Nhiệm vụ 2 — Giải thích:** chỉ ra **vì sao** khách mua → insight *dễ hiểu, hữu ích, hành
  động được* cho chiến dịch.

**Vì sao là bài toán xếp hạng (không phải phân loại):** chỉ ~6% khách mua → accuracy vô nghĩa
(đoán "không mua" hết đã đạt ~94%). Cái cần là **đưa người dễ mua lên đầu danh sách**. Do đó
metric chính là **`hits@20%`** (số người mua bắt được trong top 20%) và **`mean_hits_over_k`**
(trung bình hits ở top 15–25%, ổn định hơn); phụ trợ: ROC-AUC, PR-AUC, lift.

---

## 2. Dữ liệu

Bộ **CoIL Challenge 2000 / TIC Benchmark** (nhãn gốc "caravan policy", đổi tên thành "AIA").

| | Train | Test |
|---|---|---|
| Số khách | 5.822 | 4.000 |
| Đặc trưng | 85 (+ nhãn cột 86) | 85 (không nhãn) |
| Tỉ lệ mua | **5,98%** (348/5.822) | ẩn |

**85 đặc trưng** chia 2 khối:
- **Cột 1–43 — Nhân khẩu-xã hội học** (suy từ **mã bưu chính** → khách cùng vùng có giá trị
  giống hệt): phân nhóm KH (subtype, 41 mức), nhóm chính (main type, 10 mức), tuổi, tôn giáo,
  hôn nhân, học vấn, nghề nghiệp, tầng lớp xã hội, sở hữu nhà/xe, thu nhập, hạng sức mua.
- **Cột 44–85 — Sở hữu sản phẩm bảo hiểm:** mức **đóng phí** (44–64) và **số hợp đồng** (65–85)
  của 21 loại bảo hiểm.

**Đặc điểm quan trọng (chi phối toàn bộ thiết kế):**
- Phần lớn đặc trưng là **mã hoá thứ bậc (ordinal 0–9)** — % dân cư theo vùng, hoặc khoảng tiền
  được rời rạc hoá. Chỉ rất ít cột là nhị phân thật.
- Dữ liệu **rất thưa** (zero-inflation cao ở khối bảo hiểm).
- **Dòng trùng:** do nhân khẩu suy từ zip + rời rạc hoá thô → **651 dòng trùng hoàn toàn** trên
  85 đặc trưng (510 "profile", 5.171 profile duy nhất / 5.822). → Phải xử lý cẩn thận khi
  cross-validation (xem §4).

---

## 3. Tổng quan giải pháp

Pipeline **3 notebook tách bạch**, bàn giao qua artifact:

```
 eda.ipynb            feature_engineering.ipynb        training.ipynb
(hiểu dữ liệu)   →   (tạo + xếp hạng đặc trưng)    →   (tune + chọn model + nộp)
                              │                               ▲
                              └──  outputs/feature_set.json ──┘
```

**Nguyên tắc xuyên suốt (để con số đáng tin):**
- **Đánh giá bằng grouped OOF** — `StratifiedGroupKFold` theo *profile* (bộ 85 đặc trưng giống
  nhau) → các dòng trùng **không** nằm cả ở train lẫn validation → không rò rỉ kiểu "đã thấy
  dòng gần y hệt".
- **Encoder phụ thuộc target fit theo fold** — `TargetEncoder`/`WOE` chỉ fit trong pipeline trên
  train-fold → không rò rỉ nhãn vào val-fold.
- **Đặc trưng mới là row-wise / target-free** (kể cả frequency-encoding tính trên train) → an
  toàn để tính trên cả train+test, ghi ra parquet.
- **Metric ổn định:** `mean_hits_over_k` thay vì `hits@20%` đơn lẻ (vốn nhiễu ±vài đơn vị).

---

## 4. Giai đoạn 1 — EDA (`eda.ipynb`)

Mục tiêu: hiểu dữ liệu, phát hiện cạm bẫy, định hướng FE & chiến lược validation.

**Các bước & phát hiện chính:**

1. **Chất lượng dữ liệu** — không thiếu giá trị, đúng dải codebook, không cột hằng số.
2. **Dòng trùng (mục 2b)** — **651 dòng trùng** trên 85 đặc trưng, thuộc **510 nhóm** (405 nhóm 2
   người, 77 nhóm 3, 21 nhóm 4, 6 nhóm 5, 1 nhóm 6); chỉ **5.171 hồ sơ duy nhất / 5.822**. Trong đó
   **49 nhóm "mâu thuẫn nhãn"** (cùng đặc trưng nhưng vừa có người mua vừa không — sai số Bayes).
   → **Quyết định: KHÔNG drop** (drop bỏ 32/348 người mua, méo xác suất profile). Dùng **GroupKFold
   theo profile** để đánh giá trung thực (dòng cùng profile không nằm cả train lẫn validation).
3. **Biến mục tiêu** — mất cân bằng nặng (~6%) → khẳng định khung **ranking top-20%**.
4. **Univariate / Bivariate / Multivariate** — phân phối; tương quan với nhãn; phát hiện cặp
   *Đóng phí ↔ Số HĐ* cùng loại BH **trùng lặp mạnh** (đa cộng tuyến).
5. **Phân khúc khách hàng** — tỉ lệ mua theo main type (A3) / subtype (A1), bản đồ nhiệt
   Tuổi×Nhóm, Thu nhập×Sức mua → actionable cho Nhiệm vụ 2.
6. **Tín hiệu dự đoán nâng cao (mục 10):**
   - **10a — Adversarial validation:** huấn luyện model phân biệt train vs test → **AUC ≈ 0,50,
     PSI < 0,02** ⇒ **test KHÔNG lệch phân phối** so với train. *Hệ quả: CV nội bộ là proxy
     đáng tin cho test; không cần reweight.*
   - **10b — Độ phân giải & tổng quát hoá:** "tra cứu hồ sơ khớp-chính-xác" (CV) chỉ đạt **78**
     ≈ ngẫu nhiên (**70**) vì hồ sơ gần như duy nhất; còn mô hình thật đạt ~**162** ⇒ giá trị
     nằm ở **tổng quát hoá theo lân cận**, không phải tra cứu.
   - **10c — Tín hiệu bắc cầu FE:** % mua **tăng đơn điệu theo số loại BH đang sở hữu** (→ biện
     minh đặc trưng `agg_n_*`); **flag(>0) giữ gần hết tín hiệu** so với mức ordinal (→ biện
     minh `flag_*`); feature top **đơn điệu** (→ cây dùng raw tốt).
   - **10d — AUC đơn biến:** mạnh nhất là **Đóng phí/Số HĐ BH ô tô**, thu nhập, sức mua, cháy nổ.
   - **10e — Baseline CV trung thực:** ~**162** hits@20% (lift ~2,3x) làm mốc cho phần train.

---

## 5. Giai đoạn 2 — Feature Engineering (`feature_engineering.ipynb`)

Notebook **độc lập**, không train mô hình cuối; chỉ tạo + xếp hạng + chọn đặc trưng, rồi **bàn
giao**. Mọi quyết định chấm bằng **grouped OOF** với 2 mô hình tham chiếu (LogReg + LightGBM).

### 5.1. Tạo đặc trưng nghiệp vụ (`build_features_v2`): 85 → 125
Từ 85 đặc trưng gốc, tạo thêm **40 đặc trưng** (tất cả row-wise / target-free), gồm 8 nhóm:

| Nhóm | Số | Ví dụ | Ý nghĩa |
|---|---|---|---|
| **Tổng hợp gốc (v1)** | 5 | `agg_total_contrib`, `agg_n_contrib_types`, `agg_n_number_types`, `agg_car_related` | tổng phí / **số loại BH đang sở hữu** / mức BH liên quan xe |
| **Cờ sở hữu (flag)** | 8 | `flag_co_xe`, `flag_chay_no`, `flag_nhan_tho`, `flag_tai_san`, `flag_so_hd_oto`, `flag_da_dang_bh` | có/không từng loại BH lõi (tín hiệu ~ bằng mức ordinal — EDA 10c) |
| **Gộp theo lĩnh vực BH** | 6+6 | `dom_vehicle_*`, `dom_property_*`, `dom_life_health_*`, … (×{contrib, ntypes}) | tổng phí & số loại theo 6 lĩnh vực |
| **Tỉ lệ / chuẩn hoá** | 4 | `ratio_vehicle_share`, `ratio_contrib_per_policy`, `ratio_ntypes_balance` | cơ cấu danh mục BH |
| **Chỉ số socio** | 5 | `idx_income_level`, `idx_affluence`, `idx_education`, `idx_family`, `idx_religiosity` | tổng hợp nhân khẩu |
| **Tương tác** | 4 | `ix_pp_x_car`, `ix_income_x_pp`, `ix_affluence_x_owns`, `ix_age_x_maintype` | giao thoa có ý nghĩa |
| **Frequency-encode** | 2 | `freq_1`, `freq_5` | tần suất subtype/main type (target-free) |

→ **85 (raw) + 40 (v2) = 125 đặc trưng** ("full v2"). Tác động (grouped OOF, 2 mô hình tham chiếu):

| Bộ | #cột | LogReg `meanK` | LightGBM `meanK` |
|---|---|---|---|
| raw (85) | 85 | 163,2 | 163,2 |
| eng v1 | 98 | 171,2 | 166,8 |
| **full v2** | 125 | **177,2** | 164,2 |

→ FE v2 nâng rõ cho **mô hình tuyến tính** (163 → 177); cây ít hưởng lợi hơn (chịu được trùng lặp).

### 5.2. So sánh mã hoá biến cardinal cao (subtype col 1, main type col 5)
3 cách (grouped OOF, LightGBM):

| Encoding | `hits@20%` | `meanK` | AUC |
|---|---|---|---|
| **TargetEncoder** (mặc định) | 164 | 164,2 | 0,743 |
| WOE | 159 | 159,8 | 0,744 |
| frequency | 163 | 162,4 | 0,745 |

→ Chênh nhỏ; chọn **TargetEncoder** (fit theo fold trong pipeline → CV-safe).

### 5.3. Audit chất lượng + khử đa cộng tuyến
- **Near-zero variance** (≥99% một giá trị): **20 cột** (vd ván lướt sóng, xe tải lớn, máy nông
  nghiệp — gần như toàn 0) → gắn cờ.
- **Đa cộng tuyến (gom cụm |corr|>0,9):** **29 cụm** có >1 thành viên → gộp giảm **45 cột** (125→80);
  **VIF prune** loại thêm **2** (Tổng phí xe cộ, Số loại xe cộ) → 78. So grouped OOF:
  full 170,7 → after-corr-cluster 169,2 → after-VIF 165,6 (gần như không tụt; VIF tụt nhẹ).
- *Cặp dư thừa điển hình:* Đóng phí ↔ Số HĐ cùng loại BH; subtype ↔ main type (|corr|=0,99);
  Sở hữu nhà ↔ Nhà thuê (1,00); BHYT tư nhân ↔ BHYT quốc gia (1,00).

### 5.4. Xếp hạng importance đa phương pháp → `consensus_rank` (+ khử đa cộng tuyến)
Importance bằng **7 phương pháp** → trung bình thứ hạng (consensus) + độ ổn định `rank_std`:
**MI, ANOVA F, point-biserial, L1-logistic, LightGBM gain, Permutation (grouped), SHAP**
→ `outputs/feature_importance_full.csv` (125 feature × 7 phương pháp; chênh consensus max 123,4 vs
min 17,1; top-10 TB 105,6 vs bottom-10 24,5).

Sau đó **khử đa cộng tuyến NGAY trên ranking** (greedy: duyệt consensus cao→thấp, bỏ feature nào
|corr|>0,9 với feature đã-giữ → **giữ "thằng consensus cao hơn"**): **125 → 80 feature (bỏ 45)**.
Ví dụ bỏ: *Số HĐ ô tô* ~ *Đóng phí ô tô* (0,95); *subtype* ~ *main type* (0,99); *Sở hữu nhà* ~
*Nhà thuê* (1,00). → **Đây là `consensus_rank` bàn giao cho training.**

**Hiệu năng grouped OOF theo K** (consensus đã dedupe, TB 2 mô hình — `outputs/perf_vs_K.csv`):

| K | 5 | 10 | **15** | 20 | 25 | 30 | 40 | 60 | 80 |
|---|---|---|---|---|---|---|---|---|---|
| `meanK_avg` | 164,6 | 171,3 | **175,7** | 172,5 | 170,7 | 170,5 | 167,3 | 170,1 | 168,7 |

→ Đỉnh **K=15**, vùng tốt **K≈10–20** → chốt dải tune `K∈[10,30]`.

### 5.5. Chọn lọc đặc trưng (chẩn đoán) + kiểm định leakage
So các bộ ứng viên (grouped OOF, TB 2 mô hình):

| Bộ | #cột | `meanK` | hits_LGB | AUC |
|---|---|---|---|---|
| **RFECV (grouped)** | 45 | **179,8** | 166 | 0,766 |
| stability core (≥0,6) | 60 | 175,9 | 173 | 0,761 |
| L1-selected | 72 | 172,6 | 164 | 0,754 |
| consensus top-20 | 20 | 172,5 | 174 | 0,756 |
| full v2 | 125 | 170,7 | 164 | 0,751 |

> **Lưu ý:** training **KHÔNG** dùng `final_selected` (RFECV-45) mà dùng **`consensus_rank` (80 đã
> dedupe)** rồi **tự tune K** — linh hoạt hơn cố định 1 bộ. Phần chọn-lọc này giữ vai trò *chẩn đoán*.
- **Kiểm leakage:** đặc trưng row-wise tái lập khớp (`True`); **PSI** train↔test max **0,0052**
  (<0,1 → **không drift**); test **0 NaN**.

### 5.6. Bàn giao (`outputs/feature_set.json`)
Lưu nhiều "view": `raw` (85), `all_engineered` (40), `full_v2` (125), `after_multicollinearity` (78),
**`consensus_rank` (80 — đã xếp hạng + khử đa cộng tuyến, *cái training dùng*)**, `final_selected`
(RFECV-45, chẩn đoán). Kèm `train_fe.parquet` / `test_fe.parquet`.

---

## 6. Giai đoạn 3 — Training & chọn mô hình (`training.ipynb`)

Triết lý: **tune kỹ + đánh giá trung thực + chọn 1 mô hình đơn** (dễ giải thích cho Nhiệm vụ 2),
**không ensemble**.

### 6.1. Thiết lập
- `POOL = load_feature_set("consensus_rank")` (**80 feature — đã xếp hạng + khử đa cộng tuyến**).
- `GROUPS = profile_groups(Xtr)` → grouped OOF.
- 5 mô hình: **LightGBM, XGBoost, CatBoost, LogReg, LDA**. 40 trials/model.

### 6.2. Tuning — **K (số feature) là một hyperparameter**
Mỗi mô hình một study Optuna. **Mỗi trial chọn `K∈[10,30]`** (dùng `POOL[:K]`) **+ tham số mô
hình**, chấm bằng **grouped OOF `mean_hits_over_k`**:
- Booster (LightGBM/XGBoost/CatBoost): có **early stopping**, và `inner_es=True` — early-stop
  trên một **inner split tách từ train-fold** (không đụng val-fold được chấm).
- Tuyến tính (LogReg/LDA): tune C/penalty/shrinkage, có StandardScaler trong pipeline.

→ Việc "chọn bao nhiêu feature" trở thành quyết định **được tối ưu có kiểm soát**, không chọn tay.

**Kết quả tuning (best K + objective `meanK` 1-split + tham số chính):**

| Mô hình | best K | `meanK`(obj) | Tham số chính | Thời gian |
|---|---|---|---|---|
| LightGBM | **14** | 189,2 | lr 0,046 · num_leaves 57 · depth 11 · min_child 100 · `spw`=1,0 | 76s |
| XGBoost | 17 | 192,4 | lr 0,040 · depth 7 · min_child_w 10 · `spw`=3,97 | 64s |
| CatBoost | 14 | 193,4 | lr 0,098 · depth 4 · l2 19,6 | 156s |
| LogReg | 29 | 182,6 | C 0,35 · L1 · class_weight None | 68s |
| LDA | 30 | 182,8 | shrinkage 0,26 | 73s |

*(Đáng chú ý: 3/5 model chọn K nhỏ **14–17** — khớp đỉnh perf-vs-K ở §5.4; `spw` (cân bằng lớp)
được tune tự do, LightGBM/CatBoost chọn ~không đè trọng số → xếp hạng ít phụ thuộc resampling.)*

### 6.3. Đánh giá trung thực + chọn cuối (**1-SE rule**)
Với mỗi mô hình đã tune: chạy grouped OOF **lặp 5 seed** → `mean_hits_over_k` ± **SE**, kèm
hits@20% & AUC. Áp **1-SE rule**: trong các mô hình nằm trong 1 SE của bộ tốt nhất, ưu tiên **ít
feature hơn**, rồi **tuyến tính** (đơn giản hơn). → tránh "winner's curse" do chọn đúng đỉnh nhiễu.

### 6.4. Mô hình cuối → submission
Refit mô hình đã chọn trên **toàn bộ** train (TargetEncoder fit toàn train) → chấm điểm 4.000
test → lấy **top 800 ID** → `submission_800.txt`. Giải thích bằng **SHAP** (đầu vào Nhiệm vụ 2).

---

## 7. Kết quả

Đánh giá lặp **5 seed** (grouped OOF trung thực, `K` đã tune; sắp theo `meanK`):

| Mô hình | K | `mean_hits_over_k` ± SE | `hits@20%` | AUC | Chọn |
|---|---|---|---|---|---|
| **LightGBM** | **14** | **195,12 ± 0,98** | **206 / 348** | 0,784 | ✅ |
| XGBoost | 17 | 192,60 ± 0,95 | 199 | 0,789 | |
| CatBoost | 14 | 190,32 ± 0,93 | 191 | 0,790 | |
| LDA | 30 | 182,60 ± 0,14 | 186 | 0,758 | |
| LogReg | 29 | 181,36 ± 0,52 | 184 | 0,765 | |

**Mô hình chốt: LightGBM, K = 14 đặc trưng** (top-14 của `consensus_rank` đã khử đa cộng tuyến).
LightGBM **cao nhất** (195,12) **và** là model **duy nhất** nằm trong ngưỡng 1-SE (194,1) → **1-SE
rule chọn dứt khoát**, không tranh cãi.
- Tham số: `lr=0,046 · num_leaves=57 · max_depth=11 · min_child_samples=100 · scale_pos_weight=1,0`.
- `hits@20% = 206/348` trên train ⇒ **lift ≈ 2,96×** so với ngẫu nhiên.
- Kỳ vọng trong 800 ID: **~141 người mua** (so với ~48 nếu ngẫu nhiên) — *xem "cận trên" ở §9*.
- Sản phẩm nộp: **`submission_800.txt`** (top-5 ID điểm cao nhất: `1468, 2844, 1243, 166, 2622`).

> *So phương án trước (CatBoost K=17, **full 125 feature**: meanK 192,96 / hits 195): khử đa cộng
> tuyến + K thấp giúp **LightGBM/XGBoost bật lên** (LightGBM 187,0→195,1), CatBoost hơi giảm
> (192,96→190,3) — đúng kỳ vọng (cây vốn chịu được trùng lặp; dedupe lợi cho model nhạy với nó).*

---

> Chi tiết đầy đủ: **[docs/MODEL_INSIGHTS.md](MODEL_INSIGHTS.md)**. Tóm tắt:

**A. Vì sao — 14 yếu tố từ SHAP** (mô hình cuối, `figures/TR04_explain.png`), gộp 3 nhóm:
1. **"Đã là người mua bảo hiểm"** (mạnh nhất): Đóng phí BH ô tô, Tổng phí BH khác, Đóng phí cháy nổ,
   Số loại BH đang có, Tỉ trọng phí BH xe, Đóng phí BH thuyền → **đẩy ⬆️**.
2. **"Sung túc"**: Thu nhập×sức mua, Thu nhập TB, Học vấn cao → ⬆️; Thu nhập <30k, Học vấn thấp → kéo ⬇️.
3. **"Nhân khẩu"**: Nhóm KH chính, Đã kết hôn, Quản lý cấp trung.

**B. Nhắm AI — tỉ lệ mua theo phân khúc** (mốc nền **5,98%**):

| Phân khúc | % mua | Bội số nền |
|---|---|---|
| **Có đóng phí BH ô tô** (vs không 2,5%) | **9,3%** | 3,7× |
| Sở hữu **≥3 loại BH** (3/4/5/6 loại) | 11–15% | ~2–2,5× |
| Main type **"Người vươn lên"** | 13,1% | 2,2× |
| Subtype **"Middle class families"** | 15,0% | 2,5× |
| Subtype "Affluent young families" | 14,4% | 2,4× |

**C. Khuyến nghị hành động:** ưu tiên giao của các tín hiệu — *khách **đã có BH ô tô** + **sở hữu
≥3 loại BH** + phân khúc **sung túc/gia đình trẻ trung lưu***. Dùng `outputs/test_scores.csv` (điểm
+ thứ hạng 4.000 khách) chia tầng gọi (top 200 → 500 → 800). Góc tiếp cận: **bán chéo gói hợp nhất**.

> ⚠️ SHAP là *attribution của mô hình* (tương quan), **không phải nhân quả** — đừng ép khách mua BH
> ô tô để họ mua AIA; cả hai cùng phản ánh *khách có thói quen & khả năng mua bảo hiểm*.

