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
2. **Dòng trùng (mục 2b)** — phát hiện **651 dòng trùng** trên 85 đặc trưng (510 nhóm). Trong đó
   **49 nhóm "mâu thuẫn nhãn"** (cùng đặc trưng nhưng vừa có người mua vừa không).
   → **Quyết định: KHÔNG drop.** Đó là khách hàng thật; nhóm mâu thuẫn mang **xác suất mua theo
   profile** — chính là tín hiệu xếp hạng. Thay vào đó dùng **GroupKFold theo profile** để đánh
   giá trung thực.
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

### 5.1. Tạo đặc trưng nghiệp vụ (`build_features_v2`)
Từ 85 đặc trưng gốc, tạo thêm **~40 đặc trưng** (tất cả row-wise / target-free), gồm các nhóm:

| Nhóm | Ví dụ | Ý nghĩa |
|---|---|---|
| **Tổng hợp gốc (v1)** | `agg_total_contrib`, `agg_total_number`, `agg_n_contrib_types`, `agg_n_number_types`, `agg_car_related` | tổng phí / tổng số HĐ / **số loại BH đang sở hữu** / mức BH liên quan xe |
| **Cờ sở hữu (flag)** | `flag_co_xe`, `flag_chay_no`, `flag_nhan_tho`, `flag_tai_san`, `flag_tn_ca_nhan`, `flag_thuyen`, `flag_so_hd_oto`, `flag_da_dang_bh` | có/không từng loại BH lõi (tín hiệu ~ bằng mức ordinal — xem EDA 10c) |
| **Gộp theo lĩnh vực BH** | `dom_vehicle_*`, `dom_property_*`, `dom_life_health_*`, `dom_agriculture_*`, `dom_recreation_*`, `dom_liability_*` | tổng phí & số loại theo 6 lĩnh vực |
| **Tỉ lệ / chuẩn hoá** | `ratio_vehicle_share`, `ratio_lifehealth_share`, `ratio_contrib_per_policy`, `ratio_ntypes_balance` | cơ cấu danh mục BH |
| **Chỉ số socio** | `idx_income_level`, `idx_affluence`, `idx_education`, `idx_family`, `idx_religiosity` | tổng hợp nhân khẩu |
| **Tương tác** | `ix_pp_x_car`, `ix_income_x_pp`, `ix_affluence_x_owns`, `ix_age_x_maintype` | giao thoa có ý nghĩa |
| **Frequency-encode** | `freq_1`, `freq_5` | tần suất subtype/main type (tính trên train, **target-free**) |

→ **85 (raw) + ~40 (v2) = 125 đặc trưng** ("full v2").

### 5.2. So sánh cách mã hoá biến cardinal cao (subtype, main type)
3 cách cho cột 1 & 5: **TargetEncoder** (CV-safe) / **WOE** (CV-safe) / **frequency** (target-free)
→ chấm bằng grouped OOF, chọn cách tốt nhất. (TargetEncoder dùng làm mặc định trong pipeline.)

### 5.3. Audit chất lượng + khử đa cộng tuyến
- **Near-zero variance** (≥99% một giá trị) → gắn cờ cân nhắc loại.
- **Tương quan đa phương pháp:** Spearman/Pearson, **Cramér's V** (categorical), point-biserial/MI
  với target.
- **Khử đa cộng tuyến 2 tầng:** (a) **gom cụm** |corr|>0,9 giữ đại diện (MI cao nhất); (b)
  **VIF prune** cho khối liên tục. So grouped OOF trước/sau để chắc không tụt hiệu năng.

### 5.4. Xếp hạng importance đa phương pháp → `consensus_rank`
Tính importance bằng **7 phương pháp** rồi lấy **trung bình thứ hạng** (consensus) + độ ổn định
(`rank_std`):
1. Mutual Information 2. ANOVA F 3. point-biserial 4. L1-logistic |coef| 5. LightGBM gain
6. Permutation (grouped) 7. SHAP.
→ Xuất `outputs/feature_importance_full.csv` (125 feature × 7 phương pháp). Quét **perf theo K**
(`outputs/perf_vs_K.csv`) để thấy vùng K hiệu quả.

### 5.5. Chọn lọc & kiểm định
- So nhiều bộ ứng viên: full v2 / after-VIF / consensus top-K / **RFECV (grouped)** / L1-selected
  / **stability core** → chọn theo `meanK` (hoà → ít cột → AUC).
- **Kiểm leakage:** đặc trưng row-wise tái lập khớp; **drift PSI** train↔test thấp (ổn định).

### 5.6. Bàn giao (`outputs/feature_set.json`)
Lưu nhiều "view": `raw`, `all_engineered`, `full_v2`, `after_multicollinearity`,
**`consensus_rank`** (125 feature xếp hạng — *cái mà training dùng*), `final_selected`,
`final_name`. Kèm `train_fe.parquet` / `test_fe.parquet`.

---

## 6. Giai đoạn 3 — Training & chọn mô hình (`training.ipynb`)

Triết lý: **tune kỹ + đánh giá trung thực + chọn 1 mô hình đơn** (dễ giải thích cho Nhiệm vụ 2),
**không ensemble**.

### 6.1. Thiết lập
- `POOL = load_feature_set("consensus_rank")` (125 feature đã xếp hạng).
- `GROUPS = profile_groups(Xtr)` → grouped OOF.
- 5 mô hình: **LightGBM, XGBoost, CatBoost, LogReg, LDA**. 40 trials/model.

### 6.2. Tuning — **K (số feature) là một hyperparameter**
Mỗi mô hình một study Optuna. **Mỗi trial chọn `K∈[15,30]`** (dùng `POOL[:K]`) **+ tham số mô
hình**, chấm bằng **grouped OOF `mean_hits_over_k`**:
- Booster (LightGBM/XGBoost/CatBoost): có **early stopping**, và `inner_es=True` — early-stop
  trên một **inner split tách từ train-fold** (không đụng val-fold được chấm).
- Tuyến tính (LogReg/LDA): tune C/penalty/shrinkage, có StandardScaler trong pipeline.

→ Việc "chọn bao nhiêu feature" trở thành quyết định **được tối ưu có kiểm soát**, không chọn tay.

### 6.3. Đánh giá trung thực + chọn cuối (**1-SE rule**)
Với mỗi mô hình đã tune: chạy grouped OOF **lặp 5 seed** → `mean_hits_over_k` ± **SE**, kèm
hits@20% & AUC. Áp **1-SE rule**: trong các mô hình nằm trong 1 SE của bộ tốt nhất, ưu tiên **ít
feature hơn**, rồi **tuyến tính** (đơn giản hơn). → tránh "winner's curse" do chọn đúng đỉnh nhiễu.

### 6.4. Mô hình cuối → submission
Refit mô hình đã chọn trên **toàn bộ** train (TargetEncoder fit toàn train) → chấm điểm 4.000
test → lấy **top 800 ID** → `submission_800.txt`. Giải thích bằng **SHAP** (đầu vào Nhiệm vụ 2).

---

## 7. Kết quả

Đánh giá **grouped OOF** (trung thực) trên train:

| Mô hình | K | `mean_hits_over_k` | `hits@20%` | AUC | Chọn |
|---|---|---|---|---|---|
| **CatBoost** | **17** | **192,96 ± 1,26** | **195 / 348** | 0,786 | ✅ |
| LightGBM | 26 | 187,0 | 190 | 0,780 | |
| LDA | 18 | 183,3 | 185 | 0,757 | |
| XGBoost | 16 | 182,4 | 188 | 0,772 | |
| LogReg | 30 | 179,8 | 183 | 0,765 | |

**Mô hình chốt: CatBoost, K = 17 đặc trưng** (top-17 theo consensus). 
- `hits@20% = 195/348` trên train ⇒ **lift ≈ 2,8x** so với ngẫu nhiên.
- Kỳ vọng trong 800 ID chọn ra: **~130 người mua** (so với ~48 nếu chọn ngẫu nhiên) — *xem
  cảnh báo "cận trên" ở §9*.
- Sản phẩm nộp: **`submission_800.txt`** (800 ID, mỗi dòng 1 ID).

---

## 8. Nhiệm vụ 2 — Giải thích cho chiến dịch

Yếu tố đẩy khả năng mua AIA cao nhất (đồng thuận giữa AUC đơn biến, SHAP, consensus importance):

1. **Đã sở hữu bảo hiểm ô tô** (đóng phí & số HĐ ô tô) — tín hiệu mạnh nhất; người đã quen mua
   BH xe dễ mua chéo.
2. **Sức mua & thu nhập cao** (hạng sức mua, thu nhập TB).
3. **Độ rộng danh mục BH** — càng sở hữu nhiều loại BH càng dễ mua thêm (đơn điệu).
4. **Bảo hiểm cháy nổ / tài sản**, và một số **phân khúc nhân khẩu** (main type/subtype) có tỉ lệ
   mua vượt trội.

**Khuyến nghị hành động:** ưu tiên nhắm nhóm **đã có BH ô tô + sức mua cao + đang sở hữu nhiều
loại BH**; dùng `outputs/test_scores.csv` (điểm + thứ hạng 4.000 khách) để chia tầng chiến dịch.

---

## 9. Đánh giá trung thực & hạn chế

**Điểm mạnh (đã làm đúng):**
- ✅ **Grouped OOF** xử lý đúng rò rỉ do dòng trùng profile.
- ✅ **Encoder fit theo fold** (TargetEncoder/WOE) — không rò rỉ nhãn vào val-fold.
- ✅ Đặc trưng **row-wise / target-free** — parquet không bị leak.
- ✅ **K là hyperparameter** (không chọn tay), metric **ổn định** (`mean_hits_over_k` ± SE),
  **1-SE rule** chống winner's-curse.
- ✅ Test **không lệch** train (adversarial AUC≈0,50) → OOF là proxy hợp lệ.

**Hạn chế cần biết — con số `195` nên coi là CẬN TRÊN:**
1. **Optimism do chọn feature ngoài vòng CV (chính):** `consensus_rank` được tính trên **target
   toàn train** (cả 7 phương pháp importance dùng nhãn đầy đủ). Khi training chọn `POOL[:K]` rồi
   chấm grouped OOF, *thứ tự feature* đã "nhìn" nhãn của các dòng sẽ làm validation → OOF lạc quan
   về **chọn feature nào** (không chỉ chọn bao nhiêu). *Mức độ ở đây nhỏ* (chỉ 125 ứng viên, tín
   hiệu thật, consensus ổn định, pha loãng qua fold) nhưng **không bằng 0**. → Bản "không
   leakage" của FE notebook đúng cho *encoding/fold*, chưa tính bước *chọn feature*.
2. **Optimism do chọn max trên 5 mô hình** — báo cáo `meanK` của model thắng là max-of-5 → hơi
   lạc quan (notebook đã ghi chú "coi như cận trên").
3. **1 mô hình đơn vs ensemble:** chọn 1 model (CatBoost) là quyết định *có chủ đích* (sạch, dễ
   giải thích, parsimonious). Nhưng model-đơn-tốt-nhất-trên-OOF dễ dính winner's-curse hơn
   ensemble; **rank-average top 2–3** *có thể* ổn định hơn chút trên test ẩn (đánh đổi với tính
   diễn giải).

→ **Số thật trên 800 ID ẩn nhiều khả năng thấp hơn 195 một chút.** Submission vẫn **hợp lệ**;
chỉ là *ước lượng tự chấm* hơi cao.

**Hướng siết thêm (tuỳ chọn):** đưa bước xếp hạng feature **vào trong pipeline** (chọn top-K *bên
trong* mỗi fold bằng 1 ranker nhanh) → OOF hết optimism; hoặc đo phần chênh nested-vs-hiện-tại để
biết 195 thực rơi về đâu.

---

## 10. Cách tái lập

**Môi trường:** Python (Anaconda), **giữ `numpy < 2`** (vd 1.26.4) cho tương thích
pandas/matplotlib biên dịch cho numpy 1.x. Nếu cần SHAP, dùng bản `shap` hợp numpy 1.x.

```bash
pip install -r requirements.txt
```

**Chạy theo thứ tự** (FE phải chạy trước để sinh `outputs/feature_set.json`):
```bash
jupyter nbconvert --to notebook --execute --inplace feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace training.ipynb
# (eda.ipynb chạy độc lập, cho phân tích/biểu đồ)
```

**Regenerate notebook từ generator** (chạy từ thư mục gốc; sẽ ghi đè .ipynb, mất output cũ):
```bash
python builders/build_train_notebook.py
```

**Module dùng chung (`src/`):**

| Module | Vai trò chính |
|---|---|
| `data.py` | nạp dữ liệu từ `data/`, metadata (tên cột tiếng Việt, bảng mã) |
| `features.py` | `build_features_v2`, encoder CV-safe (TargetEncoder/WOE/frequency), khử đa cộng tuyến, `load_feature_set` |
| `cv.py` | `profile_groups`, `oof_proba` (grouped OOF), `build_pipeline` |
| `tune.py` | `tune_model_k` (K-as-hyperparameter, grouped, early stopping), `build_final_spec` |
| `metrics.py` | `hits@20%`, `mean_hits_over_k`, lift, AUC, PR-AUC |
| `models.py` | "model zoo" tham chiếu |
| `predict.py` | refit toàn train → chấm test → top-800 |

**Sản phẩm:** `submission_800.txt` (800 ID nộp bài) · `outputs/test_scores.csv` (điểm + thứ hạng
4.000 khách) · `outputs/feature_set.json` · `figures/` (EDA / FE / TR).
