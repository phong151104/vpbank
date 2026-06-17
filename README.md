# VPINS — AIA Insurance Challenge

Giải pháp Data Science cho bài toán: **giúp VPINS nhắm đúng khách hàng cho chiến dịch bán bảo
hiểm AIA**. Tài liệu này mô tả đầy đủ bối cảnh, dữ liệu, toàn bộ luồng xử lý, các quyết định kỹ
thuật quan trọng (kèm lý do), kết quả, và insight phục vụ chiến dịch.

> 📄 Báo cáo kỹ thuật sâu hơn (kèm biểu đồ): [docs/SUBMISSION_REPORT.md](docs/SUBMISSION_REPORT.md) (có bản PDF).

---

## 1. Bối cảnh & kết quả tóm tắt (cho người quản lý)

**Vấn đề.** Gọi điện/chào mời toàn bộ tệp khách hàng rất tốn kém trong khi tỉ lệ mua AIA chỉ
**~6%**. VPINS cần một cách **chấm điểm và xếp hạng** khách hàng để chiến dịch tập trung vào nhóm
triển vọng nhất.

**Hai nhiệm vụ của đề bài:**
1. **Dự đoán** — từ **4.000 khách** trong tập test, chọn ra **800 khách (20%) triển vọng nhất**.
   Điểm = số khách **thật sự mua** nằm trong 800 người ta chọn.
2. **Giải thích** — chỉ ra **vì sao** khách mua, để bộ phận kinh doanh biết nhắm vào nhóm nào và
   truyền thông thông điệp gì.

**Kết quả đạt được:**

| Chỉ số | Giá trị | Ý nghĩa kinh doanh |
|---|---|---|
| Mô hình chốt | **LightGBM**, 14 đặc trưng | Gọn, dễ giải thích |
| Bắt được người mua trong nhóm top 20% | **~206 / 348** (≈ 59%) | Nhắm 1/5 tệp đã "vợt" gần 60% người mua |
| **Lift** | **~2,96×** | Hiệu quả gấp ~3 lần chào mời ngẫu nhiên |
| Kỳ vọng trong **800 khách chọn** | **~141 người mua** | So với ~48 nếu chọn ngẫu nhiên |

→ **Sản phẩm nộp:** `submission_800.txt` (800 ID khách hàng triển vọng nhất).
→ **Insight chiến dịch:** xem [Mục 7](#7-nhiệm-vụ-2--vì-sao-khách-mua--khuyến-nghị-chiến-dịch).

---

## 2. Dữ liệu

| Tập | Số khách | Cột | Ghi chú |
|---|---|---|---|
| `data/train_data.txt` | 5.822 | ID + 85 đặc trưng + nhãn | Có nhãn để học |
| `data/test_data.txt` | 4.000 | ID + 85 đặc trưng | **Không nhãn**, để dự đoán |

- **85 đặc trưng** gồm 2 khối: **nhân khẩu học** (cột 1–42, suy ra từ **mã bưu chính**) và **sở
  hữu bảo hiểm khác** (cột 43–85: mức đóng phí + số hợp đồng của 21 loại bảo hiểm).
- **Đặc điểm quan trọng (rút từ EDA):**
  - **Mất cân bằng nặng:** chỉ **348/5.822 = 5,98%** khách mua AIA (tỉ lệ ~1:15,7).
  - **Sạch:** 0 giá trị thiếu, đúng codebook, gần như toàn bộ là biến mã hoá khoảng/đếm.
  - **Dòng trùng "profile":** do nhân khẩu học theo zip code, **651 dòng (11,2%)** trùng y hệt
    85 đặc trưng với dòng khác → ảnh hưởng cách đánh giá (xem [Mục 5.1](#51-đánh-giá-trung-thực--grouped-oof-chống-rò-rỉ)).
  - **Dư thừa:** cặp *mức đóng phí* ↔ *số hợp đồng* của cùng loại bảo hiểm gần như trùng nhau
    (tương quan ~0,99) → cần khử đa cộng tuyến.

> Đây là bộ dữ liệu **CoIL Challenge 2000 / TIC Benchmark** quen thuộc (nhãn gốc "caravan", đổi
> tên thành "AIA").

---

## 3. Bài toán được mô hình hoá thế nào

Vì mục tiêu là **chọn top 800/4.000 (= top 20%)**, đây là **bài toán XẾP HẠNG (ranking)**, không
phải phân loại theo ngưỡng 0.5. Do đó:

- **Không dùng accuracy** (đoán "tất cả không mua" đã đạt 94% nhưng vô dụng).
- **Metric chính:** `hits@20%` (số người mua bắt được trong top 20%), `mean_hits_over_k` (trung
  bình hits ở top 15–25% — ổn định hơn), và `lift@20%`.

---

## 4. Luồng giải pháp — 3 notebook

```
eda.ipynb  →  feature_engineering.ipynb  →  training.ipynb
(hiểu dữ liệu)   (tạo + xếp hạng feature)      (tune + chọn model → nộp)
                        │                              ▲
                        └── outputs/feature_set.json ──┘   (bàn giao)
```

### 4.1 `eda.ipynb` — Phân tích khám phá

Hiểu dữ liệu trước khi mô hình hoá:
- **Chất lượng dữ liệu:** kiểm tra thiếu/trùng/ngoài dải; phát hiện **dòng trùng profile** và
  **cặp đặc trưng dư thừa**.
- **Biến mục tiêu:** xác nhận mất cân bằng 5,98% → khung hoá thành bài ranking.
- **Univariate/Bivariate:** phân phối từng đặc trưng; **lift theo mức** & tương quan với nhãn →
  đặc trưng nào phân biệt người mua.
- **Phân khúc khách hàng:** tỉ lệ mua theo nhóm khách (subtype/main type), tuổi, thu nhập, sức mua.
- **Insight nâng cao:** kiểm tra **drift train↔test** (adversarial validation), xác nhận tín hiệu
  để bắc cầu sang feature engineering.

### 4.2 `feature_engineering.ipynb` — Tạo & chọn lọc đặc trưng

1. **Tạo đặc trưng nghiệp vụ** (`build_features_v2`, +40 đặc trưng, **row-wise, không leakage**):
   gộp bảo hiểm theo **lĩnh vực** (xe cộ / nhà-tài sản / nhân thọ-sức khoẻ / nông nghiệp / giải
   trí / trách nhiệm), **tỉ lệ** (tỉ trọng chi cho từng nhóm), **chỉ số tổng hợp** (affluence –
   mức sung túc, học vấn, gia đình), **tương tác** (thu nhập × sức mua), và mã hoá biến phân loại
   (TargetEncoder/WOE **CV-safe**, frequency-encode). → **85 → 125 đặc trưng**.
2. **Xếp hạng importance đa phương pháp → consensus:** 7 phương pháp (Mutual Information, ANOVA F,
   point-biserial, hệ số L1, LightGBM gain, permutation importance, SHAP) → **trung bình thứ hạng**
   để không phụ thuộc một phương pháp đơn lẻ.
3. **Khử đa cộng tuyến trên ranking:** gom cụm các đặc trưng |corr|>0,9 (giữ đại diện theo
   importance) + lọc VIF → **125 → 80 đặc trưng** (lưu thành `consensus_rank`).
4. **Bàn giao:** ghi `outputs/feature_set.json` (+ `train_fe/test_fe.parquet`) cho bước training.

### 4.3 `training.ipynb` — Huấn luyện, tinh chỉnh & chọn mô hình

- **Quét số đặc trưng K∈[10,30] như một HYPERPARAMETER** trong Optuna: mỗi thử nghiệm chọn K (lấy
  `consensus_rank` top-K) + bộ tham số mô hình, **chấm bằng grouped OOF `mean_hits_over_k`**.
- **Tinh chỉnh 5 mô hình** (~40 thử nghiệm/mô hình): **LightGBM, XGBoost, CatBoost** (có early
  stopping) + **LogReg, LDA**.
- **Chọn 1 mô hình cuối** theo **1-SE rule** (không ensemble — để dễ giải thích cho Nhiệm vụ 2).
- **Sinh file nộp:** refit mô hình trên toàn bộ train → chấm điểm 4.000 test → lấy **top 800** →
  `submission_800.txt`; kèm **SHAP** giải thích.

---

## 5. Các quyết định kỹ thuật then chốt (và **lý do**)

### 5.1 Đánh giá trung thực — grouped OOF (chống rò rỉ)
Vì có **dòng trùng profile**, nếu chia cross-validation ngẫu nhiên thì cùng một khách có thể nằm
cả ở train lẫn validation → mô hình "thấy đáp án", điểm bị **thổi phồng**. Giải pháp: dùng
**`StratifiedGroupKFold` theo profile** (các dòng cùng 85 đặc trưng buộc nằm cùng một phía) → điểm
**đáng tin**. Mọi quyết định chọn feature/model đều đo trên grouped OOF.

### 5.2 Metric ổn định — `mean_hits_over_k`
`hits@20%` đơn lẻ khá nhiễu (ít dương). Dùng **trung bình hits ở dải top 15–25%** + báo cáo **sai
số chuẩn (SE)** để so sánh mô hình công bằng, ổn định.

### 5.3 Chọn feature bằng consensus + khử đa cộng tuyến
Mỗi phương pháp importance có thiên lệch riêng → **đồng thuận 7 phương pháp** cho ranking bền hơn.
Khử đa cộng tuyến để bỏ đặc trưng dư thừa (cặp đóng phí↔số HĐ), giúp mô hình gọn & ổn định.

### 5.4 K (số feature) là hyperparameter
Thay vì cố định số feature một cách cảm tính, **để Optuna tự tối ưu K trong [10,30]** cùng tham số
mô hình. Phân tích cho thấy **chỉ ~14–20 đặc trưng đã bắt gần hết tín hiệu** ("less is more" —
đúng đặc thù bộ dữ liệu này).

### 5.5 1-SE rule & không ensemble
Khi nhiều mô hình ngang nhau trong phạm vi nhiễu (1 SE), ưu tiên mô hình **đơn giản/ít feature
hơn**. Chọn **1 mô hình đơn** thay vì ensemble để **dễ giải thích** (quan trọng cho Nhiệm vụ 2).

### 5.6 Ghi chú trung thực (tránh "nói quá")
`hits@20%=206` nên coi là **cận trên**: có optimism nhẹ do (a) xếp hạng feature trên toàn train và
(b) chọn max trên 5 mô hình. Con số trên tập nhãn ẩn sẽ thấp hơn ~vài điểm — nhưng **ưu thế của
luồng vẫn đúng**.

---

## 6. Kết quả

**So sánh 5 mô hình (grouped OOF, đã tinh chỉnh):**

| Mô hình | K | `meanK` ± SE | `hits@20%` | AUC |
|---|---|---|---|---|
| **LightGBM** ✅ (chốt) | 14 | **195,12 ± 0,98** | **206** / 348 | 0,784 |
| XGBoost | 17 | 192,60 ± 0,95 | 199 | 0,789 |
| CatBoost | 14 | 190,32 ± 0,93 | 191 | 0,790 |
| LDA (tuyến tính) | 30 | 182,60 ± 0,14 | 186 | 0,758 |
| LogReg (tuyến tính) | 29 | 181,36 ± 0,52 | 184 | 0,765 |

- **Mô hình cuối: LightGBM với 14 đặc trưng** (consensus top-14, đã khử đa cộng tuyến).
- **Lift ~2,96×** → trong 800 khách chọn kỳ vọng **~141 người mua** (so với ~48 nếu ngẫu nhiên).
- Sản phẩm: **`submission_800.txt`** (800 ID) + `outputs/test_scores.csv` (điểm xếp hạng đủ 4.000).

---

## 7. Nhiệm vụ 2 — Vì sao khách mua & khuyến nghị chiến dịch

**14 đặc trưng mô hình cuối dựa vào** (chính là các yếu tố dẫn dắt khả năng mua AIA):

| # | Yếu tố | Nhóm |
|---|---|---|
| 1 | **Đã đóng phí bảo hiểm ô tô** | Sở hữu BH khác |
| 2 | **Tổng mức đóng phí bảo hiểm khác** | Sở hữu BH khác |
| 3 | **Nhóm khách hàng chính (main type)** | Phân khúc |
| 4 | **Đã có bảo hiểm cháy nổ** | Sở hữu BH khác |
| 5 | **Thu nhập × sức mua** | Mức sung túc |
| 6 | Học vấn (thấp) | Nhân khẩu học |
| 7 | Mức thu nhập trung bình | Mức sung túc |
| 8 | Số loại bảo hiểm đang đóng phí | Đa dạng sở hữu |
| 9 | Bảo hiểm thuyền | Sở hữu BH khác |
| 10 | Đã kết hôn | Nhân khẩu học |
| 11–14 | Thu nhập <30k, quản lý cấp trung, học vấn cao, tỉ trọng chi cho BH xe | — |

**Chân dung khách hàng triển vọng nhất:** người **đã sở hữu nhiều bảo hiểm khác** (đặc biệt
**ô tô, cháy nổ, tài sản**), có **mức sung túc cao** (thu nhập × sức mua), thuộc các **phân khúc
khách hàng** nhất định, đã lập gia đình.

**Khuyến nghị hành động:**
- **Ưu tiên cross-sell**: nhắm trước nhóm đang có hợp đồng **ô tô / cháy nổ / tài sản** — tín hiệu
  mua chéo mạnh nhất.
- **Phân khúc theo mức sung túc**: kết hợp thu nhập + sức mua để lọc danh sách gọi.
- Dùng **`submission_800.txt`** làm danh sách ưu tiên cho đợt chiến dịch đầu tiên.

> Chi tiết SHAP & biểu đồ: `figures/TR04_explain.png` + [docs/SUBMISSION_REPORT.md](docs/SUBMISSION_REPORT.md).

---

## 8. Cấu trúc thư mục

```
.
├── eda.ipynb  feature_engineering.ipynb  training.ipynb   # luồng chính (chạy trực tiếp)
├── data/                     # dữ liệu gốc: train_data.txt, test_data.txt
├── submission_800.txt        # ★ KẾT QUẢ NỘP (800 ID khách triển vọng)
├── requirements.txt  README.md  .gitignore
│
├── src/                      # module tái sử dụng (import bởi notebook)
│   ├── data.py               #   nạp dữ liệu + metadata (tên cột tiếng Việt, bảng mã)
│   ├── features.py           #   feature engineering v1/v2 + TargetEncoder/WOE (CV-safe)
│   ├── metrics.py            #   hits@20%, mean_hits_over_k, lift, AUC, PR-AUC
│   ├── models.py             #   "model zoo" (mô hình tham chiếu)
│   ├── cv.py                 #   grouped OOF (StratifiedGroupKFold theo profile)
│   ├── tune.py               #   Optuna tuning (K-as-hyperparameter, early stopping)
│   └── predict.py            #   refit toàn train → chấm test → top-800
│
├── docs/                     # đề bài, mô tả thuộc tính, báo cáo nộp (SUBMISSION_REPORT.md/.pdf)
├── figures/                  # biểu đồ .png cho slide (EDA / FE / TR)
└── outputs/                  # feature_set.json, training_results.csv, test_scores.csv,
                              # feature_importance_full.csv, perf_vs_K.csv, *_fe.parquet
```

---

## 9. Cài đặt & chạy lại (reproducibility)

```bash
pip install -r requirements.txt
```

> **Môi trường:** giữ **`numpy < 2`** (vd 1.26.4) cho tương thích pandas/matplotlib; `shap` cần
> bản hợp numpy 1.x.

```bash
# Chạy theo ĐÚNG thứ tự (FE sinh feature_set.json trước, rồi training mới dùng được):
jupyter nbconvert --to notebook --execute --inplace feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace training.ipynb
```

Mọi seed được cố định → chạy lại cho cùng kết quả & cùng `submission_800.txt`.

---

## 10. Hạn chế & hướng phát triển

- **Optimism nhẹ** trong ước lượng (đã nêu ở 5.6) — số thật trên nhãn ẩn sẽ thấp hơn ~vài điểm.
- Có thể tăng số thử nghiệm Optuna / mở rộng dải K / thử ensemble (rank-average) nếu chấp nhận đánh
  đổi tính dễ giải thích.
- Có thể bổ sung **nested CV** để báo cáo con số hoàn toàn không thiên lệch.
- Khi có thêm dữ liệu hành vi (lịch sử tương tác, kênh tiếp cận), mô hình sẽ mạnh hơn nữa.
