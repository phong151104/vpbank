# VPINS — AIA Insurance Challenge

Xếp hạng khách hàng theo khả năng mua bảo hiểm **AIA**, chọn ra **top 800 / 4000** khách triển
vọng nhất (**Nhiệm vụ 1**) và phân tích **vì sao** khách mua để hỗ trợ chiến dịch (**Nhiệm vụ 2**).

> **Dữ liệu:** bộ CoIL Challenge 2000 / TIC Benchmark. Train 5.822 khách × 85 đặc trưng + nhãn;
> test 4.000 khách (không nhãn). Tỉ lệ mua ~5,98% → **bài toán xếp hạng top-K mất cân bằng**
> (đánh giá bằng `hits@20%` / `mean_hits_over_k`, không dùng accuracy).

> 📄 **Báo cáo kỹ thuật chi tiết + giải thích mô hình & insight chiến dịch:**
> [docs/SUBMISSION_REPORT.md](docs/SUBMISSION_REPORT.md) (kèm bản PDF).

## Luồng chính — 3 notebook

```
eda.ipynb  →  feature_engineering.ipynb  →  training.ipynb
(hiểu dữ liệu)   (tạo + xếp hạng feature)      (tune + chọn model + nộp)
                        │                              ▲
                        └── outputs/feature_set.json ──┘   (bàn giao)
```

1. **`eda.ipynb`** — EDA toàn diện: chất lượng dữ liệu, dòng trùng, target mất cân bằng,
   univariate/bivariate, phân khúc khách hàng, và phần *insight nâng cao* (adversarial validation,
   tín hiệu bắc cầu FE).
2. **`feature_engineering.ipynb`** — tạo đặc trưng nghiệp vụ (`build_features_v2`), xếp hạng
   importance **đa phương pháp → consensus**, khử đa cộng tuyến, validate bằng **grouped OOF**;
   bàn giao `outputs/feature_set.json` (+ `train_fe/test_fe.parquet`).
3. **`training.ipynb`** — tune từng mô hình với **K (số feature) là hyperparameter**, đánh giá
   **grouped OOF** (`mean_hits_over_k` ± SE), chọn 1 mô hình theo **1-SE rule** → sinh
   **`submission_800.txt`**.

> ⚠️ Chạy theo thứ tự: `feature_engineering.ipynb` **trước** (sinh `feature_set.json`) rồi mới
> `training.ipynb`.

## Cấu trúc thư mục

```
.
├── eda.ipynb  feature_engineering.ipynb  training.ipynb   # luồng chính (chạy trực tiếp)
├── data/                     # dữ liệu gốc: train_data.txt, test_data.txt
├── submission_800.txt                                     # KẾT QUẢ NỘP (800 ID)
├── requirements.txt  README.md  .gitignore
│
├── src/                      # module tái sử dụng (import bởi notebook)
│   ├── data.py               #   nạp dữ liệu + metadata (tên cột tiếng Việt, bảng mã)
│   ├── features.py           #   feature engineering v1/v2 + TargetEncoder/WOE (CV-safe)
│   ├── metrics.py            #   hits@20%, mean_hits_over_k, lift, AUC, PR-AUC
│   ├── models.py             #   "model zoo" (mô hình tham chiếu)
│   ├── cv.py                 #   grouped OOF (StratifiedGroupKFold theo profile)
│   ├── tune.py               #   Optuna tuning (K-as-hyperparameter, grouped, early stopping)
│   └── predict.py            #   refit toàn train → chấm test → top-800
│
├── docs/                     # đề bài, mô tả thuộc tính, báo cáo nộp (SUBMISSION_REPORT.md/.pdf)
├── figures/                  # biểu đồ .png cho slide (EDA / FE / TR)
└── outputs/                  # feature_set.json, *_fe.parquet, training_results.csv,
                              # feature_importance_full.csv, perf_vs_K.csv, test_scores.csv
```

## Cài đặt

```bash
pip install -r requirements.txt
```

> **Lưu ý môi trường:** giữ **`numpy < 2`** (vd 1.26.4) cho tương thích pandas/matplotlib đã biên
> dịch cho numpy 1.x. `shap` (nếu dùng cho biểu đồ giải thích) cần bản hợp numpy 1.x.

## Chạy

```bash
# Mở 3 notebook trong Jupyter/VS Code và Run All theo thứ tự, hoặc chạy headless:
jupyter nbconvert --to notebook --execute --inplace feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace training.ipynb
```

## Phương pháp (điểm nhấn)

- **Đánh giá trung thực — grouped OOF:** `StratifiedGroupKFold` theo *profile* (bộ 85 đặc trưng
  giống nhau) để các dòng trùng không nằm cả ở train lẫn validation → không rò rỉ.
- **Metric ổn định:** `mean_hits_over_k` (trung bình hits ở top 15–25%) + báo cáo **SE**.
- **Chọn feature:** consensus importance (7 phương pháp) → **khử đa cộng tuyến trên ranking**
  (|corr|>0.9, 125→80) → quét **K∈[10,30] như hyperparameter** trong Optuna.
- **Chọn mô hình:** **1-SE rule** — ưu tiên mô hình **đơn giản hơn** nếu nằm trong 1 SE của bộ
  tốt nhất; dùng **1 mô hình đơn** (dễ giải thích cho Nhiệm vụ 2), không ensemble.
- **Giải thích:** SHAP / hệ số mô hình → đầu vào Nhiệm vụ 2.

## Kết quả hiện tại

| Mô hình | K | grouped OOF `meanK` ± SE | `hits@20%` | AUC |
|---|---|---|---|---|
| **LightGBM** (chốt) | 14 | **195.12 ± 0.98** | **206** / 348 | 0.784 |
| XGBoost | 17 | 192.60 ± 0.95 | 199 | 0.789 |
| CatBoost | 14 | 190.32 ± 0.93 | 191 | 0.790 |
| LDA | 30 | 182.60 ± 0.14 | 186 | 0.758 |
| LogReg | 29 | 181.36 ± 0.52 | 184 | 0.765 |

→ Sản phẩm nộp: **`submission_800.txt`** (LightGBM K=14, trên `consensus_rank` đã khử đa cộng
tuyến). Lift ~2,96× → kỳ vọng ~141 người mua trong 800 chọn (so với ~48 nếu ngẫu nhiên).

> **Ghi chú trung thực:** `hits@20%=206` nên coi là **cận trên** (optimism do chọn feature trên
> toàn train + chọn max trên 5 mô hình). Số trên tập nhãn ẩn vẫn sẽ thấp hơn ~vài điểm.
