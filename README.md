# VPINS — AIA Insurance Challenge

Xếp hạng khách hàng theo khả năng mua bảo hiểm **AIA**, chọn ra **top 800 / 4000** khách triển
vọng nhất (**Nhiệm vụ 1**) và phân tích **vì sao** khách mua để hỗ trợ chiến dịch (**Nhiệm vụ 2**).

> **Dữ liệu:** bộ CoIL Challenge 2000 / TIC Benchmark. Train 5.822 khách × 85 đặc trưng + nhãn;
> test 4.000 khách (không nhãn). Tỉ lệ mua ~5,98% → **bài toán xếp hạng top-K mất cân bằng**
> (đánh giá bằng `hits@20%` / `mean_hits_over_k`, không dùng accuracy).

> 📄 **Báo cáo kỹ thuật chi tiết (từng bước):** [docs/SOLUTION.md](docs/SOLUTION.md)

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
├── eda.ipynb  feature_engineering.ipynb  training.ipynb   # luồng chính
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
├── builders/                 # generator sinh ra notebook (nguồn sự thật của .ipynb)
│   ├── build_notebook.py         → eda.ipynb
│   ├── build_fe_notebook.py      → feature_engineering.ipynb
│   └── build_train_notebook.py   → training.ipynb
│
├── docs/                     # đề bài + mô tả thuộc tính (kèm bản dịch tiếng Việt)
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
# Chạy notebook theo thứ tự (mở trong Jupyter/VS Code, Run All), hoặc:
jupyter nbconvert --to notebook --execute --inplace feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace training.ipynb

# Regenerate notebook từ generator (chạy TỪ thư mục gốc; sẽ ghi đè .ipynb, mất output cũ):
python builders/build_train_notebook.py
```

## Phương pháp (điểm nhấn)

- **Đánh giá trung thực — grouped OOF:** `StratifiedGroupKFold` theo *profile* (bộ 85 đặc trưng
  giống nhau) để các dòng trùng không nằm cả ở train lẫn validation → không rò rỉ.
- **Metric ổn định:** `mean_hits_over_k` (trung bình hits ở top 15–25%) + báo cáo **SE**.
- **Chọn feature:** consensus importance (7 phương pháp) → quét **K∈[15,30] như hyperparameter**
  trong Optuna.
- **Chọn mô hình:** **1-SE rule** — ưu tiên mô hình **đơn giản hơn** nếu nằm trong 1 SE của bộ
  tốt nhất; dùng **1 mô hình đơn** (dễ giải thích cho Nhiệm vụ 2), không ensemble.
- **Giải thích:** SHAP / hệ số mô hình → đầu vào Nhiệm vụ 2.

## Kết quả hiện tại

| Mô hình | K | grouped OOF `meanK` | `hits@20%` | AUC |
|---|---|---|---|---|
| **CatBoost** (chốt) | 17 | **192.96 ± 1.26** | **195** / 348 | 0.786 |
| LightGBM | 26 | 187.0 | 190 | 0.780 |
| LDA | 18 | 183.3 | 185 | 0.757 |
| XGBoost | 16 | 182.4 | 188 | 0.772 |
| LogReg | 30 | 179.8 | 183 | 0.765 |

→ Sản phẩm nộp: **`submission_800.txt`** (CatBoost K=17). Kỳ vọng ~130 người mua trong 800 chọn
(so với ~48 nếu chọn ngẫu nhiên).

> **Ghi chú trung thực:** con số `hits@20%=195` nên coi là **cận trên** — có optimism nhẹ do
> (a) chọn max trên 5 mô hình và (b) xếp hạng feature (`consensus_rank`) tính trên toàn train
> (feature-selection ngoài vòng CV). Số trên tập nhãn ẩn nhiều khả năng thấp hơn một chút.
