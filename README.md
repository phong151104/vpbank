# VPINS — AIA Insurance Challenge

Dự đoán & xếp hạng khách hàng có khả năng mua bảo hiểm **AIA**, chọn ra **top 800/4000** khách triển vọng nhất (Nhiệm vụ 1) và phân tích **vì sao** khách mua để hỗ trợ chiến dịch bán hàng (Nhiệm vụ 2).

> Dữ liệu: bộ CoIL Challenge 2000 / TIC Benchmark. Train 5.822 khách × 85 đặc trưng + nhãn; test 4.000 khách (không nhãn). Tỉ lệ mua ~5,98% → **bài toán xếp hạng top-K mất cân bằng**.

## Cấu trúc dự án

```
vpbank/
├── train_data.txt, test_data.txt          # dữ liệu gốc
├── attributes_description.md              # mô tả 85 đặc trưng + bảng mã (Appendix)
├── requirements.txt
│
├── eda.ipynb                  # (1) EDA toàn diện — phân tích & biểu đồ cho slide
├── model.ipynb                # (2) Mô hình hoá: model zoo → tuning → ensemble → submission → SHAP
├── run_pipeline.py            # tái lập submission KHÔNG cần notebook
├── smoke_test.py              # kiểm tra nhanh toàn pipeline
│
├── src/                       # module tái sử dụng
│   ├── data.py                #   nạp dữ liệu + metadata (tên cột tiếng Việt, bảng mã)
│   ├── features.py            #   feature engineering + TargetEncoder (CV-safe)
│   ├── metrics.py             #   hits@20%, lift@20%, recall@20%, AUC, PR-AUC
│   ├── models.py              #   model zoo (10 mô hình)
│   ├── cv.py                  #   CV/OOF runner (RepeatedStratifiedKFold)
│   ├── tune.py                #   Optuna tuning (early stopping theo fold)
│   ├── ensemble.py            #   rank-averaging + stacking
│   └── predict.py             #   refit toàn train → sinh top-800
│
├── figures/                   # biểu đồ (.png) cho slide
├── outputs/                   # test_scores.csv (điểm + thứ hạng 4.000 khách)
├── models/                    # (tuỳ chọn) lưu mô hình
└── submission_800.txt         # KẾT QUẢ NỘP: 800 ID, mỗi dòng 1 ID
```

## Cài đặt

```bash
conda activate base          # hoặc tạo môi trường mới
pip install -r requirements.txt
```

## Chạy

```bash
# Tái lập submission (tuning + ensemble), ~15-25 phút
python run_pipeline.py

# Nhanh hơn để thử
python run_pipeline.py --trials 15
python run_pipeline.py --no-tune

# Hoặc chạy notebook đầy đủ (có biểu đồ + diễn giải)
jupyter nbconvert --to notebook --execute --inplace model.ipynb
```

## Phương pháp (tóm tắt)

1. **Feature engineering** — đặc trưng tổng hợp (tổng/đếm loại bảo hiểm sở hữu, flags, tương tác); `TargetEncoder` cho cột cardinal cao (subtype, main type), fit theo từng fold để **tránh rò rỉ**.
2. **Model zoo** — 10 mô hình (LogReg, Naive Bayes, LDA, RF, ExtraTrees, HistGB, LightGBM, XGBoost, CatBoost) chạy CV (OOF).
3. **Xử lý mất cân bằng** — so sánh class-weight / SMOTE / undersample.
4. **Feature selection** — mutual information ("less is more").
5. **Tuning** — Optuna (TPE + MedianPruner), objective = OOF `hits@20%`, early stopping.
6. **Ensemble** — rank-averaging & stacking.
7. **Dự đoán** — refit toàn train → chấm điểm test → **top 800 ID**.
8. **Giải thích** — SHAP + feature importance (đầu vào Nhiệm vụ 2).

**Metric:** `hits@20%` (số người mua trong top 20% theo điểm) là chính; phụ là ROC-AUC, PR-AUC. Đánh giá bằng `RepeatedStratifiedKFold` (OOF) để trung thực.
