# Giải thích mô hình & Insight cho chiến dịch (Nhiệm vụ 2)

> Trả lời câu hỏi của giám đốc: **"Vì sao khách mua AIA, và nên nhắm ai?"** — dựa trên **SHAP**
> của mô hình cuối (LightGBM, 14 đặc trưng) + **tỉ lệ mua theo phân khúc** (EDA). Tỉ lệ mua nền
> của toàn tập là **5,98%**; mọi con số dưới đây so với mốc này.

---

## 0. TL;DR cho giám đốc

- Mô hình xếp hạng 4.000 khách test; chọn **top 800** → kỳ vọng bắt được **~141 người mua**
  (so với ~48 nếu gọi ngẫu nhiên — **gấp ~3 lần**).
- **3 nhóm động lực** khiến khách dễ mua AIA:
  1. **Đã sở hữu bảo hiểm khác** (nhất là **ô tô**, cháy nổ, và **danh mục BH rộng**) → bán chéo.
  2. **Sung túc** (thu nhập, sức mua, học vấn cao).
  3. **Một số phân khúc nhân khẩu** (gia đình trung lưu, "người vươn lên"…).
- **Nhắm ai (ưu tiên):** khách **đã có BH ô tô** + **sở hữu ≥3 loại BH** + thuộc phân khúc sung túc.
  → Dùng `outputs/test_scores.csv` (điểm + thứ hạng 4.000 khách) để chia tầng gọi.

---

## 1. Mô hình & cách đọc SHAP

- **Mô hình cuối:** LightGBM, **14 đặc trưng** (top theo importance đồng thuận, đã khử đa cộng tuyến).
- **SHAP value** = mức một đặc trưng **đẩy (+) hay kéo (−)** xác suất mua của *từng khách*.
  Biểu đồ `figures/TR04_explain.png`: mỗi hàng 1 đặc trưng (xếp theo độ quan trọng); **màu = giá
  trị đặc trưng** (đỏ = cao, xanh = thấp); **vị trí ngang = ảnh hưởng** (phải = tăng khả năng mua).

---

## 2. Vì sao khách mua — 14 yếu tố từ SHAP

| # | Yếu tố (đặc trưng) | Hướng | Diễn giải nghiệp vụ |
|---|---|---|---|
| 1 | **Đóng phí BH ô tô** | ⬆️ tăng | Đã chi cho BH xe → quen mua bảo hiểm → dễ mua chéo AIA. **Tín hiệu mạnh nhất.** |
| 2 | **Tổng phí BH khác** (`agg_total_contrib`) | ⬆️ | Tổng chi cho bảo hiểm càng lớn → càng dễ mua thêm. |
| 3 | **Nhóm KH chính** (main type) | hỗn hợp | Một số nhóm (vươn lên, trung lưu) cao hơn hẳn — xem §3. |
| 4 | **Đóng phí BH cháy nổ** | ⬆️ | Có BH tài sản/cháy nổ → ý thức bảo vệ cao → dễ mua. |
| 5 | **Thu nhập × sức mua** (`ix_income_x_pp`) | ⬆️ | Chỉ số sung túc tổng hợp — càng cao càng dễ mua. |
| 6 | **Học vấn thấp** | ⬇️ giảm | Vùng nhiều người học vấn thấp → ít mua. |
| 7 | **Thu nhập TB** | ⬆️ | Thu nhập cao → dễ mua. |
| 8 | **Số loại BH đang có** (`agg_n_contrib_types`) | ⬆️ | Danh mục BH rộng → khả năng mua thêm cao (xem §3). |
| 9 | **Đóng phí BH thuyền** | ⬆️ | Loại BH "xa xỉ" hiếm — sở hữu nó báo hiệu **giàu + thân thiện bảo hiểm**. |
| 10 | **Đã kết hôn** | ⬆️ nhẹ | Hộ đã kết hôn nhỉnh hơn chút. |
| 11 | **Thu nhập <30k** | ⬇️ | Vùng nhiều người thu nhập thấp → ít mua. |
| 12 | **Quản lý cấp trung** | ⬆️ nhẹ | Nghề ổn định, thu nhập khá. |
| 13 | **Học vấn cao** | ⬆️ | Vùng học vấn cao → dễ mua. |
| 14 | **Tỉ trọng phí BH xe** (`ratio_vehicle_share`) | ⬆️ | Danh mục thiên về BH xe → tín hiệu mua chéo. |

**Gộp lại thành 3 thông điệp:**
- **A. "Đã là người mua bảo hiểm":** ô tô (1), tổng phí (2), cháy nổ (4), số loại BH (8), tỉ trọng
  xe (14), thuyền (9) → *khách đã có thói quen & ngân sách bảo hiểm là đối tượng số 1.*
- **B. "Sung túc":** thu nhập×sức mua (5), thu nhập TB (7), học vấn cao (13) đẩy lên; thu nhập
  thấp (11), học vấn thấp (6) kéo xuống.
- **C. "Nhân khẩu":** nhóm KH chính (3), đã kết hôn (10), quản lý cấp trung (12).

---

## 3. Nhắm AI — phân khúc có tỉ lệ mua cao (số liệu thực, mốc nền 5,98%)

SHAP cho biết *vì sao*; phần này cho biết *nhắm ai* (cái SHAP không trả lời trực tiếp).

**Theo sở hữu BH ô tô** — chênh lệch lớn nhất:
| | Tỉ lệ mua | Ghi chú |
|---|---|---|
| **Có đóng phí BH ô tô** | **9,3%** (n=2.977) | **gấp ~3,7 lần** nhóm không có |
| Không có | 2,5% | |

**Theo độ rộng danh mục BH** (số loại đang sở hữu):
| Số loại BH | 0 | 1 | 2 | **3** | **4** | **5** | **6** |
|---|---|---|---|---|---|---|---|
| % mua | 5,0 | 2,5 | 5,6 | **11,1** | **13,2** | **12,5** | **15,0** |
→ Sở hữu **≥3 loại BH** → tỉ lệ mua **~2–2,5 lần** nền. (Khách chỉ có đúng 1 loại lại thấp nhất.)

**Theo Nhóm KH chính (main type):**
| Nhóm | % mua |
|---|---|
| **Người vươn lên** | **13,1%** (2,2×) |
| Hedonist thành đạt | 8,7% |
| Gia đình trung bình | 6,7% |
| (thấp) Nghỉ hưu & sùng đạo | 3,6% |

**Theo phân nhóm chi tiết (subtype, n≥30):**
| Phân nhóm | % mua |
|---|---|
| **Middle class families** | **15,0%** |
| **Affluent young families** | **14,4%** |
| High income, expensive child | 10,5% |
| Career and childcare | 10,1% |
| High status seniors | 10,0% |

*(Nguồn: `figures/07a_rate_by_segment.png`, `07b_segment_heatmaps.png`.)*

---

## 4. Khuyến nghị chiến dịch (actionable)

1. **Danh sách gọi — dùng `outputs/test_scores.csv`** (điểm + thứ hạng 4.000 khách):
   - **`submission_800.txt`** = top 800 ID ưu tiên.
   - Nếu ngân sách hẹp, chia tầng: **top 200** (nóng nhất) → **200–500** → **500–800**.
2. **Chân dung ưu tiên** (giao của các tín hiệu mạnh): *khách **đã có BH ô tô** + **đang sở hữu
   ≥3 loại BH** + thuộc nhóm **"người vươn lên" / gia đình trung lưu trẻ sung túc**.*
3. **Góc tiếp cận (messaging) theo động lực:**
   - Nhóm A (đã có BH): nhấn **bán chéo / gói hợp nhất** — "bổ sung bảo vệ sức khoẻ vào danh mục
     bạn đang có".
   - Nhóm B (sung túc): nhấn **quyền lợi cao cấp**.
   - Phân khúc gia đình: nhấn **bảo vệ cả nhà**.

---

## 5. SHAP có đủ không? — cảnh báo & những gì đã bổ sung

- **SHAP là *attribution của mô hình*, KHÔNG phải nhân quả.** "Có BH ô tô" tương quan mạnh với
  mua AIA, **không có nghĩa** ép khách mua BH ô tô thì họ sẽ mua AIA. Cả hai cùng phản ánh *khách
  có thói quen & khả năng mua bảo hiểm*.
- **Vì vậy doc này bổ sung quanh SHAP:** (a) dịch nghiệp vụ, (b) **tỉ lệ mua theo phân khúc** (để
  *nhắm ai*), (c) khuyến nghị + cách dùng output. Đây là phần khiến giải thích trở nên *dùng được*.
- **Số kỳ vọng "~141 người mua trong 800" là CẬN TRÊN** (xem [SOLUTION.md §9](SOLUTION.md)):
  có optimism ~vài điểm; con số trên nhãn ẩn sẽ thấp hơn chút. Ưu thế mô hình đã được kiểm bằng
  nested CV nên **xu hướng là đáng tin**, nhưng đừng cam kết con số tuyệt đối.
- (Tuỳ chọn nâng cao) Muốn giải thích *cá nhân hoá từng khách* (vì sao KH X ở top): dùng **SHAP
  force/waterfall** cho từng dòng — chưa làm, có thể bổ sung nếu cần.
