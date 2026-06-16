# THỬ THÁCH AI DATA SCIENTIST

VPINS là một bộ phận thuộc VPB, chuyên cung cấp các dịch vụ bảo hiểm với nhiều chính sách hấp dẫn. Hiện tại công ty đang muốn đẩy mạnh ưu đãi cho một loại bảo hiểm sức khỏe đặc biệt tên là **bảo hiểm AIA**. Trong thử thách này, bạn sẽ giúp giám đốc bộ phận hiểu rõ khách hàng của mình bằng cách hỗ trợ ông ấy xác định khách hàng nào sẵn sàng sở hữu chính sách bảo hiểm, để ông ấy có thể triển khai một chiến dịch bán hàng hiệu quả. Ông ấy mong đợi sản phẩm cuối cùng của bạn như sau:

1. Một tài liệu **tối đa 2 trang A4** (hoặc slide tối đa 10 trang) trình bày cách bạn hiểu vấn đề, phương pháp tiếp cận, quy trình làm việc và kết quả.
2. **Mã nguồn (code base)** của bạn bằng ngôn ngữ lập trình tùy chọn (ưu tiên Python), nén lại thành một file để các data scientist khác có thể tái lập công việc của bạn.


## Tuyên bố miễn trừ trách nhiệm (Disclaimer):

Thông tin trong tài liệu này là thông tin mật, có đặc quyền và chỉ dành cho người nhận được chỉ định. Không được sử dụng, công bố hoặc phân phối lại nếu chưa có sự đồng ý trước bằng văn bản của Trung tâm EDA-AI.

# MÔ TẢ TẬP DỮ LIỆU

## 1) train_data.txt:

Bạn sẽ sử dụng tập dữ liệu này để **huấn luyện và kiểm định (validate)** các mô hình dự đoán. Mỗi dòng của file gồm **86 thuộc tính**, bao gồm thông tin nhân khẩu - xã hội học (thuộc tính 1–42) và thông tin sở hữu sản phẩm (thuộc tính 43–85); thuộc tính 86 là **biến mục tiêu / nhãn** (có sở hữu bảo hiểm AIA hay không) và **ID của mỗi khách hàng** (trùng với số thứ tự dòng). Dữ liệu nhân khẩu - xã hội học được suy ra từ mã bưu chính (zip code). Tất cả khách hàng sống trong khu vực có cùng mã bưu chính sẽ có chung các thuộc tính nhân khẩu - xã hội học.

## 2) test_data.txt:

Tập dữ liệu này dùng để **dự đoán**. Nó có cùng định dạng với train_data.txt, ngoại trừ việc **thiếu biến mục tiêu** (nhãn).

## 3) attributes_description.pdf:

File này chứa mô tả chi tiết của từng thuộc tính trong dữ liệu.

# CÁC NHIỆM VỤ

## 1) Nhiệm vụ dự đoán:

Mục tiêu của nhiệm vụ này là xây dựng một mô hình **dự đoán liệu một khách hàng có mua bảo hiểm AIA hay không**. Bạn cũng được cung cấp một tập test (file test_data.txt ở trên) gồm **4000 mẫu** là những người có khả năng sở hữu bảo hiểm. Hãy **lọc ra 800 trường hợp triển vọng nhất** mong muốn mua loại bảo hiểm này. Chúng tôi sẽ dùng tập nhãn được giữ lại (hold-out) để kiểm chứng kết quả của bạn; nhãn này không được cung cấp cho bạn, nên bạn càng phát hiện được nhiều trường hợp thực sự muốn mua bảo hiểm AIA thì mô hình bạn xây dựng càng tốt.

Sản phẩm của nhiệm vụ này là một file gồm **800 dòng, mỗi dòng là ID của một khách hàng**.

## 2) Nhiệm vụ giải thích:

Mục tiêu của nhiệm vụ này là cung cấp **insight (hiểu biết sâu) về lý do vì sao khách hàng sở hữu bảo hiểm AIA**. Phần giải thích và diễn giải đi kèm sẽ được chấm điểm dựa trên tính **dễ hiểu, hữu ích và khả năng hành động** nhằm hỗ trợ hiệu quả cho giám đốc trong việc xây dựng chiến dịch bán hàng của mình.
