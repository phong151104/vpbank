# Báo cáo nộp bài - VPINS AIA Insurance Challenge

## Tóm tắt điều hành

Mục tiêu của bài toán là chọn **800 khách hàng triển vọng nhất** trong 4.000 khách hàng test để triển khai chiến dịch bán bảo hiểm AIA. Do tỷ lệ mua trong train chỉ **5,98%** (348/5.822), bài toán được xử lý như **xếp hạng top-K** thay vì phân loại bằng ngưỡng 0,5; thước đo chính là số người mua thật nằm trong top 20% danh sách.

Pipeline cuối gồm 3 bước: EDA để hiểu dữ liệu và rủi ro leakage, feature engineering để tạo tín hiệu nghiệp vụ và khử đa cộng tuyến, sau đó tune mô hình bằng grouped OOF. Mô hình chốt là **LightGBM với 14 feature**, đạt `mean_hits_over_k = 195,12 ± 0,98`, `hits@20% = 206/348`, `AUC = 0,784`. File nộp là `submission_800.txt`, gồm đúng 800 ID khách hàng duy nhất.

## 1. Hiểu bài toán và dữ liệu

Train có **5.822 khách hàng**, test có **4.000 khách hàng**. Mỗi khách có 85 đặc trưng đầu vào; train có thêm nhãn ở cột 86 cho biết khách có sở hữu AIA policy hay không. Theo `attributes_description.md`, dữ liệu được chia thành 4 nhóm chính:

| Nhóm cột | Ý nghĩa | Vai trò trong mô hình |
|---|---|---|
| 1-42 | Nhân khẩu - xã hội học suy từ zip code | Phân khúc khách hàng, tuổi, thu nhập, học vấn, gia đình |
| 43 | Hạng sức mua | Năng lực chi trả |
| 44-64 | Mức đóng phí 21 loại bảo hiểm | Cường độ đã mua/sử dụng bảo hiểm |
| 65-85 | Số hợp đồng 21 loại bảo hiểm | Độ rộng danh mục bảo hiểm |

Ba đặc điểm quyết định cách làm: dữ liệu mất cân bằng mạnh, nhiều biến là mã ordinal 0-9 và khối bảo hiểm rất thưa. Ngoài ra có **651 dòng trùng hoàn toàn theo 85 đặc trưng** vì nhiều khách cùng zip code hoặc cùng profile rời rạc hóa. Các dòng này được giữ lại vì đó vẫn là khách thật và có nhóm mâu thuẫn nhãn; thay vào đó, grouped cross-validation được dùng để các profile trùng không xuất hiện đồng thời ở train và validation.

## 2. Phương pháp và workflow

Workflow được tách thành 3 notebook có thể tái lập:

| Bước | Đã làm bên trong | Kết quả chính |
|---|---|---|
| EDA | Kiểm tra missing/schema/dải giá trị, phân phối target, dòng trùng profile, zero-inflation, tương quan feature-target và drift train-test | Không thiếu dữ liệu; target mất cân bằng; test không lệch train đáng kể; cần tối ưu ranking top 20% |
| Feature Engineering | Tạo feature tổng hợp bảo hiểm, flag sở hữu, nhóm sản phẩm, tỷ lệ danh mục, chỉ số socio, tương tác; so sánh TargetEncoder/WOE/frequency; gom cụm tương quan, VIF, ranking 7 phương pháp và dedupe theo Spearman | 85 feature gốc + 40 feature mới = 125 feature; pool training còn 80 feature đã xếp hạng và khử trùng lặp mạnh |
| Training | Tune LightGBM/XGBoost/CatBoost/LogReg/LDA bằng Optuna; tune cả số feature K; đánh giá grouped OOF 5-fold lặp nhiều seed | Chọn LightGBM K=14 bằng 1-SE rule; sinh `submission_800.txt` |

Trong EDA, train/test được xác nhận có cùng cấu trúc, không có missing và các cột nằm đúng miền mã hóa. Phần quan trọng nhất là xử lý **651 dòng trùng profile**: các dòng này được giữ lại vì đây là khách thật, nhưng group theo profile được dùng khi cross-validation để tránh mô hình nhìn thấy khách gần như giống hệt ở cả train và validation. Ngoài ra, adversarial validation và PSI được dùng để khẳng định test không bị lệch phân phối rõ rệt so với train.

Trong feature engineering, các feature mới đều là phép biến đổi **row-wise, không dùng nhãn**: tổng mức đóng phí, số loại bảo hiểm đang sở hữu, cờ có bảo hiểm ô tô/cháy nổ/nhân thọ/tài sản, tổng theo lĩnh vực bảo hiểm, tỷ trọng phí xe/sức khỏe, chỉ số thu nhập-sung túc-học vấn và tương tác như thu nhập × sức mua. Các encoder phụ thuộc target như TargetEncoder/WOE chỉ được fit bên trong từng fold, không ghi sẵn vào parquet.

Ở bước training, accuracy không được tối ưu vì dữ liệu quá mất cân bằng: dự đoán "không mua" gần như toàn bộ đã có accuracy cao nhưng vô dụng cho chiến dịch. Thay vào đó, mô hình được tối ưu theo khả năng đưa người mua thật lên đầu danh sách, đúng với cách chấm là chọn 800 khách triển vọng nhất.

## 3. Chọn biến và chống leakage

Validation được thực hiện bằng **StratifiedGroupKFold 5-fold** theo profile 85 feature gốc. Nghĩa là mỗi lượt lấy 4 fold để train và 1 fold để validation, đồng thời đảm bảo các dòng trùng profile nằm cùng một phía. Dự đoán validation được ghép lại thành OOF score cho toàn bộ train.

Cách chia này giúp đánh giá gần tình huống thật hơn: mô hình phải học quy luật tổng quát từ các nhóm khách tương tự, thay vì "ghi nhớ" một profile đã xuất hiện trong train. Điều này đặc biệt quan trọng vì nhiều đặc trưng nhân khẩu học được suy từ zip code nên nhiều khách có thông tin rất giống nhau.

Để dễ hiểu, luồng feature có thể nhìn theo 2 bước chính:

| Bước | Số feature | Đã làm gì | Đầu ra dùng để làm gì |
|---|---:|---|---|
| 1. Tạo feature | 85 → 125 | Giữ 85 feature gốc và tạo thêm 40 feature nghiệp vụ từ bảo hiểm, thu nhập, sức mua, phân khúc | Tạo không gian tín hiệu đầy đủ |
| 2. Lọc và xếp hạng | 125 → 80 | Ranking 7 phương pháp, sau đó bỏ các feature gần trùng theo `|Spearman corr| > 0,9` | Tạo `consensus_rank`, pool feature chính để train |

Trong bước lọc và xếp hạng, biến không được chọn bằng một tiêu chí đơn lẻ. Mỗi feature được chấm bằng nhiều góc nhìn: Mutual Information, ANOVA F, point-biserial correlation, L1 Logistic Regression, LightGBM gain, permutation importance và SHAP. Sau đó, thứ hạng đồng thuận được lấy để giảm thiên lệch của từng phương pháp.

Sau khi có ranking, các feature quá giống nhau được loại bớt; ví dụ trong một cụm như đóng phí bảo hiểm ô tô và số hợp đồng ô tô, đại diện có ranking tốt hơn được giữ lại. Training cuối dùng pool 80 feature này, rồi Optuna tự chọn số feature tốt nhất trong khoảng `K ∈ [10,30]`. Vì vậy mô hình không dùng cả 80 feature, mà chọn **top-K** theo ranking; kết quả cuối là **LightGBM dùng top 14 feature**.

Nhờ vậy, bộ feature cuối vừa giữ được tín hiệu dự đoán mạnh, vừa giảm trùng lặp để mô hình gọn và dễ giải thích hơn. Đây là lý do toàn bộ 125 feature không được đưa trực tiếp vào mô hình cuối.

## 4. Kết quả mô hình

Các mô hình được tune trên cùng grouped OOF, metric chính là `mean_hits_over_k`: trung bình số người mua bắt được ở top 15%, 17,5%, 20%, 22,5% và 25%. Metric này ổn định hơn việc chỉ nhìn đúng một mốc top 20%, vì một mô hình tốt sẽ xếp hạng ổn định quanh vùng top 800, chứ không chỉ may mắn ở đúng một ngưỡng.

| Mô hình | K | `meanK ± SE` | `hits@20%` | AUC | Chọn |
|---|---:|---:|---:|---:|---|
| **LightGBM** | **14** | **195,12 ± 0,98** | **206** | 0,784 | Có |
| XGBoost | 17 | 192,60 ± 0,95 | 199 | 0,789 | Không |
| CatBoost | 14 | 190,32 ± 0,93 | 191 | 0,790 | Không |
| LDA | 30 | 182,60 ± 0,14 | 186 | 0,758 | Không |
| LogReg | 29 | 181,36 ± 0,52 | 184 | 0,765 | Không |

LightGBM được chọn vì có `meanK` cao nhất và là mô hình duy nhất nằm trên ngưỡng 1-SE của mô hình tốt nhất. Với `hits@20% = 206/348`, lift trên train OOF khoảng **2,96 lần** so với chọn ngẫu nhiên; kỳ vọng trong 800 khách được chọn là khoảng **141 người mua**, so với khoảng 48 nếu chọn ngẫu nhiên theo tỷ lệ nền. Sau khi chọn, LightGBM được huấn luyện lại trên toàn bộ train, dùng để chấm điểm 4.000 khách test, sắp xếp giảm dần theo score và lấy đúng top 800 ID. File `submission_800.txt` đã kiểm tra đúng 800 dòng, ID duy nhất và nằm trong khoảng 1-4000.

So với các mô hình còn lại, LightGBM cân bằng tốt giữa hiệu năng ranking và độ gọn: chỉ cần 14 feature nhưng bắt được nhiều khách mua hơn ở vùng top danh sách. Điều này phù hợp mục tiêu kinh doanh là ưu tiên đúng khách để giảm chi phí gọi bán.

## 5. Giải thích kết quả và khuyến nghị chiến dịch

Kết quả mô hình cho thấy khách có xác suất mua AIA cao thường không phải nhóm "ngẫu nhiên giàu hơn" đơn thuần, mà là nhóm đã có **hành vi mua bảo hiểm trước đó**, có **khả năng chi trả**, và thuộc một số **phân khúc nhân khẩu phù hợp**. Đây là insight quan trọng cho chiến dịch: AIA nên được bán như một sản phẩm bổ sung trong hành trình bảo vệ tài chính, không chỉ là một sản phẩm sức khỏe đơn lẻ.

| Tín hiệu chính | Vì sao hợp lý | Hàm ý hành động |
|---|---|---|
| Đã có bảo hiểm ô tô hoặc phí bảo hiểm xe cao | Đây là dấu hiệu khách đã quen mua bảo hiểm, hiểu giá trị phòng ngừa rủi ro và có lịch sử trả phí | Ưu tiên bán chéo AIA cho nhóm có bảo hiểm xe; thông điệp nên nhấn mạnh "bổ sung bảo vệ sức khỏe cho người đã có bảo vệ tài sản" |
| Tổng phí bảo hiểm và số loại bảo hiểm đang sở hữu cao | Danh mục bảo hiểm rộng phản ánh nhu cầu bảo vệ cao hơn, khả năng chi trả tốt hơn và độ sẵn sàng mua thêm sản phẩm | Chia nhóm khách theo độ rộng danh mục; nhóm sở hữu nhiều loại bảo hiểm nên được chăm sóc bởi kịch bản tư vấn cá nhân hóa |
| Có bảo hiểm cháy nổ, tài sản, thuyền hoặc trách nhiệm | Các sản phẩm này cho thấy khách quan tâm đến rủi ro tài sản/trách nhiệm; AIA có thể được đặt vào cùng logic quản trị rủi ro | Định vị AIA như lớp bảo vệ tiếp theo cho gia đình và thu nhập, không chỉ là một ưu đãi ngắn hạn |
| Thu nhập, sức mua, học vấn và phân khúc khách hàng | Nhóm có nền tảng tài chính tốt và nhận thức cao thường dễ tiếp nhận sản phẩm bảo hiểm sức khỏe hơn | Dùng thông điệp khác nhau theo phân khúc: nhóm sung túc nhấn vào chất lượng bảo vệ; nhóm gia đình nhấn vào an tâm cho người thân |

Do đó, danh sách 800 khách trong `submission_800.txt` nên được dùng như danh sách ưu tiên bán hàng; `outputs/test_scores.csv`. `score` là điểm ưu tiên tương đối: khách hội tụ nhiều tín hiệu nên được gọi trước; thông điệp nên bám lý do xếp hạng cao như "bổ sung bảo vệ sức khỏe" cho khách đã có bảo hiểm xe/tài sản, hoặc "hoàn thiện danh mục bảo vệ" cho khách đã có nhiều hợp đồng.

**Sản phẩm bàn giao:** `submission_800.txt`, `outputs/test_scores.csv` và codebase Python/notebook để tái lập workflow.
