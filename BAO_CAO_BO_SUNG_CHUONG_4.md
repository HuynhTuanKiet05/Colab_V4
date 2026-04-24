# Bổ sung báo cáo đồ án cơ sở

Tài liệu này dùng để bổ sung trực tiếp vào báo cáo Word hiện tại. Nội dung được viết bám theo repo `Colab_V4`, các file mã nguồn đang có, và các kết quả 10-fold trong thư mục `Result/improved/`.

## 1. Các đoạn nên chỉnh lại để báo cáo khớp với sản phẩm thật

### 1.1. Chỉnh mục "Về mặt công nghệ" trong Chương 1

Đề nghị thay đoạn đang mô tả `Vite`, `Tailwind CSS` bằng đoạn sau:

Hệ thống phần mềm của đề tài được triển khai theo mô hình lai giữa ứng dụng web và dịch vụ AI. Phần giao diện người dùng được xây dựng bằng PHP kết hợp HTML, CSS và JavaScript thuần; phần suy luận mô hình được triển khai bằng Python FastAPI; dữ liệu người dùng và lịch sử dự đoán được quản lý bằng MySQL. Cách tổ chức này giúp tách biệt rõ phần hiển thị, phần xử lý nghiệp vụ và phần suy luận trí tuệ nhân tạo, đồng thời thuận lợi cho việc triển khai thử nghiệm trên môi trường học tập và demo thực tế.

### 1.2. Chỉnh mục 2.4 "Công nghệ phát triển hệ sinh thái phần mềm"

Đề nghị thay toàn bộ mục 2.4 bằng nội dung sau:

#### 2.4. Công nghệ phát triển hệ thống phần mềm

Để chuyển thuật toán dự đoán liên kết Thuốc - Bệnh thành một hệ thống có thể sử dụng, đồ án sử dụng các công nghệ sau:

- Backend AI với FastAPI: FastAPI được dùng để xây dựng các API suy luận như kiểm tra trạng thái hệ thống và dự đoán Top-K liên kết. Dịch vụ này chịu trách nhiệm nạp trọng số mô hình đã huấn luyện, tiền xử lý đầu vào và trả kết quả dưới dạng JSON.
- Ứng dụng web với PHP: Giao diện người dùng được xây dựng theo mô hình server-rendered bằng PHP. Cách triển khai này phù hợp với phạm vi đồ án cơ sở, dễ tích hợp xác thực người dùng, lịch sử tra cứu và trang quản trị.
- Cơ sở dữ liệu MySQL: MySQL được dùng để lưu tài khoản người dùng, danh mục thuốc, bệnh, protein, các liên kết đã biết và lịch sử dự đoán.
- Trực quan hóa đồ thị 3D: Hệ thống sử dụng thư viện `3d-force-graph` để hiển thị trực quan mối liên hệ giữa thuốc, bệnh và protein dưới dạng đồ thị tương tác.
- Môi trường huấn luyện mô hình: Phần huấn luyện được thực hiện bằng PyTorch và DGL trên môi trường có hỗ trợ CUDA để tăng tốc quá trình học.

Như vậy, hệ thống của đề tài không đi theo kiến trúc Frontend SPA độc lập, mà là mô hình web PHP kết nối với Python API cho tác vụ AI. Đây là lựa chọn phù hợp với mục tiêu xây dựng sản phẩm demo có thể triển khai, sử dụng và quản trị trong thực tế.

### 1.3. Chỉnh mục 3.3 "Kiến trúc hệ thống"

Đề nghị thay phần mô tả công nghệ của tầng giao diện và kiến trúc hệ thống bằng nội dung sau:

#### 3.3. Kiến trúc hệ thống

Hệ thống được thiết kế theo mô hình client-server gồm ba tầng chính:

1. Tầng giao diện (Presentation Layer)

Phần giao diện được xây dựng bằng PHP kết hợp HTML, CSS và JavaScript. Người dùng tương tác qua các trang đăng nhập, trang dự đoán, trang lịch sử và trang quản trị. Kết quả dự đoán được hiển thị dưới dạng bảng Top-K và đồ thị 3D trực quan.

2. Tầng xử lý ứng dụng (Application Layer)

Phần xử lý ứng dụng gồm hai thành phần phối hợp với nhau:

- Web PHP chịu trách nhiệm xác thực người dùng, quản lý phiên đăng nhập, điều phối giao diện và lưu lịch sử truy vấn vào cơ sở dữ liệu.
- Python FastAPI chịu trách nhiệm suy luận mô hình học sâu, tiếp nhận yêu cầu dự đoán từ web PHP và trả về kết quả ở dạng JSON.

3. Tầng dữ liệu và mô hình (Data and Model Layer)

Tầng này bao gồm cơ sở dữ liệu MySQL và các checkpoint mô hình đã huấn luyện. MySQL lưu trữ dữ liệu quản trị và lịch sử dự đoán, còn mô hình HGT cải tiến được lưu dưới dạng tệp trọng số để nạp lại khi hệ thống khởi động.

Kiến trúc trên giúp hệ thống dễ bảo trì, dễ demo và thuận lợi cho việc mở rộng trong tương lai.

### 1.4. Những câu nên tránh hoặc cần nói thận trọng hơn

- Nếu chưa có log benchmark riêng, không nên khẳng định chắc chắn mô hình "chỉ cần 4GB VRAM".
- Nếu chưa có kết quả baseline AMDGT gốc chạy lại trong cùng điều kiện, không nên viết so sánh định lượng tuyệt đối kiểu "vượt AMDGT gốc X%". Nên dùng cách diễn đạt: "mô hình cải tiến cho kết quả tốt và ổn định trên các bộ dữ liệu B, C, F; việc đối chiếu 1-1 với baseline gốc sẽ được bổ sung trong phiên bản hoàn thiện."
- Nếu vẫn giữ các tên như `GRIDecoder` hoặc `Residual Boost Fusion`, cần bảo đảm trong phần hiện thực có mô tả tương ứng với module mã nguồn. Nếu không, nên chuyển sang cách gọi bám code hơn như `pair head`, `multi-view aggregation`, `fuzzy gate`, `topology encoder`.

## 2. Nội dung đề xuất cho Chương 4

## Chương 4. Cải tiến mô hình và kết quả thực nghiệm

### 4.1. Mục tiêu cải tiến

Mô hình AMDGT gốc có ưu điểm là khai thác đồng thời thông tin tương đồng và thông tin liên kết dị thể giữa thuốc, bệnh và protein. Tuy nhiên, khi triển khai lại trong bối cảnh đề tài, nhóm nhận thấy vẫn còn một số hạn chế cần cải thiện:

- mô hình gốc chủ yếu xem thông tin topo một cách gián tiếp, chưa có nhánh đặc trưng topo tường minh;
- khả năng hòa trộn nhiều nguồn thông tin còn tương đối cứng;
- pipeline huấn luyện chưa tối ưu cho các bộ dữ liệu thưa;
- repo gốc thiên về mã nghiên cứu, chưa sẵn sàng cho suy luận thời gian thực và tích hợp website.

Từ đó, mục tiêu của chương này là trình bày các cải tiến đã thực hiện trên mô hình và đánh giá kết quả thực nghiệm bằng quy trình 10-fold cross-validation trên ba bộ dữ liệu B-dataset, C-dataset và F-dataset.

### 4.2. Thiết lập thực nghiệm

#### 4.2.1. Bộ dữ liệu sử dụng

Các thực nghiệm trong đề tài được thực hiện trên ba bộ dữ liệu đã được lưu trong thư mục dữ liệu của dự án:

- B-dataset: gồm 269 thuốc, 598 bệnh và 1021 protein.
- C-dataset: gồm 663 thuốc, 409 bệnh và 993 protein.
- F-dataset: gồm 592 thuốc, 313 bệnh và 2741 protein.

Ba bộ dữ liệu này có độ thưa và cấu trúc khác nhau, giúp đánh giá mô hình ở cả trường hợp chuẩn và trường hợp khó.

#### 4.2.2. Giao thức đánh giá

Đề tài sử dụng 10-fold cross-validation để đánh giá độ ổn định của mô hình. Ở mỗi fold, dữ liệu được chia thành tập huấn luyện và tập kiểm tra, sau đó huấn luyện mô hình đến khi đạt điểm tốt nhất theo AUC. Các độ đo được ghi nhận gồm:

- AUC
- AUPR
- Accuracy
- Precision
- Recall
- F1-score
- MCC

Trong đó, AUC và AUPR là hai chỉ số được ưu tiên nhất vì bài toán dự đoán liên kết Thuốc - Bệnh có bản chất mất cân bằng dữ liệu.

#### 4.2.3. Cấu hình huấn luyện

Các thực nghiệm được chạy trên môi trường có hỗ trợ CUDA. Cấu hình chính của từng bộ dữ liệu được tóm tắt ở Bảng 4.1.

**Bảng 4.1. Cấu hình huấn luyện chính**

| Bộ dữ liệu | Cấu hình mô hình | Learning rate | GT dim | Neighbor | Similarity view | Ghi chú |
| --- | --- | ---: | ---: | ---: | --- | --- |
| B-dataset | `vanilla_hgt + mva + mul_mlp + vector` | 0.0001 | 512 | 3 | consensus | tối ưu cho bộ dữ liệu nhỏ hơn nhưng nhiều bệnh |
| C-dataset | `vanilla_hgt + mva + mul_mlp + vector` | 0.0001 | 256 | 5 | consensus | bộ dữ liệu chuẩn, cho kết quả ổn định nhất |
| F-dataset | `rlghgt + rvg + interaction + vector` | 0.00008 | 320 | 8 | multi | dùng thêm positive weighting và multi-view |

Ngoài các tham số trên, pipeline huấn luyện còn bổ sung các cơ chế:

- trích xuất topology feature cho drug và disease;
- contrastive loss để đồng bộ các view biểu diễn;
- early stopping;
- checkpoint theo fold;
- logging chi tiết theo từng epoch;
- positive weighting cho dữ liệu thưa.

### 4.3. Các cải tiến chính của mô hình

#### 4.3.1. Bổ sung đặc trưng topo tường minh

Một cải tiến quan trọng của đề tài là tách riêng quá trình trích xuất đặc trưng topo trong file `topology_features.py`. Thay vì chỉ để HGT tự suy ra toàn bộ thông tin cấu trúc từ đồ thị, hệ thống chủ động tính toán các đặc trưng như degree chuẩn hóa, weighted degree, clustering coefficient, PageRank và average neighbor degree.

Các đặc trưng này sau đó được đưa qua `TopologyEncoder` để ánh xạ sang cùng không gian biểu diễn với các view còn lại. Cách làm này giúp mô hình khai thác tốt hơn cấu trúc toàn cục của mạng sinh học, đồng thời tăng khả năng giải thích khi phân tích vì sao một thuốc hoặc một bệnh được đánh giá là quan trọng.

#### 4.3.2. Đồng bộ nhiều view bằng contrastive learning

Mô hình cải tiến không chỉ học từ một nguồn thông tin duy nhất mà kết hợp ba nhóm biểu diễn:

- similarity view;
- association view;
- topology view.

Để hạn chế việc các view học lệch nhau quá xa, nhóm bổ sung `MultiViewContrastiveLoss`. Hàm mất mát này có vai trò kéo các biểu diễn liên quan lại gần nhau trong không gian đặc trưng, từ đó giúp quá trình hòa trộn thông tin diễn ra ổn định hơn.

#### 4.3.3. Cải tiến chiến lược fusion

Đề tài hỗ trợ hai hướng hòa trộn biểu diễn chính:

- `mva` dùng bộ tổng hợp đa view (`MultiViewAggregator`);
- `rvg` dùng biểu diễn nền kết hợp `FuzzyGate` để điều tiết mức độ đưa thông tin topo vào biểu diễn cuối.

Riêng với F-dataset, mô hình còn sử dụng `multi-view similarity`, nghĩa là không nén toàn bộ similarity về một graph duy nhất mà học trực tiếp từ nhiều view như fingerprint, GIP và consensus đối với thuốc; phenotype, GIP và consensus đối với bệnh.

#### 4.3.4. Mở rộng phần đầu ra dự đoán liên kết

Ngoài head kiểu gốc dựa trên phép nhân phần tử và MLP, phiên bản cải tiến còn hỗ trợ `interaction pair head`. Head này khai thác đồng thời:

- vector thuốc;
- vector bệnh;
- tích phần tử;
- sai khác tuyệt đối;
- độ tương đồng cosine;
- điểm bilinear.

Nhờ đó, quan hệ giữa hai thực thể được mô hình hóa phong phú hơn so với cách biểu diễn đơn giản chỉ dùng Hadamard product.

#### 4.3.5. Cải tiến pipeline huấn luyện

So với repo gốc, script `train_final.py` đã được nâng cấp đáng kể:

- có preset riêng cho từng bộ dữ liệu;
- hỗ trợ chọn backbone, fusion mode, pair head và gate mode;
- hỗ trợ positive weighting cho dữ liệu mất cân bằng;
- hỗ trợ early stopping và ReduceLROnPlateau;
- sinh file kết quả 10-fold dưới dạng CSV và lưu checkpoint tốt nhất.

Những cải tiến này không chỉ nâng cao kết quả mô hình mà còn giúp quá trình thực nghiệm có tính tái lập và dễ theo dõi hơn.

### 4.4. Kết quả thực nghiệm

Kết quả 10-fold cross-validation của mô hình cải tiến được lưu trong thư mục `Result/improved/`. Giá trị trung bình và độ lệch chuẩn được trình bày ở Bảng 4.2.

**Bảng 4.2. Kết quả trung bình 10-fold của mô hình cải tiến**

| Bộ dữ liệu | AUC | AUPR | Accuracy | Precision | Recall | F1-score | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B-dataset | 0.9354 ± 0.0039 | 0.9319 ± 0.0038 | 0.8633 ± 0.0060 | 0.8673 ± 0.0091 | 0.8580 ± 0.0086 | 0.8626 ± 0.0059 | 0.7268 ± 0.0119 |
| C-dataset | 0.9693 ± 0.0055 | 0.9718 ± 0.0047 | 0.9005 ± 0.0142 | 0.8848 ± 0.0203 | 0.9214 ± 0.0150 | 0.9026 ± 0.0133 | 0.8019 ± 0.0280 |
| F-dataset | 0.9623 ± 0.0071 | 0.9604 ± 0.0107 | 0.8932 ± 0.0250 | 0.8566 ± 0.0390 | 0.9472 ± 0.0193 | 0.8991 ± 0.0211 | 0.7921 ± 0.0457 |

Ngoài giá trị trung bình, mô hình cũng đạt một số kết quả fold tốt nhất đáng chú ý:

- B-dataset: AUC cao nhất đạt 0.9400, AUPR cao nhất đạt 0.9368.
- C-dataset: AUC cao nhất đạt 0.9740, AUPR cao nhất đạt 0.9763.
- F-dataset: AUC cao nhất đạt 0.9740, AUPR cao nhất đạt 0.9782.

#### 4.4.1. Nhận xét trên B-dataset

B-dataset cho kết quả thấp hơn hai bộ còn lại nhưng vẫn đạt AUC trung bình 0.9354 và AUPR trung bình 0.9319. Điều này cho thấy mô hình vẫn giữ được khả năng phân biệt tốt trong bối cảnh dữ liệu khó hơn. Độ lệch chuẩn nhỏ giữa các fold cũng phản ánh tính ổn định chấp nhận được của pipeline huấn luyện.

#### 4.4.2. Nhận xét trên C-dataset

C-dataset là bộ dữ liệu cho kết quả tốt nhất. AUC trung bình đạt 0.9693 và AUPR trung bình đạt 0.9718, đồng thời MCC đạt 0.8019. Đây là bộ dữ liệu thể hiện rõ hiệu quả của hướng cải tiến, đặc biệt là khả năng hòa trộn các nguồn biểu diễn và tối ưu hóa quá trình huấn luyện.

#### 4.4.3. Nhận xét trên F-dataset

F-dataset có đặc trưng thưa và khó hơn, tuy nhiên mô hình vẫn đạt AUC trung bình 0.9623 và Recall trung bình 0.9472. Kết quả này cho thấy chiến lược `rlghgt + rvg + interaction`, kết hợp `multi-view similarity` và positive weighting, có tác dụng tốt trong việc duy trì độ bao phủ cao trên dữ liệu khó.

#### 4.4.4. Đánh giá chung

Từ ba bảng kết quả trên có thể rút ra ba nhận xét chính:

- mô hình cải tiến cho chất lượng dự đoán cao và nhất quán trên nhiều bộ dữ liệu khác nhau;
- C-dataset là bộ dữ liệu phù hợp nhất để thể hiện năng lực đầy đủ của mô hình;
- với dữ liệu thưa như F-dataset, việc dùng thêm multi-view, fuzzy gate và positive weighting giúp tăng Recall rõ rệt mà vẫn giữ AUC ở mức cao.

### 4.5. Kết quả triển khai hệ thống website

Bên cạnh phần lõi học máy, đề tài còn hiện thực hóa mô hình thành một hệ thống website có thể sử dụng và demo trực tiếp. Các chức năng chính đã được triển khai gồm:

- đăng nhập và quản lý phiên người dùng;
- dự đoán theo hai chiều: Thuốc -> Bệnh và Bệnh -> Thuốc;
- lựa chọn bộ dữ liệu và số lượng kết quả Top-K;
- hiển thị danh sách kết quả cùng điểm dự đoán;
- lưu lịch sử tra cứu vào cơ sở dữ liệu;
- trang quản trị để theo dõi dữ liệu và hoạt động hệ thống;
- trực quan hóa đồ thị 3D giữa thuốc, bệnh và protein.

Điểm có ý nghĩa thực tiễn của phần triển khai này là biến một mô hình nghiên cứu vốn chỉ chạy trong môi trường terminal thành một sản phẩm có giao diện, có người dùng, có lịch sử truy vấn và có lớp quản trị dữ liệu. Đây là bước chuyển quan trọng từ nghiên cứu thuật toán sang ứng dụng phần mềm.

### 4.6. Hạn chế của thực nghiệm

Mặc dù đạt được kết quả tích cực, đồ án vẫn còn một số hạn chế:

- kết quả định lượng hiện mới lưu đầy đủ cho mô hình cải tiến; chưa có bảng chạy lại baseline AMDGT gốc trong cùng seed, cùng môi trường để đối chiếu tuyệt đối;
- hệ thống mới dừng ở mức hỗ trợ ra quyết định, chưa có giá trị thay thế kiểm chứng y sinh thực nghiệm;
- giao diện web phục vụ tốt cho mục tiêu demo nhưng vẫn có thể tiếp tục nâng cấp về hiệu năng, bảo mật và trải nghiệm người dùng;
- phần giải thích mô hình mới ở mức thống kê phụ trợ, chưa có module explainability chuyên sâu.

### 4.7. Kết luận chương

Chương này đã trình bày các cải tiến chính của mô hình so với hướng triển khai AMDGT ban đầu, bao gồm bổ sung đặc trưng topo, contrastive learning, cơ chế multi-view fusion, fuzzy gate và pipeline huấn luyện ổn định hơn. Kết quả thực nghiệm trên B-dataset, C-dataset và F-dataset cho thấy mô hình cải tiến đạt chất lượng dự đoán cao, đặc biệt nổi bật trên C-dataset với AUC trung bình 0.9693 và AUPR trung bình 0.9718.

Không chỉ dừng ở mức thực nghiệm học máy, đề tài còn xây dựng thành công một hệ thống website tích hợp mô hình AI, cơ sở dữ liệu và giao diện trực quan. Điều này khẳng định hướng đi của đề tài là đúng đắn: vừa cải tiến lõi thuật toán, vừa hiện thực hóa thành sản phẩm có khả năng ứng dụng và trình diễn trong thực tế.

## 3. Gợi ý chèn hình minh họa trong báo cáo

Nếu muốn báo cáo thuyết phục hơn, nên chèn thêm 3 hình ở cuối Chương 3 hoặc trong Chương 4:

- Hình giao diện đăng nhập và dashboard dự đoán.
- Hình bảng kết quả Top-K và đồ thị 3D sau khi suy luận.
- Hình trang quản trị hoặc lịch sử dự đoán.

Có thể đặt chú thích như sau:

- Hình 4.1. Giao diện dashboard dự đoán liên kết Thuốc - Bệnh.
- Hình 4.2. Kết quả trả về và đồ thị 3D trực quan hóa các thực thể liên quan.
- Hình 4.3. Giao diện quản trị và thống kê hoạt động hệ thống.

## 4. Lưu ý khi đưa vào bản Word

- Chỉnh lại cách gọi cho thống nhất: chỉ dùng một ngôi xưng, tốt nhất là `nhóm` hoặc `nhóm tác giả`.
- Rà lại dấu cách ở các tiêu đề như `1.1 Lý do chọn đề tài`, `2.2.2 Cơ chế chú ý tương hỗ`.
- Bổ sung đầy đủ thông tin trang bìa: sinh viên thực hiện, lớp, niên khóa, ngày tháng.
- Nếu giảng viên yêu cầu chặt chẽ hơn về so sánh baseline, nên thêm một tiểu mục ngắn: `So sánh định tính với AMDGT gốc` thay vì khẳng định định lượng tuyệt đối khi chưa có bảng baseline chạy lại.
