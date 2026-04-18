# AMDGT Context File

## 1. Tổng quan dự án
AMDGT là một repo nghiên cứu về dự đoán mối quan hệ `drug-disease association` bằng học sâu đồ thị đa mô thức. Mục tiêu chính của mô hình là tận dụng đồng thời:

- đặc trưng của drug
- đặc trưng của disease
- đặc trưng của protein
- graph similarity giữa các thực thể
- heterograph biểu diễn quan hệ drug-disease-protein
- transformer / graph transformer để học biểu diễn giàu ngữ cảnh

Repo mang tính chất **research code**: ưu tiên tái hiện thí nghiệm hơn là tối ưu kiến trúc phần mềm, vì vậy nhiều giả định được mã hóa trực tiếp trong script train và tiền xử lý.

## 2. Thành phần cốt lõi

### `train_DDA.py`
Đây là entrypoint chính để huấn luyện và đánh giá mô hình.

Nó thực hiện các bước:
1. parse tham số cấu hình bằng argparse
2. xác định thư mục dữ liệu và thư mục kết quả
3. load dữ liệu bằng `get_data(args)`
4. tiền xử lý và tạo mẫu bằng `data_processing(data, args)`
5. chia k-fold bằng `k_fold(data, args)`
6. tạo graph similarity bằng `dgl_similarity_graph(data, args)`
7. lặp qua từng fold:
   - khởi tạo model `AMNTDDA`
   - tạo heterograph theo train split của fold
   - train theo epoch
   - evaluate bằng AUC, AUPR, Accuracy, Precision, Recall, F1, MCC
8. in thống kê trung bình và độ lệch chuẩn của AUC/AUPR

### `data_preprocess.py`
Đây là file quan trọng nhất cho pipeline dữ liệu.

Các chức năng chính:
- `get_data(args)`: đọc CSV và trả về dict dữ liệu
- `data_processing(data, args)`: tạo positive/negative samples, chuẩn hóa mẫu, sinh nhãn
- `k_fold(data, args)`: chia stratified k-fold và lưu file train/test của từng fold
- `dgl_similarity_graph(data, args)`: tạo graph DGL cho drug-drug và disease-disease similarity
- `dgl_heterograph(data, drdi, args)`: tạo heterograph drug-disease-protein cho từng fold

### `model/AMNTDDA.py`
Đây là kiến trúc mô hình trung tâm.

Luồng suy luận:
1. `gt_net_drug` và `gt_net_disease` học embedding từ graph similarity
2. projection tuyến tính đưa feature drug/protein về cùng không gian
3. tạo heterograph DGL với node type: `drug`, `disease`, `protein`
4. áp dụng `HGTConv` nhiều tầng để học representation trên heterograph
5. ghép embedding từ graph similarity và HGT
6. đưa qua transformer encoder / transformer để tinh chỉnh biểu diễn
7. lấy embedding của cặp drug-disease theo sample và nhân từng phần tử
8. qua MLP để dự đoán nhị phân

## 3. Dữ liệu và ý nghĩa
Repo dùng dữ liệu dạng CSV, đã được chuẩn bị sẵn trong các thư mục dataset.

### Các dataset
- `B-dataset`
- `C-dataset`
- `F-dataset`

### Các file chính
- `DrugFingerprint.csv`: fingerprint của drug
- `DrugGIP.csv`: Gaussian interaction profile cho drug
- `DiseasePS.csv`: đặc trưng disease
- `DiseaseGIP.csv`: GIP cho disease
- `Drug_mol2vec.csv`: embedding mol2vec của drug
- `DiseaseFeature.csv`: embedding disease
- `Protein_ESM.csv`: embedding protein từ ESM
- `DrugDiseaseAssociationNumber.csv`: quan hệ drug-disease đã biết
- `DrugProteinAssociationNumber.csv`: quan hệ drug-protein đã biết
- `ProteinDiseaseAssociationNumber.csv`: quan hệ protein-disease đã biết

### Ý nghĩa thực nghiệm
Dữ liệu được dùng để tạo hai loại cấu trúc:
- **similarity graph**: drug-drug và disease-disease
- **heterograph**: drug-disease-protein

Điều này cho phép mô hình khai thác đồng thời:
- tương đồng nội tại giữa các thực thể cùng loại
- tín hiệu quan hệ chéo giữa các loại thực thể

## 4. Pipeline logic

### Bước 1: đọc dữ liệu
`get_data(args)` đọc toàn bộ CSV và lưu vào dict `data`.

### Bước 2: tạo mẫu huấn luyện
`data_processing(data, args)`:
- chuyển ma trận association về dạng chỉ mục
- tách cặp dương và âm
- shuffle với `random_seed`
- lấy số âm theo `negative_rate`
- tạo tập sample có nhãn

### Bước 3: chia k-fold
`k_fold(data, args)` dùng `StratifiedKFold` trên toàn bộ cặp drug-disease đã gán nhãn.

Lưu ý quan trọng:
- script hiện có hành vi ghi ra file `fold/i/data_train.csv` và `data_test.csv`
- thư mục `fold` phải tồn tại trước khi chạy, nếu không sẽ lỗi khi ghi file

### Bước 4: dựng graph similarity
`dgl_similarity_graph(data, args)`:
- dùng hàm `k_matrix` để giữ lại `k` láng giềng gần nhất
- chuyển sang `networkx` rồi sang `dgl.graph`
- gắn feature của node vào `ndata`

### Bước 5: dựng heterograph theo fold
`dgl_heterograph(data, drdi, args)`:
- tạo graph ba node type
- quan hệ gồm:
  - `drug -> disease`
  - `drug -> protein`
  - `disease -> protein`
- dùng cho HGTConv học biểu diễn liên miền

## 5. Kiến trúc mô hình
Mô hình `AMNTDDA` có thể hiểu như ba tầng biểu diễn:

### Tầng 1: similarity-based representation
- graph drug-drug
- graph disease-disease
- output là embedding theo từng node

### Tầng 2: heterograph-based representation
- HGTConv trên graph đa loại node
- output là embedding ngữ cảnh hóa theo liên hệ giữa drug, disease, protein

### Tầng 3: fusion and prediction
- transformer để kết hợp hai nguồn biểu diễn
- embedding cặp drug-disease được trộn bằng phép nhân Hadamard
- MLP nhị phân hóa đầu ra

## 6. Điểm cần chú ý khi làm việc với repo

### 6.1. Phụ thuộc môi trường
README gốc yêu cầu:
- Python 3.9.13
- PyTorch 1.10.0
- DGL 0.9.0
- networkx 2.8.4
- numpy 1.23.1
- scikit-learn 0.24.2

Đây là môi trường khá cũ, nên nếu chạy trên Python/PyTorch mới hơn có thể gặp lỗi API hoặc tương thích CUDA/DGL.

### 6.2. Dữ liệu cứng đường dẫn
Code đang nối đường dẫn theo kiểu:
- `data/` + dataset + `/`

Vì vậy repo phụ thuộc mạnh vào cấu trúc thư mục hiện tại.

### 6.3. Giả định GPU
Nhiều chỗ dùng:
- `device = torch.device('cuda')`

Nghĩa là code hiện mặc định chạy GPU. Nếu chạy máy không có CUDA sẽ lỗi ngay từ đầu.

### 6.4. Một số vấn đề kỹ thuật tiềm ẩn
- `StratifiedKFold(random_state=None, shuffle=False)` làm việc chia fold mang tính cố định nhưng không có seed thực sự cho fold split
- file `fold/` có thể chưa được tạo sẵn
- có khả năng mismatch shape giữa feature và layer nếu dữ liệu đầu vào không đúng format
- `torch.sparse.LongTensor(...)` là API cũ, có thể cần chỉnh nếu nâng version PyTorch
- một số code mang tính nghiên cứu, chưa có kiểm tra lỗi đầu vào đầy đủ

## 7. Khi AI bắt đầu chat mới, nên hiểu repo này thế nào
Nếu mở một cuộc chat mới để làm việc với repo AMDGT, AI nên giữ bối cảnh sau:

- đây là repo nghiên cứu cho bài toán drug-disease association prediction
- ưu tiên hiểu pipeline dữ liệu và luồng huấn luyện trước khi refactor
- thay đổi nhỏ nhưng cần đảm bảo tương thích với dữ liệu và graph DGL
- mọi sửa đổi nên cân nhắc ảnh hưởng đến shape tensor, graph construction và k-fold pipeline
- khi sửa model cần kiểm tra lại cả `train_DDA.py`, `data_preprocess.py`, `model/AMNTDDA.py`

## 8. Prompt ngắn gọn để tái sử dụng trong chat mới

> Bạn là chuyên gia CNTT đầu ngành. Hãy đọc repo AMDGT như một research code về dự đoán drug-disease association. Ưu tiên nắm pipeline dữ liệu, graph construction, mô hình AMNTDDA, và các rủi ro tương thích môi trường. Trước khi đề xuất thay đổi, hãy tóm tắt kiến trúc repo, xác định luồng dữ liệu, và chỉ ra các điểm dễ lỗi nếu chạy trên môi trường hiện đại.

## 9. Tóm tắt một dòng
AMDGT là một mô hình nghiên cứu kết hợp graph similarity, heterograph và transformer để dự đoán tương tác drug-disease, với pipeline dữ liệu và mô hình được ghép chặt trong các script Python chính.
