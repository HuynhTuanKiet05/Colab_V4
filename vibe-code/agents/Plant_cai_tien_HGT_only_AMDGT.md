# Đề xuất cải tiến **chỉ module HGT** trong AMDGT theo xu hướng nghiên cứu hiện tại

## 1) Mục tiêu tài liệu
Tài liệu này dùng để giao cho AI agent hoặc coder triển khai một bản cải tiến **chỉ tập trung vào module HGT** trong pipeline AMDGT/AMDGT-style cho bài toán dự đoán liên kết **drug–disease** trên đồ thị dị thể **drug–protein–disease**.

### Ràng buộc cứng
Chỉ được cải tiến **module số 4: Network Feature Extraction (HGT)**.

**Không thay đổi bản chất** của 5 module còn lại:
1. **Similarity Network**
2. **Similarity Feature Extraction**
3. **Biochemical Heterogeneous Network**
4. **Network Feature Extraction**  ← **được phép cải tiến**
5. **Modality Interaction Module**
6. **Prediction Module**

### Mục tiêu kỹ thuật
- Giữ nguyên input/output interface giữa module HGT và các module khác.
- Giữ nguyên pipeline dữ liệu, fusion và decoder ở mức tối đa.
- Chứng minh được rằng cải tiến nằm ở **HGT block**, không phải nhờ thay đổi nơi khác.
- Hướng cải tiến phải bám xu hướng nghiên cứu hiện nay của thế giới trong drug repurposing / DDA prediction.

---

## 2) AMDGT gốc đang làm gì ở 6 module?
Theo bài báo AMDGT, kiến trúc tổng thể gồm 6 khối lớn:

1. **Similarity Network**: xây mạng tương đồng drug và disease.
2. **Similarity Feature Extraction**: dùng Graph Transformer trên mạng similarity để lấy embedding từ góc nhìn similarity.
3. **Biochemical Heterogeneous Network**: xây mạng dị thể gồm drug, disease, protein với các feature như mol2vec, MeSH/disease embedding, ESM-2 protein embedding.
4. **Network Feature Extraction**: dùng **HGT** để học embedding ngữ cảnh trên heterogeneous network.
5. **Modality Interaction Module**: dùng transformer encoder để fusion embedding từ similarity branch và heterogeneous branch.
6. **Prediction Module**: decoder + MLP để dự đoán drug–disease association.

Ý tưởng gốc của AMDGT là thắng nhờ **đa mô thức + dual graph transformer + attention-aware fusion**, không chỉ nhờ riêng HGT. Vì vậy nếu muốn làm nghiên cứu sạch, cần giữ nguyên 5 module còn lại và chỉ thay module HGT. [1][2]

---

## 3) Chẩn đoán điểm còn có thể cải tiến của HGT gốc
HGT gốc trong AMDGT đã tốt, nhưng theo góc nhìn nghiên cứu 2024–2026, vẫn còn vài điểm có thể nâng cấp:

### 3.1 Quan hệ cạnh vẫn chưa được khai thác đủ sâu
HGT gốc có meta-relation, nhưng phần lớn vẫn là attention/message passing theo kiểu chuẩn HGT. Trong bài toán drug–protein–disease, **loại cạnh** cực kỳ quan trọng:
- drug–protein
- protein–disease
- drug–disease
- và các cạnh ngược nếu có

Xu hướng mới là **relation-aware modeling**: attention phải hiểu sâu ý nghĩa từng loại quan hệ, không chỉ phân biệt type ở mức nhẹ. [3][4]

### 3.2 Local neighborhood là chưa đủ
Nhiều mô hình hiện nay chỉ nghe hàng xóm gần (1-hop, 2-hop) nên thiếu **global topology** và **high-order semantics**. Trong bài toán DDA, các quan hệ xa theo dạng meta-path như:
- Drug → Protein → Disease
- Drug → Protein → Drug
- Disease → Protein → Disease

thường mang tín hiệu sinh học mạnh. Xu hướng mới là **local + global + high-order** chứ không chỉ neighborhood aggregation. [5][6][7][8]

### 3.3 Dùng tầng cuối duy nhất dễ làm mất thông tin
Các mô hình gần đây hay dùng **layer-wise attention / layer aggregation / jump knowledge** để giữ cả tín hiệu nông lẫn sâu, giảm over-smoothing. [6]

### 3.4 Hướng thế giới hiện tại
Nếu quy về những hướng nổi bật nhất mà **vẫn phù hợp với ràng buộc chỉ sửa HGT**, thì có 3 dòng chính:
1. **Relation-aware HGT**
2. **Local-global / meta-path guided HGT**
3. **Layer aggregation để giữ nhiều cấp thông tin**

Các hướng như pretraining lớn, contrastive learning, foundation models, cold-start optimization đang rất mạnh trên thế giới, nhưng nếu đụng vào loss/training framework hoặc thêm nhánh ngoài HGT thì sẽ vượt phạm vi “chỉ sửa HGT”. Vì vậy, trong tài liệu này ta chỉ **hấp thụ tinh thần** của các hướng đó vào bên trong HGT block. [5][8][9][10]

---

## 4) Đề xuất mô hình mới: **RLG-HGT**
Tên đề xuất:

> **RLG-HGT = Relation-aware Local-Global Heterogeneous Graph Transformer**

### Ý tưởng lõi
Thay module HGT gốc bằng một HGT mới có 3 nâng cấp nhưng vẫn giữ interface cũ:

1. **Relation-aware attention**
   - Mỗi edge type có embedding riêng.
   - Attention score và message đều được điều chế bởi relation embedding.

2. **Local–global dual branch**
   - Nhánh **local** học từ neighborhood như HGT chuẩn.
   - Nhánh **global** học từ các meta-path / context token đại diện cho quan hệ bậc cao.

3. **Layer-wise aggregation**
   - Không chỉ lấy output tầng cuối.
   - Học trọng số để trộn embedding từ nhiều tầng.

### Giữ nguyên cái gì?
- Giữ nguyên similarity branch.
- Giữ nguyên biochemical heterogeneous network và feature đầu vào.
- Giữ nguyên AttentionFusion.
- Giữ nguyên decoder + MLP + loss.
- Giữ nguyên output shape của HGT: vẫn sinh ra embedding cho **drug** và **disease** để đưa sang module fusion.

---

## 5) Toán học của HGT gốc trong AMDGT
Gọi heterogeneous biochemical network là:

\[
\mathcal{A}_N = (\mathcal{V}, \mathcal{E}, \mathcal{F})
\]

trong đó:
- \(\mathcal{V}\): tập node (drug, disease, protein)
- \(\mathcal{E}\): tập cạnh
- \(\mathcal{F}\): feature của node

Ký hiệu:
- \(\tau(v)\): loại node của \(v\)
- \(\phi(e)\): loại cạnh của \(e\)
- \(h_i^{(l-1)}\): embedding của node \(i\) tại tầng \(l-1\)

Trong HGT gốc, với head \(k\):

\[
Q_i^k = W_{Q,\tau(i)}^k h_i^{(l-1)}
\]

\[
K_j^k = W_{K,\tau(j)}^k h_j^{(l-1)}
\]

\[
V_j^k = W_{V,\tau(j)}^k h_j^{(l-1)}
\]

Attention theo meta-relation:

\[
\alpha_{ij}^k = \operatorname{softmax}_j\left(
\frac{K_j^k W^{ATT}_{\phi(i,j)} (Q_i^k)^T}{\sqrt{d_k}} \cdot
\mu_{\langle \tau(j), \phi(i,j), \tau(i) \rangle}
\right)
\]

Message:

\[
M_{ij}^k = V_j^k W^{MSG}_{\phi(i,j)}
\]

Aggregation:

\[
\hat{h}_i^{(l)} = \operatorname{Aggregate}_{j \in \mathcal{N}(i)}
\left( \alpha_{ij}^k \cdot M_{ij}^k \right)
\]

Residual + activation:

\[
h_i^{(l)} = \operatorname{ReLU}(\hat{h}_i^{(l)}) + h_i^{(l-1)}
\]

Đây là nền rất tốt, nhưng có thể mạnh hơn nếu relation semantics và global context được mô hình hóa rõ hơn. [1]

---

## 6) Toán học của module cải tiến **RLG-HGT**

## 6.1 Tổng quan
Mỗi tầng \(l\) của RLG-HGT gồm 4 bước:
1. **Relation-aware local attention**
2. **Meta-path/global context extraction**
3. **Gated local-global fusion**
4. **FFN + residual + layer aggregation**

---

## 6.2 Relation-aware local attention
Với mỗi loại cạnh \(r = \phi(i,j)\), gán một relation embedding:

\[
e_r \in \mathbb{R}^{d_r}
\]

Trên head \(k\):

\[
Q_i^k = W_{Q,\tau(i)}^k h_i^{(l-1)}
\]

\[
K_j^k = W_{K,\tau(j)}^k h_j^{(l-1)}
\]

\[
V_j^k = W_{V,\tau(j)}^k h_j^{(l-1)}
\]

Tăng cường key/value bằng relation embedding:

\[
\widetilde{K}_{j,r}^k = K_j^k + W_{K,r}^k e_r
\]

\[
\widetilde{V}_{j,r}^k = V_j^k + W_{V,r}^k e_r
\]

Attention score mới:

\[
s_{ij}^{k,local} =
\frac{(Q_i^k)^T \widetilde{K}_{j,r}^k}{\sqrt{d_k}} + b_r^k
\]

Chuẩn hóa:

\[
\alpha_{ij}^{k,local} = \operatorname{softmax}_{j \in \mathcal{N}(i)}(s_{ij}^{k,local})
\]

Message local:

\[
m_{i,local}^k = \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{k,local} \widetilde{V}_{j,r}^k
\]

Ghép nhiều head:

\[
m_i^{local} = \operatorname{Concat}_{k=1}^{H}(m_{i,local}^k)
\]

### Ý nghĩa
- Quan hệ **drug-targets-protein** và **protein-associates-disease** sẽ không còn bị đối xử quá gần nhau.
- Mỗi loại cạnh có bias và embedding riêng, giúp attention thật sự “relation-aware”.

---

## 6.3 Meta-path / global context branch
Đây là nhánh để học thông tin bậc cao và global hơn.

### Chọn meta-path
Trong phạm vi drug–protein–disease, dùng các meta-path ngắn, dễ giải thích:
- \(P_1\): Drug → Protein → Disease
- \(P_2\): Drug → Protein → Drug
- \(P_3\): Disease → Protein → Disease
- \(P_4\): Drug → Disease → Drug
- \(P_5\): Disease → Drug → Disease

Không bắt buộc dùng hết; có thể chọn tập con tùy graph thực tế.

Với node \(i\), ký hiệu tập node/path-instance đạt được theo meta-path \(p\) là:

\[
\mathcal{N}_p(i)
\]

Tạo vector đại diện cho meta-path \(p\):

\[
g_i^{(p,l)} = \operatorname{Pool}\left(\{h_u^{(l-1)} \mid u \in \mathcal{N}_p(i)\}\right)
\]

Trong đó `Pool` có thể là mean-pooling, attention-pooling hoặc top-k attentive pooling.

Tính trọng số giữa các meta-path:

\[
\beta_i^{(p,l)} =
\frac{\exp\left(a_p^T \tanh(W_p g_i^{(p,l)})\right)}
{\sum_{q \in \mathcal{P}} \exp\left(a_q^T \tanh(W_q g_i^{(q,l)})\right)}
\]

Tạo message global:

\[
m_i^{global} = \sum_{p \in \mathcal{P}} \beta_i^{(p,l)} W_p^{out} g_i^{(p,l)}
\]

### Ý nghĩa
- Nhánh local học “hàng xóm gần”.
- Nhánh global/meta-path học “quy luật bậc cao” và “ngữ nghĩa đường đi”.
- Đây là điểm rất hợp xu hướng recent papers: local-global complementary learning, meta-path guidance, high-order topology. [5][6][7][8]

---

## 6.4 Gated local-global fusion
Trộn nhánh local và global bằng gate học được:

\[
z_i^{(l)} = \sigma\left(W_z [m_i^{local} \| m_i^{global} \| h_i^{(l-1)}] + b_z\right)
\]

\[
u_i^{(l)} = z_i^{(l)} \odot m_i^{local} + (1-z_i^{(l)}) \odot m_i^{global}
\]

Chiếu về không gian ẩn và cộng residual:

\[
\tilde{h}_i^{(l)} = \operatorname{LayerNorm}\left(h_i^{(l-1)} + W_o u_i^{(l)}\right)
\]

Sau đó qua FFN:

\[
h_i^{(l)} = \operatorname{LayerNorm}\left(\tilde{h}_i^{(l)} + \operatorname{FFN}(\tilde{h}_i^{(l)})\right)
\]

### Ý nghĩa
- Gate giúp mô hình tự quyết định khi nào nên tin **local evidence**, khi nào nên tin **global/meta-path evidence**.
- Đây là cách nâng HGT mà không đụng các module fusion phía sau.

---

## 6.5 Layer-wise aggregation
Thay vì chỉ lấy \(h_i^{(L)}\), học trọng số cho từng tầng:

\[
\lambda_i^{(l)} =
\frac{\exp\left(q^T \tanh(W_l h_i^{(l)})\right)}
{\sum_{t=1}^{L} \exp\left(q^T \tanh(W_t h_i^{(t)})\right)}
\]

Output cuối:

\[
h_i^{out} = \sum_{l=1}^{L} \lambda_i^{(l)} h_i^{(l)}
\]

Sau đó lấy các embedding đầu ra tương ứng cho drug và disease:

\[
H_{RN} = \{ h_i^{out} \mid i \in \text{Drug nodes} \}
\]

\[
H_{DN} = \{ h_i^{out} \mid i \in \text{Disease nodes} \}
\]

**Giữ nguyên shape** để module AttentionFusion phía sau dùng được ngay.

---

## 7) Vì sao đề xuất này bám đúng xu hướng thế giới?

### 7.1 Relation-aware modeling
Các công trình kiểu RHGT cho thấy chỉ node-level embedding là chưa đủ; phải đưa thông tin relation/subtype vào rõ hơn để khai thác mạng dị thể sinh học sâu hơn. [3]

### 7.2 Local + global + high-order
Các công trình mới như MRDDA, HNF-DDA, MedPathEx và MAPTrans đều nhấn mạnh rằng chỉ học local neighborhood là chưa đủ. Họ thêm global structure, meta-path, high-order topology hoặc multiview interactions để tăng sức biểu diễn và tổng quát hóa. [5][6][7][8]

### 7.3 Foundation-model era nhưng trong phạm vi hợp lệ
Thế giới hiện nay đang đi rất mạnh theo foundation models và biểu diễn đa mô thức. Tuy nhiên, trong phạm vi “chỉ sửa HGT”, ta không thay feature extractor hay training framework; thay vào đó ta thiết kế HGT sao cho **khai thác tốt hơn** các feature giàu sẵn có như mol2vec, disease embeddings, ESM protein embeddings. Đây là lựa chọn thực dụng và đúng ranh giới bài toán. [9][10]

---

## 8) Phạm vi **không** làm trong vòng cải tiến này
Để đảm bảo nghiên cứu sạch, AI agent **không được** làm các việc sau trong version chính:

1. Không thay đổi Similarity Network.
2. Không thay đổi Similarity Feature Extraction.
3. Không thêm modality mới vào data pipeline.
4. Không sửa AttentionFusion.
5. Không đổi decoder từ dot-product + MLP sang kiến trúc khác.
6. Không đổi loss từ BCE/cross-entropy sang framework multitask hay contrastive toàn cục.
7. Không thêm pretraining lớn ngoài HGT.

Các ý như contrastive learning, multitask affinity prediction, pretraining là hướng rất mạnh của thế giới, nhưng nên để ở **future work**, không nên đưa vào bản “chỉ sửa HGT”.

---

## 9) Pseudocode triển khai cho AI agent
```text
Input:
    Heterogeneous graph G = (V, E)
    Node features X
    Node types tau(v)
    Edge types phi(e)
    Number of layers L
    Number of heads H
    Meta-path set P

Output:
    Drug embeddings H_RN
    Disease embeddings H_DN

Procedure RLG-HGT:
    Initialize h_i^(0) = TypeSpecificProjection(x_i)

    for l = 1..L:
        # 1) Relation-aware local branch
        for each edge (j -> i) with relation r:
            compute Q_i^k, K_j^k, V_j^k for each head k
            enhance K and V by relation embedding e_r
            compute local attention alpha_ij^(k,local)
        aggregate local messages to get m_i^local

        # 2) Global / meta-path branch
        for each node i:
            for each meta-path p in P:
                collect path-based neighborhood N_p(i)
                pool node states to get g_i^(p,l)
            compute path attention beta_i^(p,l)
            aggregate to get m_i^global

        # 3) Gated fusion
        z_i = sigmoid(Wz [m_i^local || m_i^global || h_i^(l-1)] + bz)
        u_i = z_i * m_i^local + (1-z_i) * m_i^global

        # 4) Residual + FFN
        h_i^(l) = LayerNorm(h_i^(l-1) + Wo u_i)
        h_i^(l) = LayerNorm(h_i^(l) + FFN(h_i^(l)))

    # 5) Layer-wise aggregation
    for each node i:
        h_i^out = sum_l lambda_i^(l) * h_i^(l)

    return embeddings of drug nodes and disease nodes
```

---

## 10) Hướng dẫn triển khai thực tế trên repo của bạn
Nếu áp dụng vào repo PyG/HGT hiện tại của bạn, ưu tiên cách ít phá hệ thống nhất:

### 10.1 Cấu trúc code nên thêm
- `src/model_hgt_rlg.py` hoặc `src/modules/rlg_hgt.py`
- `RelationAwareHGTLayer`
- `MetaPathGlobalBlock`
- `LayerAggregator`

### 10.2 Interface nên giữ nguyên
Model mới vẫn nên trả về:
- embedding drug
- embedding disease

đúng dimension mà decoder/fusion hiện tại đang chờ.

### 10.3 Thứ tự triển khai an toàn
**Bước 1**: làm `RelationAwareHGTLayer` trước.
- Nếu chỉ thay bước attention/message mà không đụng output shape, đây là upgrade dễ nhất.

**Bước 2**: thêm `MetaPathGlobalBlock`.
- Có thể bắt đầu bằng mean-pooling theo meta-path trước, chưa cần attention quá phức tạp.

**Bước 3**: thêm `Gated Fusion`.
- Bảo đảm local/global cùng dimension.

**Bước 4**: thêm `LayerAggregator`.
- Nếu chưa ổn định, có thể bật/tắt bằng config.

### 10.4 Config đề nghị
```yaml
model: rlg_hgt
hidden_dim: 256
num_layers: 3
num_heads: 4
use_relation_bias: true
use_global_branch: true
meta_paths:
  - drug-protein-disease
  - drug-protein-drug
  - disease-protein-disease
use_layer_aggregation: true
dropout: 0.2
```

---

## 11) Kế hoạch ablation bắt buộc
Để chứng minh đóng góp nằm ở HGT, cần chạy tối thiểu 4 phiên bản:

### A. Baseline gốc
- **AMDGT-HGT-Original**

### B. Chỉ relation-aware
- **R-HGT**
- Chỉ thêm relation embeddings vào attention/message

### C. Chỉ local-global
- **LG-HGT**
- Giữ HGT local cũ, thêm global/meta-path branch

### D. Đầy đủ
- **RLG-HGT**
- relation-aware + local-global + gated fusion + layer aggregation

### E. Tuỳ chọn
- **RLG-HGT w/o LayerAgg**
- để xem layer aggregation có thật sự đóng góp không

---

## 12) Kế hoạch thực nghiệm

### 12.1 Metric
Giữ đúng thói quen của AMDGT/repo hiện tại:
- AUC
- AUPR
- Accuracy
- Precision
- Recall
- F1
- MCC

Trong bài toán DDA, **AUPR** rất quan trọng vì dữ liệu thường thưa và mất cân bằng. [1]

### 12.2 Split
Ít nhất phải có:
1. **Random split / 10-fold CV**
2. **Drug-cold split** (nếu đủ thời gian)
3. **Disease-cold split** (nếu đủ thời gian)

Lý do: nhiều bài gần đây nhấn mạnh khả năng generalize cho unseen drugs/diseases mới là thứ có giá trị thực tế. [5][8]

### 12.3 Hyperparameter search gọn
Để không nổ chi phí:
- hidden_dim ∈ {128, 256, 512}
- num_layers ∈ {2, 3, 4}
- num_heads ∈ {4, 8}
- meta_path_topk ∈ {8, 16}
- dropout ∈ {0.1, 0.2, 0.3}

Gợi ý bắt đầu:
- C/F dataset: `hidden_dim = 256`
- B dataset: có thể thử `512` nếu tài nguyên đủ

---

## 13) Tiêu chí thành công của bản cải tiến
AI agent chỉ nên coi bản cải tiến là “thành công” nếu thỏa các điều kiện:

1. **Không làm vỡ interface** với module 5 và 6.
2. **Không dùng thêm dữ liệu ngoài scope**.
3. **AUPR tăng ổn định** qua nhiều seed, không chỉ AUC.
4. **Variance không tăng quá mạnh**.
5. Ở ablation, **RLG-HGT > HGT gốc** một cách nhất quán.
6. Nếu có cold split, bản mới phải ít nhất **không tệ hơn rõ rệt** trên cold-start.

---

## 14) Rủi ro kỹ thuật và cách né

### 14.1 Meta-path quá nhiều → nặng và nhiễu
Giải pháp:
- chỉ dùng meta-path ngắn độ dài 2 hoặc 3
- top-k sampling theo attention hoặc độ liên quan

### 14.2 Over-smoothing khi tăng số tầng
Giải pháp:
- giữ 2–4 tầng
- dùng layer aggregation
- dùng residual + layer norm

### 14.3 Relation-aware quá mạnh → overfit
Giải pháp:
- relation embedding dimension nhỏ
- dropout trên attention hoặc message
- weight decay hợp lý

### 14.4 Tăng AUC nhưng không tăng AUPR
Giải pháp:
- theo dõi AUPR là metric chính
- kiểm tra hard negatives
- đánh giá bằng nhiều seed

---

## 15) Phiên bản triển khai tối thiểu nên làm trước
Nếu thời gian có hạn, AI agent nên ưu tiên theo thứ tự này:

### Version 1
**R-HGT**
- chỉ thêm relation-aware attention/message
- dễ triển khai nhất
- rất đúng xu hướng

### Version 2
**RLG-HGT-lite**
- thêm 1 global branch đơn giản bằng meta-path mean pooling
- fusion bằng gate

### Version 3
**RLG-HGT-full**
- thêm layer-wise aggregation
- tinh chỉnh hyperparameter

Thứ tự này giúp bạn có kết quả trung gian nhanh, tránh rủi ro làm một phát quá lớn rồi không chạy được.

---

## 16) Prompt ngắn gọn giao cho AI agent
```text
Bạn hãy cải tiến duy nhất module HGT trong pipeline AMDGT-style cho bài toán drug-protein-disease link prediction.

Ràng buộc cứng:
- Không thay đổi 5 module còn lại: Similarity Network, Similarity Feature Extraction, Biochemical Heterogeneous Network, Modality Interaction, Prediction Module.
- Không thay đổi dữ liệu đầu vào, decoder, loss, hoặc fusion ngoài HGT.
- Giữ nguyên output interface của HGT để trả về embedding drug và disease với cùng shape như bản cũ.

Mục tiêu:
- Thay HGT gốc bằng Relation-aware Local-Global HGT (RLG-HGT).
- Trong local branch, attention và message phải phụ thuộc edge type embedding.
- Trong global branch, dùng meta-path ngắn để lấy high-order context.
- Dùng gated fusion để trộn local/global.
- Dùng layer-wise aggregation thay vì chỉ lấy tầng cuối.

Bắt buộc tạo các ablation:
1. HGT gốc
2. R-HGT
3. LG-HGT
4. RLG-HGT
5. RLG-HGT w/o LayerAgg (nếu có)

Metric chính: AUPR, sau đó mới đến AUC.
Ưu tiên code sạch, giữ backward compatibility, có config bật/tắt từng block.
```

---

## 17) Kết luận chốt hướng
Nếu chỉ được cải tiến **một module duy nhất là HGT**, thì hướng đáng làm nhất hiện nay là:

> **Thay HGT gốc bằng một HGT mới có relation-aware attention + local-global/meta-path context + layer aggregation, nhưng vẫn giữ nguyên interface với các module còn lại.**

Đây là lựa chọn:
- đủ mạnh để có cơ hội tăng kết quả,
- đủ sạch để chứng minh đóng góp khoa học,
- đủ hợp xu hướng thế giới 2024–2026,
- và đủ thực dụng để một AI agent/coder có thể triển khai trong repo hiện tại.

---

## 18) Tài liệu tham khảo định hướng
[1] Liu J, Guan S, Zou Q, Wu H, Tiwari P, Ding Y. **AMDGT: Attention aware multi-modal fusion using a dual graph transformer for drug–disease associations prediction**. *Knowledge-Based Systems*, 2024. DOI: 10.1016/j.knosys.2023.111329.

[2] JK-Liu7. **AMDGT GitHub repository**. README mô tả dữ liệu, môi trường và script huấn luyện.

[3] Relation-aware Heterogeneous Graph Transformer based drug repurposing (**RHGT**). *Expert Systems with Applications*, 2022.

[4] HGTDR: **Advancing drug repurposing with heterogeneous graph transformers**. *Bioinformatics*, 2024.

[5] Chen C et al. **MRDDA: a multi-relational graph neural network for drug–disease association prediction**. *Journal of Translational Medicine*, 2025.

[6] HNF-DDA: **subgraph contrastive-driven transformer-style heterogeneous network embedding for DDA prediction**. *BMC Biology*, 2025.

[7] MAPTrans: **mutual attention transformer with dynamic meta-path pruning for drug–disease association prediction**. *Briefings in Bioinformatics*, 2025.

[8] MedPathEx: **multimodal data integration and meta-path guided global-local feature fusion**. *Scientific Reports*, 2026.

[9] **Foundation models in drug discovery: Phenomenal growth today, transformative potential tomorrow?** *Drug Discovery Today*, 2025.

[10] **Foundation models in bioinformatics**. *National Science Review*, 2025.
