# AMDGT Summary

Repo AMDGT là mô hình dự đoán drug-disease association dựa trên multi-modal learning và dual graph transformer.

## Luồng chính
- `data_preprocess.py`: đọc dữ liệu, tạo mẫu dương/âm, chia k-fold, xây graph DGL
- `model/AMNTDDA.py`: gom embedding từ graph similarity + heterograph + transformer
- `train_DDA.py`: train/evaluate theo k-fold, báo metric AUC/AUPR

## Đặc điểm
- Dùng dữ liệu drug, disease, protein
- Kết hợp graph similarity và heterograph
- Đầu ra là nhãn nhị phân cho cặp drug-disease
