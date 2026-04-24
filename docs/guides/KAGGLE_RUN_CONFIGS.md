# Kaggle Run Configs

Tai lieu nay luu san cac lenh copy nhanh de chay tren Kaggle.

- Moi truong `F-dataset` va `C-dataset` dung chung duoc.
- Khong can cai lai package neu da cai xong va notebook chua reset session.
- Nho dung `result_root` khac nhau de ket qua khong de len nhau.

## 1. Clone repo

```python
%cd /kaggle/working
!rm -rf /kaggle/working/Colab_V4
!git clone https://github.com/HuynhTuanKiet05/Colab_V4.git
%cd /kaggle/working/Colab_V4
```

## 2. Cai moi truong

```python
%cd /kaggle/working/Colab_V4

!pip uninstall -y torch torchvision torchaudio dgl dglgo torchdata numpy pandas scikit-learn networkx

!pip install --no-cache-dir --force-reinstall \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121

!pip install --no-cache-dir --force-reinstall \
  numpy==1.26.4 \
  pandas==2.2.2 \
  scikit-learn==1.6.1 \
  networkx==3.2.1 \
  torchdata==0.8.0 \
  pyTelegramBotAPI \
  "jedi>=0.19.1"

!pip install --no-cache-dir --force-reinstall \
  dgl==2.4.0+cu121 \
  -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
```

## 3. Kiem tra GPU

```python
%cd /kaggle/working/Colab_V4
import torch, dgl
print(torch.__version__)
print(dgl.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
```

## 4. Smoke test F-dataset

```python
%cd /kaggle/working/Colab_V4
!python scripts/colab_train.py --dataset F-dataset --preset smoke
```

## 5. F-dataset ban safe

Day la ban da dat mean AUC khoang `0.96233`.

```python
%cd /kaggle/working/Colab_V4

!python train_final.py \
  --dataset F-dataset \
  --device cuda \
  --epochs 1000 \
  --k_fold 10 \
  --patience 220 \
  --assoc_backbone rlghgt \
  --fusion_mode rvg \
  --pair_mode interaction \
  --gate_mode vector \
  --gate_bias_init -1.2 \
  --neighbor 8 \
  --gt_out_dim 320 \
  --hgt_in_dim 320 \
  --hgt_layer 2 \
  --lambda_cl 0.06 \
  --grad_clip 5.0 \
  --save_checkpoints \
  --result_root /kaggle/working/F_dataset_auc_boost_v2_safe
```

## 6. F-dataset ban day du

```python
%cd /kaggle/working/Colab_V4

!python train_final.py \
  --dataset F-dataset \
  --device cuda \
  --epochs 1000 \
  --k_fold 10 \
  --patience 220 \
  --assoc_backbone rlghgt \
  --fusion_mode rvg \
  --pair_mode interaction \
  --gate_mode vector \
  --gate_bias_init -1.2 \
  --lambda_cl 0.06 \
  --grad_clip 5.0 \
  --save_checkpoints \
  --result_root /kaggle/working/F_dataset_auc_boost_v2
```

## 7. B-dataset ban manh nhat nen thu dau tien

`B-dataset` day hon `C/F`, nen uu tien cau hinh on dinh:
- `vanilla_hgt + mva + mul_mlp`
- `neighbor=3`
- embedding lon `512`
- khong bat `multi-view` va `positive weighting`

```python
%cd /kaggle/working/Colab_V4

!python train_final.py \
  --dataset B-dataset \
  --device cuda \
  --epochs 1000 \
  --k_fold 10 \
  --patience 220 \
  --random_seed 1234 \
  --assoc_backbone vanilla_hgt \
  --fusion_mode mva \
  --pair_mode mul_mlp \
  --gate_mode vector \
  --neighbor 3 \
  --lr 1e-4 \
  --weight_decay 1e-3 \
  --dropout 0.15 \
  --gt_out_dim 512 \
  --hgt_in_dim 512 \
  --hgt_layer 2 \
  --hgt_head 8 \
  --gt_layer 2 \
  --gt_head 2 \
  --tr_layer 2 \
  --tr_head 4 \
  --topo_hidden 192 \
  --lambda_cl 0.06 \
  --label_smoothing 0 \
  --similarity_view_mode consensus \
  --positive_weight_mode none \
  --grad_clip 5.0 \
  --save_checkpoints \
  --result_root /kaggle/working/B_dataset_max_v1
```

## 8. C-dataset ban sanh AUC 0.97

Ban nay giu sat huong cu da co ket qua rat gan `0.97`.

```python
%cd /kaggle/working/Colab_V4

!python train_final.py \
  --dataset C-dataset \
  --device cuda \
  --epochs 1000 \
  --k_fold 10 \
  --patience 300 \
  --random_seed 1234 \
  --assoc_backbone vanilla_hgt \
  --fusion_mode mva \
  --pair_mode mul_mlp \
  --gate_mode vector \
  --neighbor 5 \
  --lr 1e-4 \
  --weight_decay 1e-3 \
  --dropout 0.2 \
  --gt_out_dim 256 \
  --hgt_in_dim 256 \
  --hgt_layer 2 \
  --hgt_head 8 \
  --gt_layer 2 \
  --gt_head 2 \
  --tr_layer 2 \
  --tr_head 4 \
  --topo_hidden 128 \
  --lambda_cl 0.10 \
  --label_smoothing 0 \
  --similarity_view_mode consensus \
  --positive_weight_mode none \
  --grad_clip 5.0 \
  --save_checkpoints \
  --result_root /kaggle/working/C_dataset_auc_hunt_seed1234
```

Neu can chay lai voi seed khac:

- `--random_seed 2026`
- `--random_seed 3407`

## 9. C-dataset ban neighbor 7 de thu them

Ban nay giu `seed 1234`, chi doi `neighbor` tu `5` len `7`.

```python
%cd /kaggle/working/Colab_V4

!python train_final.py \
  --dataset C-dataset \
  --device cuda \
  --epochs 1000 \
  --k_fold 10 \
  --patience 300 \
  --random_seed 1234 \
  --assoc_backbone vanilla_hgt \
  --fusion_mode mva \
  --pair_mode mul_mlp \
  --gate_mode vector \
  --neighbor 7 \
  --lr 1e-4 \
  --weight_decay 1e-3 \
  --dropout 0.2 \
  --gt_out_dim 256 \
  --hgt_in_dim 256 \
  --hgt_layer 2 \
  --hgt_head 8 \
  --gt_layer 2 \
  --gt_head 2 \
  --tr_layer 2 \
  --tr_head 4 \
  --topo_hidden 128 \
  --lambda_cl 0.10 \
  --label_smoothing 0 \
  --similarity_view_mode consensus \
  --positive_weight_mode none \
  --grad_clip 5.0 \
  --save_checkpoints \
  --result_root /kaggle/working/C_dataset_auc_hunt_seed1234_n7
```

## 10. Zip ket qua de tai ve

```python
!zip -r /kaggle/working/F_dataset_auc_boost_v2.zip /kaggle/working/F_dataset_auc_boost_v2
```

```python
!zip -r /kaggle/working/F_dataset_auc_boost_v2_safe.zip /kaggle/working/F_dataset_auc_boost_v2_safe
```

```python
!zip -r /kaggle/working/C_dataset_auc_hunt_seed1234.zip /kaggle/working/C_dataset_auc_hunt_seed1234
```

```python
!zip -r /kaggle/working/C_dataset_auc_hunt_seed1234_n7.zip /kaggle/working/C_dataset_auc_hunt_seed1234_n7
```

```python
!zip -r /kaggle/working/B_dataset_max_v1.zip /kaggle/working/B_dataset_max_v1
```

## 11. File can tai ve neu muon gui de danh gia

- `training_log.txt`
- `10_fold_results_*.csv`
- `best_model_*.pth` neu muon giu checkpoint

## 12. Vi tri ket qua

- `F-dataset full`: `/kaggle/working/F_dataset_auc_boost_v2`
- `F-dataset safe`: `/kaggle/working/F_dataset_auc_boost_v2_safe`
- `B-dataset`: `/kaggle/working/B_dataset_max_v1`
- `C-dataset baseline`: `/kaggle/working/C_dataset_auc_hunt_seed1234`
- `C-dataset neighbor 7`: `/kaggle/working/C_dataset_auc_hunt_seed1234_n7`
