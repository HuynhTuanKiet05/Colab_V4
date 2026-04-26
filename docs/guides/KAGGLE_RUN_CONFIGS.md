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
  --result_root /kaggle/working/F_dataset_improved_safe
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
  --result_root /kaggle/working/F_dataset_improved_full
```

## 7. B-dataset ban improved manh

`B-dataset` chi co 269 thuoc (C=663, F=592). Run cu dung
`plateau scheduler + focal + EMA warmup 80` (mac dinh tuned cho C/F) va dat
mean AUC `~0.9284` - thap hon ban bao cao `0.9354`.

Dien bien cu the cua run cu:

- AUC tang nhanh epoch 1-75 roi plateau cung
- plateau scheduler giam LR ve `min_lr` truoc epoch 250
- 750 epoch cuoi chi tang AUC `0.857 -> 0.930` (LR da chet)
- EMA bat tu epoch 80 keo eval AUC ve diem chua hoi tu

Preset moi cua `B-dataset` (file `train_final.py`) da tu dong dat:

- `scheduler = cosine` thay cho `plateau`, LR decay deu trong 1000 epoch
- `use_focal = False` (CE thuong) - giam regularization cho dataset 269 thuoc
- `use_ema = False`, `use_ranking = False`
- `pair_mode = moe`
- `use_swa`, `use_dropedge`, `use_mixup`, `use_adaptive_hard_neg`

Lenh chay rut gon (preset tu lo het, khong can truyen tay tham so model):

```python
%cd /kaggle/working/Colab_V4

!python train_final.py \
  --dataset B-dataset \
  --device cuda \
  --epochs 1000 \
  --k_fold 10 \
  --patience 220 \
  --random_seed 1234 \
  --save_checkpoints \
  --result_root /kaggle/working/B_dataset_max_improved
```

Lenh chay full explicit (de doc lai cau hinh tu console):

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
  --pair_mode moe \
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
  --scheduler cosine \
  --no-use_focal \
  --no-use_ema \
  --no-use_ranking \
  --use_swa \
  --swa_start_ratio 0.70 \
  --swa_freq 10 \
  --use_dropedge \
  --dropedge_p 0.10 \
  --use_mixup \
  --mixup_alpha 0.20 \
  --use_adaptive_hard_neg \
  --hard_neg_target_ratio 0.40 \
  --hard_neg_warmup 200 \
  --save_checkpoints \
  --result_root /kaggle/working/B_dataset_max_improved
```

Muon ablation tung knob (chi de debug):

- `--scheduler plateau` -> quay ve plateau scheduler (giu nguyen 2 thay doi con lai)
- `--use_focal` -> bat lai focal loss
- `--ema_warmup_epochs 80` -> EMA bat som nhu cu

Neu sau 1000 epoch ma fold 4/9 van best o epoch 1000 thi co the train them
voi `--epochs 1500`; con lai pattern hoi tu thuong lo o khoang epoch 700-900.

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
!zip -r /kaggle/working/F_dataset_improved_full.zip /kaggle/working/F_dataset_improved_full
```

```python
!zip -r /kaggle/working/F_dataset_improved_safe.zip /kaggle/working/F_dataset_improved_safe
```

```python
!zip -r /kaggle/working/C_dataset_auc_hunt_seed1234.zip /kaggle/working/C_dataset_auc_hunt_seed1234
```

```python
!zip -r /kaggle/working/C_dataset_auc_hunt_seed1234_n7.zip /kaggle/working/C_dataset_auc_hunt_seed1234_n7
```

```python
!zip -r /kaggle/working/B_dataset_max_improved.zip /kaggle/working/B_dataset_max_improved
```

## 11. File can tai ve neu muon gui de danh gia

- `training_log.txt`
- `10_fold_results_*.csv`
- `best_model_*.pth` neu muon giu checkpoint

## 12. Vi tri ket qua

- `F-dataset full`: `/kaggle/working/F_dataset_improved_full`
- `F-dataset safe`: `/kaggle/working/F_dataset_improved_safe`
- `B-dataset`: `/kaggle/working/B_dataset_max_improved`
- `C-dataset baseline`: `/kaggle/working/C_dataset_auc_hunt_seed1234`
- `C-dataset neighbor 7`: `/kaggle/working/C_dataset_auc_hunt_seed1234_n7`

## 13. Improved pipeline duy nhat (`train_final.py`)

`train_final.py` la entrypoint improved chinh. Preset moi da tich hop truc tiep
5 cai tien (SWA + DropEdge + Mixup + MoE pair head + adaptive hard-neg) va
3 speedups (AMP auto, eval_start_epoch=100, score_every=5). Khong can chay
script improved rieng.

### 13.1. Smoke test improved (kiem tra import + training loop chay duoc)

```python
%cd /kaggle/working/Colab_V4

!python train_final.py \
  --dataset C-dataset \
  --device cuda \
  --epochs 3 \
  --k_fold 2 \
  --fold_limit 1 \
  --eval_start_epoch 1 \
  --score_every 1 \
  --no-save_checkpoints \
  --result_root /kaggle/working/_smoke_improved
```

### 13.2. C-dataset improved (target AUC ~0.98)

```python
%cd /kaggle/working/Colab_V4

!python train_final.py \
  --dataset C-dataset \
  --device cuda \
  --epochs 1000 \
  --k_fold 10 \
  --patience 220 \
  --random_seed 1234 \
  --save_checkpoints \
  --result_root /kaggle/working/C_dataset_improved_run1
```

### 13.3. F-dataset improved (target AUC ~0.975)

```python
!python train_final.py \
  --dataset F-dataset \
  --device cuda \
  --epochs 1000 \
  --k_fold 10 \
  --patience 220 \
  --random_seed 1234 \
  --save_checkpoints \
  --result_root /kaggle/working/F_dataset_improved_run1
```

### 13.4. B-dataset improved

```python
!python train_final.py \
  --dataset B-dataset \
  --device cuda \
  --epochs 1000 \
  --k_fold 10 \
  --patience 220 \
  --random_seed 1234 \
  --save_checkpoints \
  --result_root /kaggle/working/B_dataset_improved_run1
```

### 13.5. Ablation tung knob improved

```bash
# tat SWA
python train_final.py --dataset C-dataset ... --no-use_swa
# tat DropEdge
python train_final.py --dataset C-dataset ... --no-use_dropedge
# tat Mixup
python train_final.py --dataset C-dataset ... --no-use_mixup
# quay ve MLP head
python train_final.py --dataset C-dataset ... --pair_mode mul_mlp
# tat adaptive hard-neg
python train_final.py --dataset C-dataset ... --no-use_adaptive_hard_neg
```

### 13.6. Vi tri ket qua

- `B-dataset improved`: `/kaggle/working/B_dataset_improved_run1`
- `C-dataset improved`: `/kaggle/working/C_dataset_improved_run1`
- `F-dataset improved`: `/kaggle/working/F_dataset_improved_run1`
