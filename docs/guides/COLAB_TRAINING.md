# Google Colab Training Guide

Huong dan nay duoc viet lai cho repo `Colab_V4` va pipeline hien tai trong [train_final.py](C:/xampp/htdocs/DoANBase_Final/CaiTien_HGT_repo/train_final.py).

Mac dinh train hien tai:

- `assoc_backbone=vanilla_hgt`
- `fusion_mode=mva`
- `pair_mode=mul_mlp`
- `gate_mode=vector`
- `dataset=C-dataset`

Tai lieu nay uu tien:

- clone repo sach
- cai moi truong on dinh tren Colab GPU
- restart runtime dung luc
- smoke test truoc khi train dai
- luu log va checkpoint len Google Drive

## 0. Chuan bi runtime

Vao:

- `Runtime -> Change runtime type`
- chon `GPU`

Kiem tra nhanh:

```python
!nvidia-smi
import sys, platform, torch
print(sys.version)
print(platform.platform())
print("cuda available:", torch.cuda.is_available())
```

Neu `!nvidia-smi` khong ra GPU hoac `torch.cuda.is_available()` la `False` thi khong nen chay `--device cuda`.

## 1. Clone repo sach

```python
%cd /content
!rm -rf /content/Colab_V4
!git clone https://github.com/DucTri2207/Colab_V4.git
%cd /content/Colab_V4
!git log --oneline -1
```

Neu repo da ton tai va chi muon cap nhat dung ban moi nhat:

```python
%cd /content/Colab_V4
!git fetch origin
!git reset --hard origin/main
!git log --oneline -1
```

Neu ban muon mot cell an toan de dam bao thu muc `/content/Colab_V4` luon ton tai truoc moi buoc setup/train, dung cell nay:

```python
import os

%cd /content

if not os.path.exists("/content/Colab_V4"):
    !git clone https://github.com/DucTri2207/Colab_V4.git
else:
    print("Repo da ton tai, cap nhat ban moi nhat...")
    %cd /content/Colab_V4
    !git fetch origin
    !git reset --hard origin/main
    %cd /content

%cd /content/Colab_V4
!git log --oneline -1
```

## 2. Kiem tra thu muc dang co tren Colab

Neu bi loi `No such file or directory`, kiem tra nhanh:

```python
!pwd
!ls /content
!find /content -maxdepth 2 -type d | sort
```

## 3. Cai moi truong

### Cach nhanh khuyen nghi

Truoc tien, chay lai cell dam bao repo ton tai neu day la runtime moi hoac ban vua `Disconnect and delete runtime`.

```python
import os

%cd /content

if not os.path.exists("/content/Colab_V4"):
    !git clone https://github.com/DucTri2207/Colab_V4.git
else:
    print("Repo da ton tai, cap nhat ban moi nhat...")
    %cd /content/Colab_V4
    !git fetch origin
    !git reset --hard origin/main
    %cd /content

%cd /content/Colab_V4
!git log --oneline -1
```

Sau do moi setup:

```python
%cd /content/Colab_V4
!bash scripts/colab_setup.sh
```

Sau do restart runtime:

```python
import os, signal
os.kill(os.getpid(), signal.SIGKILL)
```

Khi runtime len lai:

```python
import os

%cd /content

if not os.path.exists("/content/Colab_V4"):
    !git clone https://github.com/DucTri2207/Colab_V4.git
else:
    %cd /content/Colab_V4
    !git fetch origin
    !git reset --hard origin/main
    %cd /content

%cd /content/Colab_V4
```

### Neu `colab_setup.sh` loi, cai thu cong

```python
%cd /content/Colab_V4
!python -m pip install --upgrade pip setuptools wheel
!python -m pip uninstall -y torch torchvision torchaudio dgl dglgo torchdata
!python -m pip install --no-cache-dir --force-reinstall \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
!python -m pip install --no-cache-dir --force-reinstall \
  dgl==2.4.0+cu121 \
  -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
!python -m pip install --no-cache-dir --force-reinstall -r requirements-colab.txt
```

Restart runtime sau khi cai xong:

```python
import os, signal
os.kill(os.getpid(), signal.SIGKILL)
```

## 4. Kiem tra moi truong sau khi cai

```python
%cd /content/Colab_V4
!nvidia-smi
import torch, dgl, numpy, pandas, sklearn, networkx, scipy

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("dgl:", dgl.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("sklearn:", sklearn.__version__)
print("networkx:", networkx.__version__)
print("scipy:", scipy.__version__)
```

Neu cell tren chay duoc thi moi truong da on.

## 4. Smoke test truoc khi train dai

Nen chay smoke test truoc khi train 1000 epoch.

```python
%cd /content/Colab_V4
!python scripts/colab_train.py --dataset C-dataset --preset smoke
```

## 5. Mount Google Drive de luu ket qua


```python
from google.colab import drive
drive.mount('/content/drive')

import os, time
RUN_DIR = f"/content/drive/MyDrive/Colab_V4_runs/Cdataset_{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(RUN_DIR, exist_ok=True)
print("RUN_DIR =", RUN_DIR)
```

Luu y:

- Khong viet truc tiep `"{RUN_DIR}"` trong shell command neu chua tao f-string Python truoc.
- Neu khong can luu len Drive, bo qua phan nay va khong truyen `--result_root`.

## 6. Smoke test truoc khi train dai

### Cach 1: smoke test truc tiep bang `train_final.py`

```python
%cd /content/Colab_V4
!python train_final.py \
  --dataset C-dataset \
  --device cuda \
  --epochs 1 \
  --k_fold 2 \
  --fold_limit 1 \
  --score_every 1 \
  --eval_start_epoch 1 \
  --no-save_checkpoints
```

### Cach 2: dung launcher Colab

```python
%cd /content/Colab_V4
!python scripts/colab_train.py --dataset C-dataset --preset smoke --device cuda
```

Neu smoke test pass, moi chay train dai.

## 7. Train ban mac dinh cua `Colab_V4`

Day la lenh train dung bo tham so `C-dataset` hien tai:

```python
%cd /content/Colab_V4
cmd = f"""
python train_final.py \
  --dataset C-dataset \
  --device cuda \
  --epochs 1000 \
  --k_fold 10 \
  --neighbor 5 \
  --lr 1e-4 \
  --weight_decay 1e-3 \
  --hgt_layer 2 \
  --hgt_in_dim 256 \
  --hgt_head_dim 32 \
  --gt_out_dim 256 \
  --save_checkpoints \
  --result_root "{RUN_DIR}"
"""
print(cmd)
!{cmd} 2>&1 | tee -a "{RUN_DIR}/console.log"
```

## 8. Train bien the `rvg` de doi chieu

Neu muon test nhanh fuzzy residual gate:

```python
%cd /content/Colab_V4
!python train_final.py \
  --dataset C-dataset \
  --device cuda \
  --epochs 1000 \
  --k_fold 10 \
  --neighbor 5 \
  --lr 1e-4 \
  --weight_decay 1e-3 \
  --hgt_layer 2 \
  --hgt_in_dim 256 \
  --hgt_head_dim 32 \
  --gt_out_dim 256 \
  --fusion_mode rvg \
  --pair_mode mul_mlp \
  --gate_mode vector \
  --save_checkpoints \
  --result_root "{RUN_DIR}/rvg_run"
```

## 9. Dung launcher `scripts/colab_train.py`

Launcher nay la mot wrapper cho Colab. Vi du:

```python
%cd /content/Colab_V4
!python scripts/colab_train.py --dataset C-dataset --preset full --device cuda --mount-drive
```

Preset co san:

- `smoke`
- `standard`
- `full`

## 10. Cac loi da gap va cach xu ly

### 10.1. `No such file or directory: /content/Colab_V4`

Repo chua duoc clone hoac dang o ten thu muc khac.

```python
!ls /content
%cd /content
!git clone https://github.com/DucTri2207/Colab_V4.git
%cd /content/Colab_V4
```

### 10.2. `python3: can't open file '/content/train_final.py'`

Thuong la do `%cd /content/Colab_V4` da that bai, nen notebook van dang o `/content`.

Chay:

```python
!pwd
!ls /content
```

### 10.3. `Found no NVIDIA driver on your system`

Ban dang chay `--device cuda` nhung runtime hien tai khong co GPU.

Kiem tra:

```python
!nvidia-smi
import torch
print(torch.cuda.is_available())
```

Neu khong co GPU, doi tam sang:

```python
--device cpu
```

### 10.4. `Missing dataset files in AMDGT_original/data/c-dataset`

Linux phan biet hoa thuong. Dung:

```python
--dataset C-dataset
```

khong dung:

```python
--dataset c-dataset
```

### 10.5. `tee: {RUN_DIR}/console.log: No such file or directory`

Ban dang truyen literal string `{RUN_DIR}` thay vi f-string Python.

Dung theo mau:

```python
cmd = f"""python train_final.py --result_root "{RUN_DIR}" """
!{cmd} 2>&1 | tee -a "{RUN_DIR}/console.log"
```

### 10.6. `ModuleNotFoundError: No module named 'dgl'`

DGL chua cai hoac cai hong. Chay lai:

```python
!python -m pip install --no-cache-dir --force-reinstall \
  dgl==2.4.0+cu121 \
  -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
```

Sau do restart runtime.

### 10.7. Warning dependency conflict cua Colab

Colab co nhieu package he thong khong lien quan den train.

Neu cac package sau import duoc va smoke test pass, co the tiep tuc:

- `torch`
- `torchvision`
- `torchaudio`
- `dgl`
- `numpy`
- `pandas`
- `scikit-learn`
- `networkx`
- `scipy`

## 11. Thu tu chay khuyen nghi

Thu tu an toan nhat:

1. Bat GPU runtime
2. Clone `Colab_V4`
3. Cai moi truong
4. Restart runtime
5. Kiem tra version
6. Mount Drive
7. Tao `RUN_DIR`
8. Smoke test
9. Train dai

## 12. Ghi chu quan trong

- Luon dung `C-dataset`, khong dung `c-dataset`.
- Neu repo da co san, uu tien `git fetch origin` + `git reset --hard origin/main`.
- Sau moi lan doi `torch`, `dgl`, `numpy`, nen restart runtime.
- `scripts/colab_setup.sh` la cach nhanh nhat; cai thu cong la fallback khi runtime Colab bi ban.
- Neu smoke test fail, khong nen chay 1000 epoch ngay.

## 13. Nguon install chinh thuc

- PyTorch previous versions: `https://pytorch.org/get-started/previous-versions/`
- DGL wheel index for torch 2.4 / cu121: `https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html`
