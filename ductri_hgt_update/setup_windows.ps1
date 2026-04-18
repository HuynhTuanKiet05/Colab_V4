$ErrorActionPreference = "Stop"

$pythonExe = "py -3.9"

Invoke-Expression "$pythonExe -m venv .venv"
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-py39.txt

Write-Host "Environment setup complete."
Write-Host "Run a smoke test with:"
Write-Host ".\.venv\Scripts\python.exe train_DDA.py --dataset C-dataset --epochs 1 --k_fold 2"
