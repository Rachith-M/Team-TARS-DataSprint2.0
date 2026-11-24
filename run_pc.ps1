# Anveshan A2 + recipe rows on the PC GPU (AMD ROCm or NVIDIA CUDA; both use torch.cuda).
# Usage (PowerShell, from this folder):
#   powershell -ExecutionPolicy Bypass -File .\run_pc.ps1              # final training -> submission.csv
#   powershell -ExecutionPolicy Bypass -File .\run_pc.ps1 -Validate    # Fold 1 check first, final only if it wins
#   add -Python C:\path\to\python.exe to use a specific interpreter that already has GPU torch
param([switch]$Validate, [string]$Python = '', [int]$Epochs = 8, [string]$Lr = '2e-5')
# 'Continue', not 'Stop': in Windows PowerShell 5.1 any stderr line from a native exe (pip warnings, a failed
# import) becomes a terminating error under 'Stop'. Native failures are checked via $LASTEXITCODE instead.
$ErrorActionPreference = 'Continue'
function Assert-Ok($what) { if ($LASTEXITCODE) { throw "$what failed (exit $LASTEXITCODE)" } }
$here = $PSScriptRoot
$code = Join-Path $here 'code'
$out = Join-Path $here 'runs'
$app = Join-Path $env:LOCALAPPDATA 'AnveshanNLP'
New-Item -ItemType Directory -Force $out, $app | Out-Null

function Test-GpuPython($exe) {
    if (-not $exe -or -not (Test-Path $exe)) { return $false }
    & $exe -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
    return $LASTEXITCODE -eq 0
}

# 1. Find a Python with GPU torch, else build a ROCm 7.2.1 venv (Python 3.12 + Adrenalin 26.2.2 or newer).
$candidates = @($Python, (Join-Path $app 'rocm-venv\Scripts\python.exe'), (Join-Path $app 'distill-venv\Scripts\python.exe'), 'python')
$py = $null
foreach ($c in $candidates) {
    if ($c -eq 'python') { $c = (Get-Command python -ErrorAction SilentlyContinue).Source }
    if (Test-GpuPython $c) { $py = $c; break }
}
if (-not $py) {
    Write-Host 'No GPU-enabled torch found. Installing ROCm 7.2.1 PyTorch into rocm-venv (about 3 GB)...'
    $venv = Join-Path $app 'rocm-venv'
    & py -3.12 -m venv $venv; Assert-Ok 'Creating the Python 3.12 venv'
    $py = Join-Path $venv 'Scripts\python.exe'
    $r = 'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1'
    & $py -m pip install --upgrade pip; Assert-Ok 'pip upgrade'
    & $py -m pip install --no-cache-dir "$r/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" "$r/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl" "$r/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl" "$r/rocm-7.2.1.tar.gz"; Assert-Ok 'ROCm SDK install'
    & $py -m pip install --no-cache-dir "$r/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" "$r/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" "$r/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"; Assert-Ok 'ROCm PyTorch install'
    if (-not (Test-GpuPython $py)) { throw 'torch.cuda.is_available() is False after install. Check the AMD Adrenalin driver (26.2.2+).' }
}
Write-Host "Using $py"
& $py -m pip install -q "transformers==4.57.6" "scikit-learn==1.9.1" "safetensors>=0.4.3" joblib threadpoolctl scipy huggingface_hub; Assert-Ok 'Package install'
& $py -c "import torch; print('GPU:', torch.cuda.get_device_name(0), '| HIP', torch.version.hip, '| CUDA', torch.version.cuda)"; Assert-Ok 'GPU check'

# 2. Data and pinned Bhasha checkpoint.
$train = Join-Path $code 'train.csv'; $test = Join-Path $code 'test.csv'
if ((Get-FileHash $train -Algorithm SHA256).Hash.ToLower() -ne 'db68ca1c1f3072353d72e7447d989b968652852dc24d428a4a91c0f0d9df324a') { throw 'train.csv hash mismatch' }
$ckpt = Join-Path $app 'bhasha-embed-v0-64b330ff'
if (-not (Test-Path (Join-Path $ckpt 'model.safetensors'))) {
    $ckpt = (& $py -c "from huggingface_hub import snapshot_download; print(snapshot_download('AkshitaS/bhasha-embed-v0', revision='64b330ff6703ffebdfc55c412a6a8aaf1b2728c8', cache_dir=r'$app\hf-cache', allow_patterns=['*.json','*.safetensors','vocab.txt']))" | Select-Object -Last 1)
}
Write-Host "Checkpoint $ckpt"
$plan = Join-Path $code 'sprint_results\depth_ladder\plan.npz'
Push-Location $code
try {
    # 3. Optional Fold 1 check: inner early stopping, outer refit, compare with submitted model's clean 0.8678.
    if ($Validate) {
        & $py -X utf8 -m sprint.a2_recipe --train $train --checkpoint $ckpt --plan $plan --output (Join-Path $out 'fold1') --folds 1 --epochs 12 --lr $Lr --minutes 600
        if ($LASTEXITCODE) { throw 'Fold 1 run failed' }
        $report = Get-Content (Join-Path $out 'fold1\a2_recipe.json') -Raw | ConvertFrom-Json
        $clean = $report.outer.fold1.overall; $Epochs = $report.inner.fold1.selected_epoch
        Write-Host ("Fold 1 outer clean Macro-F1 {0:N5} vs submitted 0.86783; inner picked {1} epochs" -f $clean, $Epochs)
        if ($clean -le 0.86783) { Write-Host 'Did not beat the submitted model on Fold 1. Stopping before final training.'; return }
    }
    # 4. Final: all organizer rows + recipe rows, fixed epochs, save weights, predict test.csv.
    # The LR schedule length matches the run that chose the epoch count (12 after -Validate, else = epochs, as in the pilot).
    $schedule = if ($Validate) { 12 } else { $Epochs }
    & $py -X utf8 -m sprint.a2_recipe --train $train --checkpoint $ckpt --plan $plan --output (Join-Path $out 'final') --final-epochs $Epochs --epochs $schedule --lr $Lr --test $test --minutes 600
    if ($LASTEXITCODE) { throw 'Final run failed' }
    Write-Host "Done. Submission: $(Join-Path $out 'final\final\submission.csv')  (do not upload without review)"
} finally { Pop-Location }
