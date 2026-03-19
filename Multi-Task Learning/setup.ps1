# setup.ps1 — one-shot environment setup for SPA-MSJF
# Works on any machine: auto-detects CUDA version and picks the right PyTorch wheel.
# Usage: .\setup.ps1   (run from the Multi-Task Learning folder)

$python = "py -3.12"

Write-Host "`n=== SPA-MSJF Environment Setup ===" -ForegroundColor Cyan

# --- detect CUDA version via nvidia-smi ---
$cudaVersion = $null
try {
    $smiOut = & nvidia-smi 2>$null
    if ($smiOut -match "CUDA Version:\s*(\d+\.\d+)") {
        $cudaVersion = [version]$Matches[1]
        Write-Host "Detected CUDA $cudaVersion via nvidia-smi" -ForegroundColor Green
    }
} catch {}

# --- pick the matching PyTorch wheel URL ---
if ($null -eq $cudaVersion) {
    Write-Host "No GPU detected — installing CPU-only PyTorch" -ForegroundColor Yellow
    $wheelUrl = "https://download.pytorch.org/whl/cpu"
} elseif ($cudaVersion -ge [version]"12.4") {
    Write-Host "Using CUDA 12.4 wheels" -ForegroundColor Green
    $wheelUrl = "https://download.pytorch.org/whl/cu124"
} elseif ($cudaVersion -ge [version]"12.1") {
    Write-Host "Using CUDA 12.1 wheels" -ForegroundColor Green
    $wheelUrl = "https://download.pytorch.org/whl/cu121"
} elseif ($cudaVersion -ge [version]"11.8") {
    Write-Host "Using CUDA 11.8 wheels" -ForegroundColor Green
    $wheelUrl = "https://download.pytorch.org/whl/cu118"
} else {
    Write-Host "CUDA $cudaVersion is older than 11.8 — falling back to CPU-only PyTorch" -ForegroundColor Yellow
    $wheelUrl = "https://download.pytorch.org/whl/cpu"
}

# --- install PyTorch ---
Write-Host "`nInstalling PyTorch from $wheelUrl ..." -ForegroundColor Cyan
Invoke-Expression "$python -m pip install torch torchvision --index-url $wheelUrl --no-cache-dir"

# --- install other dependencies ---
Write-Host "`nInstalling remaining dependencies ..." -ForegroundColor Cyan
Invoke-Expression "$python -m pip install -r requirements.txt --no-cache-dir"

# --- verify ---
Write-Host "`n=== Verification ===" -ForegroundColor Cyan
Invoke-Expression "$python -c `"import torch; cuda = torch.cuda.is_available(); gpu = torch.cuda.get_device_name(0) if cuda else 'N/A'; print(f'PyTorch {torch.__version__}  |  CUDA available: {cuda}  |  GPU: {gpu}')`""

Write-Host "`nSetup complete. Run training with:" -ForegroundColor Green
Write-Host "  py -3.12 train.py" -ForegroundColor White
