# Activate SGLang 2026 Conda Environment
# Usage: . .\activate_sglang2026.ps1

$envName = "sglang2026"

Write-Host "Activating Conda environment: $envName" -ForegroundColor Cyan

# Check if environment exists
$envExists = conda env list | Select-String $envName
if (-not $envExists) {
    Write-Host "[ERROR] Environment '$envName' does not exist" -ForegroundColor Red
    Write-Host "Please run: .\setup_conda_env.ps1 first" -ForegroundColor Yellow
    return
}

# Activate environment
conda activate $envName

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Environment activated: $envName" -ForegroundColor Green
    Write-Host ""
    Write-Host "Python version:" -ForegroundColor Cyan
    python --version
    Write-Host ""
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Tips:" -ForegroundColor Yellow
    Write-Host "  - Deactivate: conda deactivate" -ForegroundColor Gray
    Write-Host "  - List packages: pip list" -ForegroundColor Gray
    Write-Host "  - Install package: pip install package-name" -ForegroundColor Gray
} else {
    Write-Host "[ERROR] Failed to activate environment" -ForegroundColor Red
}

