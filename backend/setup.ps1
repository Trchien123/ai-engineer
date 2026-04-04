# PowerShell setup script for Object Detection Backend
# Run in PowerShell as: .\setup.ps1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Object Detection - Backend Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$backendPath = "E:\ai-engineer\backend"

# Check if we're in the right directory
if (!(Test-Path $backendPath)) {
    Write-Host "ERROR: Backend directory not found at $backendPath" -ForegroundColor Red
    exit 1
}

Set-Location $backendPath

# Step 1: Activate venv
Write-Host "[1/3] Activating virtual environment..." -ForegroundColor Yellow
& "$backendPath\venv\Scripts\Activate.ps1"

# Step 2: Upgrade pip, setuptools, wheel
Write-Host "[2/3] Upgrading pip, setuptools, wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel --no-cache-dir

# Step 3: Install requirements
Write-Host "[3/3] Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt --no-cache-dir

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "✓ Setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start the backend, run:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  cd E:\ai-engineer\backend" -ForegroundColor White
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  python -m uvicorn app.main:app --reload" -ForegroundColor White
Write-Host ""
