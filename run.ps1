# IdemAI Core — launch both servers in separate windows
# Run from the repo root: .\run.ps1

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Load .env if it exists
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process")
        }
    }
    Write-Host "[idemAI] Loaded .env" -ForegroundColor Green
}

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "[idemAI] WARNING: ANTHROPIC_API_KEY is not set. Aziz Agent will not work." -ForegroundColor Yellow
    Write-Host "[idemAI] Add it to a .env file in the repo root: ANTHROPIC_API_KEY=sk-ant-..." -ForegroundColor Yellow
}

# Start Recording Studio (FastAPI) in a new window
Write-Host "[idemAI] Starting Recording Studio on http://localhost:8001" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; python -m uvicorn app.recording_server:app --reload --port 8001"

# Start Dashboard (Streamlit) in a new window
Write-Host "[idemAI] Starting Dashboard on http://localhost:8501" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; python -m streamlit run app/dashboard.py"

Write-Host ""
Write-Host "Both servers starting. Open these URLs in your browser:" -ForegroundColor Green
Write-Host "  Recording Studio : http://localhost:8001/studio" -ForegroundColor White
Write-Host "  Dashboard        : http://localhost:8501" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C in each window to stop the servers." -ForegroundColor Gray
