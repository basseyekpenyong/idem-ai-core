@echo off
cd /d "%~dp0"

:: Load .env
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" if not "%%A:~0,1%"=="#" set "%%A=%%B"
    )
)

:: Create required data directories
if not exist data\processed mkdir data\processed
if not exist data\raw      mkdir data\raw
if not exist data\vocab    mkdir data\vocab

echo Starting IdemAI Core...
echo.
echo  Recording Studio : http://localhost:8001
echo  Dashboard + Aziz : http://localhost:8501
echo.
echo Press Ctrl+C in each window to stop.
echo.

start "IdemAI - Recording Server" cmd /k "python -m uvicorn app.recording_server:app --reload --port 8001"
timeout /t 3 /nobreak >nul
start "IdemAI - Dashboard" cmd /k "python -m streamlit run app/dashboard.py"
