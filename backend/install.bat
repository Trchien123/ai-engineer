@echo off
REM Install script for Object Detection Backend

cd /d E:\ai-engineer\backend

echo.
echo ========================================
echo Object Detection - Backend Setup
echo ========================================
echo.

REM Activate venv
echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip, setuptools, wheel
echo [2/3] Upgrading pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel

REM Install requirements
echo [3/3] Installing requirements...
pip install -r requirements.txt

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To start the backend, run:
echo   cd E:\ai-engineer\backend
echo   venv\Scripts\activate.bat
echo   python -m uvicorn app.main:app --reload
echo.
pause
