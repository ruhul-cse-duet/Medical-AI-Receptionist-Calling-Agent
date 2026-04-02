@echo off
setlocal

cd /d "%~dp0"

set "NGROK_BIN=ngrok\ngrok.exe"
if not exist "%NGROK_BIN%" set "NGROK_BIN=ngrok"

echo [INFO] Starting ngrok on http://localhost:8003
start "ngrok-8003" cmd /k "%NGROK_BIN% http 8003 --pooling-enabled"

echo [INFO] Starting Medical AI Receptionist on http://0.0.0.0:8003
call ".venv\Scripts\activate.bat"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload

endlocal
