@echo off
setlocal

echo === activate whisper-webui env ===
call conda activate whisper
echo === start Whisper WebUI ===
start "" http://127.0.0.1:7860
python app.py

endlocal
pause
