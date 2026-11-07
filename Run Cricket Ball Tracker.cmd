@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Set up the Python environment first using the Windows instructions in README.md.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m cbt.launcher
