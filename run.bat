@echo off
pushd "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo .venv not found - run setup.bat first.
    pause
    popd
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m harmon3 %*
popd


