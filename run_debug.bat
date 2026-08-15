@echo off
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo .venv not found - run setup.bat first.
    pause
    popd
    exit /b 1
)
".venv\Scripts\python.exe" -m harmon3 --log-level DEBUG %*
popd
pause
