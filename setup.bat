@echo off
setlocal
pushd "%~dp0"

echo === HARMON3 setup ===
echo.

if exist ".venv\Scripts\python.exe" (
    echo Reusing existing .venv
) else (
    echo Creating .venv with Python 3.12 ...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo.
        echo Could not create the venv with "py -3.12".
        echo Install Python 3.12 or edit this script to use another version.
        goto :err
    )
)

echo.
echo Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :err

echo.
echo Installing dependencies ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :err

echo.
echo Making sure the GPU build of ONNX Runtime is the one that survived ...
rem rtmlib depends on plain `onnxruntime`, and the CPU and GPU packages install over the
rem top of each other. If the CPU one landed last the CUDA provider silently disappears,
rem so it is removed and the GPU build put back. Harmless when it was already correct.
".venv\Scripts\python.exe" -m pip uninstall -y onnxruntime >nul 2>&1
".venv\Scripts\python.exe" -m pip install --force-reinstall --no-deps onnxruntime-gpu
if errorlevel 1 goto :err

echo.
echo Checking which device pose estimation will use ...
".venv\Scripts\python.exe" -c "from harmon3 import pose; d, why = pose.preferred_device('auto'); print('  pose runs on ' + why)"
if errorlevel 1 (
    echo.
    echo The pose estimator could not be imported. Pose is optional - the rest of the
    echo app works without it - but the Pose toggle will fail until this is fixed.
)

echo.
echo Verifying QtMultimedia is present ...
".venv\Scripts\python.exe" -c "from PySide6.QtMultimediaWidgets import QVideoWidget; from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput; import PySide6; print('  QtMultimedia OK, PySide6', PySide6.__version__)"
if errorlevel 1 (
    echo.
    echo QtMultimedia is missing from this PySide6 wheel.
    echo Fix: ensure requirements.txt pins the full "PySide6" metapackage (which pulls
    echo PySide6-Addons), not "PySide6-Essentials" - Essentials dropped QtMultimedia.
    goto :err
)

echo.
echo Setup complete. Launch with run.bat  (or run_debug.bat for a console + verbose logs).
popd
endlocal
exit /b 0

:err
echo.
echo *** SETUP FAILED (errorlevel %errorlevel%) ***
popd
endlocal
pause
exit /b 1
