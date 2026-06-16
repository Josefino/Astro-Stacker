@echo off
setlocal

cd /d "%~dp0"

echo Astro Stacker Windows installer
echo Working folder: %CD%
echo.

if not exist "astro_stacker_app.py" (
  echo ERROR: astro_stacker_app.py was not found in this folder.
  echo Put this installer next to astro_stacker_app.py and run it again.
  pause
  exit /b 1
)

call :detect_python

if "%PYLAUNCHER%"=="" (
  echo Python 3 was not found.
  echo Trying to install Python 3.12 with winget...

  where winget >nul 2>nul
  if errorlevel 1 (
    echo winget was not found.
    echo Install Python 3 from https://www.python.org/downloads/windows/
    echo IMPORTANT: enable "Add python.exe to PATH" during installation.
    pause
    exit /b 1
  )

  winget install --id Python.Python.3.12 -e --source winget --accept-package-agreements --accept-source-agreements
  if errorlevel 1 (
    echo Python installation failed.
    pause
    exit /b 1
  )

  rem Re-detect Python after installation. Do not hard-code a user-specific path.
  call :detect_python

  if "%PYLAUNCHER%"=="" (
    echo Python installation appears to have completed, but Python is still not available in PATH.
    echo Close this window, open a new Command Prompt, and run this installer again.
    pause
    exit /b 1
  )
)

echo Using Python:
%PYLAUNCHER% --version
if errorlevel 1 (
  echo Python was found but could not be started.
  pause
  exit /b 1
)

echo.
echo Installing Microsoft Visual C++ Redistributable 2015-2022 x64...
where winget >nul 2>nul
if not errorlevel 1 (
  winget install --id Microsoft.VCRedist.2015+.x64 -e --source winget --accept-package-agreements --accept-source-agreements
  if errorlevel 1 (
    echo WARNING: Visual C++ Redistributable installation could not be completed.
    echo If a DLL error appears, install Microsoft Visual C++ Redistributable 2015-2022 x64 manually.
  ) else (
    echo Visual C++ Redistributable check finished.
  )
) else (
  echo winget was not found, skipping automatic Visual C++ Redistributable installation.
  echo If a DLL error appears, install Microsoft Visual C++ Redistributable 2015-2022 x64 manually.
)

set "VENV_DIR=.venv-astrostacker"
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo.
  echo Creating virtual environment: %VENV_DIR%
  %PYLAUNCHER% -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
  )
)

set "PY=%CD%\%VENV_DIR%\Scripts\python.exe"

echo.
echo Upgrading pip/setuptools/wheel...
"%PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :pipfail

echo.
echo Installing Astro Stacker dependencies...
"%PY%" -m pip install --upgrade numpy opencv-python pillow astropy xisf PySide6 onnxruntime
if errorlevel 1 goto :pipfail

rem Install rawpy separately because it contains native DLL/PYD dependencies.
echo.
echo Installing rawpy...
"%PY%" -m pip install --upgrade --no-cache-dir rawpy
if errorlevel 1 goto :pipfail

echo.
set /p INSTALL_CUPY=Install optional NVIDIA CUDA/CuPy GPU support? [y/N]
if /I "%INSTALL_CUPY%"=="Y" (
  echo Installing CuPy with bundled CUDA 12 runtime DLLs...
  "%PY%" -m pip install --upgrade "cupy-cuda12x[ctk]"
  if errorlevel 1 (
    echo CuPy installation failed. The CPU version of Astro Stacker will still work.
  ) else (
    echo Verifying bundled NVIDIA CUDA runtime packages...
    "%PY%" -c "import pathlib, nvidia.cuda_runtime, nvidia.cuda_nvrtc, nvidia.cublas, nvidia.nvjitlink, nvidia.cuda_cccl; roots=[pathlib.Path(next(iter(m.__path__))).resolve() for m in (nvidia.cuda_runtime,nvidia.cuda_nvrtc,nvidia.cublas,nvidia.nvjitlink,nvidia.cuda_cccl)]; print('CUDA runtime package roots:'); [print(' ', p) for p in roots]"
    if errorlevel 1 (
      echo CUDA runtime DLL verification failed. Reinstall cupy-cuda12x[ctk].
    )
    echo Installing ONNX Runtime DirectML for DRUNet GPU inference...
    "%PY%" -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml >nul 2>nul
    "%PY%" -m pip install --upgrade onnxruntime-directml
    if errorlevel 1 (
      echo ONNX Runtime DirectML installation failed. Restoring CPU ONNX Runtime...
      "%PY%" -m pip install --upgrade onnxruntime
    ) else (
      "%PY%" -c "import onnxruntime as ort; providers=ort.get_available_providers(); print('ONNX Runtime providers:', providers); raise SystemExit(0 if 'DmlExecutionProvider' in providers else 1)"
      if errorlevel 1 (
        echo WARNING: ONNX Runtime DirectML is installed, but DmlExecutionProvider is unavailable.
        echo DRUNet will automatically use the CPU provider.
      )
    )
  )
) else (
  echo Skipping CuPy.
)

echo.
echo Verifying imports...
"%PY%" -c "import cv2, numpy, rawpy, onnxruntime; from PIL import Image; from astropy.io import fits; from xisf import XISF; from PySide6.QtWidgets import QApplication; print('Base dependencies and ONNX Runtime OK')"
if errorlevel 1 (
  echo.
  echo Dependency verification failed.
  echo Trying to repair rawpy installation...
  "%PY%" -m pip uninstall -y rawpy
  "%PY%" -m pip install --no-cache-dir rawpy
  echo.
  echo Verifying imports again...
  "%PY%" -c "import cv2, numpy, rawpy, onnxruntime; from PIL import Image; from astropy.io import fits; from xisf import XISF; from PySide6.QtWidgets import QApplication; print('Base dependencies and ONNX Runtime OK')"
  if errorlevel 1 (
    echo.
    echo Dependency verification failed again.
    echo Most likely cause: missing Microsoft Visual C++ Redistributable 2015-2022 x64,
    echo or a native DLL dependency problem on this Windows installation.
    echo Try restarting Windows and running this installer again.
    pause
    exit /b 1
  )
)

if /I "%INSTALL_CUPY%"=="Y" (
  "%PY%" -c "import cupy; print('CuPy OK:', cupy.__version__)"
  if errorlevel 1 (
    echo CuPy import failed. Astro Stacker will use CPU fallback.
  )
)

if not exist "models\drunet_color.onnx" (
  echo ERROR: models\drunet_color.onnx was not found.
  pause
  exit /b 1
)
if not exist "models\drunet_gray.onnx" (
  echo ERROR: models\drunet_gray.onnx was not found.
  pause
  exit /b 1
)
if not exist "models\cosmic_clarity_stellar.onnx" (
  echo ERROR: models\cosmic_clarity_stellar.onnx was not found.
  pause
  exit /b 1
)
echo DRUNet and AI Star Deconvolution models are ready.

> run_astro_stacker_windows.bat echo @echo off
>> run_astro_stacker_windows.bat echo cd /d "%%~dp0"
>> run_astro_stacker_windows.bat echo ".venv-astrostacker\Scripts\python.exe" "astro_stacker_app.py"

echo.
echo Installation complete.
echo Run Astro Stacker with:
echo   run_astro_stacker_windows.bat
echo.
pause
exit /b 0

:detect_python
set "PYLAUNCHER="
where py >nul 2>nul
if not errorlevel 1 set "PYLAUNCHER=py -3"
if "%PYLAUNCHER%"=="" (
  where python >nul 2>nul
  if not errorlevel 1 set "PYLAUNCHER=python"
)
exit /b 0

:pipfail
echo.
echo Python package installation failed.
echo Check your internet connection and try again.
pause
exit /b 1
