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

set "PYLAUNCHER="
where py >nul 2>nul
if %ERRORLEVEL%==0 set "PYLAUNCHER=py -3"

if "%PYLAUNCHER%"=="" (
  where python >nul 2>nul
  if %ERRORLEVEL%==0 set "PYLAUNCHER=python"
)

if "%PYLAUNCHER%"=="" (
  echo Python 3 was not found.
  echo Trying to install Python 3.12 with winget...
  where winget >nul 2>nul
  if not %ERRORLEVEL%==0 (
    echo winget was not found.
    echo Install Python 3 from https://www.python.org/downloads/windows/
    echo IMPORTANT: enable "Add python.exe to PATH" during installation.
    pause
    exit /b 1
  )
  winget install --id Python.Python.3.12 -e --source winget
  if not %ERRORLEVEL%==0 (
    echo Python installation failed.
    pause
    exit /b 1
  )
  set "PYLAUNCHER=py -3"
)

echo Using Python:
%PYLAUNCHER% --version
if not %ERRORLEVEL%==0 (
  echo Python was found but could not be started.
  pause
  exit /b 1
)

set "VENV_DIR=.venv-astrostacker"
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Creating virtual environment: %VENV_DIR%
  %PYLAUNCHER% -m venv "%VENV_DIR%"
  if not %ERRORLEVEL%==0 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
  )
)

set "PY=%CD%\%VENV_DIR%\Scripts\python.exe"

echo Upgrading pip/setuptools/wheel...
"%PY%" -m pip install --upgrade pip setuptools wheel
if not %ERRORLEVEL%==0 goto :pipfail

echo Installing Astro Stacker dependencies...
"%PY%" -m pip install --upgrade numpy opencv-python pillow astropy rawpy xisf PySide6
if not %ERRORLEVEL%==0 goto :pipfail

echo.
set /p INSTALL_CUPY=Install optional NVIDIA CUDA/CuPy GPU support? [y/N] 
if /I "%INSTALL_CUPY%"=="Y" (
  echo Installing CuPy for CUDA 12.x...
  "%PY%" -m pip install --upgrade cupy-cuda12x
  if not %ERRORLEVEL%==0 (
    echo CuPy installation failed. The CPU version of Astro Stacker will still work.
  )
) else (
  echo Skipping CuPy.
)

echo.
echo Verifying imports...
"%PY%" -c "import cv2, numpy, rawpy; from PIL import Image; from astropy.io import fits; from xisf import XISF; from PySide6.QtWidgets import QApplication; print('Base dependencies OK')"
if not %ERRORLEVEL%==0 (
  echo Dependency verification failed.
  pause
  exit /b 1
)

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

:pipfail
echo.
echo Python package installation failed.
echo Check your internet connection and try again.
pause
exit /b 1
