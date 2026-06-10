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
    echo Please close this window, open a new Command Prompt, and run this installer again.
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
echo Installing Microsoft Visual C++ Redistributable x64...
where winget >nul 2>nul
if not errorlevel 1 (
  winget install --id Microsoft.VCRedist.2015+.x64 -e --source winget --accept-package-agreements --accept-source-agreements
  echo Visual C++ Redistributable check finished.
) else (
  echo winget was not found, skipping automatic Visual C++ Redistributable installation.
  echo If rawpy fails with a DLL error, install Microsoft Visual C++ Redistributable 2015-2022 x64 manually.
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
"%PY%" -m pip install --upgrade numpy opencv-python pillow astropy xisf PySide6
if errorlevel 1 goto :pipfail

rem Install rawpy separately because it contains native DLL/PYD dependencies.
echo.
echo Installing rawpy...
"%PY%" -m pip install --upgrade --no-cache-dir rawpy
if errorlevel 1 goto :pipfail

echo.
set /p INSTALL_CUPY=Install optional NVIDIA CUDA/CuPy GPU support? [y/N]
if /I "%INSTALL_CUPY%"=="Y" (
  echo Installing CuPy for CUDA 12.x...
  "%PY%" -m pip install --upgrade cupy-cuda12x
  if errorlevel 1 (
    echo CuPy installation failed. The CPU version of Astro Stacker will still work.
  )
) else (
  echo Skipping CuPy.
)

echo.
echo Verifying imports...
"%PY%" -c "import cv2, numpy, rawpy; from PIL import Image; from astropy.io import fits; from xisf import XISF; from PySide6.QtWidgets import QApplication; print('Base dependencies OK')"
if errorlevel 1 (
  echo.
  echo Dependency verification failed.
  echo Trying to repair rawpy installation...
  "%PY%" -m pip uninstall -y rawpy
  "%PY%" -m pip install --no-cache-dir rawpy
  echo.
  echo Verifying imports again...
  "%PY%" -c "import cv2, numpy, rawpy; from PIL import Image; from astropy.io import fits; from xisf import XISF; from PySide6.QtWidgets import QApplication; print('Base dependencies OK')"
  if errorlevel 1 (
    echo.
    echo Dependency verification failed again.
    echo Most likely cause: missing Microsoft Visual C++ Redistributable 2015-2022 x64,
    echo or a rawpy DLL dependency problem on this Windows installation.
    echo Try restarting Windows and running this installer again.
    pause
    exit /b 1
  )
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
