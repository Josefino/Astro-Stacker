@echo off
setlocal EnableExtensions

cd /d "%~dp0\.."
set "ROOT=%CD%"
set "PACKAGING=%ROOT%\packaging"
set "VENV=%ROOT%\.venv-build-windows"
set "PY=%VENV%\Scripts\python.exe"
set "DIST=%ROOT%\dist_installer"
set "WORK=%ROOT%\build_installer"
set "RELEASE=%ROOT%\release"

echo ============================================================
echo Astro Stacker 2.8 - Windows installer build
echo ============================================================
echo.

if not exist "%ROOT%\astro_stacker_app.py" (
  echo ERROR: astro_stacker_app.py was not found.
  pause
  exit /b 1
)

set "PYLAUNCHER="
where py >nul 2>nul
if not errorlevel 1 py -3.12 --version >nul 2>nul
if not errorlevel 1 set "PYLAUNCHER=py -3.12"
if "%PYLAUNCHER%"=="" (
  where python >nul 2>nul
  if not errorlevel 1 set "PYLAUNCHER=python"
)
if "%PYLAUNCHER%"=="" (
  echo ERROR: Python 3.12 was not found.
  echo Install 64-bit Python 3.12 from https://www.python.org/
  pause
  exit /b 1
)

if not exist "%PY%" (
  echo Creating build environment...
  %PYLAUNCHER% -m venv "%VENV%"
  if errorlevel 1 goto :failed
)

echo Installing base build dependencies...
"%PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :failed
"%PY%" -m pip install --upgrade -r "%PACKAGING%\requirements-base.txt"
if errorlevel 1 goto :failed

echo Preparing icons...
"%PY%" "%PACKAGING%\make_icons.py"
if errorlevel 1 goto :failed

if exist "%DIST%" rmdir /s /q "%DIST%"
if exist "%WORK%" rmdir /s /q "%WORK%"
if not exist "%RELEASE%" mkdir "%RELEASE%"

echo.
echo Building CPU application...
"%PY%" -m PyInstaller --noconfirm --clean ^
  --distpath "%DIST%" ^
  --workpath "%WORK%\cpu" ^
  "%PACKAGING%\AstroStacker-Windows-CPU.spec"
if errorlevel 1 goto :failed

echo Verifying CPU application...
if not exist "%DIST%\AstroStacker_CPU\AstroStacker.exe" goto :failed

echo.
echo Installing CUDA build dependencies...
"%PY%" -m pip install --upgrade -r "%PACKAGING%\requirements-windows-cuda.txt"
if errorlevel 1 goto :failed

echo Testing CuPy import. A missing NVIDIA GPU is allowed on the build PC...
"%PY%" -c "import cupy; print('CuPy build version:', cupy.__version__)"
if errorlevel 1 goto :failed

echo.
echo Building NVIDIA CUDA application...
"%PY%" -m PyInstaller --noconfirm --clean ^
  --distpath "%DIST%" ^
  --workpath "%WORK%\cuda" ^
  "%PACKAGING%\AstroStacker-Windows-CUDA.spec"
if errorlevel 1 goto :failed

echo Verifying CUDA application...
if not exist "%DIST%\AstroStacker_CUDA\AstroStacker.exe" goto :failed
where /r "%DIST%\AstroStacker_CUDA" nvrtc*.dll >nul 2>nul
if errorlevel 1 (
  echo ERROR: The CUDA build does not contain nvrtc DLL files.
  echo Check the cupy-cuda12x[ctk] installation and PyInstaller collection.
  goto :failed
)
where /r "%DIST%\AstroStacker_CUDA" cudart*.dll >nul 2>nul
if errorlevel 1 (
  echo ERROR: The CUDA build does not contain cudart DLL files.
  echo Check the cupy-cuda12x[ctk] installation and PyInstaller collection.
  goto :failed
)

call :find_inno

if "%ISCC%"=="" (
  echo.
  echo Inno Setup 6 is not installed.
  where winget >nul 2>nul
  if not errorlevel 1 (
    echo Installing Inno Setup 6 with winget...
    winget install --id JRSoftware.InnoSetup -e --source winget ^
      --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
      echo Automatic Inno Setup installation failed.
    ) else (
      call :find_inno
    )
  )
)

if "%ISCC%"=="" (
  echo.
  echo CPU and CUDA applications were built successfully, but Inno Setup
  echo could not be installed or found.
  echo Install it manually with:
  echo   winget install --id JRSoftware.InnoSetup -e
  echo Then run this script again. Existing build folders may be reused.
  pause
  exit /b 2
)

echo.
echo Creating the final installer...
pushd "%PACKAGING%"
"%ISCC%" "AstroStacker.iss"
set "INNO_RESULT=%ERRORLEVEL%"
popd
if not "%INNO_RESULT%"=="0" goto :failed

echo.
echo Build complete:
echo   %RELEASE%\AstroStacker28_Setup.exe
echo.
pause
exit /b 0

:failed
echo.
echo ERROR: Installer build failed.
echo Review the messages above for the failing package or build step.
pause
exit /b 1

:find_inno
set "ISCC="
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if exist "%LOCALAPPDATA%\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Inno Setup 6\ISCC.exe"
if "%ISCC%"=="" (
  where ISCC.exe >nul 2>nul
  if not errorlevel 1 set "ISCC=ISCC.exe"
)
if "%ISCC%"=="" (
  for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1" /v InstallLocation 2^>nul ^| find /i "InstallLocation"') do (
    if exist "%%B\ISCC.exe" set "ISCC=%%B\ISCC.exe"
  )
)
if "%ISCC%"=="" (
  for /f "tokens=2,*" %%A in ('reg query "HKLM\Software\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1" /v InstallLocation 2^>nul ^| find /i "InstallLocation"') do (
    if exist "%%B\ISCC.exe" set "ISCC=%%B\ISCC.exe"
  )
)
if not "%ISCC%"=="" echo Inno Setup compiler found: "%ISCC%"
exit /b 0
