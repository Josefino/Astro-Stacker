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
set "REDIST_DIR=%PACKAGING%\redist"
set "VCREDIST=%REDIST_DIR%\vc_redist.x64.exe"

echo ============================================================
echo Astro Stacker 3.0 - Windows installer build
echo ============================================================
echo.

if not exist "%ROOT%\astro_stacker_app.py" (
  echo ERROR: astro_stacker_app.py was not found.
  pause
  exit /b 1
)
if not exist "%ROOT%\models\drunet_color.onnx" (
  echo ERROR: models\drunet_color.onnx was not found.
  pause
  exit /b 1
)
if not exist "%ROOT%\models\drunet_gray.onnx" (
  echo ERROR: models\drunet_gray.onnx was not found.
  pause
  exit /b 1
)
if not exist "%ROOT%\models\cosmic_clarity_stellar.onnx" (
  echo ERROR: models\cosmic_clarity_stellar.onnx was not found.
  pause
  exit /b 1
)
if not exist "%ROOT%\MANUAL_EN.html" goto :missing_payload
if not exist "%ROOT%\MANUAL_CZ.html" goto :missing_payload
if not exist "%ROOT%\AS_Stacker_PI_Plugin\AS_Stacker_PI.js" goto :missing_payload
if not exist "%ROOT%\AS_Stacker_PI_Plugin\astro_stacker_cli.py" goto :missing_payload
if not exist "%ROOT%\AS_Stacker_PI_Plugin\astro_stacker_app.py" goto :missing_payload
fc /b "%ROOT%\astro_stacker_app.py" "%ROOT%\AS_Stacker_PI_Plugin\astro_stacker_app.py" >nul
if errorlevel 1 (
  echo ERROR: PixInsight wrapper contains an outdated astro_stacker_app.py.
  echo Synchronize AS_Stacker_PI_Plugin before building the installer.
  pause
  exit /b 1
)
fc /b "%ROOT%\astro_stacker_cli.py" "%ROOT%\AS_Stacker_PI_Plugin\astro_stacker_cli.py" >nul
if errorlevel 1 (
  echo ERROR: PixInsight wrapper contains an outdated astro_stacker_cli.py.
  echo Synchronize AS_Stacker_PI_Plugin before building the installer.
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
"%PY%" -m pip uninstall -y onnxruntime-gpu onnxruntime-directml cupy-cuda12x ^
  nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cublas-cu12 ^
  nvidia-nvjitlink-cu12 nvidia-cuda-cccl-cu12 >nul 2>nul
"%PY%" -m pip install --upgrade -r "%PACKAGING%\requirements-base.txt"
if errorlevel 1 goto :failed

echo Preparing icons...
"%PY%" "%PACKAGING%\make_icons.py"
if errorlevel 1 goto :failed

if not exist "%REDIST_DIR%" mkdir "%REDIST_DIR%"
if not exist "%VCREDIST%" (
  echo Downloading Microsoft Visual C++ Redistributable x64...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing 'https://aka.ms/vc14/vc_redist.x64.exe' -OutFile '%VCREDIST%'"
  if errorlevel 1 (
    echo ERROR: Microsoft Visual C++ Redistributable could not be downloaded.
    echo Download it manually from:
    echo   https://aka.ms/vc14/vc_redist.x64.exe
    echo and save it as:
    echo   %VCREDIST%
    goto :failed
  )
)
for %%F in ("%VCREDIST%") do if %%~zF LSS 1000000 (
  echo ERROR: vc_redist.x64.exe is incomplete or invalid.
  goto :failed
)

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
call :verify_payload "%DIST%\AstroStacker_CPU" "CPU"
if errorlevel 1 goto :failed

echo.
echo Installing CUDA build dependencies...
"%PY%" -m pip uninstall -y onnxruntime
if errorlevel 1 goto :failed
"%PY%" -m pip install --upgrade -r "%PACKAGING%\requirements-windows-cuda.txt"
if errorlevel 1 goto :failed

echo Testing CuPy import. A missing NVIDIA GPU is allowed on the build PC...
"%PY%" -c "import cupy; print('CuPy build version:', cupy.__version__)"
if errorlevel 1 goto :failed
"%PY%" -c "import onnxruntime as ort; providers=ort.get_available_providers(); print('ONNX Runtime providers:', providers); assert 'DmlExecutionProvider' in providers"
if errorlevel 1 (
  echo ERROR: onnxruntime-directml does not provide DmlExecutionProvider.
  goto :failed
)

echo.
echo Building NVIDIA CUDA application...
"%PY%" -m PyInstaller --noconfirm --clean ^
  --distpath "%DIST%" ^
  --workpath "%WORK%\cuda" ^
  "%PACKAGING%\AstroStacker-Windows-CUDA.spec"
if errorlevel 1 goto :failed

echo Verifying CUDA application...
if not exist "%DIST%\AstroStacker_CUDA\AstroStacker.exe" goto :failed
call :verify_payload "%DIST%\AstroStacker_CUDA" "CUDA"
if errorlevel 1 goto :failed
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
where /r "%DIST%\AstroStacker_CUDA" cublas*.dll >nul 2>nul
if errorlevel 1 (
  echo ERROR: The CUDA build does not contain cuBLAS DLL files.
  echo Check the cupy-cuda12x[ctk] installation and PyInstaller collection.
  goto :failed
)
where /r "%DIST%\AstroStacker_CUDA" nvJitLink*.dll >nul 2>nul
if errorlevel 1 (
  echo ERROR: The CUDA build does not contain nvJitLink DLL files.
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
echo   %RELEASE%\AstroStacker30_Setup.exe
echo.
pause
exit /b 0

:missing_payload
echo ERROR: A required application, manual, or PixInsight wrapper file is missing.
echo Restore the complete release source tree and run the build again.
pause
exit /b 1

:verify_payload
set "PAYLOAD_ROOT=%~1"
set "PAYLOAD_NAME=%~2"
where /r "%PAYLOAD_ROOT%" drunet_color.onnx >nul 2>nul
if errorlevel 1 (
  echo ERROR: %PAYLOAD_NAME% build does not contain drunet_color.onnx.
  exit /b 1
)
where /r "%PAYLOAD_ROOT%" drunet_gray.onnx >nul 2>nul
if errorlevel 1 (
  echo ERROR: %PAYLOAD_NAME% build does not contain drunet_gray.onnx.
  exit /b 1
)
where /r "%PAYLOAD_ROOT%" cosmic_clarity_stellar.onnx >nul 2>nul
if errorlevel 1 (
  echo ERROR: %PAYLOAD_NAME% build does not contain cosmic_clarity_stellar.onnx.
  exit /b 1
)
where /r "%PAYLOAD_ROOT%" MANUAL_EN.html >nul 2>nul
if errorlevel 1 (
  echo ERROR: %PAYLOAD_NAME% build does not contain MANUAL_EN.html.
  exit /b 1
)
where /r "%PAYLOAD_ROOT%" MANUAL_CZ.html >nul 2>nul
if errorlevel 1 (
  echo ERROR: %PAYLOAD_NAME% build does not contain MANUAL_CZ.html.
  exit /b 1
)
where /r "%PAYLOAD_ROOT%" AS_Stacker_PI.js >nul 2>nul
if errorlevel 1 (
  echo ERROR: %PAYLOAD_NAME% build does not contain the PixInsight wrapper.
  exit /b 1
)
echo %PAYLOAD_NAME% payload verification passed.
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
