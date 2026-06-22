@echo off
setlocal EnableExtensions

cd /d "%~dp0\.."
set "ROOT=%CD%"
set "PACKAGING=%ROOT%\packaging"
set "VENV=%ROOT%\.venv-build-windows-lite"
set "PY=%VENV%\Scripts\python.exe"
set "DIST=%ROOT%\dist_installer_lite"
set "WORK=%ROOT%\build_installer_lite"
set "RELEASE=%ROOT%\release"
set "REDIST_DIR=%PACKAGING%\redist"
set "VCREDIST=%REDIST_DIR%\vc_redist.x64.exe"

echo ============================================================
echo Astro Stacker 3.1 Lite - small Windows installer
echo CPU only, without CUDA and AI denoise
echo ============================================================
echo.

if not exist "%ROOT%\astro_stacker_app.py" (
  echo ERROR: astro_stacker_app.py was not found.
  pause
  exit /b 1
)
if not exist "%ROOT%\AstroStacker_intro.png" (
  echo ERROR: AstroStacker_intro.png was not found.
  pause
  exit /b 1
)
if not exist "%ROOT%\MANUAL_EN.html" (
  if exist "%ROOT%\AS_Stacker_PI_Plugin\MANUAL_EN.html" (
    echo MANUAL_EN.html will be taken from AS_Stacker_PI_Plugin.
  ) else (
    echo WARNING: MANUAL_EN.html was not found. The Lite installer will be built without it.
  )
)
if not exist "%ROOT%\MANUAL_CZ.html" (
  if exist "%ROOT%\AS_Stacker_PI_Plugin\MANUAL_CZ.html" (
    echo MANUAL_CZ.html will be taken from AS_Stacker_PI_Plugin.
  ) else (
    echo WARNING: MANUAL_CZ.html was not found. The Lite installer will be built without it.
  )
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
  echo Creating isolated Lite build environment...
  %PYLAUNCHER% -m venv "%VENV%"
  if errorlevel 1 goto :failed
)

echo Installing Lite build dependencies...
"%PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :failed
"%PY%" -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml cupy cupy-cuda12x torch ^
  nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cublas-cu12 ^
  nvidia-nvjitlink-cu12 nvidia-cuda-cccl-cu12 >nul 2>nul
"%PY%" -m pip install --upgrade -r "%PACKAGING%\requirements-windows-lite.txt"
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
    echo Save it manually as:
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
echo Building Lite CPU application...
"%PY%" -m PyInstaller --noconfirm --clean ^
  --distpath "%DIST%" ^
  --workpath "%WORK%" ^
  "%PACKAGING%\AstroStacker-Windows-Lite.spec"
if errorlevel 1 goto :failed

if not exist "%DIST%\AstroStacker_Lite\AstroStacker.exe" goto :failed
where /r "%DIST%\AstroStacker_Lite" AstroStacker_intro.png >nul 2>nul
if errorlevel 1 (
  echo ERROR: Lite build does not contain AstroStacker_intro.png.
  goto :failed
)

echo Verifying that AI and CUDA payloads are absent...
where /r "%DIST%\AstroStacker_Lite" *.onnx >nul 2>nul
if not errorlevel 1 (
  echo ERROR: Lite build unexpectedly contains an ONNX model.
  goto :failed
)
where /r "%DIST%\AstroStacker_Lite" onnxruntime*.dll >nul 2>nul
if not errorlevel 1 (
  echo ERROR: Lite build unexpectedly contains ONNX Runtime.
  goto :failed
)
where /r "%DIST%\AstroStacker_Lite" cupy*.pyd >nul 2>nul
if not errorlevel 1 (
  echo ERROR: Lite build unexpectedly contains CuPy.
  goto :failed
)
where /r "%DIST%\AstroStacker_Lite" cudart*.dll >nul 2>nul
if not errorlevel 1 (
  echo ERROR: Lite build unexpectedly contains CUDA runtime DLLs.
  goto :failed
)
where /r "%DIST%\AstroStacker_Lite" cublas*.dll >nul 2>nul
if not errorlevel 1 (
  echo ERROR: Lite build unexpectedly contains cuBLAS DLLs.
  goto :failed
)
where /r "%DIST%\AstroStacker_Lite" torch*.dll >nul 2>nul
if not errorlevel 1 (
  echo ERROR: Lite build unexpectedly contains PyTorch.
  goto :failed
)

call :find_inno
if "%ISCC%"=="" (
  where winget >nul 2>nul
  if not errorlevel 1 (
    echo Installing Inno Setup 6 with winget...
    winget install --id JRSoftware.InnoSetup -e --source winget ^
      --accept-package-agreements --accept-source-agreements
    if not errorlevel 1 call :find_inno
  )
)
if "%ISCC%"=="" (
  echo ERROR: Inno Setup 6 could not be installed or found.
  echo Install it with: winget install --id JRSoftware.InnoSetup -e
  pause
  exit /b 2
)

echo Creating AstroStacker31_Lite_Setup.exe...
pushd "%PACKAGING%"
"%ISCC%" "AstroStacker-Lite.iss"
set "INNO_RESULT=%ERRORLEVEL%"
popd
if not "%INNO_RESULT%"=="0" goto :failed

echo.
echo Lite installer created successfully:
echo   %RELEASE%\AstroStacker31_Lite_Setup.exe
echo.
pause
exit /b 0

:failed
echo.
echo ERROR: Lite installer build failed.
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
