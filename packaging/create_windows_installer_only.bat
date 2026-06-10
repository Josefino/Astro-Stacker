@echo off
setlocal EnableExtensions

cd /d "%~dp0\.."
set "ROOT=%CD%"
set "PACKAGING=%ROOT%\packaging"
set "DIST=%ROOT%\dist_installer"
set "RELEASE=%ROOT%\release"

echo ============================================================
echo Astro Stacker 2.8 - create installer from existing builds
echo ============================================================
echo.

if not exist "%DIST%\AstroStacker_CPU\AstroStacker.exe" (
  echo ERROR: The CPU build was not found:
  echo   %DIST%\AstroStacker_CPU\AstroStacker.exe
  echo Run packaging\build_windows_installer.bat first.
  pause
  exit /b 1
)

if not exist "%DIST%\AstroStacker_CUDA\AstroStacker.exe" (
  echo ERROR: The CUDA build was not found:
  echo   %DIST%\AstroStacker_CUDA\AstroStacker.exe
  echo Run packaging\build_windows_installer.bat first.
  pause
  exit /b 1
)

if not exist "%RELEASE%" mkdir "%RELEASE%"

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
  echo.
  echo ERROR: Inno Setup 6 could not be installed or found.
  echo Install it manually:
  echo   winget install --id JRSoftware.InnoSetup -e
  pause
  exit /b 2
)

echo Creating AstroStacker28_Setup.exe...
pushd "%PACKAGING%"
"%ISCC%" "AstroStacker.iss"
set "INNO_RESULT=%ERRORLEVEL%"
popd
if not "%INNO_RESULT%"=="0" (
  echo.
  echo ERROR: Inno Setup compilation failed.
  pause
  exit /b 1
)

echo.
echo Installer created successfully:
echo   %RELEASE%\AstroStacker28_Setup.exe
echo.
pause
exit /b 0

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
