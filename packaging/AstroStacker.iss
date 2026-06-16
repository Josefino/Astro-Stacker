#define MyAppName "Astro Stacker"
#define MyAppVersion "3.0"
#define MyAppPublisher "Josef Ladra"
#define MyAppExeName "AstroStacker.exe"

[Setup]
AppId={{F13C9E79-C71C-4D6B-BD20-F5A16E15C4AA}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL=https://github.com/Josefino/Astro-Stacker
AppSupportURL=https://github.com/Josefino/Astro-Stacker/issues
AppUpdatesURL=https://github.com/Josefino/Astro-Stacker/releases
DefaultDirName={autopf}\Astro Stacker
DefaultGroupName=Astro Stacker
AllowNoIcons=yes
OutputDir=..\release
OutputBaseFilename=AstroStacker30_Setup
SetupIconFile=icons\AstroStacker.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
UninstallDisplayIcon={app}\{#MyAppExeName}
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "czech"; MessagesFile: "compiler:Languages\Czech.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist_installer\AstroStacker_CPU\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Check: not UseCudaBuild
Source: "..\dist_installer\AstroStacker_CUDA\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Check: UseCudaBuild
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\MANUAL_EN.html"; DestDir: "{app}\Documentation"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\MANUAL_CZ.html"; DestDir: "{app}\Documentation"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\AS_Stacker_PI_Plugin\*"; DestDir: "{app}\AS_Stacker_PI_Plugin"; Excludes: "models\*"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "redist\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{group}\Astro Stacker"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\Astro Stacker"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Installing or updating Microsoft Visual C++ Runtime..."; Flags: waituntilterminated
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,Astro Stacker}"; Flags: nowait postinstall skipifsilent

[Code]
var
  GpuPage: TWizardPage;
  CudaCheckBox: TNewCheckBox;
  CudaInfoLabel: TNewStaticText;

function HasNvidiaGPU: Boolean;
begin
  Result :=
    FileExists(ExpandConstant('{sys}\nvidia-smi.exe')) or
    FileExists(ExpandConstant('{win}\System32\nvidia-smi.exe')) or
    RegKeyExists(HKLM64, 'SOFTWARE\NVIDIA Corporation');
end;

function UseCudaBuild: Boolean;
begin
  Result := CudaCheckBox.Checked;
end;

procedure InitializeWizard;
begin
  GpuPage := CreateCustomPage(
    wpSelectDir,
    'GPU acceleration',
    'Choose whether to install NVIDIA CUDA support.'
  );

  CudaCheckBox := TNewCheckBox.Create(GpuPage);
  CudaCheckBox.Parent := GpuPage.Surface;
  CudaCheckBox.Left := 0;
  CudaCheckBox.Top := 16;
  CudaCheckBox.Width := GpuPage.SurfaceWidth;
  CudaCheckBox.Caption := 'Install NVIDIA CUDA GPU support';
  CudaCheckBox.Checked := HasNvidiaGPU;

  CudaInfoLabel := TNewStaticText.Create(GpuPage);
  CudaInfoLabel.Parent := GpuPage.Surface;
  CudaInfoLabel.Left := 0;
  CudaInfoLabel.Top := CudaCheckBox.Top + CudaCheckBox.Height + 16;
  CudaInfoLabel.Width := GpuPage.SurfaceWidth;
  CudaInfoLabel.Height := 80;
  CudaInfoLabel.AutoSize := False;
  CudaInfoLabel.WordWrap := True;
  CudaInfoLabel.Caption :=
    'Enable this only for a Windows computer with a supported NVIDIA GPU. ' +
    'The current NVIDIA display driver must be installed separately. ' +
    'Without this option Astro Stacker uses the CPU implementation.';
end;
