#define MyAppName "Astro Stacker Lite"
#define MyAppVersion "3.1"
#define MyAppPublisher "Josef Ladra"
#define MyAppExeName "AstroStacker.exe"

[Setup]
AppId={{A12330D4-FA32-4698-B22A-29E5C0CE8564}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL=https://github.com/Josefino/Astro-Stacker
AppSupportURL=https://github.com/Josefino/Astro-Stacker/issues
AppUpdatesURL=https://github.com/Josefino/Astro-Stacker/releases
DefaultDirName={autopf}\Astro Stacker Lite
DefaultGroupName=Astro Stacker Lite
AllowNoIcons=yes
OutputDir=..\release
OutputBaseFilename=AstroStacker31_Lite_Setup
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
Source: "..\dist_installer_lite\AstroStacker_Lite\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\MANUAL_EN.html"; DestDir: "{app}\Documentation"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\MANUAL_CZ.html"; DestDir: "{app}\Documentation"; Flags: ignoreversion skipifsourcedoesntexist
Source: "redist\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{group}\Astro Stacker Lite"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\Astro Stacker Lite"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Installing or updating Microsoft Visual C++ Runtime..."; Flags: waituntilterminated
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,Astro Stacker Lite}"; Flags: nowait postinstall skipifsilent
