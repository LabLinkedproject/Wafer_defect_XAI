@echo off
rem Double-click on the Windows laptop: builds WaferDefectXAI.exe here. No install needed.
cd /d "%~dp0"
set CSC=%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe
if not exist "%CSC%" set CSC=%WINDIR%\Microsoft.NET\Framework\v4.0.30319\csc.exe
if not exist "%CSC%" (
  echo C# compiler not found. .NET Framework 4 is required.
  pause
  exit /b 1
)
"%CSC%" /nologo /codepage:65001 /target:winexe /out:WaferDefectXAI.exe /r:System.Windows.Forms.dll WaferDefectXAI.cs
if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)
echo Done: WaferDefectXAI.exe
pause
