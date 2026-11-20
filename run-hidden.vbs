' Launches run.bat with its console window hidden. Used as the action of the
' elevated "Tellur" scheduled task (and safe to double-click directly): the
' task supplies the elevated token, run.bat sees it's already admin and skips
' its UAC self-elevation, and no black console window ever flashes.
Dim shell, fso, bat
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
bat = fso.GetParentFolderName(WScript.ScriptFullName) & "\run.bat"
shell.Run """" & bat & """", 0, False
