@echo off
REM Backup dos dados privados (versao Windows do backup.sh)
REM Uso: clique duplo, ou "backup.bat D:\" no cmd para salvar no pendrive
cd /d "%~dp0"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set STAMP=%%i
set DEST=%~1
if "%DEST%"=="" set DEST=%USERPROFILE%
set OUT=%DEST%\camargo_backup_%STAMP%.tar.gz

tar -czf "%OUT%" data/app.db data/facts.json knowledge/perfil_marcelo_camargo.md dataset/17_feedback.jsonl dataset/00_feedback_negativo.jsonl knowledge/doc_*.md 2>nul

echo.
echo Backup criado: %OUT%
echo Conteudo:
tar -tzf "%OUT%"
echo.
pause
