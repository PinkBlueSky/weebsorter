@echo off
REM ============================================================
REM  anime-tagger launcher
REM  Assumes uv is installed and on PATH.
REM  HF_HOME is set inside main.py to .\models\
REM ============================================================
setlocal
cd /d "%~dp0"
uv run anime-tagger
if %ERRORLEVEL% neq 0 (
    echo.
    echo anime-tagger exited with error code %ERRORLEVEL%.
)
pause
endlocal
