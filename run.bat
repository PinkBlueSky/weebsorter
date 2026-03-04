@echo off
REM ============================================================
REM  anime-tagger launcher
REM  Assumes uv is installed and on PATH.
REM  HF_HOME is set inside main.py to .\models\
REM ============================================================
setlocal
cd /d "%~dp0"

REM Run directly from the project .venv — no network connection needed.
if not exist ".venv\Scripts\python.exe" goto :no_venv

.venv\Scripts\python.exe -m anime_tagger.main
if %ERRORLEVEL% neq 0 (
    echo.
    echo anime-tagger exited with error code %ERRORLEVEL%.
)
goto :done

:no_venv
echo ERROR: .venv not found. Run this once to set it up:
echo   uv sync
echo (uv installer: powershell -c "irm https://astral.sh/uv/install.ps1 | iex")

:done
pause
endlocal
