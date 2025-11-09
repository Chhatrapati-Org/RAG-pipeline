@echo off
echo.
echo ============================================================
echo    PS04 RAG System - GUI Launcher
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or higher
    pause
    exit /b 1
)

REM Run the launcher
python launch_gui.py

pause
