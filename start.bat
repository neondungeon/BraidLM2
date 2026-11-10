@echo off
title BraidLM
setlocal EnableDelayedExpansion

echo.
echo  ============================================================
echo   BraidLM  ^|  Local LLM Interface  ^|  Powered by Ollama
echo  ============================================================
echo.

:: ── 1. Python ───────────────────────────────────────────────────────────────
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo  [ERROR] Python not found.
    echo          Install Python 3.10+ from https://python.org
    pause & exit /b 1
)

:: ── 2. Ollama ────────────────────────────────────────────────────────────────
ollama --version >nul 2>&1
if !errorlevel! neq 0 (
    echo  [ERROR] Ollama not found.
    echo          Install it from https://ollama.com/download then re-run this launcher.
    pause & exit /b 1
)

:: ── 3. Start Ollama serve if not already running ────────────────────────────
curl -s http://localhost:11434 >nul 2>&1
if !errorlevel! neq 0 (
    echo  [INFO] Starting Ollama server in background...
    start /b "" ollama serve
    timeout /t 3 /nobreak >nul
)

:: ── 4. Pull models (no-op and instant if already cached) ────────────────────
echo  [1/3] Ensuring Coder model is available...
ollama pull qwen2.5-coder:3b
echo  [2/3] Ensuring General model is available...
ollama pull qwen2.5:3b

:: ── 5. Python dependencies ───────────────────────────────────────────────────
echo  [3/3] Installing Python dependencies...
pip install -r requirements.txt --quiet
if !errorlevel! neq 0 (
    echo  [ERROR] pip install failed. Check your Python environment.
    pause & exit /b 1
)

echo.
echo  Launching BraidLM...
echo.
python main.py

if !errorlevel! neq 0 (
    echo.
    echo  [ERROR] BraidLM exited with an error. See output above.
    pause
)
endlocal
