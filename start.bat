@echo off
title BraidLM
setlocal EnableDelayedExpansion

echo.
echo  ============================================================
echo   BraidLM  ^|  Local LLM Interface  ^|  Powered by Ollama
echo  ============================================================
echo.

:: 1. Python
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo  [ERROR] Python not found. Install Python 3.10+ from https://python.org
    pause & exit /b 1
)

:: 2. Ollama
ollama --version >nul 2>&1
if !errorlevel! neq 0 (
    echo  [ERROR] Ollama not found.
    echo  Ollama is required. Opening https://ollama.com/download ...
    start https://ollama.com/download
    echo  Install Ollama, then re-run this launcher.
    pause & exit /b 1
)

:: 3. Start the Ollama server if it is not already running
curl -s http://localhost:11434 >nul 2>&1
if !errorlevel! neq 0 (
    echo  [INFO] Starting Ollama server in background...
    start /b "" ollama serve
    timeout /t 3 /nobreak >nul
)

:: 4. Python dependencies
echo  Installing Python dependencies...
pip install -r requirements.txt --quiet
if !errorlevel! neq 0 (
    echo  [ERROR] pip install failed. Check your Python environment.
    pause & exit /b 1
)

:: Models are NOT force-pulled any more: removing a model in BraidLM would be
:: undone on the next launch. If no model is installed, BraidLM offers to install
:: the two defaults (qwen2.5-coder:3b, qwen2.5:3b) or to browse Hugging Face.

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
