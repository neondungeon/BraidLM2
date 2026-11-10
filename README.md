# 🧵 BraidLM

A desktop chat interface for running **Qwen2.5** models entirely locally,
built on the [`nocolour/Ollama-Desktop`](https://github.com/nocolour/Ollama-Desktop)
architecture (MIT) and rebranded as BraidLM.

---

## Why Ollama?

| Problem (previous build)         | How Ollama fixes it                                      |
|----------------------------------|----------------------------------------------------------|
| Models re-downloaded on relaunch | Cached in `~/.ollama/models` forever after first pull    |
| Slow mode-switching (re-load)    | Switching models = changing a string; zero disk I/O      |
| Raw transformers on CPU          | GGUF-quantised weights; faster & lower RAM than FP32     |

---

## Quick Start (Windows)

1. Install **Ollama** → https://ollama.com/download
2. Double-click **`start.bat`**

The launcher will:
- Start `ollama serve` if it isn't running
- Pull both Qwen models (once; ~1.9 GB each)
- Install Python deps (`PySide6`, `ollama`, `pygments`)
- Launch BraidLM

---

## Modes

| Toggle   | Model                  | Best for                        |
|----------|------------------------|---------------------------------|
| **Coder** *(default)* | `qwen2.5-coder:3b` | Code generation & debugging |
| **General**           | `qwen2.5:3b`       | Q&A, writing, reasoning     |

Switching modes is instant — Ollama keeps both models warm.

---

## Keyboard Shortcuts

| Key             | Action          |
|-----------------|-----------------|
| `Enter`         | Send message    |
| `Shift + Enter` | New line        |

---

## Project Structure

```
BraidLM/
├── main.py          ← entry point
├── window.py        ← main window, chat UI, mode toggle, signal wiring
├── worker.py        ← OllamaWorker (QObject, runs in QThread)
├── renderer.py      ← markdown → HTML with Pygments syntax highlighting
├── requirements.txt
└── start.bat        ← Windows one-click launcher
```

Based on [nocolour/Ollama-Desktop](https://github.com/nocolour/Ollama-Desktop) (MIT).
