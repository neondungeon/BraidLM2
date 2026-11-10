"""
worker.py — BraidLM Ollama inference worker
============================================
Runs inside a QThread so the GUI stays responsive.

Ollama is used as the LLM backend because:
  • It caches models in ~/.ollama/models — zero re-downloads after the first pull.
  • Switching models costs nothing: just a name string in the API call.
  • GGUF-quantised Qwen2.5 weights are better optimised than raw HuggingFace
    transformers checkpoints for CPU inference.
  • The streaming API is trivial: iterate chunks from ollama.chat(stream=True).

Models
------
  coder   → qwen2.5-coder:3b   (Qwen2.5-Coder-3B-Instruct, GGUF)
  general → qwen2.5:3b          (Qwen2.5-3B-Instruct, GGUF)
"""

from PySide6.QtCore import QObject, Signal
import ollama

# ── Model IDs on the Ollama registry ───────────────────────────────────────
MODELS: dict[str, str] = {
    "coder":   "qwen2.5-coder:3b",
    "general": "qwen2.5:3b",
}

# ── Per-mode system prompts ─────────────────────────────────────────────────
SYSTEM_PROMPTS: dict[str, str] = {
    "coder": (
        "You are BraidLM, an expert coding assistant powered by Qwen2.5-Coder. "
        "Write clean, efficient, well-commented code that actually works. "
        "Always wrap code in fenced blocks with a language tag (e.g. ```python). "
        "After writing code, briefly explain what it does and any edge cases."
    ),
    "general": (
        "You are BraidLM, a helpful and knowledgeable AI assistant powered by Qwen2.5. "
        "Answer questions clearly, accurately, and at the right level of depth."
    ),
}


class OllamaWorker(QObject):
    """
    Streams an Ollama response and forwards tokens to the GUI via Qt signals.

    Usage (from MainWindow):
        worker = OllamaWorker(mode, history_snapshot)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.token  .connect(gui_slot)
        worker.done   .connect(gui_slot)
        worker.error  .connect(gui_slot)
        thread.start()
    """

    token = Signal(str)   # One decoded chunk of text
    done  = Signal(str)   # Full response text, emitted once at the end
    error = Signal(str)   # Human-readable error message

    def __init__(self, mode: str, messages: list[dict], parent=None) -> None:
        super().__init__(parent)
        self.mode     = mode
        self.messages = messages   # Snapshot of history; no race conditions

    def run(self) -> None:
        """Called by QThread.started; blocks until generation is complete."""
        try:
            full_messages = [
                {"role": "system", "content": SYSTEM_PROMPTS[self.mode]}
            ] + self.messages

            accumulated = ""
            for chunk in ollama.chat(
                model=MODELS[self.mode],
                messages=full_messages,
                stream=True,
            ):
                tok = chunk["message"]["content"]
                accumulated += tok
                self.token.emit(tok)

            self.done.emit(accumulated)

        except ollama.ResponseError as exc:
            # Model not pulled yet or Ollama version mismatch
            self.error.emit(
                f"Ollama error: {exc.error}\n\n"
                f"Try:  ollama pull {MODELS[self.mode]}"
            )
        except Exception as exc:
            # Ollama not running, network issue, etc.
            self.error.emit(
                f"{type(exc).__name__}: {exc}\n\n"
                "Make sure Ollama is installed and running.\n"
                "Start it with:  ollama serve"
            )
