"""
worker.py: BraidLM background workers
=====================================
Everything slow (chat generation, context compaction, image generation, model
downloads, Hugging Face queries) runs off the GUI thread and talks back through
Qt signals.  Every worker can be cancelled from the GUI thread at any moment,
even while it is still waiting for the first token.

Ollama is the LLM backend; Hugging Face is the model source.
"""
from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

import context_guard as cg
import imagegen
from ollama_client import (ModelInfo, OllamaClient, OllamaError, PullProgress, StreamHandle)

# ── Built-in modes (unchanged from the original BraidLM) ────────────────────
MODELS: dict[str, str] = {
    "coder":   "qwen2.5-coder:3b",
    "general": "qwen2.5:3b",
}

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

GENERIC_SYSTEM = (
    "You are BraidLM, a helpful and knowledgeable AI assistant running locally through Ollama. "
    "Answer clearly and accurately. Put code in fenced blocks with a language tag."
)


def mode_for_model(model_name: str) -> str | None:
    for mode, name in MODELS.items():
        if name == model_name:
            return mode
    return None


def system_prompt_for(model_name: str) -> str:
    mode = mode_for_model(model_name)
    return SYSTEM_PROMPTS[mode] if mode else GENERIC_SYSTEM


def friendly_error(msg: str, model: str = "") -> str:
    low = msg.lower()
    if "more system memory" in low or "out of memory" in low:
        return (msg + "\n\nThis model does not fit in memory at the current context size. Lower the "
                "context (status bar → ctx), or choose a smaller / more heavily quantised model in ＋.")
    if "not found" in low and model:
        return f"{msg}\n\nThe model '{model}' is not installed. Use ＋ to install it."
    if "cannot reach ollama" in low or "connection" in low:
        return msg + "\n\nMake sure Ollama is installed and running (start it with:  ollama serve)."
    return msg


# ── Chat ────────────────────────────────────────────────────────────────────

@dataclass
class ChatJob:
    model: str
    system: str
    messages: list                      # API-form messages not yet folded into the summary
    summary: str
    ctx: int
    ctx_ceiling: int
    est: cg.TokenEstimator
    force_compact: bool = False
    think_capable: bool = False
    options: dict = field(default_factory=dict)


class ChatWorker(QObject):
    token = Signal(str)
    thinking = Signal(str)
    notice = Signal(str)                # context-guard actions the user should see
    ctx_changed = Signal(int)
    compacted = Signal(str, int)        # new summary, number of input messages it now covers
    done = Signal(object)
    stopped = Signal(object)
    error = Signal(str)

    def __init__(self, client: OllamaClient, job: ChatJob, parent=None) -> None:
        super().__init__(parent)
        self.client = client
        self.job = job
        self.handle = StreamHandle()

    def request_stop(self) -> None:
        self.handle.cancel()

    @Slot()
    def run(self) -> None:
        j = self.job
        acc: list[str] = []
        think: list[str] = []
        stats: dict = {}
        prep = None
        try:
            def summarize(prev: str, chunk: str, ctx: int) -> str:
                user = ((f"Existing summary:\n{prev}\n\n" if prev else "")
                        + f"New conversation to fold into the summary:\n{chunk}\n\nWrite the updated summary.")
                payload = {"model": j.model,
                           "messages": [{"role": "system", "content": cg.SUMMARY_SYSTEM},
                                        {"role": "user", "content": user}],
                           "options": {"num_ctx": ctx, "temperature": 0.2, "num_predict": 700}}
                if j.think_capable:
                    payload["think"] = False
                return self.client.chat_once(payload, self.handle)

            prep = cg.prepare(j.system, j.summary, j.messages, j.ctx, j.ctx_ceiling, j.est,
                              summarize, j.force_compact)
            if self.handle.cancelled:
                self.stopped.emit(self._payload(acc, think, stats, prep))
                return
            for n in prep.notices:
                self.notice.emit(n)
            if prep.ctx != j.ctx:
                self.ctx_changed.emit(prep.ctx)
            if prep.compacted:
                self.compacted.emit(prep.summary, prep.consumed)

            payload = {"model": j.model, "messages": prep.messages,
                       "options": {"num_ctx": prep.ctx, **j.options}}
            for ev in self.client.chat_stream(payload, self.handle):
                msg = ev.get("message") or {}
                if msg.get("thinking"):
                    think.append(msg["thinking"])
                    self.thinking.emit(msg["thinking"])
                if msg.get("content"):
                    acc.append(msg["content"])
                    self.token.emit(msg["content"])
                if ev.get("done"):
                    stats = ev

            if self.handle.cancelled:
                self.stopped.emit(self._payload(acc, think, stats, prep))
                return
            out = self._payload(acc, think, stats, prep)
            try:
                out["actual_ctx"] = self.client.running_context(j.model)
            except OllamaError:
                out["actual_ctx"] = 0
            self.done.emit(out)
        except OllamaError as exc:
            if self.handle.cancelled:
                self.stopped.emit(self._payload(acc, think, stats, prep))
            else:
                self.error.emit(friendly_error(str(exc), j.model))
        except Exception as exc:                                  # never kill the thread silently
            traceback.print_exc()
            self.error.emit(f"{type(exc).__name__}: {exc}")

    def _payload(self, acc, think, stats, prep) -> dict:
        return {
            "text": "".join(acc), "thinking": "".join(think),
            "prompt_eval_count": int(stats.get("prompt_eval_count") or 0),
            "eval_count": int(stats.get("eval_count") or 0),
            "eval_duration": int(stats.get("eval_duration") or 0),
            "total_duration": int(stats.get("total_duration") or 0),
            "est_sent": prep.est_tokens if prep else 0,
            "ctx": prep.ctx if prep else self.job.ctx,
        }


# ── Image generation ────────────────────────────────────────────────────────

class ImageWorker(QObject):
    progress = Signal(int, int)
    status = Signal(str)
    done = Signal(object)
    stopped = Signal()
    error = Signal(str)

    def __init__(self, client: OllamaClient, model: imagegen.ImageModel, prompt: str, parent=None) -> None:
        super().__init__(parent)
        self.client, self.model, self.prompt = client, model, prompt
        self.handle = StreamHandle()
        self.cancel = threading.Event()

    def request_stop(self) -> None:
        self.cancel.set()
        self.handle.cancel()

    @Slot()
    def run(self) -> None:
        try:
            if self.model.kind == "ollama":
                self.status.emit(f"Generating image with {self.model.name}…")
                png = imagegen.generate_ollama(self.client, self.model.name, self.prompt, self.handle,
                                               lambda d, t: self.progress.emit(d, t))
                path = imagegen.save_png_bytes(png, self.prompt)
            else:
                ok, why = imagegen.diffusers_status()
                if not ok:
                    raise imagegen.ImageGenError(why)
                self.status.emit("Freeing GPU memory held by chat models…")
                try:
                    for m in self.client.ps():
                        self.client.unload(m.get("name") or m.get("model"))
                except OllamaError:
                    pass
                self.status.emit(f"Loading {self.model.name} (first run can take a minute)…")
                img = imagegen.engine().generate(self.model, self.prompt,
                                                 lambda d, t: self.progress.emit(d, t), self.cancel)
                path = imagegen.save_pil(img, self.prompt)
            self.done.emit({"path": str(path), "prompt": self.prompt, "model": self.model.name})
        except imagegen.ImageGenCancelled:
            self.stopped.emit()
        except imagegen.ImageGenError as exc:
            self.error.emit(str(exc))
        except Exception as exc:
            if self.cancel.is_set():
                self.stopped.emit()
            else:
                traceback.print_exc()
                self.error.emit(f"{type(exc).__name__}: {exc}")


# ── Model install ───────────────────────────────────────────────────────────

class PullWorker(QObject):
    """ollama pull <name> with aggregated progress, then a load test."""
    progress = Signal(object, object, str)       # done bytes, total bytes, status text
    done = Signal()
    cancelled = Signal()
    error = Signal(str)
    verify_failed = Signal(str)

    def __init__(self, client: OllamaClient, name: str, verify: bool = True, parent=None) -> None:
        super().__init__(parent)
        self.client, self.name, self.verify = client, name, verify
        self.handle = StreamHandle()

    def request_stop(self) -> None:
        self.handle.cancel()

    @Slot()
    def run(self) -> None:
        agg = PullProgress()
        try:
            ok = False
            for ev in self.client.pull_stream(self.name, self.handle):
                d, t, st = agg.feed(ev)
                self.progress.emit(d, t, st)
                if st == "success":
                    ok = True
            if self.handle.cancelled:
                self.cancelled.emit()
                return
            if not ok:
                self.error.emit("The download ended before Ollama reported success.")
                return
            if self.verify:
                self.progress.emit(1, 1, "Verifying that the model loads…")
                try:
                    info = self.client.info(self.name)
                    if info.is_chat:
                        self.client.chat_once({"model": self.name, "stream": True,
                                               "messages": [{"role": "user", "content": "Hi"}],
                                               "options": {"num_predict": 1, "num_ctx": 2048}}, self.handle)
                except OllamaError as exc:
                    if self.handle.cancelled:
                        self.cancelled.emit()
                        return
                    self.verify_failed.emit(str(exc))
                    return
            self.done.emit()
        except OllamaError as exc:
            if self.handle.cancelled:
                self.cancelled.emit()
            else:
                self.error.emit(friendly_error(str(exc), self.name))
        except Exception as exc:
            traceback.print_exc()
            self.error.emit(f"{type(exc).__name__}: {exc}")


class DownloadWorker(QObject):
    """Download a diffusers model from Hugging Face (resumable)."""
    progress = Signal(object, object, str)
    done = Signal()
    cancelled = Signal()
    error = Signal(str)

    def __init__(self, hub, plan, dest, parent=None) -> None:
        super().__init__(parent)
        self.hub, self.plan, self.dest = hub, plan, dest
        self.cancel = threading.Event()

    def request_stop(self) -> None:
        self.cancel.set()

    @Slot()
    def run(self) -> None:
        import hf_hub
        try:
            self.hub.download_plan(self.plan, self.dest,
                                   lambda d, t, n: self.progress.emit(d, t, n), self.cancel)
            self.done.emit()
        except hf_hub.HubError as exc:
            if self.cancel.is_set():
                self.cancelled.emit()
            else:
                self.error.emit(str(exc))
        except Exception as exc:
            traceback.print_exc()
            self.error.emit(f"{type(exc).__name__}: {exc}")


# ── Tiny "run this off the GUI thread" helper ───────────────────────────────

class _Relay(QObject):
    dispatch = Signal(object, object)

    def __init__(self) -> None:
        super().__init__()
        self.dispatch.connect(self._run)

    @Slot(object, object)
    def _run(self, fn, arg) -> None:
        try:
            fn(arg)
        except RuntimeError:
            pass                      # widget was closed before the result arrived
        except Exception:
            traceback.print_exc()


_RELAY: _Relay | None = None


class _Task(QRunnable):
    def __init__(self, fn, on_result, on_error, relay: _Relay) -> None:
        super().__init__()
        self.fn, self.on_result, self.on_error, self.relay = fn, on_result, on_error, relay

    def run(self) -> None:
        try:
            result = self.fn()
        except Exception as exc:
            if self.on_error:
                self.relay.dispatch.emit(self.on_error, str(exc))
            return
        if self.on_result:
            self.relay.dispatch.emit(self.on_result, result)


def submit(fn: Callable, on_result: Callable | None = None, on_error: Callable | None = None) -> None:
    """Run ``fn()`` on a worker thread; deliver its result/exception on the GUI thread.
    Must be called from the GUI thread."""
    global _RELAY
    if _RELAY is None:
        _RELAY = _Relay()
    QThreadPool.globalInstance().start(_Task(fn, on_result, on_error, _RELAY))


def start_worker(worker: QObject, owner: list) -> "QThread":
    """Run ``worker.run`` on its own QThread.  The thread and worker are kept
    alive in ``owner`` until the thread finishes, then released."""
    from PySide6.QtCore import QThread
    th = QThread()
    worker.moveToThread(th)
    th.started.connect(worker.run)
    entry = (th, worker)
    owner.append(entry)

    def cleanup() -> None:
        th.quit()

    def release() -> None:
        try:
            owner.remove(entry)
        except ValueError:
            pass
        th.deleteLater()

    th.finished.connect(release)
    for sig in ("done", "stopped", "cancelled", "error", "verify_failed"):
        s = getattr(worker, sig, None)
        if s is not None:
            s.connect(cleanup)
    th.start()
    return th
