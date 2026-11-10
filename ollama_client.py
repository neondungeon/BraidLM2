"""
ollama_client.py: a small, cancellable client for the Ollama HTTP API
=====================================================================
Why not the official ``ollama`` Python package?  BraidLM needs to be able to
*abort a request that is still waiting for its first token* (long prompt
processing, model load).  With a plain generator that is impossible, because
the worker thread is blocked inside ``next()``.  Here every streaming call is
bound to a :class:`StreamHandle` whose ``cancel()`` closes the socket from any
thread; Ollama notices the disconnect and stops generating.

Only documented endpoints are used:
    GET  /api/version  /api/tags  /api/ps
    POST /api/show  /api/chat  /api/generate  /api/pull
    DELETE /api/delete
"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx


class OllamaError(RuntimeError):
    """An error reported by (or while talking to) the Ollama server."""


def normalize_host(host: str | None) -> str:
    host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").strip()
    if "://" not in host:
        host = "http://" + host
    # OLLAMA_HOST=0.0.0.0:11434 is a *bind* address; connect via loopback.
    host = host.replace("//0.0.0.0", "//127.0.0.1")
    # Bare scheme+host without a port -> Ollama's default port
    if re.fullmatch(r"https?://[^/:]+", host):
        host += ":11434"
    return host.rstrip("/")


class StreamHandle:
    """Lets another thread abort an in-flight streaming request."""

    def __init__(self) -> None:
        self.cancelled = False
        self._lock = threading.Lock()
        self._client: httpx.Client | None = None
        self._resp: httpx.Response | None = None

    def _attach(self, client: httpx.Client, resp: httpx.Response | None) -> None:
        with self._lock:
            self._client, self._resp = client, resp

    def cancel(self) -> None:
        self.cancelled = True
        with self._lock:
            targets = [self._resp, self._client]
        for t in targets:
            try:
                if t is not None:
                    t.close()
            except Exception:
                pass


# ── Model metadata ──────────────────────────────────────────────────────────

_IMAGE_CAPS = {"image", "image_generation", "imagegen", "image-generation"}
_IMAGE_NAME = re.compile(r"(^|/)(x/)?(z-image|flux|stable-diffusion|sdxl|sd3)", re.I)


@dataclass
class ModelInfo:
    name: str
    size: int = 0                       # bytes on disk (from /api/tags)
    capabilities: frozenset = frozenset()
    family: str = ""
    param_size: str = ""
    quant: str = ""
    arch: str = ""
    ctx_max: int = 0                    # trained maximum context
    kv_bytes_per_token: int = 0         # f16 KV cache cost, 0 = unknown
    default_num_ctx: int = 0            # num_ctx baked into the model, 0 = none

    @property
    def vision(self) -> bool:
        return "vision" in self.capabilities

    @property
    def tools(self) -> bool:
        return "tools" in self.capabilities

    @property
    def thinking(self) -> bool:
        return "thinking" in self.capabilities

    @property
    def is_image(self) -> bool:
        """True for image-*generation* models served by Ollama.

        Ollama's experimental image generation (v0.32.5 and earlier, macOS) was
        advertised through the ``x/`` namespace; the exact capability string is
        not documented, so several signals are accepted."""
        if self.capabilities & _IMAGE_CAPS:
            return True
        if "completion" in self.capabilities or "embedding" in self.capabilities:
            return False
        return bool(_IMAGE_NAME.search(self.name)) and not self.capabilities - _IMAGE_CAPS

    @property
    def is_embedding(self) -> bool:
        return "embedding" in self.capabilities and "completion" not in self.capabilities

    @property
    def is_chat(self) -> bool:
        return not self.is_image and not self.is_embedding


def _first_int(v: Any) -> int:
    if isinstance(v, (list, tuple)):
        return sum(int(x) for x in v if isinstance(x, (int, float)))
    return int(v) if isinstance(v, (int, float)) else 0


def kv_bytes_per_token(mi: dict, arch: str) -> int:
    """Bytes of f16 KV cache one token costs, from GGUF architecture keys.

    This *over*-estimates for models with sliding-window or shared-KV layers,
    which is the safe direction when deciding whether a context size fits."""
    n_layers = _first_int(mi.get(f"{arch}.block_count"))
    n_head = _first_int(mi.get(f"{arch}.attention.head_count"))
    n_kv = _first_int(mi.get(f"{arch}.attention.head_count_kv")) or n_head
    emb = _first_int(mi.get(f"{arch}.embedding_length"))
    if not (n_layers and n_head and emb):
        return 0
    head_dim = emb // n_head
    key_len = _first_int(mi.get(f"{arch}.attention.key_length")) or head_dim
    val_len = _first_int(mi.get(f"{arch}.attention.value_length")) or key_len
    # if head_count_kv was a per-layer list we already summed it (== layers*n_kv)
    layers_kv = n_kv if isinstance(mi.get(f"{arch}.attention.head_count_kv"), (list, tuple)) \
        else n_layers * n_kv
    return int(layers_kv * (key_len + val_len) * 2)


def parse_show(name: str, data: dict, size: int = 0) -> ModelInfo:
    mi = data.get("model_info") or {}
    details = data.get("details") or {}
    arch = str(mi.get("general.architecture") or details.get("family") or "")
    params = data.get("parameters") or ""
    m = re.search(r"^\s*num_ctx\s+(\d+)", params, re.M)
    return ModelInfo(
        name=name,
        size=size,
        capabilities=frozenset(str(c).lower() for c in (data.get("capabilities") or [])),
        family=str(details.get("family") or arch),
        param_size=str(details.get("parameter_size") or ""),
        quant=str(details.get("quantization_level") or ""),
        arch=arch,
        ctx_max=_first_int(mi.get(f"{arch}.context_length")) if arch else 0,
        kv_bytes_per_token=kv_bytes_per_token(mi, arch) if arch else 0,
        default_num_ctx=int(m.group(1)) if m else 0,
    )


# ── Client ──────────────────────────────────────────────────────────────────

class OllamaClient:
    def __init__(self, host: str | None = None) -> None:
        self.base = normalize_host(host)
        self._timeout = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)

    # -- plain requests ------------------------------------------------------
    def _request(self, method: str, path: str, payload: dict | None = None,
                 timeout: float | None = 30.0) -> Any:
        try:
            with httpx.Client(base_url=self.base, timeout=timeout) as c:
                r = c.request(method, path, json=payload)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.base}: {exc}") from exc
        if r.status_code >= 400:
            raise OllamaError(_error_text(r))
        if not r.content:
            return {}
        try:
            return r.json()
        except ValueError:
            return {}

    def version(self) -> str:
        return str(self._request("GET", "/api/version", timeout=5).get("version", ""))

    def tags(self) -> list[dict]:
        return list(self._request("GET", "/api/tags", timeout=15).get("models") or [])

    def ps(self) -> list[dict]:
        return list(self._request("GET", "/api/ps", timeout=5).get("models") or [])

    def show(self, name: str) -> dict:
        return self._request("POST", "/api/show", {"model": name}, timeout=30)

    def info(self, name: str, size: int = 0) -> ModelInfo:
        return parse_show(name, self.show(name), size)

    def delete(self, name: str) -> None:
        self._request("DELETE", "/api/delete", {"model": name}, timeout=60)

    def unload(self, name: str) -> None:
        """Ask Ollama to free a model's memory right now (documented trick)."""
        try:
            self._request("POST", "/api/generate", {"model": name, "keep_alive": 0}, timeout=15)
        except OllamaError:
            pass

    def running_context(self, name: str) -> int:
        """The context length Ollama *actually* allocated for a loaded model."""
        for m in self.ps():
            if name in (m.get("name"), m.get("model")):
                return int(m.get("context_length") or 0)
        return 0

    # -- streaming -----------------------------------------------------------
    def stream(self, path: str, payload: dict, handle: StreamHandle | None = None) -> Iterator[dict]:
        """Yield decoded NDJSON objects.

        The blocking HTTP work runs in a pump thread; this generator only waits
        on a queue, polling ``handle.cancelled``.  That is what makes Stop work
        even while the server has not yet sent response headers (a state in
        which closing an httpx client from another thread does not unblock the
        reader).  Once cancelled, the pump closes its connection as soon as it
        regains control, which makes Ollama stop generating."""
        handle = handle or StreamHandle()
        if handle.cancelled:
            return
        q: queue.Queue = queue.Queue(maxsize=256)
        _END = object()

        def put(item) -> bool:
            while not handle.cancelled:
                try:
                    q.put(item, timeout=0.2)
                    return True
                except queue.Full:
                    continue
            return False

        def pump() -> None:
            client = httpx.Client(base_url=self.base, timeout=self._timeout)
            handle._attach(client, None)
            try:
                with client.stream("POST", path, json=payload) as resp:
                    handle._attach(client, resp)
                    if handle.cancelled:
                        return
                    if resp.status_code >= 400:
                        resp.read()
                        put(OllamaError(_error_text(resp)))
                        return
                    for line in resp.iter_lines():
                        if handle.cancelled:
                            return
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(obj, dict) and obj.get("error"):
                            put(OllamaError(str(obj["error"])))
                            return
                        if not put(obj):
                            return
            except Exception as exc:
                if not handle.cancelled:
                    put(OllamaError(f"Connection to Ollama failed: {exc}"))
            finally:
                try:
                    client.close()
                except Exception:
                    pass
                try:
                    q.put_nowait(_END)
                except queue.Full:
                    pass

        threading.Thread(target=pump, daemon=True, name="ollama-pump").start()
        while True:
            if handle.cancelled:
                return
            try:
                item = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is _END:
                return
            if isinstance(item, OllamaError):
                raise item
            yield item

    def chat_stream(self, payload: dict, handle: StreamHandle | None = None) -> Iterator[dict]:
        payload = {**payload, "stream": True}
        return self.stream("/api/chat", payload, handle)

    def chat_once(self, payload: dict, handle: StreamHandle | None = None) -> str:
        """Run a chat request to completion and return the text (used for summaries)."""
        out = []
        for obj in self.chat_stream(payload, handle):
            out.append((obj.get("message") or {}).get("content") or "")
        return "".join(out)

    def generate_stream(self, payload: dict, handle: StreamHandle | None = None) -> Iterator[dict]:
        return self.stream("/api/generate", {**payload, "stream": True}, handle)

    def pull_stream(self, name: str, handle: StreamHandle | None = None) -> Iterator[dict]:
        return self.stream("/api/pull", {"model": name, "stream": True}, handle)


def _error_text(resp: httpx.Response) -> str:
    try:
        j = resp.json()
        if isinstance(j, dict) and j.get("error"):
            return str(j["error"])
    except ValueError:
        pass
    return f"HTTP {resp.status_code}: {resp.text[:300]}"


class PullProgress:
    """Aggregates Ollama's per-layer pull events into one overall percentage."""

    def __init__(self) -> None:
        self.layers: dict[str, list[int]] = {}   # digest -> [completed, total]
        self.status = ""

    def feed(self, ev: dict) -> tuple[int, int, str]:
        self.status = str(ev.get("status") or self.status)
        d = ev.get("digest")
        if d and ev.get("total"):
            self.layers[d] = [int(ev.get("completed") or 0), int(ev["total"])]
        done = sum(v[0] for v in self.layers.values())
        total = sum(v[1] for v in self.layers.values())
        return done, total, self.status
