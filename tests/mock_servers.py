"""Mock Ollama and Hugging Face servers (stdlib only) used by the tests.

MockOllama follows the documented Ollama API (docs.ollama.com) and can *simulate the
silent-truncation behaviour* reported in ollama/ollama#14259: when the prompt is longer
than the window it returns HTTP 200 with prompt_eval_count == ctx//2 + 2."""
from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 1x1 transparent PNG
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


def _model(name, size, caps, family="llama", ctx_max=32768, kv=True, extra=None):
    mi = {"general.architecture": family, f"{family}.context_length": ctx_max}
    if kv:
        mi.update({f"{family}.block_count": 28, f"{family}.attention.head_count": 16,
                   f"{family}.attention.head_count_kv": 2, f"{family}.embedding_length": 2048})
    return {"name": name, "model": name, "size": size, "capabilities": caps,
            "details": {"family": family, "parameter_size": "3B", "quantization_level": "Q4_K_M"},
            "model_info": mi, "parameters": "", **(extra or {})}


class MockOllama:
    def __init__(self):
        self.models = {
            "qwen2.5-coder:3b": _model("qwen2.5-coder:3b", 1_900_000_000, ["completion", "tools"], "qwen2"),
            "qwen2.5:3b": _model("qwen2.5:3b", 1_900_000_000, ["completion", "tools"], "qwen2"),
        }
        self.loaded: dict[str, int] = {}          # name -> context actually allocated
        self.chat_calls: list[dict] = []
        self.generate_calls: list[dict] = []
        self.pulled: list[str] = []
        self.first_token_delay = 0.0
        self.token_delay = 0.0
        self.words = 12
        self.simulate_truncation = True
        self.ignore_requested_ctx: int | None = None   # emulate "ollama ignores requested num_ctx"
        self.fail_chat_for: set[str] = set()
        self.image_works = True
        self.reply_override: str | None = None
        self.server: ThreadingHTTPServer | None = None

    # -- helpers -------------------------------------------------------------
    def add(self, name, size, caps, **kw):
        self.models[name] = _model(name, size, caps, **kw)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def start(self):
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, obj, code=200):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(n) or b"{}")

            def _stream_start(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

            def _line(self, obj):
                data = (json.dumps(obj) + "\n").encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            def _end(self):
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

            def do_GET(self):
                if self.path == "/api/version":
                    return self._json({"version": "0.35.1"})
                if self.path == "/api/tags":
                    return self._json({"models": [
                        {"name": n, "model": n, "size": m["size"], "details": m["details"]}
                        for n, m in outer.models.items()]})
                if self.path == "/api/ps":
                    return self._json({"models": [
                        {"name": n, "model": n, "context_length": c, "size": 1, "size_vram": 1}
                        for n, c in outer.loaded.items()]})
                self._json({"error": "not found"}, 404)

            def do_DELETE(self):
                b = self._body()
                if b.get("model") in outer.models:
                    del outer.models[b["model"]]
                    return self._json({})
                self._json({"error": f"model '{b.get('model')}' not found"}, 404)

            def do_POST(self):
                b = self._body()
                if self.path == "/api/show":
                    m = outer.models.get(b.get("model"))
                    if not m:
                        return self._json({"error": f"model '{b.get('model')}' not found"}, 404)
                    return self._json({k: v for k, v in m.items() if k not in ("name", "model", "size")})
                if self.path == "/api/pull":
                    return self._pull(b)
                if self.path == "/api/chat":
                    return self._chat(b)
                if self.path == "/api/generate":
                    return self._generate(b)
                self._json({"error": "not found"}, 404)

            def _pull(self, b):
                name = b["model"]
                if "doesnotexist" in name:
                    return self._json({"error": "pull model manifest: file does not exist"}, 500)
                self._stream_start()
                self._line({"status": "pulling manifest"})
                for d, total in (("sha256:aaa", 1000), ("sha256:bbb", 3000)):
                    for i in range(1, 5):
                        self._line({"status": f"pulling {d[:12]}", "digest": d, "total": total,
                                    "completed": total * i // 4})
                        time.sleep(0.01)
                self._line({"status": "verifying sha256 digest"})
                self._line({"status": "success"})
                outer.pulled.append(name)
                outer.add(name, 4000, ["completion"], family="llama")
                self._end()

            def _tokens(self, msgs):
                return sum(len(m.get("content") or "") for m in msgs) // 4 + 8 * len(msgs) \
                    + 1200 * sum(len(m.get("images") or []) for m in msgs)

            def _chat(self, b):
                outer.chat_calls.append(b)
                name = b["model"]
                m = outer.models.get(name)
                if not m:
                    return self._json({"error": f"model '{name}' not found"}, 404)
                if name in outer.fail_chat_for:
                    return self._json({"error": "unknown model architecture: 'bogus'"}, 500)
                req_ctx = int((b.get("options") or {}).get("num_ctx") or 4096)
                ctx = outer.ignore_requested_ctx or req_ctx
                outer.loaded[name] = ctx
                prompt = self._tokens(b["messages"])
                evaluated = prompt
                if outer.simulate_truncation and prompt > ctx:
                    evaluated = ctx // 2 + 2
                last = b["messages"][-1].get("content") or ""
                sysmsg = (b["messages"][0].get("content") or "") if b["messages"][0]["role"] == "system" else ""
                if "compress chat transcripts" in sysmsg:
                    reply = "- Summary: user and assistant discussed earlier topics."
                elif outer.reply_override is not None:
                    reply = outer.reply_override
                else:
                    reply = "Echo: " + " ".join(last.split()[:6])
                words = reply.split(" ") if outer.words else [reply]
                self._stream_start()
                if outer.first_token_delay:
                    time.sleep(outer.first_token_delay)
                if "think" in name and "compress" not in sysmsg:
                    self._line({"model": name, "message": {"role": "assistant", "content": "",
                                                          "thinking": "Let me think. "}, "done": False})
                try:
                    for i, w in enumerate(words):
                        self._line({"model": name, "message": {"role": "assistant",
                                    "content": w + ("" if i == len(words) - 1 else " ")}, "done": False})
                        if outer.token_delay:
                            time.sleep(outer.token_delay)
                    self._line({"model": name, "message": {"role": "assistant", "content": ""}, "done": True,
                                "done_reason": "stop", "prompt_eval_count": evaluated,
                                "eval_count": len(words), "eval_duration": 500_000_000,
                                "total_duration": 900_000_000})
                    self._end()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            def _generate(self, b):
                outer.generate_calls.append(b)
                name = b["model"]
                if b.get("keep_alive") == 0:
                    outer.loaded.pop(name, None)
                    return self._json({"model": name, "done": True})
                m = outer.models.get(name)
                if not m:
                    return self._json({"error": f"model '{name}' not found"}, 404)
                is_img = "image" in m["capabilities"]
                self._stream_start()
                try:
                    if is_img and not outer.image_works:
                        self._line({"error": "image generation is not supported"})
                    elif is_img:
                        for i in range(1, 4):
                            self._line({"model": name, "created_at": "x", "response": "", "done": False,
                                        "completed": i, "total": 3})
                            time.sleep(0.01)
                        self._line({"model": name, "done": True, "image": base64.b64encode(TINY_PNG).decode()})
                    else:
                        self._line({"model": name, "response": "I only produce text.", "done": False})
                        self._line({"model": name, "response": "", "done": True})
                    self._end()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()


class MockHF:
    """Hub API: /api/models search, /api/models/<repo>/tree/main (paginated), /<repo>/resolve/main/<file>."""

    def __init__(self):
        self.files: dict[str, dict[str, bytes]] = {}       # repo -> path -> bytes
        self.meta: dict[str, dict] = {}                    # repo -> search metadata
        self.page_size = 3
        self.gated: set[str] = set()
        self.last_search: dict = {}
        self.range_requests = 0
        self.server: ThreadingHTTPServer | None = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def start(self):
        outer = self
        from urllib.parse import parse_qs, urlparse, unquote

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, code, body: bytes, ctype="application/json", extra=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                u = urlparse(self.path)
                q = parse_qs(u.query)
                path = unquote(u.path)
                if path == "/api/models":
                    outer.last_search = q
                    term = (q.get("search") or [""])[0].lower()
                    out = [{"id": r, **outer.meta.get(r, {})} for r in outer.files if term in r.lower()]
                    return self._send(200, json.dumps(out).encode())
                if path.startswith("/api/models/") and path.endswith("/tree/main"):
                    repo = path[len("/api/models/"):-len("/tree/main")]
                    if repo in outer.gated and "Authorization" not in self.headers:
                        return self._send(401, b"{}")
                    if repo not in outer.files:
                        return self._send(404, b"{}")
                    entries = [{"type": "file", "path": p, "size": len(d), "oid": "x"}
                               for p, d in sorted(outer.files[repo].items())]
                    entries.append({"type": "directory", "path": "somedir"})
                    page = int((q.get("cursor") or ["0"])[0])
                    chunk = entries[page * outer.page_size:(page + 1) * outer.page_size]
                    extra = {}
                    if (page + 1) * outer.page_size < len(entries):
                        extra["Link"] = f'<{outer.url}{u.path}?recursive=true&cursor={page + 1}>; rel="next"'
                    return self._send(200, json.dumps(chunk).encode(), extra=extra)
                if "/resolve/main/" in path:
                    repo, fp = path.lstrip("/").split("/resolve/main/", 1)
                    if repo in outer.gated and "Authorization" not in self.headers:
                        return self._send(401, b"{}")
                    data = outer.files.get(repo, {}).get(fp)
                    if data is None:
                        return self._send(404, b"nope")
                    rng = self.headers.get("Range")
                    if rng:
                        outer.range_requests += 1
                        start = int(rng.split("=")[1].split("-")[0])
                        if start >= len(data):
                            return self._send(416, b"")
                        return self._send(206, data[start:], "application/octet-stream")
                    return self._send(200, data, "application/octet-stream")
                self._send(404, b"{}")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
