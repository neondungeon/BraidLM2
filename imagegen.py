"""
imagegen.py: "draw me a cat" -> an actual picture, on any OS
============================================================
Two backends, picked automatically per model:

  ollama     The model lives in Ollama and answers /api/generate with a base64
             ``image`` field (Ollama's experimental image generation).  NOTE:
             per Ollama's v0.32.6 release notes this feature was *removed* from
             current releases (it only ever ran on macOS), so BraidLM reports
             that clearly instead of failing mysteriously.

  diffusers  Text-to-image models downloaded from Hugging Face by the model
             browser and run locally with PyTorch + diffusers (Windows, Linux,
             macOS).  Heavy dependencies are optional:  requirements-imagegen.txt

Chat models never get hijacked: a request is routed to an image model only when
it is an *imperative* image request ("generate an image of ...", "draw me a
...", "/image ...") and not a programming question ("how do I create an image
in Python?").
"""
from __future__ import annotations

import base64
import inspect
import json
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import settings
from ollama_client import OllamaClient, OllamaError, StreamHandle


class ImageGenError(RuntimeError):
    pass


class ImageGenCancelled(RuntimeError):
    pass


# ── intent detection ────────────────────────────────────────────────────────

_NOUN = (r"(?:images?|pictures?|photos?|photographs?|illustrations?|drawings?|paintings?|artworks?|"
         r"renders?|renderings?|logos?|wallpapers?|portraits?|sketch(?:es)?|posters?|icons?|avatars?|"
         r"banners?|concept art|digital art|cartoons?)")
_VERB = r"(?:generate|create|make|draw|paint|render|produce|design|illustrate|imagine|visuali[sz]e|sketch)"
_LEAD = (r"^\s*(?:(?:please|pls|hey|ok|okay|now|next|and)[,\s]+)*"
         r"(?:(?:can|could|would|will) you\s+(?:please\s+)?|i (?:want|need|would like|'d like) (?:you to\s+)?|"
         r"let'?s\s+|go ahead and\s+)?")
_IMPERATIVE = re.compile(_LEAD + _VERB + r"\b(?:\s+me)?[^.?!\n]{0,45}?\b" + _NOUN + r"\b", re.I)
_DRAW_BARE = re.compile(_LEAD + r"(?:draw|paint|sketch|illustrate)\b(?:\s+me)?\s+(?:an?|the|some|my)\s+\S+", re.I)
_SLASH = re.compile(r"^\s*/(?:image|img|imagine|draw)\s+(.+)", re.I | re.S)
_DEV = re.compile(
    r"\b(python|javascript|typescript|java|c\+\+|c#|rust|golang|code|function|script|program|api|library|"
    r"css|html|svg|canvas|pillow|pil|opencv|numpy|matplotlib|docker|dockerfile|react|vue|command|cli|"
    r"imagemagick|ffmpeg|resize|crop|convert|thumbnail|base64|tag|upload|pipeline|algorithm|class|"
    r"library|package|npm|pip|sdk|bash|powershell|sql|jupyter|notebook|dataset|tensor|pytorch)\b", re.I)
_QUESTION = re.compile(r"^\s*(?:how|what|why|where|when|which|is|are|does|do|did)\b", re.I)


def looks_like_image_request(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if _SLASH.match(t):
        return True
    if len(t) > 600 or _QUESTION.match(t) or _DEV.search(t):
        return False
    return bool(_IMPERATIVE.match(t) or _DRAW_BARE.match(t))


_STRIP = re.compile(_LEAD + _VERB + r"\b(?:\s+me)?\s+(?:(?:an?|the|some)\s+)?"
                    r"(?:(?:detailed|beautiful|high[- ]quality|realistic|new|quick)\s+)*"
                    r"(?:images?|pictures?)\s+(?:of|showing|depicting|featuring|with|that shows)\s+", re.I)


def clean_prompt(text: str) -> str:
    """Drop only the generic command words ("generate an image of ...") and keep the
    subject.  Style-bearing nouns (logo, portrait, photo, poster...) are left in."""
    t = text.strip()
    m = _SLASH.match(t)
    if m:
        return m.group(1).strip()
    stripped = _STRIP.sub("", t, count=1).strip()
    return stripped if len(stripped) >= 3 else t


# ── model registry ──────────────────────────────────────────────────────────

@dataclass
class ImageModel:
    kind: str                   # "ollama" | "diffusers"
    name: str                   # Ollama model name, or Hugging Face repo id
    path: str = ""
    variant: str | None = None
    size: int = 0

    @property
    def key(self) -> str:
        return f"diffusers:{self.name}" if self.kind == "diffusers" else self.name

    @property
    def label(self) -> str:
        return f"🎨 {self.name}"


def diffusers_dir(repo_id: str) -> Path:
    return settings.subdir("image_models") / repo_id.replace("/", "__")


def list_local_diffusers() -> list[ImageModel]:
    out = []
    root = settings.subdir("image_models")
    for d in sorted(root.iterdir()) if root.exists() else []:
        meta = d / "braidlm_meta.json"
        if d.is_dir() and meta.is_file():
            try:
                j = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append(ImageModel("diffusers", j.get("repo_id") or d.name.replace("__", "/"),
                                  str(d), j.get("variant"), int(j.get("size") or 0)))
    return out


def remove_diffusers(model: ImageModel) -> None:
    release_engine()
    shutil.rmtree(model.path, ignore_errors=True)


def images_dir() -> Path:
    return settings.subdir("images")


def save_png_bytes(data: bytes, prompt: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:40] or "image"
    path = images_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}.png"
    i = 1
    while path.exists():
        path = path.with_name(f"{path.stem}-{i}.png")
        i += 1
    path.write_bytes(data)
    return path


# ── backend 1: Ollama-native ────────────────────────────────────────────────

REMOVED_HINT = ("Ollama's experimental image generation was removed in v0.32.6 (Aug 2026) and was only "
                "available on macOS before that. Install a diffusers image model from the ＋ browser "
                "(Image generation tab) to generate images on this machine.")


def generate_ollama(client: OllamaClient, model: str, prompt: str, handle: StreamHandle,
                    progress: Callable[[int, int], None] | None = None) -> bytes:
    img_b64, text = None, []
    try:
        for ev in client.generate_stream({"model": model, "prompt": prompt}, handle):
            if progress and ev.get("total"):
                progress(int(ev.get("completed") or 0), int(ev["total"]))
            if isinstance(ev.get("image"), str) and ev["image"]:
                img_b64 = ev["image"]
            elif isinstance(ev.get("images"), list) and ev["images"]:
                img_b64 = ev["images"][-1]
            if ev.get("response"):
                text.append(ev["response"])
    except OllamaError as exc:
        raise ImageGenError(f"{exc}\n\n{REMOVED_HINT}") from exc
    if handle.cancelled:
        raise ImageGenCancelled()
    if not img_b64:
        if "".join(text).strip():
            raise ImageGenError(f"'{model}' answered with text instead of an image, so it is not an image model.")
        raise ImageGenError(f"Ollama returned no image for '{model}'.\n\n{REMOVED_HINT}")
    if img_b64.startswith("data:"):
        img_b64 = img_b64.split(",", 1)[-1]
    return base64.b64decode(img_b64)


# ── backend 2: diffusers ────────────────────────────────────────────────────

INSTALL_HINT = ("Image generation from Hugging Face models needs extra packages:\n\n"
                "    pip install -r requirements-imagegen.txt\n\n"
                "On Windows/Linux with an NVIDIA GPU, first install the CUDA build of PyTorch from "
                "https://pytorch.org/get-started/locally/ (the default PyPI wheel on Windows is CPU-only).")


def diffusers_status() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        return False, f"Missing Python package: {exc.name}\n\n{INSTALL_HINT}"
    return True, ""


def default_params(repo_id: str) -> dict:
    n = repo_id.lower()
    big = any(k in n for k in ("xl", "flux", "z-image", "sd3", "stable-diffusion-3", "qwen-image", "hidream",
                               "kolors", "pixart", "sana", "playground"))
    small_turbo = "sd-turbo" in n
    if "z-image" in n and "turbo" in n:
        steps, guidance = 9, 0.0
    elif any(k in n for k in ("turbo", "lightning", "schnell", "klein", "lcm", "hyper")):
        steps, guidance = 4, 0.0
    elif "flux" in n:
        steps, guidance = 28, 3.5
    else:
        steps, guidance = 30, 6.5
    size = 1024 if big and not small_turbo else 512
    return {"steps": int(settings.get("image_steps", steps)),
            "guidance": float(settings.get("image_guidance", guidance)),
            "size": int(settings.get("image_size", size))}


class DiffusersEngine:
    def __init__(self) -> None:
        self.pipe = None
        self.key: str | None = None
        self.low_vram = False
        self.device = "cpu"
        self.lock = threading.Lock()

    @staticmethod
    def _pick_device(torch):
        if torch.cuda.is_available():
            dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return "cuda", dt
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    def release(self) -> None:
        with self.lock:
            self.pipe, self.key = None, None
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load(self, model: ImageModel, low_vram: bool = False):
        import torch
        from diffusers import DiffusionPipeline
        device, dtype = self._pick_device(torch)
        kwargs = {"torch_dtype": dtype}
        if model.variant:
            kwargs["variant"] = model.variant
        pipe = DiffusionPipeline.from_pretrained(model.path, **kwargs)
        if device == "cuda" and low_vram:
            pipe.enable_model_cpu_offload()
        elif device == "cuda":
            try:
                free, _total = torch.cuda.mem_get_info()
                if model.size and model.size * 1.25 > free:
                    pipe.enable_model_cpu_offload()
                    low_vram = True
                else:
                    pipe.to("cuda")
            except Exception:
                pipe.to("cuda")
        else:
            pipe.to(device)
        self.device, self.low_vram = device, low_vram
        return pipe

    def generate(self, model: ImageModel, prompt: str, progress: Callable[[int, int], None],
                 cancel: threading.Event, seed: int | None = None):
        ok, why = diffusers_status()
        if not ok:
            raise ImageGenError(why)
        import torch
        params = default_params(model.name)
        with self.lock:
            if self.pipe is None or self.key != model.key:
                self.pipe = None
        if self.pipe is None:
            try:
                pipe = self._load(model)
            except Exception as exc:
                raise ImageGenError(f"Could not load '{model.name}': {exc}") from exc
            with self.lock:
                self.pipe, self.key = pipe, model.key
        pipe = self.pipe
        steps = params["steps"]
        gen = torch.Generator("cpu").manual_seed(int(seed if seed is not None else time.time_ns() % 2**31))

        def on_step(p, i, t, cb_kwargs):
            if cancel.is_set():
                p._interrupt = True
            progress(i + 1, steps)
            return cb_kwargs

        wanted = dict(prompt=prompt, num_inference_steps=steps, guidance_scale=params["guidance"],
                      width=params["size"], height=params["size"], generator=gen,
                      callback_on_step_end=on_step)
        sig = inspect.signature(pipe.__call__).parameters
        takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.values())
        kwargs = {k: v for k, v in wanted.items() if k in sig or (takes_any and k == "prompt")}
        try:
            result = pipe(**kwargs)
        except torch.cuda.OutOfMemoryError:           # type: ignore[attr-defined]
            if self.low_vram:
                raise ImageGenError("Out of GPU memory even with CPU offloading. Close other GPU programs "
                                    "or choose a smaller image model.")
            self.release()
            try:
                pipe = self._load(model, low_vram=True)
            except Exception as exc:
                raise ImageGenError(f"Could not reload '{model.name}' in low-VRAM mode: {exc}") from exc
            with self.lock:
                self.pipe, self.key = pipe, model.key
            result = pipe(**kwargs)
        except Exception as exc:
            if cancel.is_set():
                raise ImageGenCancelled() from exc
            raise ImageGenError(f"Image generation failed: {exc}") from exc
        if cancel.is_set():
            raise ImageGenCancelled()
        return result.images[0]


_ENGINE = DiffusersEngine()


def engine() -> DiffusersEngine:
    return _ENGINE


def release_engine() -> None:
    _ENGINE.release()


def engine_loaded() -> bool:
    return _ENGINE.pipe is not None


def save_pil(image, prompt: str) -> Path:
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return save_png_bytes(buf.getvalue(), prompt)
