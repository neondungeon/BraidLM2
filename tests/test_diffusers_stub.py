"""Diffusers backend tested with FAKE torch/diffusers/transformers modules.

This verifies BraidLM's own logic (device choice, kwarg filtering, progress,
cancel, registry, saving). It does NOT prove a real model renders correctly."""
import sys
import threading
import types

import pytest

import imagegen

PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
       b"\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa7\x9a\xa0\xa0\x00\x00\x00\x00IEND\xaeB`\x82")


class FakeImage:
    def save(self, buf, format="PNG"):
        buf.write(PNG)


@pytest.fixture
def fakes(monkeypatch):
    calls = {}
    torch = types.ModuleType("torch")
    torch.float16 = "f16"; torch.float32 = "f32"; torch.bfloat16 = "bf16"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None,
                                       OutOfMemoryError=type("OOM", (RuntimeError,), {}))
    torch.backends = types.SimpleNamespace()
    class Gen:
        def __init__(self, dev): pass
        def manual_seed(self, s): calls["seed"] = s; return self
    torch.Generator = Gen

    class Pipe:
        def __init__(self): self._interrupt = False
        @classmethod
        def from_pretrained(cls, path, **kw):
            calls["load"] = (path, kw); calls["loads"] = calls.get("loads", 0) + 1
            return cls()
        def to(self, d): calls["device"] = d; return self
        def __call__(self, prompt, num_inference_steps=10, guidance_scale=1.0, width=64, height=64,
                     generator=None, callback_on_step_end=None):       # note: no 'negative_prompt'
            calls["call"] = dict(prompt=prompt, steps=num_inference_steps, size=(width, height))
            for i in range(num_inference_steps):
                callback_on_step_end(self, i, 0, {})
                if self._interrupt:
                    break
            calls["ran"] = i + 1
            return types.SimpleNamespace(images=[FakeImage()])

    diff = types.ModuleType("diffusers"); diff.DiffusionPipeline = Pipe
    for n, m in (("torch", torch), ("diffusers", diff), ("transformers", types.ModuleType("transformers"))):
        monkeypatch.setitem(sys.modules, n, m)
    imagegen.release_engine()
    yield calls
    imagegen.release_engine()


def model(tmp_path):
    return imagegen.ImageModel("diffusers", "acme/sd-turbo", str(tmp_path), "fp16", 1000)


def test_status_ok_and_generate_with_progress(fakes, tmp_path):
    assert imagegen.diffusers_status() == (True, "")
    seen = []
    img = imagegen.engine().generate(model(tmp_path), "a cat", lambda d, t: seen.append((d, t)), threading.Event())
    assert fakes["load"][1]["variant"] == "fp16" and fakes["device"] == "cpu"
    assert fakes["call"]["size"] == (512, 512) and fakes["call"]["steps"] == 4
    assert seen[-1] == (4, 4)
    path = imagegen.save_pil(img, "a cat")
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_pipeline_is_cached_between_calls(fakes, tmp_path):
    m = model(tmp_path)
    for _ in range(2):
        imagegen.engine().generate(m, "x", lambda d, t: None, threading.Event())
    assert fakes["loads"] == 1


def test_cancel_interrupts_and_raises(fakes, tmp_path):
    ev = threading.Event()
    def prog(d, t):
        if d == 2:
            ev.set()
    with pytest.raises(imagegen.ImageGenCancelled):
        imagegen.engine().generate(model(tmp_path), "x", prog, ev)
    assert fakes["ran"] < 4


def test_missing_packages_message(monkeypatch):
    for n in ("torch", "diffusers", "transformers"):
        monkeypatch.setitem(sys.modules, n, None)
    ok, why = imagegen.diffusers_status()
    assert not ok and "requirements-imagegen.txt" in why


def test_worker_end_to_end(fakes, tmp_path, ollama, app=None):
    from PySide6.QtWidgets import QApplication
    import time
    from ollama_client import OllamaClient
    from worker import ImageWorker
    qa = QApplication.instance() or QApplication([])
    w = ImageWorker(OllamaClient(ollama.url), model(tmp_path), "a lighthouse")
    out = []
    w.done.connect(out.append, Qt_direct()); w.error.connect(lambda m: out.append(("err", m)), Qt_direct())
    w.run()
    assert out and out[0]["path"].endswith(".png"), out


def Qt_direct():
    from PySide6.QtCore import Qt
    return Qt.ConnectionType.DirectConnection
