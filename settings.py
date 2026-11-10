"""
settings.py: BraidLM persistent settings and on-disk locations
==============================================================
Everything BraidLM stores lives under one folder (default ``~/.braidlm``,
override with the ``BRAIDLM_HOME`` environment variable):

    config.json        last-used model, context preference, ...
    images/            generated images
    image_models/      diffusion models downloaded from Hugging Face
    conversations/     transcripts written when a conversation is ended
    backups/           copies of files made *before* BraidLM edits them
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def home_dir() -> Path:
    p = Path(os.environ.get("BRAIDLM_HOME") or (Path.home() / ".braidlm"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def subdir(name: str) -> Path:
    p = home_dir() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cfg_path() -> Path:
    return home_dir() / "config.json"


def load() -> dict[str, Any]:
    with _LOCK:
        try:
            return json.loads(_cfg_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}


def get(key: str, default: Any = None) -> Any:
    return load().get(key, default)


def set_value(key: str, value: Any) -> None:
    data = load()
    data[key] = value
    with _LOCK:
        try:
            _cfg_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass  # settings are a convenience; never crash because of them
