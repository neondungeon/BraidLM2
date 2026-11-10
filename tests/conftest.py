import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pytest


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIDLM_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("BRAIDLM_VRAM_GB", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    yield


@pytest.fixture
def ollama():
    from mock_servers import MockOllama
    m = MockOllama().start()
    yield m
    m.stop()


@pytest.fixture
def hf():
    from mock_servers import MockHF
    m = MockHF().start()
    yield m
    m.stop()
