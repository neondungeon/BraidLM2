"""Offscreen GUI tests for MainWindow against the mock Ollama server."""
import time
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

import imagegen
import settings
import window as W
from ollama_client import OllamaClient


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait(app, cond, timeout=8.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    app.processEvents()
    return cond()


@pytest.fixture(autouse=True)
def _hardware(monkeypatch):
    monkeypatch.setenv("BRAIDLM_VRAM_GB", "12")
    monkeypatch.setenv("BRAIDLM_RAM_GB", "32")


@pytest.fixture
def win(app, ollama):
    w = W.MainWindow(OllamaClient(ollama.url), interactive=False)
    w.show()
    yield w
    w.close()
    app.processEvents()


def send(app, w, text):
    w._input.setPlainText(text)
    w._send()
    assert wait(app, lambda: not w._busy)


def test_startup_selects_builtin_and_shows_buttons(win):
    assert win._selected == "qwen2.5-coder:3b"
    assert set(win._mode_buttons) == {"coder", "general"}
    assert win._btn_plus.text() == "＋"
    # ＋ sits to the left of Coder/General
    assert win._btn_plus.geometry().x() < win._mode_buttons["coder"].geometry().x() < win._mode_buttons["general"].geometry().x()


def test_send_streams_and_records_history(app, win):
    send(app, win, "hello there friend")
    roles = [m["role"] for m in win._history]
    assert roles == ["user", "assistant"]
    assert win._history[1]["content"].startswith("Echo:")
    assert win._ctx_used > 0


def test_stop_during_generation_keeps_partial(app, win, ollama):
    ollama.token_delay = 0.15
    ollama.words = 60
    win._input.setPlainText("count slowly")
    win._send()
    assert wait(app, lambda: len(win._stream_buf) > 3)
    win.stop_or_end()
    assert wait(app, lambda: not win._busy)
    assert win._history[-1]["role"] == "assistant" and win._history[-1].get("stopped")
    assert win._history[0]["role"] == "user"       # conversation not ended by a mid-answer stop


def test_stop_before_first_token_restores_input(app, win, ollama):
    ollama.first_token_delay = 3.0
    win._input.setPlainText("slow start")
    win._send()
    assert wait(app, lambda: win._busy)
    time.sleep(0.2)
    t0 = time.time()
    win.stop_or_end()
    assert wait(app, lambda: not win._busy, 3)
    assert time.time() - t0 < 2.5
    assert win._history == [] and win._input.toPlainText() == "slow start"


def test_stop_when_idle_ends_conversation_and_saves_transcript(app, win):
    send(app, win, "remember the number 42")
    win.stop_or_end()
    assert win._history == [] and win._summary == ""
    files = list(settings.subdir("conversations").glob("*.md"))
    assert len(files) == 1
    txt = files[0].read_text(encoding="utf-8")
    assert "remember the number 42" in txt and "## BraidLM" in txt


def test_attach_text_file_and_apply_with_undo(app, win, ollama, tmp_path, monkeypatch):
    f = tmp_path / "hello.py"
    f.write_bytes(b"print('hi')\r\n")
    win.add_attachments([str(f)])
    assert len(win._pending) == 1 and win._chip_holder.isVisibleTo(win)
    ollama.reply_override = "Here you go:\n```python\nprint('hello world')\n```\nChanged the greeting."
    win._input.setPlainText("make it friendlier")
    win._send()
    assert wait(app, lambda: not win._busy)
    sent = ollama.chat_calls[-1]["messages"]
    assert "<attached_file" in sent[-1]["content"] and "print('hi')" in sent[-1]["content"]
    assert "COMPLETE updated" in sent[0]["content"]
    assert win._pending == []
    html = win._chat.toHtml()
    assert "Apply to file" in win._chat.document().toPlainText()
    monkeypatch.setattr(win, "_confirm_diff", lambda *a: True)
    win.apply_block(1, 0)
    assert f.read_bytes() == b"print('hello world')\r\n"        # CRLF preserved
    assert list(settings.subdir("backups").glob("*hello.py"))
    assert "Undo apply" in win._chat.document().toPlainText()
    win._undo(1, 0)
    assert f.read_bytes() == b"print('hi')\r\n"


def test_image_attachment_rejected_for_non_vision_model(app, win, tmp_path):
    from PySide6.QtGui import QImage
    p = tmp_path / "x.png"
    QImage(10, 10, QImage.Format.Format_RGB32).save(str(p))
    win.add_attachments([str(p)])
    assert win._pending == []
    assert any("cannot look at images" in m["content"] for m in win._history if m["role"] == "notice")


def test_image_attachment_sent_to_vision_model(app, ollama, tmp_path):
    ollama.add("gemma3:4b", 3_000_000_000, ["completion", "vision"], family="gemma3")
    w = W.MainWindow(OllamaClient(ollama.url), interactive=False)
    w.select_model("gemma3:4b")
    from PySide6.QtGui import QImage
    p = tmp_path / "x.png"
    QImage(64, 64, QImage.Format.Format_RGB32).save(str(p))
    w.add_attachments([str(p)])
    assert len(w._pending) == 1
    send(app, w, "what is this")
    assert ollama.chat_calls[-1]["messages"][-1]["images"]
    w.close()


def test_attach_hidden_for_image_model(app, ollama):
    ollama.add("x/z-image-turbo", 1_000_000_000, ["image"])
    w = W.MainWindow(OllamaClient(ollama.url), interactive=False)
    w.show()
    w.select_model(w._image_models[0])
    assert not w._btn_attach.isVisible()
    w.select_model("qwen2.5:3b")
    assert w._btn_attach.isVisible()
    w.close()


def test_image_model_selected_generates_image(app, ollama):
    ollama.add("x/z-image-turbo", 1_000_000_000, ["image"])
    w = W.MainWindow(OllamaClient(ollama.url), interactive=False)
    w.select_model(w._image_models[0])
    send(app, w, "a red fox in the snow")
    last = w._history[-1]
    assert last["kind"] == "image" and Path(last["image_path"]).read_bytes()[:4] == b"\x89PNG"
    assert ollama.generate_calls[-1]["prompt"] == "a red fox in the snow"
    assert ollama.chat_calls == []
    w.close()


def test_image_request_routed_from_text_model_but_questions_are_not(app, ollama):
    ollama.add("x/z-image-turbo", 1_000_000_000, ["image"])
    w = W.MainWindow(OllamaClient(ollama.url), interactive=False)
    send(app, w, "Please generate an image of a lighthouse at dusk")
    assert w._history[-1]["kind"] == "image" and ollama.chat_calls == []
    assert w._selected == "qwen2.5-coder:3b"
    send(app, w, "How do I resize an image in Python?")
    assert len(ollama.chat_calls) == 1
    w.close()


def test_image_request_without_image_model_gives_tip_and_answers(app, win, ollama):
    send(app, win, "draw me a picture of a cat")
    assert any("install an image model" in m["content"] for m in win._history if m["role"] == "notice")
    assert len(ollama.chat_calls) == 1


def test_image_failure_when_ollama_removed_support(app, ollama):
    ollama.add("x/z-image-turbo", 1_000_000_000, ["image"])
    ollama.image_works = False
    w = W.MainWindow(OllamaClient(ollama.url), interactive=False)
    w.select_model(w._image_models[0])
    send(app, w, "a cat")
    notice = [m["content"] for m in w._history if m["role"] == "notice"][-1]
    assert "v0.32.6" in notice
    assert w._input.toPlainText() == "a cat"
    w.close()


def test_error_restores_input(app, win, ollama):
    ollama.fail_chat_for = {"qwen2.5-coder:3b"}
    send(app, win, "will fail")
    assert win._input.toPlainText() == "will fail"
    assert [m["role"] for m in win._history] == ["notice"]


def test_context_guard_raises_ctx_and_warns_on_truncation(app, win, ollama):
    big = "word " * 3000                      # ~15k chars ≈ 4.7k tokens > 4k window
    send(app, win, big)
    assert ollama.chat_calls[-1]["options"]["num_ctx"] > 4096
    assert not any("silently cut" in m["content"] for m in win._history if m["role"] == "notice")
    # Ollama ignoring our ctx request -> silent truncation must be reported
    ollama.ignore_requested_ctx = 4096
    send(app, win, "word " * 4000)
    texts = [m["content"] for m in win._history if m["role"] == "notice"]
    assert any("silently cut" in t for t in texts)
    assert any("allocated only 4,096" in t for t in texts)
    assert win._force_compact is True


def test_compaction_folds_old_turns_into_summary(app, win, ollama):
    ollama.words = 3
    win._set_ctx_choice(2048)                # manual, no auto-raise
    for i in range(8):
        send(app, win, f"message number {i} " + "filler " * 120)
    assert win._summary and win._summary_upto > 0
    sent = ollama.chat_calls[-1]["messages"]
    assert "Summary of the earlier part" in sent[0]["content"]
    assert all(set(m) <= {"role", "content", "images"} for m in sent)


def test_model_switch_disabled_while_busy_and_persisted(app, win, ollama):
    win.select_model("qwen2.5:3b")
    assert settings.get("last_model") == "qwen2.5:3b"
    ollama.token_delay = 0.1
    ollama.words = 40
    win._input.setPlainText("go")
    win._send()
    assert wait(app, lambda: win._busy)
    win.select_model("qwen2.5-coder:3b")
    assert win._selected == "qwen2.5:3b"
    win.stop_or_end()
    assert wait(app, lambda: not win._busy)


def test_removed_builtin_is_dimmed_and_chat_still_works(app, ollama):
    del ollama.models["qwen2.5-coder:3b"]
    w = W.MainWindow(OllamaClient(ollama.url), interactive=False)
    assert w._selected == "qwen2.5:3b"
    assert "not installed" in w._mode_buttons["coder"].toolTip()
    w.close()


def test_no_models_and_offline_states(app, ollama):
    ollama.models.clear()
    w = W.MainWindow(OllamaClient(ollama.url), interactive=False)
    assert w._selected is None and not w._btn_send.isEnabled()
    w.close()
    w = W.MainWindow(OllamaClient("http://127.0.0.1:1"), interactive=False)
    assert w._offline and "Cannot reach Ollama" in w._offline
    w.close()


def test_browser_installs_and_removes_gguf(app, ollama, hf):
    from model_browser import ModelBrowser
    import hardware
    hw = hardware.Hardware(ram_total=32 * 1024 ** 3, ram_free=20 * 1024 ** 3)
    b = ModelBrowser(OllamaClient(ollama.url), hw)
    changed = []
    b.modelsChanged.connect(lambda: changed.append(1))
    import hf_hub
    hf.files["acme/tiny-gguf"] = {"tiny-Q4_K_M.gguf": b"x" * 4000, "tiny-Q8_0.gguf": b"x" * 8000}
    b._hub = lambda: hf_hub.HubClient(None, hf.url)
    b.query.setText("hf.co/acme/tiny-gguf:Q4_K_M")
    b.search()
    assert wait(app, lambda: b._gguf is not None)
    assert b.table.rowCount() >= 1
    assert b.btn_install.isEnabled()
    b.install()
    assert wait(app, lambda: b._active is None and changed)
    assert any(n.startswith("hf.co/acme/tiny-gguf") for n in ollama.models)
    assert wait(app, lambda: b.inst_list.count() >= 3) or b.refresh_installed() is None
    wait(app, lambda: any("tiny-gguf" in b.inst_list.item(i).text() for i in range(b.inst_list.count())))
    for i in range(b.inst_list.count()):
        if "tiny-gguf" in b.inst_list.item(i).text():
            b.inst_list.setCurrentRow(i)
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    b.remove_selected()
    assert wait(app, lambda: not any(n.startswith("hf.co/acme") for n in ollama.models))
    b.close()
