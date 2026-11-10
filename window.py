"""
window.py: BraidLM main application window
==========================================
Architecture is based on nocolour/Ollama-Desktop (MIT): https://github.com/nocolour/Ollama-Desktop

What this window adds on top of a plain Ollama chat:
  * ＋ model browser (Hugging Face search, fit check, install / remove)
  * Context Guard (live context meter, auto-raise, compaction, truncation warning)
  * Attach file (text/code/PDF/DOCX, images for vision models) + "Apply to file…" with diff + undo
  * Image generation (image model selected, or an image request while one is installed)
  * Stop: cancels a running reply; when idle it ENDS the conversation (saves a transcript)
"""
from __future__ import annotations

import html as html_module
import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QKeyEvent, QTextCharFormat, QColor, QTextCursor
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QFileDialog, QHBoxLayout, QInputDialog,
                               QLabel, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton,
                               QSizePolicy, QStatusBar, QTextBrowser, QTextEdit, QToolButton,
                               QVBoxLayout, QWidget)

import attachments as att
import context_guard as cg
import fileops
import hardware as hw_mod
import imagegen
import settings
from model_browser import ModelBrowser
from ollama_client import ModelInfo, OllamaClient, OllamaError
from renderer import (close_open_fence, extract_code_blocks, full_page_html, image_html,
                      markdown_to_html, thinking_html)
from worker import (MODELS, ChatJob, ChatWorker, ImageWorker, PullWorker, start_worker,
                    system_prompt_for)

# ── Stylesheet ──────────────────────────────────────────────────────────────
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0e0e1a;
    color: #dde1e7;
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
}

/* Chat display */
QTextBrowser {
    background-color: #131320;
    border: 1px solid #1e1e38;
    border-radius: 8px;
    padding: 8px;
    color: #dde1e7;
    font-size: 13px;
    line-height: 1.6;
    selection-background-color: #3a3a70;
}

/* Input box */
QTextEdit {
    background-color: #1a1a2e;
    border: 1px solid #2a2a48;
    border-radius: 8px;
    padding: 8px;
    color: #dde1e7;
    font-size: 13px;
    selection-background-color: #3a3a70;
}
QTextEdit:focus { border-color: #5a4fcf; }

/* Buttons */
QPushButton {
    background-color: #242440;
    color: #dde1e7;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 13px;
    min-height: 32px;
    min-width: 80px;
}
QPushButton:hover   { background-color: #2e2e56; }
QPushButton:pressed { background-color: #5a4fcf; }
QPushButton:disabled { background-color: #161628; color: #444466; }

/* Status bar */
QStatusBar {
    background-color: #0b0b16;
    color: #636390;
    font-size: 11px;
    padding: 3px 10px;
    border-top: 1px solid #1e1e36;
}
QStatusBar::item { border: none; }

/* Slim scrollbars */
QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #2e2e56;
    border-radius: 3px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #5a4fcf; }
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical { height: 0; }

QLabel { color: #9090b8; font-size: 12px; }
QToolButton { background:#242440; color:#dde1e7; border:none; border-radius:5px; padding:3px 8px; font-size:11px; }
QToolButton:hover { background:#2e2e56; }
QComboBox { background:#242440; color:#dde1e7; border:none; border-radius:6px; padding:6px 10px; min-height:20px; }
QComboBox QAbstractItemView { background:#1a1a2e; color:#dde1e7; selection-background-color:#5a4fcf; }
QProgressBar { background:#161628; border:none; border-radius:4px; text-align:center; color:#dde1e7; font-size:10px; max-height:14px; min-width:150px; }
QProgressBar::chunk { background:#5a4fcf; border-radius:4px; }
QDialog, QTabWidget::pane { background:#0e0e1a; }
QListWidget, QTableWidget, QLineEdit { background:#1a1a2e; color:#dde1e7; border:1px solid #2a2a48; border-radius:6px; }
QLineEdit { padding:5px 8px; }
"""

_LABEL_USER = '<span style="font-weight:bold;font-size:12px;color:#e05c7a;">You</span>'
_LABEL_BOT = '<span style="font-weight:bold;font-size:12px;color:#7c9ef5;">🧵 BraidLM</span>'
_HR = '<hr style="border:none;border-top:1px solid #1e1e38;margin:8px 0;">'
API_KEYS = ("role", "content", "images")


def api_view(history: list[dict]) -> list[dict]:
    """History as Ollama wants it: UI-only entries and keys removed."""
    out = []
    for m in history:
        if m.get("role") not in ("user", "assistant"):
            continue
        d = {k: m[k] for k in API_KEYS if m.get(k)}
        d.setdefault("content", "")
        out.append(d)
    return out


def save_transcript(history: list[dict], model: str) -> Path | None:
    """Write the conversation to ~/.braidlm/conversations/<timestamp>.md."""
    turns = [m for m in history if m.get("role") in ("user", "assistant")]
    if not turns:
        return None
    lines = [f"# BraidLM conversation – {time.strftime('%Y-%m-%d %H:%M')}", f"Model: {model}", ""]
    for m in turns:
        who = "You" if m["role"] == "user" else "BraidLM"
        body = m.get("display") if m["role"] == "user" and m.get("display") is not None else m.get("content", "")
        lines += [f"## {who}", "", str(body).strip(), ""]
        for a in m.get("attachments") or []:
            lines.append(f"_attached: {a.name}_")
        if m.get("image_path"):
            lines.append(f"_image: {m['image_path']}_")
    d = settings.subdir("conversations")
    p = d / f"{time.strftime('%Y%m%d-%H%M%S')}.md"
    i = 1
    while p.exists():
        p = d / f"{time.strftime('%Y%m%d-%H%M%S')}-{i}.md"
        i += 1
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


class MainWindow(QMainWindow):
    """Top-level BraidLM window."""

    def __init__(self, client: OllamaClient | None = None, interactive: bool = True) -> None:
        super().__init__()
        self.setWindowTitle("BraidLM")
        self.resize(1040, 760)
        self.setMinimumSize(700, 520)
        self.setStyleSheet(DARK_STYLE)
        self.setAcceptDrops(True)

        self.client = client or OllamaClient()
        self.interactive = interactive
        self.hw = hw_mod.detect()
        self._threads: list = []
        self._browser: ModelBrowser | None = None

        # model state
        self._tags: list[dict] = []
        self._infos: dict[str, ModelInfo] = {}
        self._chat_models: list[str] = []
        self._image_models: list[imagegen.ImageModel] = []
        self._selected: str | imagegen.ImageModel | None = None    # chat model name or ImageModel
        self._offline = ""

        # conversation state
        self._history: list[dict] = []
        self._summary = ""
        self._summary_upto = 0
        self._pending: list[att.Attachment] = []
        self._applied: dict[tuple[int, int], tuple[str, str]] = {}
        self._est = cg.TokenEstimator()
        self._ctx_choice = settings.get("ctx_choice", "auto")
        self._ctx_cur = 0
        self._ctx_actual = 0
        self._ctx_used = 0
        self._force_compact = False
        self._busy = False
        self._worker = None
        self._stream_buf = ""
        self._stream_think = ""
        self._last_user: tuple[str, list] = ("", [])

        self._build_ui()
        self.refresh_models()
        QTimer.singleShot(0, self._startup)

    # ── UI construction ─────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(14, 14, 14, 10)
        vbox.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(8)
        title = QLabel("🧵 BraidLM")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color:#8b7cf5;letter-spacing:0.5px;")
        header.addWidget(title)
        header.addStretch()
        lbl = QLabel("Model:")
        lbl.setStyleSheet("font-size:12px;color:#636390;margin-right:4px;")
        header.addWidget(lbl)
        self._btn_plus = QPushButton("＋")
        self._btn_plus.setToolTip("Browse Hugging Face: install and remove models")
        self._btn_plus.setFixedWidth(44)
        self._btn_plus.setStyleSheet("font-size:18px;padding:0;")
        self._btn_plus.clicked.connect(self.open_browser)
        header.addWidget(self._btn_plus)
        self._mode_buttons: dict[str, QPushButton] = {}
        for mode in MODELS:
            b = QPushButton(mode.capitalize())
            b.setFixedWidth(90)
            b.clicked.connect(lambda checked=False, m=mode: self._on_mode_clicked(m))
            self._mode_buttons[mode] = b
            header.addWidget(b)
        self._combo = QComboBox()
        self._combo.setMinimumWidth(200)
        self._combo.activated.connect(self._on_combo)
        header.addWidget(self._combo)
        vbox.addLayout(header)

        self._chat = QTextBrowser()
        self._chat.setReadOnly(True)
        self._chat.setOpenLinks(False)
        self._chat.setOpenExternalLinks(False)
        self._chat.anchorClicked.connect(self._on_anchor)
        self._chat.setFont(QFont("Consolas", 12))
        self._chat.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        vbox.addWidget(self._chat, stretch=1)

        self._chips = QHBoxLayout()
        self._chips.setSpacing(6)
        self._chip_holder = QWidget()
        self._chip_holder.setLayout(self._chips)
        self._chip_holder.setVisible(False)
        vbox.addWidget(self._chip_holder)

        row = QHBoxLayout()
        row.setSpacing(10)
        self._input = QTextEdit()
        self._input.setFixedHeight(84)
        self._input.setFont(QFont("Segoe UI", 12))
        self._input.installEventFilter(self)
        row.addWidget(self._input, stretch=1)

        col = QVBoxLayout()
        col.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(6)
        self._btn_attach = QPushButton("📎")
        self._btn_attach.setToolTip("Attach a file (drag & drop also works)")
        self._btn_attach.setFixedWidth(44)
        self._btn_attach.setMinimumWidth(44)
        self._btn_attach.setStyleSheet("padding:0;")
        self._btn_attach.clicked.connect(self.attach_dialog)
        self._btn_send = QPushButton("⬆  Send")
        self._btn_send.setFixedWidth(100)
        self._btn_send.clicked.connect(self._send)
        top.addWidget(self._btn_attach)
        top.addWidget(self._btn_send)
        col.addLayout(top)
        bot = QHBoxLayout()
        bot.setSpacing(6)
        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setFixedWidth(100)
        self._btn_stop.setToolTip("While BraidLM is answering: stop the answer.\n"
                                  "Otherwise: end this conversation (a transcript is saved).  Esc also stops an answer.")
        self._btn_stop.setStyleSheet("QPushButton{background:#4a2030;} QPushButton:hover{background:#6a2a42;}")
        self._btn_stop.clicked.connect(self.stop_or_end)
        self._btn_clear = QPushButton("✕")
        self._btn_clear.setToolTip("Clear the chat without saving")
        self._btn_clear.setFixedWidth(44)
        self._btn_clear.setStyleSheet("padding:0;")
        self._btn_clear.clicked.connect(self._clear)
        bot.addWidget(self._btn_clear)
        bot.addWidget(self._btn_stop)
        col.addLayout(bot)
        row.addLayout(col)
        vbox.addLayout(row)

        sb = QStatusBar()
        self.setStatusBar(sb)
        self._ctx_btn = QToolButton()
        self._ctx_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self._ctx_btn)
        self._ctx_actions = {}
        for label, val in [("Auto (recommended)", "auto")] + [(f"{c // 1024}k tokens", c) for c in hw_mod.CTX_LADDER if c <= 131072]:
            a = menu.addAction(label)
            a.setCheckable(True)
            a.triggered.connect(lambda checked=False, v=val: self._set_ctx_choice(v))
            self._ctx_actions[val] = a
        self._ctx_btn.setMenu(menu)
        self._ctx_bar = QProgressBar()
        self._ctx_bar.setRange(0, 1000)
        self._ctx_bar.setToolTip("How full the model's context window is")
        sb.addPermanentWidget(self._ctx_bar)
        sb.addPermanentWidget(self._ctx_btn)
        self._update_ctx_widgets()

    # ── startup / models ────────────────────────────────────────────────────
    def _startup(self) -> None:
        while self._offline and self.interactive:
            r = QMessageBox.critical(self, "Ollama is not reachable",
                                     f"{self._offline}\n\nStart Ollama (ollama serve) and retry.",
                                     QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Close)
            if r == QMessageBox.StandardButton.Close:
                self.close()
                return
            self.refresh_models()
        if not self._offline and self._selected is None and self.interactive:
            self._offer_defaults()

    def _offer_defaults(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("No models installed")
        box.setText("BraidLM needs at least one model. Install the two small built-in ones "
                    f"({', '.join(MODELS.values())}, ≈4 GB total) or browse Hugging Face?")
        b1 = box.addButton("Install defaults", QMessageBox.ButtonRole.AcceptRole)
        b2 = box.addButton("Browse Hugging Face", QMessageBox.ButtonRole.ActionRole)
        box.addButton("Not now", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is b1:
            self._install_builtin(MODELS["coder"])
        elif box.clickedButton() is b2:
            self.open_browser()

    def refresh_models(self) -> None:
        try:
            self._tags = self.client.tags()
            self._offline = ""
        except OllamaError as exc:
            self._tags, self._offline = [], str(exc)
        chat, image = [], []
        for t in self._tags:
            name = t.get("name") or t.get("model")
            if not name:
                continue
            info = self._infos.get(name)
            if info is None:
                try:
                    info = self.client.info(name, int(t.get("size") or 0))
                    self._infos[name] = info
                except OllamaError:
                    info = ModelInfo(name, int(t.get("size") or 0), frozenset({"completion"}))
            if info.is_image:
                image.append(imagegen.ImageModel("ollama", name, size=info.size))
            elif info.is_chat:
                chat.append(name)
        image += imagegen.list_local_diffusers()
        self._chat_models, self._image_models = chat, image
        self._infos = {k: v for k, v in self._infos.items() if k in {t.get("name") for t in self._tags}}

        keep = self._selected
        if isinstance(keep, str) and keep not in chat:
            keep = None
        if isinstance(keep, imagegen.ImageModel) and keep.key not in {m.key for m in image}:
            keep = None
        if keep is None:
            keep = self._initial_selection()
        self._selected = keep
        self._rebuild_selector()
        self._update_controls()
        if self._offline:
            self._set_status("Ollama is not reachable")

    def _initial_selection(self):
        last = settings.get("last_model", "")
        for n in self._chat_models:
            if n == last:
                return n
        for m in self._image_models:
            if m.key == last:
                return m
        for n in MODELS.values():
            if n in self._chat_models:
                return n
        if self._chat_models:
            return self._chat_models[0]
        return self._image_models[0] if self._image_models else None

    def _rebuild_selector(self) -> None:
        builtin = set(MODELS.values())
        for mode, b in self._mode_buttons.items():
            installed = MODELS[mode] in self._chat_models
            active = self._selected == MODELS[mode]
            if active:
                b.setStyleSheet("background:#5a4fcf;color:#ffffff;font-weight:bold;")
            elif installed:
                b.setStyleSheet("background:#242440;color:#dde1e7;font-weight:600;")
            else:
                b.setStyleSheet("background:#161628;color:#555577;font-weight:600;")
            b.setToolTip(MODELS[mode] + ("" if installed else "  (not installed – click to install)"))
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItem("More models ▾", None)
        for n in self._chat_models:
            if n not in builtin:
                self._combo.addItem(n, n)
        for m in self._image_models:
            self._combo.addItem(m.label, m)
        idx = 0
        for i in range(1, self._combo.count()):
            d = self._combo.itemData(i)
            if (isinstance(d, str) and d == self._selected) or \
               (isinstance(d, imagegen.ImageModel) and isinstance(self._selected, imagegen.ImageModel)
                    and d.key == self._selected.key):
                idx = i
        self._combo.setCurrentIndex(idx)
        self._combo.blockSignals(False)

    def _on_mode_clicked(self, mode: str) -> None:
        name = MODELS[mode]
        if name in self._chat_models:
            self.select_model(name)
        elif self.interactive and not self._busy:
            if QMessageBox.question(self, "Install model",
                                    f"{name} is not installed. Download it now?") == QMessageBox.StandardButton.Yes:
                self._install_builtin(name)

    def _on_combo(self, idx: int) -> None:
        d = self._combo.itemData(idx)
        if d is not None:
            self.select_model(d)
        else:
            self._rebuild_selector()

    def select_model(self, model) -> None:
        if self._busy:
            return
        self._selected = model
        key = model.key if isinstance(model, imagegen.ImageModel) else model
        settings.set_value("last_model", key)
        self._ctx_cur = 0
        self._ctx_used = 0
        self._ctx_actual = 0
        self._rebuild_selector()
        self._update_controls()
        self._set_status(self._ready_text())

    def _install_builtin(self, name: str) -> None:
        if self._busy:
            return
        w = PullWorker(self.client, name)
        w.progress.connect(self._on_pull_progress)
        w.done.connect(self._on_pull_done)
        w.cancelled.connect(self._on_pull_end)
        w.error.connect(self._on_pull_error)
        w.verify_failed.connect(self._on_pull_error)
        self._worker = w
        self._pull_name = name
        self._set_busy(True)
        start_worker(w, self._threads)

    def _on_pull_progress(self, done, total, text) -> None:
        if total:
            self._ctx_bar.setValue(int(done * 1000 / total))
            self._ctx_bar.setFormat(f"{done * 100 // total}%")
        self._set_status(f"Installing {self._pull_name}: {text}")

    def _on_pull_done(self) -> None:
        self._worker = None
        self._set_busy(False)
        self.refresh_models()
        if self._selected is None and self._chat_models:
            self.select_model(self._chat_models[0])
        self._set_status(f"Installed {self._pull_name}")

    def _on_pull_end(self) -> None:
        self._worker = None
        self._set_busy(False)
        self._set_status("Install cancelled")

    def _on_pull_error(self, msg: str) -> None:
        self._worker = None
        self._set_busy(False)
        self._add_notice(f"Could not install {self._pull_name}: {msg}")
        self._set_status("Install failed")

    def open_browser(self) -> None:
        if self._browser is None:
            self._browser = ModelBrowser(self.client, self.hw, self)
            self._browser.modelsChanged.connect(self.refresh_models)
        self._browser.show()
        self._browser.raise_()
        self._browser.activateWindow()

    # ── helpers about the selection ─────────────────────────────────────────
    def _current_info(self) -> ModelInfo | None:
        return self._infos.get(self._selected) if isinstance(self._selected, str) else None

    def _is_image_selected(self) -> bool:
        return isinstance(self._selected, imagegen.ImageModel)

    def _model_label(self) -> str:
        if isinstance(self._selected, imagegen.ImageModel):
            return self._selected.name
        return self._selected or "no model"

    def _ready_text(self) -> str:
        if self._selected is None:
            return "No model installed – use ＋ to get one"
        if self._is_image_selected():
            return f"Ready · image model · {self._selected.name}"
        mode = next((m for m, n in MODELS.items() if n == self._selected), None)
        return f"Ready · {mode + ' mode · ' if mode else ''}{self._selected}"

    def _update_controls(self) -> None:
        info = self._current_info()
        can_chat = isinstance(self._selected, str)
        self._btn_attach.setVisible(can_chat)
        if info and info.vision:
            self._btn_attach.setToolTip("Attach a file or image (this model can see images)")
        elif can_chat:
            self._btn_attach.setToolTip("Attach a text, code, PDF or Word file (this model cannot see images)")
        if not can_chat and self._pending:
            self._pending.clear()
            self._render_chips()
        if self._selected is None:
            self._input.setPlaceholderText("No model installed. Click ＋ to browse Hugging Face.")
        elif self._is_image_selected():
            self._input.setPlaceholderText("Describe the image you want…    Enter → generate")
        else:
            self._input.setPlaceholderText("Ask anything…    Enter → send   |   Shift+Enter → new line"
                                           + ("   |   or ask for an image" if self._preferred_image_model() else ""))
        self._set_busy(self._busy)
        self._update_ctx_widgets()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        ready = not busy and self._selected is not None
        self._btn_send.setEnabled(ready)
        self._btn_attach.setEnabled(not busy)
        self._btn_plus.setEnabled(True)
        self._combo.setEnabled(not busy)
        for b in self._mode_buttons.values():
            b.setEnabled(not busy)
        self._btn_clear.setEnabled(not busy)

    def _set_status(self, msg: str) -> None:
        self.statusBar().showMessage(msg)

    # ── context meter ───────────────────────────────────────────────────────
    def _set_ctx_choice(self, v) -> None:
        self._ctx_choice = v
        settings.set_value("ctx_choice", v)
        self._ctx_cur = 0
        self._update_ctx_widgets()

    def _plan_ctx(self) -> tuple[int, int]:
        """(starting ctx, ceiling) for the next request."""
        if self._ctx_choice != "auto":
            c = int(self._ctx_choice)
            return c, c
        info = self._current_info()
        weights = info.size if info and info.size else 2 * 1024 ** 3
        kv = info.kv_bytes_per_token if info else 0
        cmax = info.ctx_max if info and info.ctx_max else 32768
        ceiling = hw_mod.recommend_ctx(weights, kv, cmax, self.hw, soft_cap=min(cmax, 131072))
        start = min(ceiling, 4096)
        return max(start, min(self._ctx_cur or start, ceiling)), ceiling

    def _update_ctx_widgets(self) -> None:
        for val, a in self._ctx_actions.items():
            a.setChecked(val == self._ctx_choice or (val != "auto" and str(val) == str(self._ctx_choice)))
        ctx = self._ctx_actual or self._ctx_cur or self._plan_ctx()[0]
        if self._ctx_choice == "auto":
            self._ctx_btn.setText(f"ctx: Auto ({ctx // 1024}k) ▾")
        else:
            self._ctx_btn.setText(f"ctx: {int(self._ctx_choice) // 1024}k ▾")
        frac = min(1.0, self._ctx_used / ctx) if ctx else 0
        self._ctx_bar.setValue(int(frac * 1000))
        self._ctx_bar.setFormat(f"{self._ctx_used:,} / {ctx:,}")
        colour = "#5a4fcf" if frac < 0.7 else ("#d9a400" if frac < 0.9 else "#e05555")
        self._ctx_bar.setStyleSheet(f"QProgressBar::chunk{{background:{colour};border-radius:4px;}}")

    # ── attachments ─────────────────────────────────────────────────────────
    def attach_dialog(self) -> None:
        info = self._current_info()
        paths, _ = QFileDialog.getOpenFileNames(self, "Attach file(s)", "",
                                                att.file_dialog_filter(bool(info and info.vision)))
        self.add_attachments(paths)

    def add_attachments(self, paths: list[str]) -> None:
        if not isinstance(self._selected, str) or self._busy:
            return
        info = self._current_info()
        for p in paths:
            try:
                a = att.load_attachment(p, bool(info and info.vision))
            except (att.AttachmentError, OSError) as exc:
                self._add_notice(f"Could not attach {Path(p).name}: {exc}")
                continue
            self._pending.append(a)
        self._render_chips()

    def _render_chips(self) -> None:
        while self._chips.count():
            it = self._chips.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for a in self._pending:
            b = QPushButton(f"📎 {a.name} ({a.pretty_size()})  ✕")
            b.setStyleSheet("min-height:20px;padding:2px 10px;font-size:11px;background:#1d1d36;")
            b.setToolTip(a.note or "Click to remove")
            b.clicked.connect(lambda checked=False, x=a: self._remove_pending(x))
            self._chips.addWidget(b)
        self._chips.addStretch(1)
        self._chip_holder.setVisible(bool(self._pending))

    def _remove_pending(self, a) -> None:
        if a in self._pending:
            self._pending.remove(a)
        self._render_chips()

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        self.add_attachments([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    # ── key handling ────────────────────────────────────────────────────────
    def eventFilter(self, obj: QObject, event) -> bool:
        if obj is self._input and isinstance(event, QKeyEvent):
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                self._send()
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Escape and self._busy:
            self._stop_generation()
            return
        super().keyPressEvent(e)

    # ── sending ─────────────────────────────────────────────────────────────
    def _preferred_image_model(self) -> imagegen.ImageModel | None:
        pref = settings.get("image_model", "")
        for m in self._image_models:
            if m.key == pref:
                return m
        return self._image_models[0] if self._image_models else None

    def _send(self) -> None:
        text = self._input.toPlainText().strip()
        if self._busy or self._selected is None:
            return
        pending = list(self._pending)
        if not text and not pending:
            return
        if self._is_image_selected():
            self._start_image(text, self._selected)
            return
        if not pending and imagegen.looks_like_image_request(text):
            m = self._preferred_image_model()
            if m is not None:
                self._start_image(text, m)
                return
            self._add_notice("Tip: to generate images, install an image model with ＋ → "
                             "Image models. Answering as text for now.")
        self._start_chat(text, pending)

    def _start_chat(self, text: str, pending: list) -> None:
        info = self._current_info()
        body = att.format_for_prompt(text, pending)
        entry = {"role": "user", "content": body, "display": text, "attachments": pending}
        imgs = [a.image_b64 for a in pending if a.is_image and a.image_b64]
        if imgs:
            entry["images"] = imgs
        self._last_user = (text, pending)
        self._history.append(entry)
        self._input.clear()
        self._pending = []
        self._render_chips()
        self._rebuild_chat_html()
        self._begin_stream_ui()

        has_att = any(m.get("attachments") for m in self._history)
        system = system_prompt_for(self._selected) + ((" " + att.EDIT_HINT) if has_att else "")
        ctx, ceiling = self._plan_ctx()
        job = ChatJob(model=self._selected, system=system,
                      messages=api_view(self._history[self._summary_upto:]),
                      summary=self._summary, ctx=ctx, ctx_ceiling=ceiling, est=self._est,
                      force_compact=self._force_compact,
                      think_capable=bool(info and info.thinking))
        self._force_compact = False
        w = ChatWorker(self.client, job)
        w.token.connect(self._on_token)
        w.thinking.connect(self._on_thinking)
        w.notice.connect(self._add_notice)
        w.ctx_changed.connect(self._on_ctx_changed)
        w.compacted.connect(self._on_compacted)
        w.done.connect(self._on_done)
        w.stopped.connect(self._on_stopped)
        w.error.connect(self._on_error)
        self._worker = w
        self._set_busy(True)
        self._set_status("Generating…  (Stop or Esc to cancel)")
        start_worker(w, self._threads)

    def _begin_stream_ui(self) -> None:
        self._stream_buf = ""
        self._stream_think = ""
        cur = self._chat.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.insertHtml(f"<br>{_LABEL_BOT}<br>")
        self._chat.setTextCursor(cur)
        self._chat.ensureCursorVisible()

    def _insert_stream(self, text: str, colour: str, italic: bool = False) -> None:
        cur = self._chat.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(colour))
        fmt.setFontWeight(QFont.Weight.Normal)
        fmt.setFontItalic(italic)
        cur.insertText(text, fmt)
        self._chat.setTextCursor(cur)
        self._chat.ensureCursorVisible()

    def _on_token(self, tok: str) -> None:
        self._stream_buf += tok
        self._insert_stream(tok, "#c8d4e8")

    def _on_thinking(self, tok: str) -> None:
        self._stream_think += tok
        self._insert_stream(tok, "#7a7aa8", True)

    def _on_ctx_changed(self, ctx: int) -> None:
        self._ctx_cur = ctx
        self._update_ctx_widgets()

    def _on_compacted(self, summary: str, consumed: int) -> None:
        self._summary = summary
        self._summary_upto += consumed

    def _finish_worker(self) -> None:
        self._worker = None
        self._stream_buf = ""
        self._stream_think = ""
        self._set_busy(False)

    def _on_done(self, p: dict) -> None:
        text, think = p.get("text", ""), p.get("thinking", "")
        ctx = int(p.get("ctx") or 0)
        actual = int(p.get("actual_ctx") or 0)
        pec = int(p.get("prompt_eval_count") or 0)
        est_sent = int(p.get("est_sent") or 0)
        truncated = cg.detect_truncation(pec, est_sent, actual or ctx)
        if not text.strip() and not think.strip():
            self._add_notice("The model returned an empty reply.")
        self._history.append({"role": "assistant", "content": text, "thinking": think})
        if truncated:
            self._force_compact = True
            self._add_notice(
                "⚠ Ollama silently cut off the start of this conversation (its numbers match the "
                "overflow signature). Context Guard will compact the history on your next message. "
                "Raise the context in the status bar to avoid this.")
        elif pec > 0 and est_sent > 0:
            self._est.observe(est_sent, pec)
        if actual and ctx and actual < ctx:
            self._add_notice(f"Ollama allocated only {actual:,} tokens of context although {ctx:,} were requested.")
        self._ctx_actual = actual or ctx
        self._ctx_used = pec + int(p.get("eval_count") or 0)
        self._rebuild_chat_html()
        self._finish_worker()
        self._update_ctx_widgets()
        self._set_status(self._ready_text())

    def _on_stopped(self, p: dict) -> None:
        text, think = p.get("text", ""), p.get("thinking", "")
        if text.strip() or think.strip():
            self._history.append({"role": "assistant", "content": text, "thinking": think, "stopped": True})
        else:
            self._drop_failed_turn()
        self._rebuild_chat_html()
        self._finish_worker()
        self._set_status("Stopped")

    def _on_error(self, msg: str) -> None:
        self._drop_failed_turn()
        self._add_notice(f"⚠ {msg}")
        self._finish_worker()
        self._set_status("Error – " + msg.splitlines()[0][:80])

    def _drop_failed_turn(self) -> None:
        if self._history and self._history[-1].get("role") == "user":
            self._history.pop()
            text, atts = self._last_user
            self._input.setPlainText(text)
            self._pending = list(atts)
            self._render_chips()
        self._rebuild_chat_html()

    # ── image generation ────────────────────────────────────────────────────
    def _start_image(self, text: str, model: imagegen.ImageModel) -> None:
        prompt = imagegen.clean_prompt(text) if text else ""
        if not prompt:
            return
        self._history.append({"role": "user", "content": text, "display": text, "kind": "image"})
        self._input.clear()
        self._last_user = (text, [])
        self._rebuild_chat_html()
        w = ImageWorker(self.client, model, prompt)
        w.status.connect(self._set_status)
        w.progress.connect(self._on_image_progress)
        w.done.connect(self._on_image_done)
        w.stopped.connect(self._on_image_stopped)
        w.error.connect(self._on_image_error)
        self._img_model = model
        self._worker = w
        self._set_busy(True)
        self._set_status(f"Generating image with {model.name}…")
        start_worker(w, self._threads)

    def _on_image_progress(self, d: int, t: int) -> None:
        if t:
            self._ctx_bar.setFormat(f"image step {d}/{t}")
            self._ctx_bar.setValue(int(d * 1000 / t))

    def _on_image_done(self, r: dict) -> None:
        settings.set_value("image_model", self._img_model.key)
        self._history.append({"role": "assistant", "kind": "image", "image_path": r["path"],
                              "content": f"[Generated an image for: {r['prompt']}]"})
        self._rebuild_chat_html()
        self._finish_worker()
        self._update_ctx_widgets()
        self._set_status(f"Image saved to {r['path']}")

    def _on_image_stopped(self) -> None:
        self._drop_failed_turn()
        self._finish_worker()
        self._update_ctx_widgets()
        self._set_status("Image generation stopped")

    def _on_image_error(self, msg: str) -> None:
        self._drop_failed_turn()
        self._add_notice(f"⚠ {msg}")
        self._finish_worker()
        self._update_ctx_widgets()
        self._set_status("Image generation failed")

    # ── stop / end / clear ──────────────────────────────────────────────────
    def _stop_generation(self) -> None:
        if self._worker is not None and hasattr(self._worker, "request_stop"):
            self._worker.request_stop()
            self._set_status("Stopping…")

    def stop_or_end(self) -> None:
        """Busy: stop the running job.  Idle: END the conversation (save + reset)."""
        if self._busy:
            self._stop_generation()
            return
        path = save_transcript(self._history, self._model_label())
        had = bool(path)
        self._reset_conversation()
        if had:
            self._set_status(f"Conversation ended – transcript saved to {path}")
        else:
            self._set_status("No conversation to end")

    def _reset_conversation(self) -> None:
        self._history.clear()
        self._summary, self._summary_upto = "", 0
        self._applied.clear()
        self._pending = []
        self._render_chips()
        self._ctx_used = 0
        self._ctx_cur = 0
        self._ctx_actual = 0
        self._force_compact = False
        self._chat.clear()
        self._update_ctx_widgets()

    def _clear(self) -> None:
        if self._busy:
            return
        self._reset_conversation()
        self._set_status(self._ready_text())

    # ── chat rendering ──────────────────────────────────────────────────────
    def _add_notice(self, text: str) -> None:
        self._history.append({"role": "notice", "content": text})
        self._set_status(text.splitlines()[0][:120])
        self._rebuild_chat_html()
        if self._busy and isinstance(self._worker, ChatWorker):     # keep the live answer visible
            buf, think = self._stream_buf, self._stream_think
            self._begin_stream_ui()
            if think:
                self._on_thinking(think)
            if buf:
                self._on_token(buf)
            self._stream_buf, self._stream_think = buf, think

    def _edit_candidates(self) -> list[str]:
        seen: list[str] = []
        for m in self._history:
            for a in m.get("attachments") or []:
                if a.editable and a.path not in seen:
                    seen.append(a.path)
        return seen

    def _rebuild_chat_html(self) -> None:
        parts: list[str] = []
        can_apply = bool(self._edit_candidates())
        first = True
        for i, msg in enumerate(self._history):
            role = msg["role"]
            if role == "notice":
                parts.append('<div style="color:#d9a400;font-size:11px;margin:6px 0;">'
                             + html_module.escape(msg["content"]).replace("\n", "<br>") + "</div>")
                continue
            if not first and role == "user":
                parts.append(_HR)
            first = False
            if role == "user":
                esc = html_module.escape(msg.get("display", msg["content"])).replace("\n", "<br>")
                chips = "".join(f'<br><span style="color:#9090b8;font-size:11px;">📎 {html_module.escape(a.name)}</span>'
                                for a in msg.get("attachments") or [])
                parts.append(f'{_LABEL_USER}<br><span style="color:#e8d4d8;">{esc}</span>{chips}')
            elif msg.get("kind") == "image":
                url = QUrl.fromLocalFile(msg["image_path"]).toString()
                parts.append(f"<br>{_LABEL_BOT}<br>" + image_html(url, os.path.basename(msg["image_path"]), f"braid://open/{i}/0"))
            else:
                think = thinking_html(msg["thinking"]) if msg.get("thinking") else ""

                def actions(j: int, i=i) -> str:
                    links = [f'<a href="braid://copy/{i}/{j}">Copy</a>']
                    if can_apply:
                        links.append(f'<a href="braid://apply/{i}/{j}">Apply to file…</a>')
                    links.append(f'<a href="braid://save/{i}/{j}">Save as…</a>')
                    done = [v for (mi, bj), v in self._applied.items() if mi == i and bj == j]
                    if done:
                        links.append(f'<a href="braid://undo/{i}/{j}">Undo apply ({html_module.escape(Path(done[0][0]).name)})</a>')
                    return " &nbsp;·&nbsp; ".join(links)

                body = markdown_to_html(close_open_fence(msg.get("content", "")), actions)
                stopped = '<br><span style="color:#9090b8;font-size:11px;">■ stopped</span>' if msg.get("stopped") else ""
                parts.append(f"<br>{_LABEL_BOT}<br>{think}{body}{stopped}")
        self._chat.setHtml(full_page_html("".join(parts)))
        sb = self._chat.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── link actions (copy / apply / save / undo / open) ────────────────────
    def _on_anchor(self, url: QUrl) -> None:
        if url.scheme() != "braid":
            return
        try:
            i, j = (int(x) for x in url.path().strip("/").split("/")[:2])
            msg = self._history[i]
        except (ValueError, IndexError):
            return
        act = url.host()
        if act == "open":
            QDesktopServices.openUrl(QUrl.fromLocalFile(msg.get("image_path", "")))
            return
        if act == "undo":
            self._undo(i, j)
            return
        blocks = extract_code_blocks(msg.get("content", ""))
        if j >= len(blocks):
            return
        code = blocks[j]
        if act == "copy":
            QApplication.clipboard().setText(code.code)
            self._set_status("Copied to clipboard")
        elif act == "save":
            self._save_as(code)
        elif act == "apply":
            self.apply_block(i, j)

    def _save_as(self, block) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save code as")
        if path:
            try:
                Path(path).write_text(block.code, encoding="utf-8")
                self._set_status(f"Saved {path}")
            except OSError as exc:
                self._add_notice(f"Could not save: {exc}")

    def _pick_target(self, cands: list[str], guess: int) -> str | None:
        if len(cands) == 1:
            return cands[0]
        names = [Path(c).name + "   —   " + c for c in cands]
        item, ok = QInputDialog.getItem(self, "Apply to which file?", "File:", names, max(guess, 0), False)
        return cands[names.index(item)] if ok else None

    def _confirm_diff(self, path: str, old: str, new: str) -> bool:
        diff = fileops.unified_diff(old, new, Path(path).name)
        add, rem = fileops.diff_stats(old, new)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Apply changes to {Path(path).name}?")
        dlg.resize(820, 560)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(f"{path}\n+{add} / −{rem} lines.  A backup is saved first; you can undo from the chat."))
        view = QTextBrowser()
        view.setHtml(fileops.diff_to_html(diff))
        lay.addWidget(view, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        no, yes = QPushButton("Cancel"), QPushButton("Apply")
        no.clicked.connect(dlg.reject)
        yes.clicked.connect(dlg.accept)
        yes.setEnabled(bool(diff))
        row.addWidget(no)
        row.addWidget(yes)
        lay.addLayout(row)
        return dlg.exec() == QDialog.DialogCode.Accepted

    def apply_block(self, i: int, j: int) -> None:
        cands = self._edit_candidates()
        if not cands:
            return
        block = extract_code_blocks(self._history[i].get("content", ""))[j]
        target = self._pick_target(cands, fileops.guess_target(block.code, block.lang, cands))
        if not target:
            return
        try:
            old, _nl = fileops.read_text_exact(target)
            new = fileops.normalise_new_text(old, block.code)
            if not self._confirm_diff(target, old, new):
                return
            backup = fileops.apply_with_backup(target, block.code)
        except OSError as exc:
            self._add_notice(f"Could not write {target}: {exc}")
            return
        self._applied[(i, j)] = (target, str(backup))
        self._set_status(f"Applied to {target} (backup: {backup})")
        self._rebuild_chat_html()

    def _undo(self, i: int, j: int) -> None:
        rec = self._applied.pop((i, j), None)
        if rec:
            try:
                fileops.undo(*rec)
                self._set_status(f"Restored {rec[0]}")
            except OSError as exc:
                self._add_notice(f"Undo failed: {exc}")
            self._rebuild_chat_html()

    # ── shutdown ────────────────────────────────────────────────────────────
    def closeEvent(self, e) -> None:
        if self._worker is not None and hasattr(self._worker, "request_stop"):
            self._worker.request_stop()
        if self._browser is not None:
            self._browser.close()
        for th, _w in list(self._threads):
            th.quit()
            th.wait(2000)
        super().closeEvent(e)
