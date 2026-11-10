"""
window.py — BraidLM main application window
============================================
Architecture is based on nocolour/Ollama-Desktop (MIT):
  https://github.com/nocolour/Ollama-Desktop

Adapted and rebranded as BraidLM with:
  • Coder / General mode toggle (Qwen2.5-Coder:3b ↔ Qwen2.5:3b)
  • BraidLM purple-dark theme
  • Rebuilt chat renderer using renderer.py for markdown + syntax highlight
  • Simplified module layout (no dialogs, single-window focus)

Layout
------
  ┌──────────────────────────────────────────────┐
  │  🧵 BraidLM              [General]  [Coder ◀] │  ← header
  ├──────────────────────────────────────────────┤
  │                                              │
  │         Chat display (QTextBrowser)          │  ← scrollable HTML view
  │                                              │
  ├──────────────────────────────────────────────┤
  │  [Input …………………………………]     [⬆ Send]          │
  │                             [✕ Clear]        │  ← input row
  └──────────────────────────────────────────────┘
  │ Status bar                                   │

Threading contract
------------------
  OllamaWorker runs inside a QThread.
  All GUI updates happen on the main thread via Qt signals (token / done / error).
"""

import html as html_module

from PySide6.QtCore import QObject, Qt, QThread
from PySide6.QtGui import QFont, QKeyEvent, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollBar,
    QSizePolicy,
    QStatusBar,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from renderer import full_page_html, markdown_to_html
from worker import MODELS, OllamaWorker

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
"""

# ── CSS injected into the QTextBrowser's HTML for formatted messages ────────
_MSG_LABEL_USER = (
    '<span style="font-weight:bold;font-size:12px;color:#e05c7a;">You</span>'
)
_MSG_LABEL_BOT = (
    '<span style="font-weight:bold;font-size:12px;color:#7c9ef5;">🧵 BraidLM</span>'
)
_HR = '<hr style="border:none;border-top:1px solid #1e1e38;margin:8px 0;">'


class MainWindow(QMainWindow):
    """Top-level BraidLM window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BraidLM")
        self.resize(1040, 760)
        self.setMinimumSize(680, 500)
        self.setStyleSheet(DARK_STYLE)

        # State
        self._mode:          str         = "coder"   # Active model mode
        self._history:       list[dict]  = []        # Conversation history
        self._stream_buf:    str         = ""        # Accumulates streamed tokens
        self._busy:          bool        = False
        self._worker:        OllamaWorker | None = None
        self._worker_thread: QThread     | None = None

        self._build_ui()
        self._update_mode_buttons()
        self._set_status(f"Ready  ·  coder mode  ·  {MODELS['coder']}")

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(14, 14, 14, 10)
        vbox.setSpacing(10)

        # ── Header ──────────────────────────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(8)

        title = QLabel("🧵 BraidLM")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color:#8b7cf5;letter-spacing:0.5px;")
        header.addWidget(title)
        header.addStretch()

        mode_lbl = QLabel("Mode:")
        mode_lbl.setStyleSheet("font-size:12px;color:#636390;margin-right:4px;")
        header.addWidget(mode_lbl)

        self._btn_general = QPushButton("General")
        self._btn_general.setFixedWidth(90)
        self._btn_general.clicked.connect(lambda: self._switch_mode("general"))
        header.addWidget(self._btn_general)

        self._btn_coder = QPushButton("Coder")
        self._btn_coder.setFixedWidth(90)
        self._btn_coder.clicked.connect(lambda: self._switch_mode("coder"))
        header.addWidget(self._btn_coder)

        vbox.addLayout(header)

        # ── Chat display ─────────────────────────────────────────────────────
        self._chat = QTextBrowser()
        self._chat.setReadOnly(True)
        self._chat.setOpenExternalLinks(False)
        self._chat.setFont(QFont("Consolas", 12))
        self._chat.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        vbox.addWidget(self._chat, stretch=1)

        # ── Input row ────────────────────────────────────────────────────────
        input_row = QHBoxLayout()
        input_row.setSpacing(10)

        self._input = QTextEdit()
        self._input.setPlaceholderText(
            "Ask anything…    Enter → send   |   Shift+Enter → new line"
        )
        self._input.setFixedHeight(84)
        self._input.setFont(QFont("Segoe UI", 12))
        self._input.installEventFilter(self)
        input_row.addWidget(self._input, stretch=1)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(6)

        self._btn_send = QPushButton("⬆  Send")
        self._btn_send.setFixedWidth(100)
        self._btn_send.clicked.connect(self._send)
        btn_col.addWidget(self._btn_send)

        btn_clear = QPushButton("✕  Clear")
        btn_clear.setFixedWidth(100)
        btn_clear.clicked.connect(self._clear)
        btn_col.addWidget(btn_clear)

        btn_col.addStretch()
        input_row.addLayout(btn_col)
        vbox.addLayout(input_row)

        # ── Status bar ───────────────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

    # ── Enter-key interception ─────────────────────────────────────────────

    def eventFilter(self, obj: QObject, event) -> bool:
        if obj is self._input and isinstance(event, QKeyEvent):
            enter = Qt.Key.Key_Return
            shift = Qt.KeyboardModifier.ShiftModifier
            if event.key() in (enter, Qt.Key.Key_Enter) and not (event.modifiers() & shift):
                self._send()
                return True
        return super().eventFilter(obj, event)

    # ── Mode toggle ────────────────────────────────────────────────────────

    def _switch_mode(self, mode: str) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        self._update_mode_buttons()
        self._set_status(f"Ready  ·  {mode} mode  ·  {MODELS[mode]}")

    def _update_mode_buttons(self) -> None:
        _active   = "background:#5a4fcf;color:#ffffff;font-weight:bold;"
        _inactive = "background:#242440;color:#dde1e7;font-weight:600;"
        self._btn_coder  .setStyleSheet(_active if self._mode == "coder"   else _inactive)
        self._btn_general.setStyleSheet(_active if self._mode == "general" else _inactive)

    # ── Sending messages ───────────────────────────────────────────────────

    def _send(self) -> None:
        text = self._input.toPlainText().strip()
        if not text or self._busy:
            return

        self._input.clear()
        self._history.append({"role": "user", "content": text})

        # Show user message in chat
        self._append_user_bubble(text)
        # Reserve the bot label + streaming area
        self._start_bot_bubble()

        self._busy = True
        self._stream_buf = ""
        self._btn_send.setEnabled(False)
        self._set_status("Generating…")

        # Launch worker thread
        self._worker = OllamaWorker(self._mode, list(self._history))
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.token.connect(self._on_token)
        self._worker.done .connect(self._on_done)
        self._worker.error.connect(self._on_error)

        self._worker_thread.start()

    # ── Chat rendering ─────────────────────────────────────────────────────

    def _append_user_bubble(self, text: str) -> None:
        """Insert a formatted user message at the end of the chat display."""
        cursor = self._chat.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Separator between turns (skip before very first message)
        if self._chat.document().characterCount() > 1:
            cursor.insertHtml(_HR)

        escaped = html_module.escape(text).replace("\n", "<br>")
        cursor.insertHtml(
            f"{_MSG_LABEL_USER}<br>"
            f'<span style="color:#e8d4d8;">{escaped}</span><br>'
        )
        self._chat.setTextCursor(cursor)
        self._chat.ensureCursorVisible()

    def _start_bot_bubble(self) -> None:
        """Insert the BraidLM label; streaming tokens will follow."""
        cursor = self._chat.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(f"<br>{_MSG_LABEL_BOT}<br>")
        self._chat.setTextCursor(cursor)
        self._chat.ensureCursorVisible()

    def _on_token(self, token: str) -> None:
        """Append a raw token as plain text during streaming."""
        self._stream_buf += token
        cursor = self._chat.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        # Use a fixed char format so Qt doesn't inherit the bold label style
        from PySide6.QtGui import QTextCharFormat, QColor
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#c8d4e8"))
        fmt.setFontWeight(QFont.Weight.Normal)
        cursor.insertText(token, fmt)
        self._chat.setTextCursor(cursor)
        self._chat.ensureCursorVisible()

    def _on_done(self, full_response: str) -> None:
        """
        Generation complete.  Rebuild the entire chat as HTML so the final
        bot response gets full markdown + syntax-highlighted code rendering.
        The plain-text streaming content is replaced by the rich HTML version.
        """
        self._history.append({"role": "assistant", "content": full_response})
        self._rebuild_chat_html()
        self._cleanup_worker()
        self._set_status(f"Ready  ·  {self._mode} mode  ·  {MODELS[self._mode]}")

    def _on_error(self, msg: str) -> None:
        """Display an error in the bot bubble and unlock the UI."""
        cursor = self._chat.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        from PySide6.QtGui import QTextCharFormat, QColor
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#e05c7a"))
        cursor.insertText(f"\n⚠  {msg}", fmt)
        self._chat.setTextCursor(cursor)
        self._cleanup_worker()
        self._set_status(f"Error — {msg[:60]}")

    def _rebuild_chat_html(self) -> None:
        """
        Re-render the entire conversation history as a single styled HTML page.
        Called after each completed response so code blocks get syntax highlighting.
        Scroll position is restored to the bottom after rebuild.
        """
        parts: list[str] = []
        first = True

        for msg in self._history:
            if not first:
                parts.append(_HR)
            first = False

            role    = msg["role"]
            content = msg["content"]

            if role == "user":
                escaped = html_module.escape(content).replace("\n", "<br>")
                parts.append(
                    f"{_MSG_LABEL_USER}<br>"
                    f'<span style="color:#e8d4d8;">{escaped}</span>'
                )
            elif role == "assistant":
                body = markdown_to_html(content)
                parts.append(f"{_MSG_LABEL_BOT}<br>{body}")

        self._chat.setHtml(full_page_html("".join(parts)))
        # Scroll to bottom after re-render
        sb = self._chat.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── Clear ──────────────────────────────────────────────────────────────

    def _clear(self) -> None:
        if self._busy:
            return
        self._history.clear()
        self._stream_buf = ""
        self._chat.clear()

    # ── Worker cleanup ─────────────────────────────────────────────────────

    def _cleanup_worker(self) -> None:
        self._busy = False
        self._btn_send.setEnabled(True)
        self._stream_buf = ""
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread.wait()
            self._worker_thread = None
            self._worker = None

    # ── Status bar ─────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        self._status_bar.showMessage(msg)
