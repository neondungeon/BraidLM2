"""
model_browser.py: the "＋" dialog – browse, install and remove Hugging Face models
=================================================================================
* Language models (GGUF) are installed *through Ollama* (``hf.co/<repo>:<quant>``)
  so they appear in the normal model list and run on the same server.
* Image models (diffusers) are downloaded into ~/.braidlm/image_models and run by
  BraidLM's own backend (Ollama cannot generate images on every platform).

What BraidLM adds on top of ``ollama pull hf.co/...``:
  - searching Hugging Face from the GUI (the Ollama CLI has no search command),
  - an estimated *fit verdict per quantisation, before anything is downloaded*,
  - a recommended quant, a post-install load test, and one-click removal.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QProgressBar, QPushButton, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)

import hardware as hw_mod
import hf_hub
import imagegen
import settings
from ollama_client import OllamaClient, OllamaError
from worker import DownloadWorker, PullWorker, start_worker, submit

GiB = 1024 ** 3
_FIT_COLOURS = {"gpu": "#4caf50", "cpu": "#d9a400", "partial": "#d9a400", "no": "#e05555"}


def human(n: int) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return ""


class ModelBrowser(QDialog):
    modelsChanged = Signal()

    def __init__(self, client: OllamaClient, hw: hw_mod.Hardware, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Models – Hugging Face")
        self.resize(780, 640)
        self.setModal(False)
        self.client, self.hw = client, hw
        self._threads: list = []
        self._repos: list = []
        self._gguf: hf_hub.GgufRepo | None = None
        self._plan: hf_hub.DiffusersPlan | None = None
        self._active = None                       # running install worker
        self._pending_quant: str | None = None
        self._gen = 0                             # invalidates stale async results

        root = QVBoxLayout(self)
        self.hw_label = QLabel("Your hardware: " + hw.describe())
        self.hw_label.setWordWrap(True)
        self.hw_label.setStyleSheet("color:#9aa;")
        root.addWidget(self.hw_label)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self._build_discover()
        self._build_installed()
        self.tabs.currentChanged.connect(lambda i: self.refresh_installed() if i == 1 else None)
        self.refresh_installed()

    # ── Discover tab ────────────────────────────────────────────────────────
    def _build_discover(self) -> None:
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        self.kind = QComboBox()
        self.kind.addItem("Language models (GGUF → Ollama)", "gguf")
        self.kind.addItem("Image models (diffusers → BraidLM)", "image")
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.query = QLineEdit()
        self.query.setPlaceholderText("Search Hugging Face, or paste hf.co/user/repo:Q4_K_M")
        self.query.returnPressed.connect(self.search)
        self.btn_search = QPushButton("Search")
        self.btn_search.clicked.connect(self.search)
        row.addWidget(self.kind)
        row.addWidget(self.query, 1)
        row.addWidget(self.btn_search)
        v.addLayout(row)

        self.results = QListWidget()
        self.results.currentRowChanged.connect(self._repo_selected)
        v.addWidget(self.results, 2)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Quantisation", "Size", "Will it run on this machine?"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._update_install_enabled)
        v.addWidget(self.table, 2)

        self.info = QLabel("")
        self.info.setWordWrap(True)
        v.addWidget(self.info)

        trow = QHBoxLayout()
        self.token_label = QLabel("HF token (only for gated models):")
        self.token = QLineEdit(settings.get("hf_token", "") or "")
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        trow.addWidget(self.token_label)
        trow.addWidget(self.token, 1)
        v.addLayout(trow)

        self.warn = QLabel("")
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("color:#d9a400;")
        v.addWidget(self.warn)

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        v.addWidget(self.bar)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        v.addWidget(self.status)

        brow = QHBoxLayout()
        brow.addStretch(1)
        self.btn_install = QPushButton("Install")
        self.btn_install.setEnabled(False)
        self.btn_install.clicked.connect(self.install)
        self.btn_cancel = QPushButton("Cancel download")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self.cancel_install)
        brow.addWidget(self.btn_cancel)
        brow.addWidget(self.btn_install)
        v.addLayout(brow)
        self.tabs.addTab(w, "Discover")
        self._kind_changed()

    def _hub(self) -> hf_hub.HubClient:
        tok = self.token.text().strip()
        if tok:
            settings.set_value("hf_token", tok)
        return hf_hub.HubClient(tok or None)

    def _kind_changed(self) -> None:
        image = self.kind.currentData() == "image"
        self.token_label.setVisible(image)
        self.token.setVisible(image)
        self.table.setVisible(not image)
        self.results.clear()
        self.table.setRowCount(0)
        self._gguf = self._plan = None
        self.info.setText("")
        self.warn.setText("")
        if image:
            ok, why = imagegen.diffusers_status()
            if not ok:
                self.warn.setText("Image models need extra Python packages before they can run "
                                  "(you can still download them now):\n" + why)
        self._update_install_enabled()

    def search(self) -> None:
        text = self.query.text().strip()
        kind = self.kind.currentData()
        parsed = hf_hub.parse_repo_input(text) if text else None
        self._gen += 1
        gen = self._gen
        self.btn_search.setEnabled(False)
        self.status.setText("Searching Hugging Face…")
        hub = self._hub()
        self._pending_quant = parsed[1] if parsed else None

        def work():
            if parsed:
                return [hf_hub.RepoSummary(parsed[0])]
            return hub.search(text, kind)

        def ok(repos):
            if gen != self._gen:
                return
            self.btn_search.setEnabled(True)
            self._repos = repos
            self.results.clear()
            for r in repos:
                it = QListWidgetItem(f"{r.repo_id}   ⬇ {r.downloads:,}  ♥ {r.likes:,}" + ("   🔒 gated" if r.gated else ""))
                it.setData(Qt.ItemDataRole.UserRole, r.repo_id)
                self.results.addItem(it)
            self.status.setText(f"{len(repos)} result(s)." if repos else "No results.")
            if parsed and repos:
                self.results.setCurrentRow(0)

        def bad(msg):
            if gen != self._gen:
                return
            self.btn_search.setEnabled(True)
            self.status.setText(msg)

        submit(work, ok, bad)

    def _repo_selected(self, row: int) -> None:
        self._gguf = self._plan = None
        self.table.setRowCount(0)
        self.info.setText("")
        self._update_install_enabled()
        if row < 0 or row >= len(self._repos):
            return
        repo_id = self._repos[row].repo_id
        kind = self.kind.currentData()
        self._gen += 1
        gen = self._gen
        hub = self._hub()
        self.status.setText(f"Reading file list of {repo_id}…")

        def ok(res):
            if gen != self._gen:
                return
            self.status.setText("")
            if kind == "image":
                self._plan = res
                self.info.setText(f"{repo_id}: {len(res.files)} files, {human(res.total)} to download"
                                  + (f" (fp16 weights)" if res.variant else "") + ".")
            else:
                self._gguf = res
                self._fill_table(res)
            self._update_install_enabled()

        def bad(msg):
            if gen == self._gen:
                self.status.setText(msg)

        submit(lambda: hub.diffusers_plan(repo_id) if kind == "image" else hub.gguf_repo(repo_id), ok, bad)

    def _fill_table(self, repo: hf_hub.GgufRepo) -> None:
        rec = hf_hub.recommend_variant(repo.variants, self.hw)
        want = -1
        if self._pending_quant:
            for i, v in enumerate(repo.variants):
                if v.quant.lower() == self._pending_quant.lower():
                    want = i
        self.table.setRowCount(len(repo.variants))
        for i, v in enumerate(repo.variants):
            fit = hw_mod.classify(v.size, self.hw, 4096)
            q = QTableWidgetItem(v.quant + ("  ★ recommended" if i == rec else ""))
            s = QTableWidgetItem(human(v.size))
            f = QTableWidgetItem(f"{fit.label} – {fit.detail}")
            f.setForeground(QColor(_FIT_COLOURS[fit.level]))
            for c, it in enumerate((q, s, f)):
                self.table.setItem(i, c, it)
        self.table.selectRow(want if want >= 0 else rec)
        notes = []
        if repo.has_mmproj:
            notes.append("This repo ships a vision projector (mmproj); Ollama may not load it from hf.co references.")
        notes.append("Fit is estimated at 4k context; BraidLM's Context Guard adapts the real context later.")
        self.info.setText(" ".join(notes))

    def _selected_variant(self):
        if not self._gguf:
            return None
        rows = self.table.selectionModel().selectedRows()
        return self._gguf.variants[rows[0].row()] if rows else None

    def _update_install_enabled(self) -> None:
        busy = self._active is not None
        if self.kind.currentData() == "image":
            ok = self._plan is not None
        else:
            ok = self._selected_variant() is not None
        self.btn_install.setEnabled(ok and not busy)

    # ── install ─────────────────────────────────────────────────────────────
    def _set_busy(self, busy: bool) -> None:
        self.bar.setVisible(busy)
        self.btn_cancel.setVisible(busy)
        self.btn_cancel.setEnabled(True)
        for w in (self.kind, self.query, self.btn_search, self.results, self.table):
            w.setEnabled(not busy)
        self._update_install_enabled()

    def install(self) -> None:
        if self._active is not None:
            return
        if self.kind.currentData() == "image":
            plan = self._plan
            if plan is None:
                return
            if plan.total > 0 and QMessageBox.question(
                    self, "Install image model",
                    f"Download {human(plan.total)} for {plan.repo_id}?") != QMessageBox.StandardButton.Yes:
                return
            dest = imagegen.diffusers_dir(plan.repo_id)
            w = DownloadWorker(self._hub(), plan, dest)
            self._name = plan.repo_id
        else:
            v = self._selected_variant()
            if v is None or self._gguf is None:
                return
            fit = hw_mod.classify(v.size, self.hw, 4096)
            if fit.level == "no" and QMessageBox.warning(
                    self, "Probably too large",
                    f"{self._gguf.repo_id} {v.quant} {fit.detail}.\n\nInstall anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return
            ref = hf_hub.ollama_ref(self._gguf.repo_id, v)
            w = PullWorker(self.client, ref)
            w.verify_failed.connect(self._on_verify_failed)
            self._name = ref
        w.progress.connect(self._progress)
        w.done.connect(self._install_done)
        w.cancelled.connect(self._on_cancelled)
        w.error.connect(self._on_failed)
        self._active = w
        self._set_busy(True)
        self.bar.setRange(0, 0)
        self.status.setText(f"Installing {self._name}…")
        start_worker(w, self._threads)

    def _progress(self, done, total, text: str) -> None:
        if total:
            self.bar.setRange(0, 1000)
            self.bar.setValue(int(done * 1000 / total))
            self.status.setText(f"{text or 'Downloading'} – {human(done)} / {human(total)}")
        else:
            self.bar.setRange(0, 0)
            self.status.setText(text)

    def cancel_install(self) -> None:
        if self._active is not None:
            self.btn_cancel.setEnabled(False)
            self._active.request_stop()

    def _install_end(self, msg: str) -> None:
        self._active = None
        self._set_busy(False)
        self.status.setText(msg)
        self.refresh_installed()
        self.modelsChanged.emit()

    def _install_done(self) -> None:
        self._install_end(f"Installed {self._name}.")

    def _on_cancelled(self) -> None:
        self._install_end("Download cancelled.")

    def _on_failed(self, msg: str) -> None:
        self._install_end("Failed: " + msg)

    def _on_verify_failed(self, msg: str) -> None:
        ref = self._name
        self._install_end("Downloaded, but the model failed to load.")
        r = QMessageBox.question(
            self, "Model does not load",
            f"{ref} was downloaded but Ollama could not run it:\n\n{msg}\n\nRemove it?")
        if r == QMessageBox.StandardButton.Yes:
            self._remove_ollama(ref)

    def _remove_ollama(self, name: str) -> None:
        def ok(_):
            self.status.setText(f"Removed {name}.")
            self.refresh_installed()
            self.modelsChanged.emit()
        submit(lambda: self.client.delete(name), ok, lambda m: self.status.setText("Remove failed: " + m))

    def cancel_all(self) -> None:
        if self._active is not None:
            self._active.request_stop()

    # ── Installed tab ───────────────────────────────────────────────────────
    def _build_installed(self) -> None:
        w = QWidget()
        v = QVBoxLayout(w)
        self.inst_list = QListWidget()
        self.inst_list.currentRowChanged.connect(lambda r: self.btn_remove.setEnabled(r >= 0))
        v.addWidget(self.inst_list, 1)
        self.inst_total = QLabel("")
        v.addWidget(self.inst_total)
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_remove = QPushButton("Remove selected")
        self.btn_remove.setEnabled(False)
        self.btn_remove.clicked.connect(self.remove_selected)
        row.addWidget(self.btn_remove)
        v.addLayout(row)
        self.tabs.addTab(w, "Installed")

    def refresh_installed(self) -> None:
        def work():
            try:
                tags = self.client.tags()
                err = ""
            except OllamaError as exc:
                tags, err = [], str(exc)
            return tags, err, imagegen.list_local_diffusers()

        def ok(res):
            tags, err, imgs = res
            self.inst_list.clear()
            total = 0
            for t in tags:
                size = int(t.get("size") or 0)
                total += size
                it = QListWidgetItem(f"{t.get('name')}    {human(size)}    (Ollama)")
                it.setData(Qt.ItemDataRole.UserRole, ("ollama", t.get("name")))
                self.inst_list.addItem(it)
            for m in imgs:
                total += m.size
                it = QListWidgetItem(f"🎨 {m.name}    {human(m.size)}    (image · BraidLM)")
                it.setData(Qt.ItemDataRole.UserRole, ("diffusers", m.name))
                self.inst_list.addItem(it)
            self.inst_total.setText(err or f"{self.inst_list.count()} model(s), {human(total)} on disk.")
            self.btn_remove.setEnabled(False)

        submit(work, ok, lambda m: self.inst_total.setText(m))

    def remove_selected(self) -> None:
        it = self.inst_list.currentItem()
        if it is None:
            return
        kind, name = it.data(Qt.ItemDataRole.UserRole)
        if QMessageBox.question(self, "Remove model", f"Delete {name} from disk?") != QMessageBox.StandardButton.Yes:
            return
        if kind == "ollama":
            self._remove_ollama(name)
        else:
            for m in imagegen.list_local_diffusers():
                if m.name == name:
                    imagegen.remove_diffusers(m)
            self.refresh_installed()
            self.modelsChanged.emit()

    def closeEvent(self, e) -> None:
        self.cancel_all()
        super().closeEvent(e)
