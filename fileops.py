"""
fileops.py: apply a model's edit to a file on disk, safely
==========================================================
A model's answer is never written to disk directly.  The user first sees a
diff; on approval the original is copied to ``~/.braidlm/backups`` and the
change is written; an Undo restores the backup byte-for-byte.
"""
from __future__ import annotations

import difflib
import html
import shutil
import time
from pathlib import Path

import settings


def read_text_exact(path: str) -> tuple[str, str]:
    """Return (text, newline) where newline is the file's dominant line ending."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    nl = "\r\n" if raw.count(b"\r\n") > raw.count(b"\n") / 2 and b"\r\n" in raw else "\n"
    return text.replace("\r\n", "\n"), nl


def normalise_new_text(old_text: str, new_text: str) -> str:
    """Models rarely reproduce the trailing newline; keep the file's convention."""
    new_text = new_text.replace("\r\n", "\n")
    if old_text.endswith("\n") and not new_text.endswith("\n"):
        new_text += "\n"
    if not old_text.endswith("\n") and new_text.endswith("\n") and old_text:
        new_text = new_text.rstrip("\n")
    return new_text


def unified_diff(old: str, new: str, name: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"{name} (current)", tofile=f"{name} (proposed)", n=3))


def diff_stats(old: str, new: str) -> tuple[int, int]:
    added = removed = 0
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def diff_to_html(diff: str) -> str:
    if not diff:
        return '<p style="color:#9090b8;">No differences: the proposed text is identical to the file.</p>'
    out = ['<pre style="font-family:Consolas,monospace;font-size:12px;">']
    for line in diff.splitlines():
        esc = html.escape(line)
        if line.startswith("+++") or line.startswith("---"):
            out.append(f'<span style="color:#9090b8;">{esc}</span>')
        elif line.startswith("@@"):
            out.append(f'<span style="color:#7c9ef5;">{esc}</span>')
        elif line.startswith("+"):
            out.append(f'<span style="color:#7ee0a0;background:#16301f;">{esc}</span>')
        elif line.startswith("-"):
            out.append(f'<span style="color:#f08a9c;background:#3a1a24;">{esc}</span>')
        else:
            out.append(esc)
    out.append("</pre>")
    return "\n".join(out)


def apply_with_backup(path: str, new_text: str) -> Path:
    """Back up ``path`` then overwrite it with ``new_text``.  Returns the backup path."""
    src = Path(path)
    old_text, nl = read_text_exact(path)
    final = normalise_new_text(old_text, new_text).replace("\n", nl)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = settings.subdir("backups") / f"{stamp}_{src.name}"
    i = 1
    while backup.exists():
        backup = settings.subdir("backups") / f"{stamp}_{i}_{src.name}"
        i += 1
    shutil.copy2(src, backup)
    with open(src, "w", encoding="utf-8", newline="") as f:
        f.write(final)
    return backup


def undo(path: str, backup: str) -> None:
    if not Path(backup).is_file():
        raise FileNotFoundError(f"Backup {backup} no longer exists.")
    shutil.copy2(backup, path)


def guess_target(code: str, lang: str, candidates: list[str]) -> int:
    """Pick which attached file a code block most likely replaces (index into candidates)."""
    if not candidates:
        return -1
    ext_map = {"python": ".py", "py": ".py", "javascript": ".js", "js": ".js", "typescript": ".ts",
               "ts": ".ts", "html": ".html", "css": ".css", "json": ".json", "yaml": ".yml", "yml": ".yml",
               "bash": ".sh", "sh": ".sh", "java": ".java", "c": ".c", "cpp": ".cpp", "go": ".go",
               "rust": ".rs", "rs": ".rs", "sql": ".sql", "markdown": ".md", "md": ".md", "xml": ".xml",
               "toml": ".toml", "ruby": ".rb", "php": ".php", "csharp": ".cs", "cs": ".cs"}
    want = ext_map.get((lang or "").lower())
    scored = []
    for i, c in enumerate(candidates):
        p = Path(c)
        s = 0
        if want and p.suffix.lower() == want:
            s += 2
        if p.name in code[:300]:
            s += 3
        scored.append((s, i))
    best = max(scored, key=lambda t: (t[0], t[1]))      # ties -> most recently attached
    return best[1]
