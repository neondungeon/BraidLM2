"""
attachments.py: turn a file the user picked into something a model can read
===========================================================================
text / code / data   -> read as text (UTF-8, falling back to Windows-1252)
PDF                  -> text via ``pypdf`` (pip install pypdf)
DOCX                 -> text from word/document.xml (no extra dependency)
images               -> base64 for vision models (downscaled to keep prompts small)

The attached text is wrapped in an ``<attached_file>`` block so (a) the model
sees where the file starts and ends, and (b) context_guard can recognise and
summarise it away when the conversation is compacted.
"""
from __future__ import annotations

import base64
import html
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
TEXT_EXT = {
    ".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".conf", ".env", ".xml", ".html", ".htm", ".css", ".scss", ".js", ".jsx", ".ts", ".tsx", ".mjs",
    ".py", ".pyi", ".ipynb", ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".go", ".rs",
    ".rb", ".php", ".swift", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd", ".sql", ".r", ".lua",
    ".pl", ".scala", ".dart", ".vue", ".svelte", ".tex", ".log", ".diff", ".patch", ".gradle", ".make",
    ".mk", ".cmake", ".dockerfile", ".gitignore", ".proto", ".graphql", ".tf", ".hcl", ".vb", ".asm",
}
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_SIDE = 1568


class AttachmentError(ValueError):
    pass


@dataclass
class Attachment:
    path: str
    name: str
    kind: str                      # "text" | "image" | "pdf" | "docx"
    size: int
    text: str | None = None
    image_b64: str | None = None
    note: str = ""
    editable: bool = False         # plain text on disk we can safely write back to

    @property
    def is_image(self) -> bool:
        return self.kind == "image"

    def pretty_size(self) -> str:
        n = float(self.size)
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} GB"


def file_dialog_filter(allow_images: bool) -> str:
    exts = " ".join(f"*{e}" for e in sorted(TEXT_EXT | {".pdf", ".docx"}))
    parts = [f"Documents and code ({exts})"]
    if allow_images:
        parts.insert(0, "Images (" + " ".join(f"*{e}" for e in sorted(IMAGE_EXT)) + ")")
    parts.append("All files (*)")
    return ";;".join(parts)


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig",):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("cp1252", errors="replace")


def _looks_binary(raw: bytes) -> bool:
    head = raw[:8192]
    return b"\x00" in head and raw[:2] not in (b"\xff\xfe", b"\xfe\xff")


def read_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise AttachmentError(f"{path.name} is not a valid .docx file.") from exc
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab/>", "\t", xml)
    xml = re.sub(r"<w:br[^>]*/>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    return html.unescape(text).strip()


def read_pdf(path: Path) -> tuple[str, str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise AttachmentError("Reading PDFs needs the 'pypdf' package:  pip install pypdf") from exc
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise AttachmentError("This PDF is encrypted.")
        pages = [(p.extract_text() or "").strip() for p in reader.pages]
    except AttachmentError:
        raise
    except Exception as exc:
        raise AttachmentError(f"Could not read the PDF: {exc}") from exc
    text = "\n\n".join(f"--- page {i + 1} ---\n{t}" for i, t in enumerate(pages) if t)
    note = "" if text else "No extractable text (a scanned PDF?). Use a vision model on page images instead."
    return text, note


def prepare_image_b64(path: Path, max_side: int = MAX_IMAGE_SIDE) -> str:
    """Load, downscale and re-encode an image (PNG if it has transparency, else JPEG)."""
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
    from PySide6.QtGui import QImage
    img = QImage(str(path))
    if img.isNull():
        raise AttachmentError(f"Could not decode image {path.name}.")
    if max(img.width(), img.height()) > max_side:
        img = img.scaled(max_side, max_side, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    fmt = "PNG" if img.hasAlphaChannel() else "JPEG"
    img.save(buf, fmt, 90)
    data = bytes(QByteArray(buf.data()))
    return base64.b64encode(data).decode("ascii")


def load_attachment(path: str, allow_images: bool) -> Attachment:
    p = Path(path)
    if not p.is_file():
        raise AttachmentError(f"{p.name or path} is not a file.")
    size = p.stat().st_size
    if size > MAX_FILE_BYTES:
        raise AttachmentError(f"{p.name} is {size / 1048576:.0f} MB; the limit is {MAX_FILE_BYTES // 1048576} MB.")
    ext = p.suffix.lower()

    if ext in IMAGE_EXT:
        if not allow_images:
            raise AttachmentError("This model cannot look at images. Pick a vision model (e.g. gemma3, "
                                  "qwen2.5vl, llava) to attach pictures.")
        return Attachment(str(p), p.name, "image", size, image_b64=prepare_image_b64(p))
    if ext == ".pdf":
        text, note = read_pdf(p)
        return Attachment(str(p), p.name, "pdf", size, text=text, note=note)
    if ext == ".docx":
        return Attachment(str(p), p.name, "docx", size, text=read_docx(p))

    raw = p.read_bytes()
    if _looks_binary(raw):
        raise AttachmentError(f"{p.name} looks like a binary file BraidLM cannot read.")
    return Attachment(str(p), p.name, "text", size, text=_decode(raw), editable=True)


def format_for_prompt(prompt: str, atts: list[Attachment]) -> str:
    """User prompt + attached files as one message body."""
    blocks = []
    for a in atts:
        if a.kind == "image" or a.text is None:
            continue
        body = a.text if a.text else f"(no text could be extracted: {a.note})"
        blocks.append(f'<attached_file name="{a.name}" path="{a.path}">\n{body}\n</attached_file>')
    if not blocks:
        return prompt
    return (prompt.strip() or "Here are the attached file(s).") + "\n\n" + "\n\n".join(blocks)


EDIT_HINT = (
    "When the user asks you to change or rewrite an attached file, reply with the COMPLETE updated "
    "file in a single fenced code block (with a language tag), then a short list of what you changed. "
    "Never abbreviate the file with placeholders like '... rest unchanged'."
)
