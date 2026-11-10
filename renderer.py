"""
renderer.py: BraidLM markdown to HTML converter
================================================
Converts the model's markdown output to styled HTML for display
inside a QTextBrowser widget.
"""

import html
import re
from dataclasses import dataclass
from typing import Callable

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name
from pygments.util import ClassNotFound

# ── One shared formatter so CSS is generated once ──────────────────────────
_FMT = HtmlFormatter(style="monokai", noclasses=False, wrapcode=True)
_PYGMENTS_CSS: str = _FMT.get_style_defs(".highlight")

# ── Regex patterns for prose parsing ───────────────────────────────────────
_CODE_BLOCK   = re.compile(r"```(\w*)\n?(.*?)```",           re.DOTALL)
_HEADER       = re.compile(r"^(#{1,6})\s+(.+)$",            re.MULTILINE)
_BOLD         = re.compile(r"\*\*(.+?)\*\*",                 re.DOTALL)
_ITALIC       = re.compile(r"\*(?=\S)(.+?)(?<=\S)\*")
_INLINE_CODE  = re.compile(r"`([^`]+)`")
_LIST_ITEM    = re.compile(r"^[ \t]*[-*]\s+(.+)$",          re.MULTILINE)
_ORDERED_ITEM = re.compile(r"^[ \t]*\d+\.\s+(.+)$",         re.MULTILINE)


# ── Core public API ─────────────────────────────────────────────────────────

@dataclass
class CodeBlock:
    lang: str
    code: str


def extract_code_blocks(text: str) -> list[CodeBlock]:
    """Fenced code blocks in document order (same rule the renderer uses)."""
    return [CodeBlock(m.group(1).strip(), m.group(2)) for m in _CODE_BLOCK.finditer(text)]


def close_open_fence(text: str) -> str:
    """A stopped/streamed answer may end inside a code fence; close it so it renders."""
    if text.count("```") % 2 == 1:
        return text.rstrip("\n") + "\n```"
    return text


def markdown_to_html(text: str, block_actions: Callable[[int], str] | None = None) -> str:
    """
    Convert a markdown string to an HTML fragment (no <html>/<body> wrapper).
    Code blocks are syntax-highlighted; prose gets basic inline formatting.

    ``block_actions(i)`` may return extra HTML (e.g. Copy / Apply links) that is
    placed under the i-th code block.
    """
    parts: list[str] = []
    last = 0

    for idx, m in enumerate(_CODE_BLOCK.finditer(text)):
        # Prose before this code block
        before = text[last:m.start()]
        if before.strip():
            parts.append(_prose_to_html(before))

        # Syntax-highlighted code block
        lang = m.group(1).strip()
        code = m.group(2)
        parts.append(_highlight_block(code, lang))
        if block_actions:
            extra = block_actions(idx)
            if extra:
                parts.append(f'<p class="actions">{extra}</p>')
        last = m.end()

    # Remaining prose after the last code block
    tail = text[last:]
    if tail.strip():
        parts.append(_prose_to_html(tail))

    return "".join(parts)


def thinking_html(text: str) -> str:
    """Dimmed block for a reasoning model's chain of thought."""
    esc = html.escape(text.strip()).replace("\n", "<br>")
    return (f'<div class="thinking"><span style="color:#7a7aa8;font-size:11px;">💭 Thinking</span><br>'
            f'<span style="color:#8a8ab0;font-style:italic;">{esc}</span></div>')


def image_html(url: str, caption: str, open_href: str, width: int = 420) -> str:
    return (f'<p><img src="{html.escape(url)}" width="{width}"><br>'
            f'<span style="color:#9090b8;font-size:11px;">{html.escape(caption)}</span> &nbsp; '
            f'<a href="{html.escape(open_href)}" style="color:#8b7cf5;">Open / save…</a></p>')


def full_page_html(body: str) -> str:
    """Wrap an HTML body fragment in a complete <html> document with BraidLM CSS."""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
/* ── Base ── */
body {{
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 13px;
    color: #dde1e7;
    background: transparent;
    margin: 4px 0;
    padding: 0;
    line-height: 1.65;
}}
p   {{ margin: 0 0 8px; }}
ul, ol {{ margin: 4px 0 8px 20px; padding: 0; }}
li  {{ margin: 2px 0; }}
h1  {{ font-size: 1.4em; color: #8b7cf5; margin: 12px 0 4px; }}
h2  {{ font-size: 1.2em; color: #8b7cf5; margin: 10px 0 4px; }}
h3  {{ font-size: 1.1em; color: #8b7cf5; margin:  8px 0 4px; }}
h4, h5, h6 {{ font-size: 1.0em; color: #8b7cf5; margin: 6px 0 4px; }}
strong {{ color: #e8d4d8; }}
em     {{ color: #c8d4e8; font-style: italic; }}
code {{
    font-family: Consolas, 'Courier New', monospace;
    font-size: 12px;
    background: #2a2a4a;
    padding: 2px 5px;
    border-radius: 3px;
    color: #e8c8f0;
}}
/* ── Code blocks (Pygments) ── */
.code-block {{
    background: #272822;   /* Monokai */
    border-radius: 6px;
    margin: 8px 0;
    overflow-x: auto;
}}
.highlight {{ background: #272822; }}
.actions {{ margin: 0 0 10px; font-size: 11px; color: #636390; }}
.actions a {{ color: #8b7cf5; text-decoration: none; }}
.thinking {{ margin: 4px 0 8px; padding-left: 8px; border-left: 2px solid #3a3a60; }}
pre {{ margin: 0; padding: 10px 12px; overflow-x: auto; }}
/* ── Pygments token colours ── */
{_PYGMENTS_CSS}
</style>
</head>
<body>{body}</body>
</html>"""


# ── Internal helpers ────────────────────────────────────────────────────────

def _highlight_block(code: str, lang: str) -> str:
    """Return a <div class="code-block"> with Pygments-highlighted code."""
    try:
        lexer = get_lexer_by_name(lang or "text", stripall=True)
    except ClassNotFound:
        lexer = TextLexer(stripall=True)
    highlighted = highlight(code, lexer, _FMT)
    return f'<div class="code-block">{highlighted}</div>'


def _prose_to_html(text: str) -> str:
    """
    Convert markdown prose (no code blocks) to HTML.
    Order matters: escape HTML first, then apply markdown patterns.
    """
    # 1. HTML-escape raw text to prevent injection
    out = html.escape(text)

    # 2. Unordered lists (collect consecutive items into <ul>)
    out = _collect_list(out, _LIST_ITEM, "ul")

    # 3. Ordered lists
    out = _collect_list(out, _ORDERED_ITEM, "ol")

    # 4. Headers  (## Title → <h2>Title</h2>)
    out = _HEADER.sub(
        lambda m: f"<h{len(m.group(1))}>{m.group(2)}</h{len(m.group(1))}>",
        out,
    )

    # 5. Inline formatting (applied to escaped HTML, so group 1 contains
    #    already-escaped text which is safe to embed in tags)
    out = _BOLD       .sub(r"<strong>\1</strong>", out)
    out = _ITALIC     .sub(r"<em>\1</em>",         out)
    out = _INLINE_CODE.sub(r"<code>\1</code>",     out)

    # 6. Paragraphs: split on blank lines, convert single \n to <br>
    paragraphs = re.split(r"\n{2,}", out)
    paras_html = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        # Don't double-wrap already-wrapped block elements
        if re.match(r"^<(h\d|ul|ol|li)", p):
            paras_html.append(p)
        else:
            paras_html.append(f"<p>{p.replace(chr(10), '<br>')}</p>")

    return "\n".join(paras_html)


def _collect_list(text: str, pattern: re.Pattern, tag: str) -> str:
    """
    Find runs of list items matching `pattern` and wrap them in <ul> or <ol>.
    Items between non-matching lines are left as-is.
    """
    lines  = text.split("\n")
    result = []
    in_list = False

    for line in lines:
        m = pattern.match(line)
        if m:
            if not in_list:
                result.append(f"<{tag}>")
                in_list = True
            result.append(f"<li>{m.group(1)}</li>")
        else:
            if in_list:
                result.append(f"</{tag}>")
                in_list = False
            result.append(line)

    if in_list:
        result.append(f"</{tag}>")

    return "\n".join(result)
