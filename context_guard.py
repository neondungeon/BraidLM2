"""
context_guard.py: never let Ollama silently truncate a conversation
====================================================================
THE PROBLEM (documented in ollama/ollama issues #14259, #4967, #3839 and
re-measured on v0.34-0.35):  when a prompt is longer than the context window,
Ollama drops text from the *front* of the prompt (system prompt first), answers
HTTP 200, and tells the caller nothing -- the response has no "truncated" field
and ``prompt_eval_count`` reports the post-truncation number.  The model then
confidently answers from whatever is left.

WHAT BRAIDLM DOES INSTEAD
  1. Estimates the size of every request *before* sending it (conservative).
  2. If it will not fit: first raises ``num_ctx`` (if the hardware allows),
     then compacts the oldest turns into a summary, then trims an oversized
     attachment in the middle -- and always tells the user what it did.
  3. After each reply, checks Ollama's own token counts for the signature of
     truncation (a prompt cut to ~half the window) and warns loudly.
  4. Learns from Ollama's real token counts so the estimate gets safer.

Everything here is pure Python (no Qt, no network) so it is unit-testable.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

from hardware import CTX_LADDER

IMAGE_TOKENS = 1200          # conservative cost of one attached image
MSG_OVERHEAD = 8             # chat-template tokens per message


class TokenEstimator:
    """Cheap, deliberately pessimistic token counter.

    English averages ~4 chars/token and code ~3; we assume 3.2 for ASCII and
    one token per non-ASCII character (CJK etc.).  ``factor`` is raised whenever
    Ollama reports more tokens than we predicted, so the guard only ever gets
    *safer* during a session."""

    ASCII_CPT = 3.2

    def __init__(self) -> None:
        self.factor = 1.0

    def text(self, s: str) -> int:
        if not s:
            return 0
        if s.isascii():
            raw = len(s) / self.ASCII_CPT
        else:
            n_ascii = len(s.encode("ascii", "ignore"))
            raw = n_ascii / self.ASCII_CPT + (len(s) - n_ascii)
        return int(math.ceil(raw * self.factor))

    def message(self, m: dict) -> int:
        return (self.text(m.get("content") or "") + MSG_OVERHEAD
                + IMAGE_TOKENS * len(m.get("images") or []))

    def total(self, msgs: list[dict]) -> int:
        return sum(self.message(m) for m in msgs)

    def observe(self, estimated: int, actual: int) -> None:
        """Feed back Ollama's real prompt token count (only ever raises the factor)."""
        if estimated > 0 and actual > estimated:
            self.factor = min(3.0, self.factor * min(actual / estimated, 2.0))


def reply_reserve(ctx: int) -> int:
    """Tokens kept free for the model's answer."""
    return min(max(384, ctx // 6), 2048)


def prompt_budget(ctx: int) -> int:
    return max(256, ctx - reply_reserve(ctx))


def detect_truncation(prompt_eval_count: int, est_sent: int, ctx: int) -> bool:
    """True if Ollama's numbers show the prompt was cut.

    Measured signature (v0.34/0.35): an over-long prompt is cut to about half the
    window (+2..3 tokens): 4096 -> 2050, 32768 -> 16387, 65536 -> 32770.
    ``est_sent`` is our own pessimistic estimate of what we sent."""
    if prompt_eval_count <= 0 or ctx <= 0:
        return False
    signature = abs(prompt_eval_count - (ctx // 2 + 2)) <= 5
    if signature and est_sent > prompt_eval_count * 1.15:
        return True
    return est_sent > ctx * 0.98 and prompt_eval_count < est_sent * 0.8


# ── text helpers ────────────────────────────────────────────────────────────

_ATTACH_BLOCK = re.compile(r'<attached_file name="([^"]*)"[^>]*>.*?</attached_file>', re.S)


def strip_attachment_blocks(text: str) -> str:
    return _ATTACH_BLOCK.sub(lambda m: f"[attached file {m.group(1)} - content omitted]", text)


def trim_middle(text: str, max_tokens: int, est: TokenEstimator) -> tuple[str, bool]:
    """Shrink ``text`` to about ``max_tokens`` by removing the middle (keeps head + tail)."""
    if est.text(text) <= max_tokens:
        return text, False
    per_tok = max(est.text(text), 1) / max(len(text), 1)          # tokens per char
    keep_chars = max(200, int(max_tokens / per_tok * 0.92))
    head = int(keep_chars * 0.65)
    tail = keep_chars - head
    removed = len(text) - head - tail
    marker = (f"\n\n[… BraidLM removed {removed:,} characters from the middle of this text "
              f"so it fits the model's context window …]\n\n")
    return text[:head] + marker + text[len(text) - tail:], True


def split_for_compaction(msgs: list[dict], est: TokenEstimator, keep_tokens: int
                         ) -> tuple[list[dict], list[dict]]:
    """Split ``msgs`` into (old, keep).  ``keep`` is the newest tail that fits in
    ``keep_tokens`` and always starts on a user turn; the last message is always kept."""
    if not msgs:
        return [], []
    total, i = 0, len(msgs)
    while i > 0:
        t = est.message(msgs[i - 1])
        if total + t > keep_tokens and i < len(msgs):
            break
        total += t
        i -= 1
    while i < len(msgs) - 1 and msgs[i].get("role") != "user":
        i += 1
    return msgs[:i], msgs[i:]


SUMMARY_SYSTEM = (
    "You compress chat transcripts. Write a faithful, dense summary that preserves: "
    "the user's goals, decisions made, facts, names, numbers, file names, code identifiers, "
    "constraints and unresolved questions. Do not add anything new. "
    "Output only the summary, as short bullet points."
)


def transcript(msgs: list[dict]) -> str:
    lines = []
    for m in msgs:
        who = "User" if m.get("role") == "user" else "Assistant"
        body = strip_attachment_blocks(m.get("content") or "")
        if m.get("images"):
            body += f" [{len(m['images'])} image(s) attached - omitted]"
        lines.append(f"{who}: {body}")
    return "\n\n".join(lines)


def chunk_transcript(msgs: list[dict], est: TokenEstimator, max_tokens: int) -> list[str]:
    """Break old messages into transcript chunks that each fit ``max_tokens``."""
    chunks, cur, cur_t = [], [], 0
    for m in msgs:
        piece = transcript([m])
        t = est.text(piece)
        if t > max_tokens:                      # one huge message: trim it
            piece, _ = trim_middle(piece, max_tokens, est)
            t = est.text(piece)
        if cur and cur_t + t > max_tokens:
            chunks.append("\n\n".join(cur))
            cur, cur_t = [], 0
        cur.append(piece)
        cur_t += t
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


Summarizer = Callable[[str, str, int], str]     # (previous_summary, new_transcript, ctx) -> summary


def compact(prior_summary: str, old: list[dict], est: TokenEstimator, ctx: int,
            summarize: Summarizer) -> str:
    """Fold ``old`` messages (and any earlier summary) into one summary string."""
    summary = prior_summary
    for chunk in chunk_transcript(old, est, max(512, int(prompt_budget(ctx) * 0.5))):
        summary = summarize(summary, chunk, ctx).strip() or summary
    return summary


# ── the orchestrator ────────────────────────────────────────────────────────

@dataclass
class Prepared:
    messages: list[dict]                      # exactly what will be sent (incl. system)
    ctx: int                                  # num_ctx to request
    est_tokens: int                           # our estimate of ``messages``
    summary: str                              # (possibly new) running summary
    consumed: int                             # how many *input* msgs the summary now covers
    notices: list[str] = field(default_factory=list)
    compacted: bool = False


def _system_with_summary(system: str, summary: str) -> str:
    if not summary:
        return system
    return (f"{system}\n\nSummary of the earlier part of this conversation "
            f"(older messages were compacted to save context):\n{summary}")


def prepare(system: str, summary: str, msgs: list[dict], ctx: int, ctx_ceiling: int,
            est: TokenEstimator, summarize: Summarizer | None,
            force_compact: bool = False) -> Prepared:
    """Return a request guaranteed (by estimate) to fit the context window.

    ``msgs``     the not-yet-summarised conversation (API form: role/content[/images])
    ``summary``  running summary of messages already compacted away ('' if none)
    """
    notices: list[str] = []
    msgs = [dict(m) for m in msgs]
    consumed = 0
    compacted = False

    def build() -> list[dict]:
        return [{"role": "system", "content": _system_with_summary(system, summary)}] + msgs

    # 1. raise the context window if the hardware allows it
    need = est.total(build())
    start_ctx = ctx
    while need > prompt_budget(ctx) and ctx < ctx_ceiling:
        nxt = next((c for c in CTX_LADDER if c > ctx), ctx)
        nxt = min(nxt, ctx_ceiling)
        if nxt <= ctx:
            break
        ctx = nxt
    if ctx != start_ctx:
        notices.append(f"Context window raised {start_ctx:,} → {ctx:,} tokens to fit this request.")

    # 2. compact old turns
    if (need > prompt_budget(ctx) or force_compact) and len(msgs) > 1:
        keep_tokens = int(prompt_budget(ctx) * 0.45)
        old, keep = split_for_compaction(msgs, est, keep_tokens)
        if old and summarize is not None:
            summary = compact(summary, old, est, ctx, summarize)
            msgs = keep
            consumed = len(old)
            compacted = True
            notices.append(f"Compacted {len(old)} earlier message(s) into a summary "
                           f"(context was {need:,}/{ctx:,} tokens).")
        elif old:                                # no summariser available: drop, but say so
            msgs = keep
            consumed = len(old)
            compacted = True
            notices.append(f"Dropped {len(old)} oldest message(s) to fit the context window.")

    # 3. a single message (usually an attachment) is still too big: trim its middle
    need = est.total(build())
    if need > prompt_budget(ctx) and msgs:
        others = need - est.message(msgs[-1])
        room = max(256, prompt_budget(ctx) - others - MSG_OVERHEAD)
        last = msgs[-1]
        new_text, trimmed = trim_middle(last.get("content") or "", room, est)
        if trimmed:
            msgs[-1] = {**last, "content": new_text}
            notices.append("The newest message was too large for the context window, so its middle "
                           "section was removed. Pick a bigger context or attach a smaller file "
                           "for full coverage.")

    full = build()
    return Prepared(full, ctx, est.total(full), summary, consumed, notices, compacted)
