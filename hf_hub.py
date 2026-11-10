"""
hf_hub.py: browse, size-up and download open models from Hugging Face
=====================================================================
Two kinds of model are supported:

  * "gguf"   language / vision-language models.  Installed by handing Ollama a
             ``hf.co/<repo>:<quant>`` reference (Ollama pulls it itself).  We add
             what ``ollama pull`` does not: search, per-quantisation file sizes
             and a does-it-fit-my-machine verdict *before* anything downloads.
  * "image"  text-to-image models in diffusers format.  Downloaded here (with
             resume + progress) and run by imagegen.py.

Only the public Hub HTTP API is used.  Set ``HF_ENDPOINT`` to use a mirror.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import httpx

import hardware as hw_mod


class HubError(RuntimeError):
    pass


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


# ── parsing ─────────────────────────────────────────────────────────────────

_QUANT_RE = re.compile(
    r"(?<![A-Za-z0-9])((?:UD-)?(?:IQ\d(?:_[A-Z]+)*|Q\d(?:_K)?(?:_[A-Z0-9]+)?|MXFP4(?:_MOE)?|BF16|F16|F32))"
    r"(?![A-Za-z0-9])", re.I)
_SHARD_RE = re.compile(r"-\d{5}-of-\d{5}")

QUANT_PREFERENCE = ["Q4_K_M", "Q5_K_M", "Q4_K_S", "Q6_K", "Q4_0", "IQ4_XS", "Q5_K_S", "Q8_0",
                    "Q3_K_M", "IQ3_M", "Q3_K_L", "Q3_K_S", "IQ3_XS", "Q2_K", "IQ2_M", "F16", "BF16"]


def extract_quant(filename: str) -> str | None:
    base = Path(filename).name
    base = re.sub(r"\.gguf$", "", base, flags=re.I)
    base = _SHARD_RE.sub("", base)
    found = _QUANT_RE.findall(base)
    return found[-1].upper() if found else None


def parse_repo_input(text: str) -> tuple[str, str | None] | None:
    """Accepts 'user/repo', 'hf.co/user/repo:Q4_K_M', a huggingface.co URL or an
    'ollama run hf.co/...' command.  Returns (repo_id, quant|None) or None."""
    t = text.strip()
    t = re.sub(r"^ollama\s+(run|pull)\s+", "", t, flags=re.I)
    t = re.sub(r"^https?://", "", t)
    t = re.sub(r"^(www\.)?(hf\.co|huggingface\.co)/", "", t, flags=re.I)
    t = t.split("?")[0].split("#")[0]
    quant = None
    if ":" in t:
        t, quant = t.rsplit(":", 1)
        quant = quant.strip() or None
    t = re.sub(r"/(tree|blob|resolve)/.*$", "", t).strip("/")
    if re.fullmatch(r"[\w.\-]+/[\w.\-]+", t):
        return t, quant
    return None


# ── data ────────────────────────────────────────────────────────────────────

@dataclass
class RepoSummary:
    repo_id: str
    downloads: int = 0
    likes: int = 0
    pipeline_tag: str = ""
    tags: list = field(default_factory=list)
    gated: bool = False
    last_modified: str = ""


@dataclass
class GgufVariant:
    quant: str                    # label used in the hf.co/<repo>:<quant> reference
    size: int                     # bytes (all shards)
    files: list = field(default_factory=list)

    @property
    def ref_suffix(self) -> str:
        return "" if self.quant == "(default)" else f":{self.quant}"


@dataclass
class GgufRepo:
    repo_id: str
    variants: list
    has_mmproj: bool = False


@dataclass
class DiffusersPlan:
    repo_id: str
    files: list                    # [(path, size)]
    variant: str | None            # "fp16" or None
    total: int


def ollama_ref(repo_id: str, variant: GgufVariant | None) -> str:
    return f"hf.co/{repo_id}" + (variant.ref_suffix if variant else "")


# ── choosing ────────────────────────────────────────────────────────────────

def recommend_variant(variants: list, hw: hw_mod.Hardware, ctx: int = 4096) -> int:
    """Index of the variant to pre-select: among the best-fitting ones, the
    highest-quality by a sensible preference order (Q4_K_M first, like Ollama)."""
    if not variants:
        return -1
    fits = [hw_mod.classify(v.size, hw, ctx) for v in variants]
    best = max(f.rank for f in fits)
    cands = [i for i, f in enumerate(fits) if f.rank == best]

    def key(i: int):
        q = variants[i].quant.replace("UD-", "")
        pref = QUANT_PREFERENCE.index(q) if q in QUANT_PREFERENCE else len(QUANT_PREFERENCE)
        return (pref, -variants[i].size)
    return min(cands, key=key)


_DIFF_SKIP_EXT = (".bin", ".ckpt", ".pt", ".pth", ".onnx", ".msgpack", ".h5", ".ot", ".gguf",
                  ".png", ".jpg", ".jpeg", ".webp", ".gif", ".md", ".mp4", ".pdf")


def select_diffusers_files(tree: list[dict]) -> tuple[list, str | None]:
    """Pick the files a diffusers pipeline needs from a repo listing.

    Skips duplicate formats (.bin/.ckpt/onnx/flax), repo-root single-file
    checkpoints, and picks the fp16 variant when the repo ships one."""
    files = [(e["path"], int((e.get("lfs") or {}).get("size") or e.get("size") or 0))
             for e in tree if e.get("type") == "file"]
    paths = {p for p, _ in files}
    if "model_index.json" not in paths:
        raise HubError("This repository is not in diffusers format (no model_index.json).")
    has_fp16 = any(".fp16." in p and p.endswith(".safetensors") for p in paths)
    out = []
    for p, s in files:
        low = p.lower()
        if low.startswith((".git", "onnx/", "openvino", "flax_")) or "/onnx/" in low:
            continue
        if low.endswith(_DIFF_SKIP_EXT):
            continue
        if low.endswith(".safetensors"):
            if "/" not in p:                       # single-file checkpoint at repo root: not needed
                continue
            folder = p.rsplit("/", 1)[0] + "/"
            folder_has_fp16 = any(q.startswith(folder) and ".fp16." in q and q.endswith(".safetensors")
                                  for q in paths)
            # per folder: if an fp16 build exists keep only that, otherwise keep the plain weights
            if folder_has_fp16 and ".fp16." not in p:
                continue
            out.append((p, s))
        else:
            out.append((p, s))
    return out, ("fp16" if has_fp16 else None)


# ── client ──────────────────────────────────────────────────────────────────

class HubClient:
    def __init__(self, token: str | None = None, base: str | None = None) -> None:
        self.base = (base or endpoint()).rstrip("/")
        self.token = token or os.environ.get("HF_TOKEN") or None

    def _headers(self) -> dict:
        h = {"User-Agent": "BraidLM/2.0"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _get(self, url: str, params=None, timeout: float = 20.0) -> httpx.Response:
        try:
            r = httpx.get(url, params=params, headers=self._headers(), timeout=timeout,
                          follow_redirects=True)
        except httpx.HTTPError as exc:
            raise HubError(f"Could not reach Hugging Face: {exc}") from exc
        if r.status_code in (401, 403):
            raise HubError("Hugging Face denied access (gated or private model). Accept the model's "
                           "license on huggingface.co and provide an access token.")
        if r.status_code == 404:
            raise HubError("Not found on Hugging Face.")
        if r.status_code >= 400:
            raise HubError(f"Hugging Face returned HTTP {r.status_code}.")
        return r

    # -- search --------------------------------------------------------------
    def search(self, query: str, kind: str = "gguf", limit: int = 30, sort: str = "downloads"
               ) -> list[RepoSummary]:
        params: list[tuple[str, str]] = [
            ("limit", str(limit)), ("sort", sort), ("direction", "-1"),
        ]
        if query.strip():
            params.append(("search", query.strip()))
        if kind == "image":
            params += [("filter", "diffusers"), ("pipeline_tag", "text-to-image")]
        else:
            params.append(("filter", "gguf"))
        for f in ("downloads", "likes", "pipeline_tag", "tags", "gated", "lastModified"):
            params.append(("expand[]", f))
        data = self._get(f"{self.base}/api/models", params).json()
        out = []
        for m in data if isinstance(data, list) else []:
            rid = m.get("id") or m.get("modelId")
            if not rid:
                continue
            out.append(RepoSummary(rid, int(m.get("downloads") or 0), int(m.get("likes") or 0),
                                   m.get("pipeline_tag") or "", list(m.get("tags") or []),
                                   bool(m.get("gated")), str(m.get("lastModified") or "")))
        return out

    # -- file listings -------------------------------------------------------
    def tree(self, repo_id: str) -> list[dict]:
        url = f"{self.base}/api/models/{repo_id}/tree/main"
        params: dict | None = {"recursive": "true"}
        entries: list[dict] = []
        for _ in range(50):                                    # follow pagination
            r = self._get(url, params)
            entries += r.json()
            nxt = r.links.get("next", {}).get("url")
            if not nxt:
                break
            url, params = nxt, None
        return entries

    def gguf_repo(self, repo_id: str) -> GgufRepo:
        entries = self.tree(repo_id)
        groups: dict[str, GgufVariant] = {}
        mmproj = False
        for e in entries:
            p = e.get("path", "")
            if e.get("type") != "file" or not p.lower().endswith(".gguf"):
                continue
            name = Path(p).name.lower()
            if name.startswith("mmproj") or "mmproj" in name:
                mmproj = True
                continue
            size = int((e.get("lfs") or {}).get("size") or e.get("size") or 0)
            q = extract_quant(p) or "(default)"
            g = groups.setdefault(q, GgufVariant(q, 0, []))
            g.size += size
            g.files.append(p)
        variants = sorted(groups.values(), key=lambda v: v.size)
        if not variants:
            raise HubError("No GGUF files found in this repository.")
        return GgufRepo(repo_id, variants, mmproj)

    def diffusers_plan(self, repo_id: str) -> DiffusersPlan:
        files, variant = select_diffusers_files(self.tree(repo_id))
        return DiffusersPlan(repo_id, files, variant, sum(s for _, s in files))

    # -- download ------------------------------------------------------------
    def download_plan(self, plan: DiffusersPlan, dest: Path, progress=None,
                      cancel: threading.Event | None = None) -> None:
        """Download every file of ``plan`` into ``dest`` (resumable).

        ``progress(done_bytes, total_bytes, filename)`` is called regularly."""
        cancel = cancel or threading.Event()
        dest.mkdir(parents=True, exist_ok=True)
        total = plan.total or sum(s for _, s in plan.files)
        done = 0
        for path, size in plan.files:
            if cancel.is_set():
                raise HubError("Download cancelled.")
            target = dest / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and (not size or target.stat().st_size == size):
                done += size or target.stat().st_size
                if progress:
                    progress(done, total, path)
                continue
            done = self._download_file(plan.repo_id, path, size, target, done, total, progress, cancel)
        (dest / "braidlm_meta.json").write_text(json.dumps({
            "repo_id": plan.repo_id, "variant": plan.variant, "size": total,
            "installed_at": int(time.time())}), encoding="utf-8")

    def _download_file(self, repo_id, path, size, target: Path, done, total, progress, cancel) -> int:
        url = f"{self.base}/{repo_id}/resolve/main/{quote(path)}"
        part = target.with_name(target.name + ".part")
        for attempt in range(4):
            have = part.stat().st_size if part.exists() else 0
            headers = self._headers()
            if have:
                headers["Range"] = f"bytes={have}-"
            try:
                with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                                  timeout=httpx.Timeout(30.0, read=60.0)) as r:
                    if r.status_code in (401, 403):
                        raise HubError("Hugging Face denied access: this model is gated. Accept its license "
                                       "on huggingface.co and paste an access token.")
                    if r.status_code == 416:                 # part file already complete
                        break
                    if r.status_code >= 400:
                        raise HubError(f"Download failed for {path}: HTTP {r.status_code}")
                    mode = "ab" if (have and r.status_code == 206) else "wb"
                    if mode == "wb":
                        have = 0
                    base_done = done + have
                    got = have
                    with open(part, mode) as f:
                        for chunk in r.iter_bytes(1024 * 1024):
                            if cancel.is_set():
                                raise HubError("Download cancelled.")
                            f.write(chunk)
                            got += len(chunk)
                            if progress:
                                progress(base_done + (got - have), total, path)
                break
            except HubError:
                raise
            except httpx.HTTPError as exc:
                if attempt == 3:
                    raise HubError(f"Network error while downloading {path}: {exc}") from exc
                time.sleep(1.5 * (attempt + 1))
        if size and part.stat().st_size != size:
            raise HubError(f"Size mismatch for {path} (got {part.stat().st_size}, expected {size}).")
        part.replace(target)
        return done + (size or target.stat().st_size)
