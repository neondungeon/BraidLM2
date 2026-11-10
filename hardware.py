"""
hardware.py: what can this machine actually run?
================================================
Detects RAM / GPU memory (stdlib only, no psutil) and answers two questions
that Ollama's own ``pull`` does not ask *before* you spend an hour downloading:

  1. Will a model file of N bytes fit on this machine?          -> classify()
  2. How much context can I afford for a model I already have?  -> recommend_ctx()

All numbers are *estimates*.  They are deliberately conservative: it is much
better to say "partial offload" and be pleasantly surprised than the reverse.

Overrides (useful for AMD/Intel GPUs on Windows, which have no portable query):
    BRAIDLM_VRAM_GB=12      BRAIDLM_RAM_GB=32
"""
from __future__ import annotations

import ctypes
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

GiB = 1024 ** 3
MiB = 1024 ** 2

CTX_LADDER = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]


@dataclass
class Gpu:
    name: str
    total: int           # bytes
    free: int = 0


@dataclass
class Hardware:
    ram_total: int
    ram_free: int
    gpus: list = field(default_factory=list)
    unified: bool = False            # Apple Silicon: GPU shares system RAM
    note: str = ""

    @property
    def vram_total(self) -> int:
        return sum(g.total for g in self.gpus)

    def accel_budget(self) -> int:
        """Bytes usable for weights + KV cache on the accelerator."""
        if self.unified:
            return int(self.ram_total * (0.75 if self.ram_total >= 36 * GiB else 0.66))
        return int(self.vram_total * 0.92)       # headroom for driver/display

    def ram_budget(self) -> int:
        """Bytes of system RAM Ollama can reasonably spill into."""
        return max(0, int(self.ram_total * 0.85) - 2 * GiB)

    def describe(self) -> str:
        parts = []
        if self.unified:
            parts.append(f"Apple Silicon, {self.ram_total / GiB:.0f} GB unified memory")
        else:
            if self.gpus:
                parts.append(", ".join(f"{g.name} ({g.total / GiB:.0f} GB)" for g in self.gpus))
            else:
                parts.append("No GPU detected")
            parts.append(f"{self.ram_total / GiB:.0f} GB RAM")
        if self.note:
            parts.append(self.note)
        return "  ·  ".join(parts)


# ── Detection ───────────────────────────────────────────────────────────────

def _ram_linux() -> tuple[int, int]:
    total = free = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            total = int(line.split()[1]) * 1024
        elif line.startswith("MemAvailable:"):
            free = int(line.split()[1]) * 1024
    return total, free


def _ram_windows() -> tuple[int, int]:
    class MEMSTAT(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    st = MEMSTAT()
    st.dwLength = ctypes.sizeof(MEMSTAT)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))   # type: ignore[attr-defined]
    return int(st.ullTotalPhys), int(st.ullAvailPhys)


def _ram_macos(run) -> tuple[int, int]:
    total = int(run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout.strip())
    free = total // 2
    try:
        out = run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page = int(re.search(r"page size of (\d+)", out).group(1))
        pages = sum(int(re.search(rf"{k}:\s+(\d+)", out).group(1))
                    for k in ("Pages free", "Pages inactive", "Pages speculative"))
        free = pages * page
    except Exception:
        pass
    return total, free


def _nvidia(run) -> list[Gpu]:
    try:
        out = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                   "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=6)
    except Exception:
        return []
    if getattr(out, "returncode", 1) != 0:
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[1].isdigit():
            gpus.append(Gpu(parts[0], int(parts[1]) * MiB, int(parts[2]) * MiB if parts[2].isdigit() else 0))
    return gpus


def _amdgpu_sysfs(root: str = "/sys/class/drm") -> list[Gpu]:
    gpus = []
    try:
        for card in sorted(Path(root).glob("card[0-9]*")):
            tot = card / "device" / "mem_info_vram_total"
            if tot.exists():
                total = int(tot.read_text().strip())
                used_f = card / "device" / "mem_info_vram_used"
                used = int(used_f.read_text().strip()) if used_f.exists() else 0
                if total > 512 * MiB:
                    gpus.append(Gpu(f"AMD GPU ({card.name})", total, max(0, total - used)))
    except (OSError, ValueError):
        pass
    return gpus


def detect(run=subprocess.run, plat: str | None = None, env=None) -> Hardware:
    plat = plat or sys.platform
    env = os.environ if env is None else env
    try:
        if plat.startswith("linux"):
            total, free = _ram_linux()
        elif plat == "win32":
            total, free = _ram_windows()
        elif plat == "darwin":
            total, free = _ram_macos(run)
        else:
            total, free = 8 * GiB, 4 * GiB
    except Exception:
        total, free = 8 * GiB, 4 * GiB

    unified = plat == "darwin" and platform.machine().lower() in ("arm64", "aarch64")
    gpus: list[Gpu] = [] if unified else _nvidia(run)
    if not gpus and not unified and plat.startswith("linux"):
        gpus = _amdgpu_sysfs()

    note = ""
    if env.get("BRAIDLM_RAM_GB"):
        try:
            total = int(float(env["BRAIDLM_RAM_GB"]) * GiB)
            free = min(free, total)
        except ValueError:
            pass
    if env.get("BRAIDLM_VRAM_GB"):
        try:
            v = int(float(env["BRAIDLM_VRAM_GB"]) * GiB)
            gpus = [Gpu("GPU (manual)", v, v)]
            unified = False
            note = "VRAM set manually"
        except ValueError:
            pass
    return Hardware(ram_total=total, ram_free=free, gpus=gpus, unified=unified, note=note)


# ── Fit estimation ──────────────────────────────────────────────────────────

@dataclass
class Fit:
    level: str          # "gpu" | "partial" | "cpu" | "no"
    needed: int         # bytes
    label: str
    detail: str

    @property
    def rank(self) -> int:
        return {"gpu": 3, "cpu": 2, "partial": 2, "no": 0}[self.level]


def fallback_kv_per_token(weights: int) -> int:
    """Conservative KV-cache cost when the architecture is unknown (~8B Q4 -> 120 KB)."""
    return max(32 * 1024, weights // 40000)


def kv_cache_bytes(ctx: int, kv_per_token: int, weights: int = 0) -> int:
    return ctx * (kv_per_token or fallback_kv_per_token(weights))


def estimate_needed(weights: int, ctx: int = 4096, kv_per_token: int = 0) -> int:
    """Weights + KV cache at ``ctx`` + ~0.5 GB of runtime buffers."""
    return int(weights * 1.05 + kv_cache_bytes(ctx, kv_per_token, weights) + 512 * MiB)


def classify(weights: int, hw: Hardware, ctx: int = 4096, kv_per_token: int = 0) -> Fit:
    need = estimate_needed(weights, ctx, kv_per_token)
    accel, ram = hw.accel_budget(), hw.ram_budget()
    gb = need / GiB
    if hw.unified:
        if need <= accel:
            return Fit("gpu", need, "Fits (unified memory)", f"needs ≈{gb:.1f} GB of {accel / GiB:.0f} GB usable")
        if need <= ram:
            return Fit("partial", need, "Tight (may swap)", f"needs ≈{gb:.1f} GB; GPU limit is {accel / GiB:.0f} GB")
        return Fit("no", need, "Too large", f"needs ≈{gb:.1f} GB, machine has {hw.ram_total / GiB:.0f} GB")
    if accel > 0:
        if need <= accel:
            return Fit("gpu", need, "Fits in GPU", f"needs ≈{gb:.1f} GB of {accel / GiB:.0f} GB VRAM")
        if need <= accel + ram:
            return Fit("partial", need, "Partial GPU offload (slower)",
                       f"needs ≈{gb:.1f} GB; {accel / GiB:.0f} GB VRAM + RAM spill")
        return Fit("no", need, "Too large", f"needs ≈{gb:.1f} GB, VRAM+RAM is {(accel + ram) / GiB:.0f} GB")
    if need <= ram:
        return Fit("cpu", need, "CPU only (slow)", f"needs ≈{gb:.1f} GB of {ram / GiB:.0f} GB usable RAM")
    return Fit("no", need, "Too large", f"needs ≈{gb:.1f} GB, usable RAM is {ram / GiB:.0f} GB")


def recommend_ctx(weights: int, kv_per_token: int, ctx_max: int, hw: Hardware,
                  soft_cap: int = 32768) -> int:
    """Largest ladder context whose weights + KV cache fit the accelerator.

    * never exceeds the model's trained limit (``ctx_max``) or ``soft_cap``
    * if the weights alone already spill out of VRAM, stays modest (8k): a big
      KV cache in system RAM makes every token slower
    """
    cap = min(ctx_max or soft_cap, soft_cap)
    accel, ram = hw.accel_budget(), hw.ram_budget()
    if accel > 0:
        budget = accel
        if estimate_needed(weights, 2048, kv_per_token) > accel:
            cap = min(cap, 8192)
            budget = accel + ram
    else:
        budget = int(ram * 0.8)
        cap = min(cap, 16384)
    best = CTX_LADDER[0]
    for c in CTX_LADDER:
        if c > cap:
            break
        if estimate_needed(weights, c, kv_per_token) <= budget:
            best = c
    return best


def next_ctx(current: int, ceiling: int) -> int:
    """Next ladder step above ``current`` (or ``current`` if already at ceiling)."""
    for c in CTX_LADDER:
        if c > current:
            return min(c, ceiling) if ceiling >= c else current
    return current
