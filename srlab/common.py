"""Shared paths, hashing and JSON helpers. Every artifact written by the lab goes through here
so that provenance (content hashes, config hashes) is computed the same way everywhere."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ml-latest-small"
PROC = ROOT / "data" / "processed"
ART = ROOT / "artifacts"
CKPT = ART / "checkpoints"
CASES = ART / "cases"
TRACES = ART / "traces"
REPORTS = ART / "reports"
for _p in (PROC, ART, CKPT, CASES, TRACES, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

GLOBAL_SEED = 20261022


def sha256_file(path: Path | str, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def sha256_obj(obj: Any, n: int = 16) -> str:
    s = json.dumps(obj, sort_keys=True, default=_default, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def _default(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serializable: {type(o)}")


def dump_json(obj: Any, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def load_json(path: Path | str) -> Any:
    with open(path) as f:
        return json.load(f)


def hardware() -> dict:
    import torch
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
    }


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
