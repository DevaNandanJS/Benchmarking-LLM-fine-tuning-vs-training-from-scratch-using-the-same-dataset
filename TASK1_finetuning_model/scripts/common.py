"""Shared helpers for Track 1 (fine-tuning) scripts.

Encodes the global conventions from plan/finetuning_execution_plan.md §0:
  - fixed seed, logged in every run config
  - config-as-file before training (configs/run_<name>.json)
  - metrics.jsonl logging format for every training run
"""
from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── UTF-8 stdout/stderr reconfiguration ─────────────────────────────────────
# Ensures print() with non-ASCII characters (e.g. arrows in log messages)
# never crashes with UnicodeEncodeError on Windows default console (CP1252)
# or other non-UTF-8 environments. No-ops silently on streams that don't
# support reconfigure (e.g. piped/redirected output in some CI environments).
for _stream in (sys.stdout, sys.stderr):
    try:
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SEED = 42

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACK_DIR = Path(__file__).resolve().parents[1]



def set_seed(seed: int = SEED) -> int:
    """Determinism convention: seed python/numpy/torch, return the seed used."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


def path(rel: str) -> Path:
    """Resolve a path relative to TASK1_finetuning_model/ (robust to CWD)."""
    return TRACK_DIR / rel


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_config(config: dict, run_name: str) -> Path:
    """Config-as-file convention: write configs/run_<run_name>.json before training."""
    cfg = dict(config)
    cfg.setdefault("seed", SEED)
    out = TRACK_DIR / "configs" / f"run_{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return out


class MetricsLogger:
    """Append-only metrics.jsonl writer (per-run), §0 logging format.

    One JSON object per logged step:
    {"step": int, "epoch": float, "train_loss": float,
     "val_loss": float|null, "lr": float|null, "timestamp": iso8601}
    """

    def __init__(self, run_name: str):
        self.path = TRACK_DIR / "logs" / run_name / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def log(
        self,
        *,
        step: int,
        epoch: float,
        train_loss: float,
        val_loss: float | None = None,
        lr: float | None = None,
    ) -> None:
        rec = {
            "step": int(step),
            "epoch": round(float(epoch), 4),
            "train_loss": round(float(train_loss), 6),
            "val_loss": None if val_loss is None else round(float(val_loss), 6),
            "lr": None if lr is None else float(lr),
            "timestamp": iso_now(),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
