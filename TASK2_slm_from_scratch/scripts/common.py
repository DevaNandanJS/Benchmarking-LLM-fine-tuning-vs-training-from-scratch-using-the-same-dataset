"""Shared helpers for Track 2 (training from scratch) scripts.

Mirrors TASK1_finetuning_model/scripts/common.py exactly in interface but with
TRACK_DIR pointing to TASK2_slm_from_scratch/ — this is intentional and critical.
Do NOT import track1's common.py from here: its TRACK_DIR would resolve to
TASK1_finetuning_model/, routing all logs and configs to the wrong track.

Global conventions (plan/scratch_slm_execution_plan.md §0):
  - SEED = 42 — logged in every run config
  - dump_config() — write configs/run_<name>.json BEFORE training starts
  - MetricsLogger — one metrics.jsonl per run, one JSON object per step
"""
from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── UTF-8 stdout/stderr (Windows default console can be CP1252) ──────────────
for _stream in (sys.stdout, sys.stderr):
    try:
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SEED = 42

# Resolve upward: scripts/ → TASK2_slm_from_scratch/ → llm_task/
SCRIPTS_DIR: Path = Path(__file__).resolve().parent
TRACK_DIR:   Path = SCRIPTS_DIR.parent          # TASK2_slm_from_scratch/
REPO_ROOT:   Path = TRACK_DIR.parent            # llm_task/


def set_seed(seed: int = SEED) -> int:
    """Determinism convention: seed python/numpy/torch, return seed used."""
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
    """Resolve a path relative to TASK2_slm_from_scratch/ (robust to CWD)."""
    return TRACK_DIR / rel


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_config(config: dict, run_name: str) -> Path:
    """Config-as-file: write configs/run_<run_name>.json BEFORE training."""
    cfg = dict(config)
    cfg.setdefault("seed", SEED)
    out = TRACK_DIR / "configs" / f"run_{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"[config] Written: {out}")
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
            "step":       int(step),
            "epoch":      round(float(epoch), 4),
            "train_loss": round(float(train_loss), 6),
            "val_loss":   None if val_loss is None else round(float(val_loss), 6),
            "lr":         None if lr is None else float(lr),
            "timestamp":  iso_now(),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
