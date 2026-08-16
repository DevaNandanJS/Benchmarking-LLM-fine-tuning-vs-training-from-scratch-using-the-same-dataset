"""Phase 4 — Training Loop Implementation (Training from Scratch)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# \u2500\u2500 Bootstrap: make scripts/ importable regardless of CWD \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import SEED, MetricsLogger, dump_config, iso_now, set_seed  # noqa: E402
from config import (  # noqa: E402
    BEST_VAL_DIR,
    DATASET_STATS_JSON,
    EVAL_DIR,
    LAST_CKPT_DIR,
    SWEEP_RESULTS_CSV,
    TOKENIZER_DIR,
    TRAIN_PT,
    VAL_PT,
)
from model import GPT, GPTConfig  # noqa: E402

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# \u00a71 \u2014 Fixed training hyperparameters
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

BATCH_SIZE    = 16      # reduce to 8 or 4 if OOM on a T4 GPU
TOTAL_STEPS   = 2000    # generous upper bound; best-val checkpoint decides the real end
WARMUP_RATIO  = 0.05    # first 5% of steps: linear LR ramp-up from 0 to peak
MIN_LR_RATIO  = 0.1     # cosine decays to 10% of peak LR, not 0 (avoids dead zone)
MAX_GRAD_NORM = 1.0     # global gradient norm clipping \u2014 more important here than in
                         # Track 1 fine-tuning because we start from random init
WEIGHT_DECAY  = 0.1     # applied to 2D+ params only (see make_param_groups)
LOG_EVERY     = 25      # steps between train-loss-only metrics.jsonl entries
EVAL_EVERY    = 100     # steps between full val passes AND checkpoint-save checks
                         # (save check is nested inside eval \u2014 SAVE_EVERY eliminated)

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# \u00a72 \u2014 Sweep configurations
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

# Three named runs covering two sweep axes, one variable changed at a time:
#   small  \u2192 base:        architecture size (n_layer 4\u21926, n_embd 128\u2192192)
#   base   \u2192 base_highlr: learning rate (3e-4 \u2192 6e-4), architecture fixed
SWEEP_CONFIGS: dict[str, dict] = {
    "small": {
        "n_layer":       4,
        "n_embd":        128,
        "n_head":        4,
        "learning_rate": 3e-4,
        "sweep_axis":    "architecture",
    },
    "base": {
        "n_layer":       6,
        "n_embd":        192,
        "n_head":        4,
        "learning_rate": 3e-4,
        "sweep_axis":    "architecture",
    },
    "base_highlr": {
        "n_layer":       6,
        "n_embd":        192,
        "n_head":        4,
        "learning_rate": 6e-4,
        "sweep_axis":    "learning_rate",
    },
}

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# \u00a73 \u2014 Weight-decay parameter split
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

def make_param_groups(model: GPT) -> list[dict]:
    """Split parameters into weight-decay and no-weight-decay groups."""
    decay:    list = []
    no_decay: list = []
    decay_names:    list[str] = []
    no_decay_names: list[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:      # Linear.weight, Embedding.weight
            decay.append(param)
            decay_names.append(name)
        else:                      # biases (dim=1), LayerNorm weight/bias (dim=1)
            no_decay.append(param)
            no_decay_names.append(name)

    decay_count    = sum(p.numel() for p in decay)
    no_decay_count = sum(p.numel() for p in no_decay)
    print(f"[phase4] weight-decay group:    {len(decay_names):3d} tensors, {decay_count:,} params")
    print(f"[phase4] no-weight-decay group: {len(no_decay_names):3d} tensors, {no_decay_count:,} params")

    return [
        {"params": decay,    "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# \u00a74 \u2014 LR schedule: linear warmup + cosine decay
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

def get_lr_schedule(optimizer, total_steps: int, warmup_steps: int):
    """Return a LambdaLR with linear warmup then cosine decay to MIN_LR_RATIO."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear ramp: 0 \u2192 1 over warmup_steps
            return float(step) / float(max(1, warmup_steps))
        # Cosine decay: 1 \u2192 0 over remaining steps, floored at MIN_LR_RATIO
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1.0 \u2192 0.0
        return max(MIN_LR_RATIO, cosine)

    return LambdaLR(optimizer, lr_lambda)

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# \u00a75 \u2014 Validation pass
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

def evaluate(model: GPT, val_loader, device) -> float:
    """Full validation pass — returns mean cross-entropy loss over the val set."""
    import torch

    model.eval()
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids = input_ids.to(device)
            labels    = labels.to(device)
            _, loss   = model(input_ids, labels=labels)
            total_loss += loss.item()
            n_batches  += 1

    model.train()
    return total_loss / max(1, n_batches)

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# \u00a76 \u2014 Sweep CSV append
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

def append_sweep_row(
    run_name:        str,
    sweep_cfg:       dict,
    total_steps_run: int,
    final_train_loss: float,
    best_val_loss:   float,
    best_val_step:   int,
) -> None:
    """Append one row to eval/sweep_results.csv. Creates the file with header if it does not yet exist."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SWEEP_RESULTS_CSV.exists()
    with SWEEP_RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_name", "n_layer", "n_embd", "n_head", "learning_rate",
                "sweep_axis", "total_steps_run", "final_train_loss",
                "best_val_loss", "best_val_step", "timestamp",
            ])
        writer.writerow([
            run_name,
            sweep_cfg["n_layer"],
            sweep_cfg["n_embd"],
            sweep_cfg["n_head"],
            sweep_cfg["learning_rate"],
            sweep_cfg.get("sweep_axis", ""),
            total_steps_run,
            round(final_train_loss, 6),
            round(best_val_loss, 6),
            best_val_step,
            iso_now(),
        ])
    print(f"[phase4] sweep row appended \u2192 {SWEEP_RESULTS_CSV}")

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# \u00a77 \u2014 Runtime config reader (mirrors model.py \u00a76)
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

def _load_vocab_and_block_size() -> tuple[int, int]:
    """Read vocab_size and block_size from Phase 1/2 outputs."""
    vocab_size = 1024
    block_size = 256

    vocab_json = TOKENIZER_DIR / "vocab.json"
    if vocab_json.exists():
        with vocab_json.open(encoding="utf-8") as f:
            vocab_data = json.load(f)
        vocab_size = len(vocab_data)
        print(f"[phase4] vocab_size={vocab_size} (from {vocab_json})")
    else:
        print(
            f"[phase4] WARNING: {vocab_json} not found. "
            f"Falling back to vocab_size={vocab_size}. "
            "Run Phase 1 on Colab first for the real value."
        )

    if DATASET_STATS_JSON.exists():
        with DATASET_STATS_JSON.open(encoding="utf-8") as f:
            stats = json.load(f)
        block_size = stats.get("context_length", block_size)
        print(f"[phase4] block_size={block_size} (from {DATASET_STATS_JSON})")
    else:
        print(
            f"[phase4] WARNING: {DATASET_STATS_JSON} not found. "
            f"Falling back to block_size={block_size}. "
            "Run Phase 2 on Colab first for the real value."
        )

    return vocab_size, block_size

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# \u00a78 \u2014 Main
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

def main() -> None:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, TensorDataset

    # \u2500\u2500 Parse arguments \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    parser = argparse.ArgumentParser(description="Track 2 Phase 4 \u2014 Training Loop")
    parser.add_argument(
        "--run",
        choices=list(SWEEP_CONFIGS.keys()),
        default="base",
        help=(
            "Sweep config to run: small | base | base_highlr. "
            "small/base sweep architecture size; base/base_highlr sweep LR. "
            "Default: base."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Smoke-test mode: truncate to 4 chunks, run 5 steps on CPU. "
            "Exercises all code paths (including evaluate()) before Colab. "
            "Does not write checkpoints or update sweep_results.csv."
        ),
    )
    args = parser.parse_args()
    run_name  = args.run
    smoke     = args.smoke_test
    sweep_cfg = SWEEP_CONFIGS[run_name]

    set_seed(SEED)
    print(f"[phase4] run={run_name}  smoke={smoke}  seed={SEED}")

    # \u2500\u2500 Device + AMP setup \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if smoke:
        device  = torch.device("cpu")
        use_amp = False
        print("[phase4] SMOKE MODE \u2014 CPU fp32, 4 chunks, 5 steps")
    else:
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = device.type == "cuda"

    dtype = torch.float16 if use_amp else torch.float32
    print(f"[phase4] device={device}  dtype={dtype}  AMP={use_amp}")

    # \u2500\u2500 Read Phase 1/2 outputs for vocab_size and block_size \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    vocab_size, block_size = _load_vocab_and_block_size()

    # \u2500\u2500 Build GPTConfig from sweep entry + Phase 1/2 outputs \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    config = GPTConfig(
        vocab_size  = vocab_size,
        block_size  = block_size,
        n_layer     = sweep_cfg["n_layer"],
        n_head      = sweep_cfg["n_head"],
        n_embd      = sweep_cfg["n_embd"],
        dropout     = 0.1,
        bias        = True,
        tie_weights = True,
    )
    lr = sweep_cfg["learning_rate"]

    # Derived training schedule values
    batch_sz     = 2 if smoke else BATCH_SIZE
    total_steps  = 5 if smoke else TOTAL_STEPS
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))

    # \u2500\u2500 Dump run config BEFORE any compute (config-as-file convention \u00a70) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    run_cfg: dict = {
        "phase":            4,
        "run_name":         run_name,
        "sweep_axis":       sweep_cfg.get("sweep_axis", ""),
        "n_layer":          config.n_layer,
        "n_embd":           config.n_embd,
        "n_head":           config.n_head,
        "vocab_size":       config.vocab_size,
        "block_size":       config.block_size,
        "dropout":          config.dropout,
        "tie_weights":      config.tie_weights,
        "learning_rate":    lr,
        "batch_size":       batch_sz,
        "total_steps":      total_steps,
        "warmup_steps":     warmup_steps,
        "warmup_ratio":     WARMUP_RATIO,
        "min_lr_ratio":     MIN_LR_RATIO,
        "max_grad_norm":    MAX_GRAD_NORM,
        "weight_decay":     WEIGHT_DECAY,
        "log_every":        LOG_EVERY,
        "eval_every":       EVAL_EVERY,
        "batch_strategy":   "dataloader_shuffle",
        "amp":              use_amp,
        "smoke_test":       smoke,
        "seed":             SEED,
        "timestamp":        iso_now(),
    }
    if not smoke:
        dump_config(run_cfg, f"phase4_{run_name}")
        print(f"[phase4] config dumped to configs/run_phase4_{run_name}.json")

    # Load dataset tensors
    if smoke:
        import torch as _torch
        rng = _torch.Generator().manual_seed(SEED)
        n_chunks = 4
        # Random token IDs in [0, vocab_size); labels are input_ids shifted by 1
        _ids = _torch.randint(0, vocab_size, (n_chunks, block_size), generator=rng)
        train_input_ids = _ids.clone()
        train_labels    = _ids.clone()
        val_input_ids   = _ids.clone()
        val_labels      = _ids.clone()
        print(f"[phase4] SMOKE: synthetic {n_chunks} train / {n_chunks} val chunks "
              f"(vocab_size={vocab_size}, block_size={block_size})")
    else:
        if not TRAIN_PT.exists():
            raise FileNotFoundError(
                f"[phase4] {TRAIN_PT} not found. "
                "Run Phase 1 (train_tokenizer.py) and Phase 2 (build_dataset.py) "
                "on Colab first, then commit + push before re-running this script."
            )

        print(f"[phase4] loading tensors from {TRAIN_PT} / {VAL_PT} ...")
        train_data = torch.load(TRAIN_PT, weights_only=True)
        val_data   = torch.load(VAL_PT,   weights_only=True)

        train_input_ids: torch.Tensor = train_data["input_ids"]   # (N_train, block_size)
        train_labels:    torch.Tensor = train_data["labels"]       # (N_train, block_size)
        val_input_ids:   torch.Tensor = val_data["input_ids"]     # (N_val, block_size)
        val_labels:      torch.Tensor = val_data["labels"]         # (N_val, block_size)

    # DataLoader construction
    train_dataset = TensorDataset(train_input_ids, train_labels)
    val_dataset   = TensorDataset(val_input_ids,   val_labels)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_sz, shuffle=True, generator=g,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_sz, shuffle=False,
    )

    steps_per_epoch = max(1, len(train_loader))
    print(
        f"[phase4] train chunks={len(train_dataset)}  "
        f"batches/epoch={steps_per_epoch}  "
        f"val chunks={len(val_dataset)}"
    )

    # Cyclic stateful iterator: when exhausted (StopIteration), a new
    # iterator is created \u2014 which triggers the DataLoader's re-shuffle.
    # List wrapper avoids the nonlocal-assignment restriction in closures.
    _train_iter: list = [iter(train_loader)]

    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next (input_ids, labels) batch, cycling the loader."""
        try:
            batch = next(_train_iter[0])
        except StopIteration:
            _train_iter[0] = iter(train_loader)   # re-shuffle on new iterator
            batch = next(_train_iter[0])
        return batch[0], batch[1]                 # input_ids, labels

    # \u2500\u2500 Model \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    print(
        f"\n[phase4] building GPT: n_layer={config.n_layer}  "
        f"n_embd={config.n_embd}  n_head={config.n_head}  "
        f"vocab_size={config.vocab_size}  block_size={config.block_size}"
    )
    model = GPT(config).to(device)
    param_info = model.count_parameters()
    print(f"[phase4] total trainable params: {param_info['total']:,}")

    # \u2500\u2500 Optimizer with weight-decay split \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    param_groups = make_param_groups(model)
    optimizer    = AdamW(param_groups, lr=lr)

    # \u2500\u2500 LR schedule \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    scheduler = get_lr_schedule(optimizer, total_steps, warmup_steps)
    print(
        f"[phase4] total_steps={total_steps}  warmup_steps={warmup_steps}  "
        f"peak_lr={lr:.2e}  floor_lr={lr * MIN_LR_RATIO:.2e}"
    )

    # AMP GradScaler
    # enabled=use_amp: same gate as autocast below — cannot drift out of sync.
    # GradScaler(enabled=False) is a pass-through on CPU (no scaling, no crash).
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[phase4] AMP GradScaler: {'enabled (fp16)' if use_amp else 'disabled (CPU/fp32)'}")

    # Metrics logger
    logger = MetricsLogger(run_name)
    print(f"[phase4] metrics -> {logger.path}")
    print(f"[phase4] LOG_EVERY={LOG_EVERY} steps  EVAL_EVERY={EVAL_EVERY} steps")

    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    best_val_loss:   float = float("inf")
    best_val_step:   int   = -1
    val_loss:        float = float("inf")
    last_train_loss: float = float("nan")

    model.train()
    print(f"\n[phase4] starting training: {total_steps} steps ...\n")

    for step in range(1, total_steps + 1):

        input_ids, labels = get_batch()
        input_ids = input_ids.to(device)
        labels    = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # \u2500\u2500 Mixed-precision forward pass \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            _, loss = model(input_ids, labels=labels)

        if not math.isfinite(loss.item()):
            raise RuntimeError(
                f"[phase4] NaN/Inf loss at step {step}. "
                "Check: LR not too high, gradient clipping is applied before "
                "optimizer.step(), AMP scaler is correctly configured. "
                "See Appendix A of scratch_slm_execution_plan.md for the full "
                "failure-mode checklist."
            )

        # \u2500\u2500 Backward: scale \u2192 backward \u2192 unscale \u2192 clip \u2192 step \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        last_train_loss = loss.item()
        current_lr      = scheduler.get_last_lr()[0]

        # \u2500\u2500 Per-step log (train_loss only; val_loss=None between eval points) \u2500\u2500\u2500\u2500
        if step % LOG_EVERY == 0:
            logger.log(
                step=step,
                epoch=round(step / steps_per_epoch, 4),
                train_loss=last_train_loss,
                lr=current_lr,
            )

        # \u2500\u2500 Eval pass + conditional checkpoint (save NESTED inside eval) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        eval_now = (step % EVAL_EVERY == 0) or (smoke and step == total_steps)
        if eval_now:
            val_loss = evaluate(model, val_loader, device)

            # Log the eval point with val_loss (overwrites the step-log if
            # step is a multiple of both LOG_EVERY and EVAL_EVERY \u2014 that's
            # fine; the logged val_loss is what matters at eval steps).
            logger.log(
                step=step,
                epoch=round(step / steps_per_epoch, 4),
                train_loss=last_train_loss,
                val_loss=val_loss,
                lr=current_lr,
            )
            print(
                f"[phase4] step {step:5d}  "
                f"train_loss={last_train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"lr={current_lr:.2e}"
            )

            # \u2500\u2500 Best-val checkpoint (skipped in smoke mode) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            if not smoke and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_step = step
                ckpt_dir = BEST_VAL_DIR / run_name
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_dir / "best_val.pt")
                # Save config alongside checkpoint: Phase 5 can load both
                # without needing to look up the run config separately.
                best_cfg = {
                    **run_cfg,
                    "best_val_step":  step,
                    "best_val_loss":  round(val_loss, 6),
                }
                (ckpt_dir / "best_val_config.json").write_text(
                    json.dumps(best_cfg, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    f"[phase4] \u2713 new best val_loss={val_loss:.4f} "
                    f"\u2192 {ckpt_dir / 'best_val.pt'}"
                )

    # \u2500\u2500 Post-loop: save \u201clast\u201d checkpoint (audit artifact, NOT used by Phase 5) \u2500\u2500\u2500\u2500
    # Phase 5 evaluation always loads best_val.pt, not this file.
    # Keeping the last checkpoint allows manual inspection if needed.
    if not smoke:
        last_dir = LAST_CKPT_DIR / run_name
        last_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), last_dir / "last_ckpt.pt")
        print(f"[phase4] last checkpoint \u2192 {last_dir / 'last_ckpt.pt'}")

    # \u2500\u2500 Summary \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    print(f"\n[phase4] training complete")
    print(f"[phase4] final_train_loss = {last_train_loss:.4f}")
    print(f"[phase4] best_val_loss    = {best_val_loss:.4f}  (step {best_val_step})")

    if smoke:
        print("\n[phase4] SMOKE MODE complete \u2014 no checkpoints or CSV written")
        print("[phase4] Smoke test passed: shapes, loss, gradients, LR schedule, "
              "evaluate() (model.eval/train cycle) all exercised \u2713")
        return

    # \u2500\u2500 Append to sweep_results.csv \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    append_sweep_row(
        run_name        = run_name,
        sweep_cfg       = sweep_cfg,
        total_steps_run = total_steps,
        final_train_loss= last_train_loss,
        best_val_loss   = best_val_loss,
        best_val_step   = best_val_step,
    )
    print(f"[phase4] Phase 4 run complete.")
    print(f"[phase4] Commit logs/, checkpoints/, eval/ back to the repo before Phase 5.")

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# \u00a79 \u2014 Entry point
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

if __name__ == "__main__":
    main()
