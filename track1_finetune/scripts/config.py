"""Shared path constants for all Track 1 scripts.

Put every path that could break if mis-typed in more than one place here.
Import this module instead of hardcoding literals.

Usage::

    from scripts.config import RAW_PDF, EXTRACTED_DIR
"""
from pathlib import Path

# ── Repository / track roots ────────────────────────────────────────────────
# Resolve upward from this file: scripts/ → track1_finetune/ → llm_task/
SCRIPTS_DIR: Path = Path(__file__).resolve().parent
TRACK_DIR: Path = SCRIPTS_DIR.parent
REPO_ROOT: Path = TRACK_DIR.parent

# ── Input ───────────────────────────────────────────────────────────────────
# The PDF has a space in its name — define it ONCE here so every script gets
# the correct path by importing, not by re-typing the literal.
RAW_PDF: Path = REPO_ROOT / "data" / "raw" / "LLM4log - LLM task doc.pdf"

# ── Output directories ───────────────────────────────────────────────────────
DATA_DIR: Path = REPO_ROOT / "data"
EXTRACTED_DIR: Path = DATA_DIR / "extracted"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# ── Phase 1 output files ─────────────────────────────────────────────────────
CLEAN_TXT: Path = EXTRACTED_DIR / "document_clean.txt"
STATS_JSON: Path = EXTRACTED_DIR / "stats.json"
MANIFEST_JSON: Path = EXTRACTED_DIR / "extraction_manifest.json"
HYPHEN_LOG: Path = EXTRACTED_DIR / "hyphen_join_decisions.txt"
RAW_PAGES_DIR: Path = EXTRACTED_DIR / "raw_pages"

# ── Phase 2 output files ─────────────────────────────────────────────────────
CONFIGS_DIR: Path = TRACK_DIR / "configs"
MODEL_CHOICE_MD: Path = CONFIGS_DIR / "model_choice.md"
# Full list of model.named_modules() — authoritative source for Phase 4 target_modules.
MODEL_ARCH_JSON: Path = CONFIGS_DIR / "model_architecture.json"

# ── Phase 3 output files ─────────────────────────────────────────────────────
TRAIN_PT: Path = PROCESSED_DIR / "track1_train.pt"
VAL_PT: Path = PROCESSED_DIR / "track1_val.pt"
DATASET_STATS_JSON: Path = PROCESSED_DIR / "dataset_stats.json"
SPLIT_STRATEGY_MD: Path = CONFIGS_DIR / "split_strategy.md"

# ── Phase 4 output files ─────────────────────────────────────────────────────
# Authoritative trainable-parameter count + sanity-check loss (required DoD deliverable).
# Written by wrap_lora.py at run time on Colab.
TRAINABLE_PARAMS_JSON: Path = CONFIGS_DIR / "trainable_params.json"

# ── Phase 5 output directories / files ───────────────────────────────────────
LOGS_DIR: Path = TRACK_DIR / "logs"
CHECKPOINTS_DIR: Path = TRACK_DIR / "checkpoints"
# Best adapter saved here (adapter only — not the frozen base weights).
# train.py creates BEST_VAL_DIR / <run_name>/ for each sweep point.
BEST_VAL_DIR: Path = CHECKPOINTS_DIR / "best_val"

# ── Phase 6 output files ─────────────────────────────────────────────────────
EVAL_DIR: Path = TRACK_DIR / "eval"
# Loss curves plot (both train and val on one chart, labelled axes + legend).
LOSS_CURVE_PNG: Path = EVAL_DIR / "loss_curve.png"
# Tabulated sweep comparison — one row per sweep run (r4 / r8 / r16).
SWEEP_RESULTS_CSV: Path = EVAL_DIR / "sweep_results.csv"
# Final held-out numbers: cross-entropy loss, perplexity, BPB.
FINAL_METRICS_JSON: Path = EVAL_DIR / "final_metrics.json"
# 3-5 sentence human interpretation of the loss curve (required deliverable).
LOSS_CURVE_INTERP_MD: Path = EVAL_DIR / "loss_curve_interpretation.md"

# ── Phase 7 output files ─────────────────────────────────────────────────────
GENERATIONS_DIR: Path = TRACK_DIR / "generations"
# ≥5 prompt/completion pairs with one-sentence annotations.
TRACK1_SAMPLES_MD: Path = GENERATIONS_DIR / "track1_samples.md"

# ── Phase 8 output files (repo root — shared across tracks) ──────────────────
SHARED_EVAL_DIR: Path = REPO_ROOT / "shared_eval"

