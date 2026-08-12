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
