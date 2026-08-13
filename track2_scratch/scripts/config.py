"""Path constants for all Track 2 scripts.

Import this module instead of hardcoding path literals. Every path that
appears in more than one script belongs here.

Usage::

    from config import CLEAN_TXT, VOCAB_SWEEP_CSV, TOKENIZER_DIR
"""
from pathlib import Path

# ── Track roots ──────────────────────────────────────────────────────────────
SCRIPTS_DIR: Path = Path(__file__).resolve().parent
TRACK_DIR:   Path = SCRIPTS_DIR.parent          # track2_scratch/
REPO_ROOT:   Path = TRACK_DIR.parent            # llm_task/

# ── Shared input (produced by Track 1 Phase 1 — do not re-extract) ───────────
DATA_DIR:       Path = REPO_ROOT / "data"
EXTRACTED_DIR:  Path = DATA_DIR / "extracted"
PROCESSED_DIR:  Path = DATA_DIR / "processed"
CLEAN_TXT:      Path = EXTRACTED_DIR / "document_clean.txt"
T1_STATS_JSON:  Path = EXTRACTED_DIR / "stats.json"   # Track 1 extraction stats

# ── Track 2 configs ───────────────────────────────────────────────────────────
CONFIGS_DIR:           Path = TRACK_DIR / "configs"
TOKENIZER_CHOICE_MD:   Path = CONFIGS_DIR / "tokenizer_choice.md"

# ── Phase 1 — tokenizer ───────────────────────────────────────────────────────
TOKENIZER_DIR:         Path = TRACK_DIR / "tokenizer"       # vocab.json + merges.txt saved here
EVAL_DIR:              Path = TRACK_DIR / "eval"
VOCAB_SWEEP_CSV:       Path = EVAL_DIR / "vocab_sweep.csv"

# ── Phase 2 — dataset ─────────────────────────────────────────────────────────
TRAIN_PT:              Path = PROCESSED_DIR / "track2_train.pt"
VAL_PT:                Path = PROCESSED_DIR / "track2_val.pt"
DATASET_STATS_JSON:    Path = PROCESSED_DIR / "track2_dataset_stats.json"

# ── Phase 3 — model architecture ─────────────────────────────────────────────
MODEL_CONFIG_JSON:     Path = CONFIGS_DIR / "run_phase3_model.json"
TRAINABLE_PARAMS_JSON: Path = CONFIGS_DIR / "trainable_params.json"

# ── Phase 4 — training ────────────────────────────────────────────────────────
LOGS_DIR:              Path = TRACK_DIR / "logs"
CHECKPOINTS_DIR:       Path = TRACK_DIR / "checkpoints"
BEST_VAL_DIR:          Path = CHECKPOINTS_DIR / "best_val"
LAST_CKPT_DIR:         Path = CHECKPOINTS_DIR / "last"

# ── Phase 5 — evaluation ──────────────────────────────────────────────────────
LOSS_CURVE_PNG:        Path = EVAL_DIR / "loss_curve.png"
SWEEP_RESULTS_CSV:     Path = EVAL_DIR / "sweep_results.csv"
FINAL_METRICS_JSON:    Path = EVAL_DIR / "final_metrics.json"
LOSS_CURVE_INTERP_MD:  Path = EVAL_DIR / "loss_curve_interpretation.md"

# ── Phase 6 — generations ─────────────────────────────────────────────────────
GENERATIONS_DIR:       Path = TRACK_DIR / "generations"
TRACK2_SAMPLES_MD:     Path = GENERATIONS_DIR / "track2_samples.md"

# ── Phase 7 — shared eval (repo root) ────────────────────────────────────────
SHARED_EVAL_DIR:       Path = REPO_ROOT / "shared_eval"
