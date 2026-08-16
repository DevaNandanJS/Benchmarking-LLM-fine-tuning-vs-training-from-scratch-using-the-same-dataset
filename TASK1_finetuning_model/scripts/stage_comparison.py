"""Stage Track 1 evaluation metrics and artifacts for cross-track comparison."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import iso_now  # noqa: E402
from config import (  # noqa: E402
    EVAL_DIR,
    FINAL_METRICS_JSON,
    LOSS_CURVE_PNG,
    SHARED_EVAL_DIR,
    TRAINABLE_PARAMS_JSON,
)

COMPARISON_NOTES_MD = SHARED_EVAL_DIR / "comparison_notes.md"
T1_METRICS_JSON = SHARED_EVAL_DIR / "finetuning_final_metrics.json"
T1_LOSS_CURVE_PNG = SHARED_EVAL_DIR / "finetuning_loss_curve.png"

def main() -> None:
    SHARED_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Copy eval artifacts into shared_eval/
    if not FINAL_METRICS_JSON.exists():
        print(
            f"[phase8] ERROR: {FINAL_METRICS_JSON} not found. "
            "Run evaluate.py (Phase 6) first."
        )
        sys.exit(1)

    shutil.copy2(FINAL_METRICS_JSON, T1_METRICS_JSON)
    print(f"[phase8] copied -> {T1_METRICS_JSON}")

    if LOSS_CURVE_PNG.exists():
        shutil.copy2(LOSS_CURVE_PNG, T1_LOSS_CURVE_PNG)
        print(f"[phase8] copied -> {T1_LOSS_CURVE_PNG}")
    else:
        print(f"[phase8] WARNING: {LOSS_CURVE_PNG} not found — skipping plot copy")

    # Load Track 1 metrics for notes
    t1 = json.loads(FINAL_METRICS_JSON.read_text(encoding="utf-8"))

    # Load trainable param count if available
    trainable_pct = "TBD (run wrap_lora.py on Colab)"
    trainable_count = "TBD"
    if TRAINABLE_PARAMS_JSON.exists():
        tp = json.loads(TRAINABLE_PARAMS_JSON.read_text(encoding="utf-8"))
        trainable_pct = f"{tp.get('trainable_percentage', '?'):.3f}%"
        trainable_count = f"{tp.get('trainable_parameters', '?'):,}"

    # Write comparison_notes.md
    notes = f"""# Cross-Track Comparison Notes

> **Status:** Track 1 complete. Track 2 fields are TBD — fill in once Track 2 finishes.
> Last updated: {iso_now()}

---

## Primary Comparison Metric: Bits-Per-Byte (BPB)

BPB is the only metric directly comparable across tracks. It normalises loss by
raw text bytes rather than token count, making it tokenizer-agnostic. Lower = better.

| Track | Model | Strategy | BPB | Perplexity | Best Val CE Loss |
|---|---|---|---|---|---|
| **Track 1** | SmolLM2-135M (LoRA) | Fine-tune pretrained | **{t1.get('bpb', 'TBD')}** | {t1.get('perplexity', 'TBD')} | {t1.get('mean_ce_loss', 'TBD')} |
| **Track 2** | [TBD] (scratch) | Train from random init | **TBD** | TBD | TBD |

---

## Efficiency Comparison

| Metric | Track 1 (LoRA) | Track 2 (Scratch) |
|---|---|---|
| Trainable parameters | {trainable_count} ({trainable_pct}) | TBD (full model) |
| Total parameters | ~135M (frozen base + adapters) | TBD |
| Training time | TBD (from Phase 5 logs) | TBD |
| GPU memory peak | TBD (torch.cuda.max_memory_allocated) | TBD |

---

## Qualitative Generation Comparison

*(Fill in after both tracks have generated samples in Phase 7)*

| Aspect | Track 1 (LoRA) | Track 2 (Scratch) |
|---|---|---|
| Domain vocabulary | TBD | TBD |
| Fluency | TBD | TBD |
| Coherence over ~80 tokens | TBD | TBD |
| Hallucination tendency | TBD | TBD |

---

## One-Paragraph Summary

*(Write after both tracks are complete)*

Track 1 started with a model that already understands language (grammar, vocabulary,
common phrasings) and used LoRA to adapt only {trainable_pct} of its parameters to
the domain document. Track 2 started from random weights and had to learn both
language fundamentals and domain content from the same ~65K-token document.
The BPB comparison ({t1.get('bpb', 'TBD')} vs TBD) quantifies which strategy
produced a model better at predicting the source text, controlling for vocabulary
size differences. The trainable-parameter count comparison ({trainable_count} for Track 1
vs TBD for Track 2) quantifies the compute cost of each approach.

---

## Files in this directory

| File | Contents |
|---|---|
| `finetuning_final_metrics.json` | CE loss, perplexity, BPB for the best Track 1 checkpoint |
| `finetuning_loss_curve.png` | Train/val loss curve for the best Track 1 run |
| `comparison_notes.md` | This file |
| `slm_final_metrics.json` | *(TBD — Track 2)* |
| `slm_loss_curve.png` | *(TBD — Track 2)* |
"""
    COMPARISON_NOTES_MD.write_text(notes, encoding="utf-8")
    print(f"[phase8] comparison notes written -> {COMPARISON_NOTES_MD}")

    # Definition-of-Done assertions
    assert T1_METRICS_JSON.exists(), f"FAIL: {T1_METRICS_JSON} not written"
    assert COMPARISON_NOTES_MD.exists(), f"FAIL: {COMPARISON_NOTES_MD} not written"
    print("\n[phase8] [SUCCESS] all Definition-of-Done assertions passed")
    print(f"[phase8]   BPB={t1.get('bpb', '?')}  perplexity={t1.get('perplexity', '?')}")
    print("[phase8] Phase 8 complete. Fill in Track 2 fields once Track 2 finishes.")

if __name__ == "__main__":
    main()
