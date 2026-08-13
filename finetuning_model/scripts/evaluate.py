"""Phase 6 — Quantitative Evaluation.

Goal: produce the numbers and plots that go directly into the write-up's
"training and validation loss curves" and cross-track comparison sections.

Outputs:
    finetuning_model/eval/loss_curve.png              — train+val loss vs step
    finetuning_model/eval/final_metrics.json          — loss, perplexity, BPB
    finetuning_model/eval/loss_curve_interpretation.md — 3-5 sentence interpretation

BPB (bits-per-byte) — why and how:
    Different tokenizers have different vocabulary sizes, making raw
    cross-entropy loss non-comparable across models. BPB normalises by the
    *byte* length of the text, making it tokenizer-agnostic:

        BPB = (sum_CE_nats / utf8_byte_length_of_val_text) / ln(2)

    where sum_CE_nats is the SUM (not mean) of cross-entropy over all
    validation tokens, and utf8_byte_length_of_val_text is the raw byte
    length of the validation text span (boundary: onwards in document_clean.txt).

    BPB validity requirement: every validation token must be scored exactly
    ONCE, i.e., val windows must NOT overlap. Verified: Phase 3 built val
    chunks with val_stride=256=context_length (val_overlap_pct=0.0 confirmed
    in data/processed/dataset_stats.json). The BPB formula is valid as-is.

    Byte boundary alignment: naive re-tokenization + slicing at `boundary:`
    token index can drift by a character or two if the tokenizer produces
    slightly different token alignments on a second pass. Instead, we use
    tokenizer(..., return_offsets_mapping=True) to get the exact character
    offset of the boundary token and slice the raw text at that character.
    This ensures the byte count and token count refer to exactly the same
    text span.

Smoke-test mode (--smoke):
    Loads a tiny dummy metrics.jsonl (2 rows) and the real val tensors (first
    2 chunks) to verify shapes and JSON output without needing a full run.

Run on Colab (from repo root after git pull):
    !python finetuning_model/scripts/evaluate.py --run r8  # or whichever is best

Definition of Done (plan §Phase 6):
    [ ] eval/loss_curve.png produced and legible (labelled axes, legend)
    [ ] eval/final_metrics.json contains loss, perplexity, and BPB
    [ ] Written interpretation exists and references specific curve features
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import SEED, iso_now, set_seed  # noqa: E402
from config import (  # noqa: E402
    BEST_VAL_DIR,
    CLEAN_TXT,
    DATASET_STATS_JSON,
    EVAL_DIR,
    FINAL_METRICS_JSON,
    LOGS_DIR,
    LOSS_CURVE_INTERP_MD,
    LOSS_CURVE_PNG,
    MODEL_ARCH_JSON,
    SWEEP_RESULTS_CSV,
    VAL_PT,
)

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


def load_metrics_jsonl(run_name: str) -> list[dict]:
    """Load all logged steps from logs/<run_name>/metrics.jsonl."""
    path = LOGS_DIR / run_name / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"metrics.jsonl not found at {path}. "
            f"Run train.py --run {run_name} first."
        )
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def pick_best_run() -> str:
    """Read sweep_results.csv and return the run_name with lowest best_val_loss."""
    import csv
    if not SWEEP_RESULTS_CSV.exists():
        return "r8"   # default if sweep hasn't been run
    best_run, best_loss = "r8", float("inf")
    with SWEEP_RESULTS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            loss = float(row["best_val_loss"])
            if loss < best_loss:
                best_loss = loss
                best_run = row["run_name"]
    print(f"[phase6] best run from sweep_results.csv: {best_run} (val_loss={best_loss:.4f})")
    return best_run


def plot_loss_curve(records: list[dict], run_name: str) -> None:
    """Plot train and val loss vs step, save to eval/loss_curve.png."""
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — works in Colab cells
    import matplotlib.pyplot as plt

    steps_train = [r["step"] for r in records if r.get("train_loss") is not None]
    losses_train = [r["train_loss"] for r in records if r.get("train_loss") is not None]
    steps_val = [r["step"] for r in records if r.get("val_loss") is not None]
    losses_val = [r["val_loss"] for r in records if r.get("val_loss") is not None]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps_train, losses_train, label="Train loss", linewidth=1.5, color="#2563eb")
    if steps_val:
        ax.plot(steps_val, losses_val, label="Val loss", linewidth=2.0,
                color="#dc2626", marker="o", markersize=4)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title(
        f"Track 1 — LoRA Fine-Tuning Loss Curve\n"
        f"SmolLM2-135M, {run_name.upper()}, context_length=256",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(LOSS_CURVE_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[phase6] loss curve saved -> {LOSS_CURVE_PNG}")


def get_val_text_bytes(boundary_token_idx: int) -> tuple[int, str]:
    """Return (byte_count, val_text_str) for the validation text span.

    Uses tokenizer return_offsets_mapping to find the exact character offset
    of the boundary token rather than re-slicing by token index. This avoids
    the small character-alignment drift that naive slicing can introduce on a
    second tokenization pass.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    text = CLEAN_TXT.read_text(encoding="utf-8")

    # Tokenize with character-level offset mapping.
    # return_offsets_mapping gives (start_char, end_char) for each token.
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"]

    # Safety: boundary may be beyond encoded tokens if trailing whitespace differs.
    actual_boundary = min(boundary_token_idx, len(offsets) - 1)
    char_start = offsets[actual_boundary][0]   # character where val region begins

    val_text = text[char_start:]
    byte_len = len(val_text.encode("utf-8"))
    print(
        f"[phase6] val text span: chars {char_start}–{len(text)}  "
        f"({byte_len:,} UTF-8 bytes)"
    )
    return byte_len, val_text


def compute_final_metrics(run_name: str, device) -> dict:
    """Load best checkpoint, compute CE loss, perplexity, and BPB on val set."""
    import torch
    from peft import PeftModel
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt_dir = BEST_VAL_DIR / run_name
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found at {ckpt_dir}. "
            f"Run train.py --run {run_name} first."
        )

    print(f"[phase6] loading best checkpoint from {ckpt_dir} ...")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base_model, str(ckpt_dir))
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    val_data = torch.load(VAL_PT, weights_only=True)
    val_dataset = TensorDataset(val_data["input_ids"], val_data["labels"])
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Sum CE nats over all tokens (not mean) — required for BPB formula.
    total_ce_nats = 0.0
    total_tokens = 0

    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            labels = labels.clone()
            labels[labels == pad_id] = -100

            out = model(input_ids=input_ids, labels=labels)
            # out.loss is mean CE over non-masked tokens in this batch.
            # Recover sum by multiplying by the number of non-masked tokens.
            non_masked = (labels != -100).sum().item()
            total_ce_nats += out.loss.item() * non_masked
            total_tokens += non_masked

    mean_ce = total_ce_nats / total_tokens
    perplexity = math.exp(mean_ce)

    # BPB — load dataset stats to get boundary token index
    stats = json.loads(DATASET_STATS_JSON.read_text(encoding="utf-8"))
    boundary = stats["split_boundary_token_idx"]
    byte_len, _ = get_val_text_bytes(boundary)

    bpb = (total_ce_nats / byte_len) / math.log(2)

    print(f"[phase6] mean_ce_loss={mean_ce:.4f}  perplexity={perplexity:.2f}  BPB={bpb:.4f}")
    return {
        "run_name": run_name,
        "checkpoint_dir": str(ckpt_dir),
        "val_chunks": len(val_dataset),
        "total_val_tokens_scored": total_tokens,
        "mean_ce_loss": round(mean_ce, 6),
        "perplexity": round(perplexity, 4),
        "bpb": round(bpb, 6),
        "bpb_note": (
            "BPB = (sum_CE_nats / utf8_byte_length_of_val_text) / ln(2). "
            "Val windows are non-overlapping (val_stride=context_length=256, "
            "val_overlap_pct=0.0 per dataset_stats.json) so each token is "
            "scored exactly once — BPB formula is valid. "
            "Byte boundary aligned via return_offsets_mapping (not re-sliced "
            "by token index) to avoid character-offset drift."
        ),
    }


def write_interpretation_template(metrics: dict, records: list[dict]) -> None:
    """Write a 3-5 sentence interpretation skeleton populated with real stats."""
    # Find the step where val loss was lowest (best epoch)
    val_records = [r for r in records if r.get("val_loss") is not None]
    if val_records:
        best_rec = min(val_records, key=lambda r: r["val_loss"])
        best_step = best_rec["step"]
        best_epoch = best_rec.get("epoch", "?")
        best_val_loss = best_rec["val_loss"]

        # Detect overfitting: val rising while train falling post best
        post_best = [r for r in val_records if r["step"] > best_step]
        overfitting_note = ""
        if len(post_best) >= 2:
            post_losses = ", ".join(f"{r['val_loss']:.4f}" for r in post_best[:3])
            overfitting_note = (
                f"After step {best_step} the validation loss began rising "
                f"({post_losses}), "
                "while training loss continued to decrease — a clear early-overfitting "
                "signal consistent with the small (~430-chunk) training set."
            )
        else:
            overfitting_note = "No clear overfitting divergence was observed within the training run."

        first_train_loss = records[0]["train_loss"] if records else float("nan")
        last_train_loss = records[-1]["train_loss"] if records else float("nan")
    else:
        best_step, best_epoch, best_val_loss = "?", "?", "?"
        overfitting_note = "[complete after Colab run]"
        first_train_loss = last_train_loss = float("nan")

    interpretation = f"""# Loss Curve Interpretation — Track 1 (LoRA Fine-Tuning, {metrics.get('run_name', '?').upper()})

> **Note:** This interpretation was templated by evaluate.py and populated with
> real statistics. Revise the phrasing to reflect your own reading of the curve.

## Curve Summary

The training loss decreased from **{first_train_loss:.4f}** at step 1 to
**{last_train_loss:.4f}** by the final logged step, indicating that the LoRA
adapter successfully updated its weights and the model learned from the training
signal rather than staying at its pretrained loss baseline.

The validation loss reached its minimum at **step {best_step}** (epoch {best_epoch}),
where val_loss = **{best_val_loss:.4f}** (perplexity ≈ {metrics.get('perplexity', '?')}).
{overfitting_note}

## Metric Summary

| Metric | Value |
|---|---|
| Final mean CE loss (val) | {metrics.get('mean_ce_loss', '?')} |
| Perplexity | {metrics.get('perplexity', '?')} |
| Bits-per-byte (BPB) | {metrics.get('bpb', '?')} |
| Best val checkpoint step | {best_step} |

## Interpretation

The curve shape is consistent with a **[healthy convergence / early overfitting / underfitting]**
pattern — revise this based on the actual shape:

- **Healthy convergence:** both curves fall together and plateau; val follows train closely.
- **Early overfitting (likely at this scale):** train continues falling after val bottoms out;
  the divergence point is the step reported above.
- **Underfitting:** both curves plateau at a high loss and neither falls significantly.

The BPB of **{metrics.get('bpb', '?')}** is the primary cross-track comparison metric.
Lower BPB means the model assigns more probability mass to the actual text per byte,
regardless of vocabulary size. This value will be compared directly to Track 2's BPB
once that track completes.
"""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    LOSS_CURVE_INTERP_MD.write_text(interpretation, encoding="utf-8")
    print(f"[phase6] interpretation template saved -> {LOSS_CURVE_INTERP_MD}")


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="Phase 6 — Quantitative evaluation")
    parser.add_argument(
        "--run", default=None,
        help=(
            "Run name to evaluate (r4 / r8 / r16). "
            "If omitted, reads sweep_results.csv to pick the best run."
        ),
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-test: skip checkpoint load, use dummy data to verify output shapes.",
    )
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[phase6] device={device}")

    run_name = args.run or pick_best_run()
    print(f"[phase6] evaluating run: {run_name}")

    # ── 1. Load metrics and plot ───────────────────────────────────────────
    if args.smoke:
        # Stub two records for smoke testing shape/JSON output
        records = [
            {"step": 1, "epoch": 0.1, "train_loss": 9.5, "val_loss": None, "lr": 2e-4},
            {"step": 5, "epoch": 1.0, "train_loss": 8.2, "val_loss": 8.8, "lr": 1.8e-4},
        ]
        print("[phase6] SMOKE: using stub records")
    else:
        records = load_metrics_jsonl(run_name)
        print(f"[phase6] loaded {len(records)} logged steps from metrics.jsonl")

    plot_loss_curve(records, run_name)

    # ── 2. Compute final metrics ───────────────────────────────────────────
    if args.smoke:
        metrics = {
            "run_name": run_name,
            "checkpoint_dir": "smoke-test-stub",
            "val_chunks": 2,
            "total_val_tokens_scored": 512,
            "mean_ce_loss": 8.8,
            "perplexity": round(math.exp(8.8), 4),
            "bpb": round((8.8 * 512 / 5000) / math.log(2), 6),
            "bpb_note": "smoke test stub values — not real",
        }
        print("[phase6] SMOKE: using stub metrics")
    else:
        metrics = compute_final_metrics(run_name, device)

    metrics["timestamp"] = iso_now()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_METRICS_JSON.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[phase6] final_metrics.json saved -> {FINAL_METRICS_JSON}")

    # ── 3. Write interpretation template ─────────────────────────────────
    write_interpretation_template(metrics, records)

    # ── 4. Definition-of-Done assertions ─────────────────────────────────
    assert LOSS_CURVE_PNG.exists(), f"FAIL: {LOSS_CURVE_PNG} not written"
    assert FINAL_METRICS_JSON.exists(), f"FAIL: {FINAL_METRICS_JSON} not written"
    assert LOSS_CURVE_INTERP_MD.exists(), f"FAIL: {LOSS_CURVE_INTERP_MD} not written"
    loaded = json.loads(FINAL_METRICS_JSON.read_text())
    for key in ("mean_ce_loss", "perplexity", "bpb"):
        assert key in loaded, f"FAIL: {key} missing from final_metrics.json"

    print("\n[phase6] ✅ all Definition-of-Done assertions passed")
    print(f"[phase6]   loss={loaded['mean_ce_loss']}  perp={loaded['perplexity']}  BPB={loaded['bpb']}")
    print("[phase6] Phase 6 complete. Edit loss_curve_interpretation.md with your curve reading.")


if __name__ == "__main__":
    main()
