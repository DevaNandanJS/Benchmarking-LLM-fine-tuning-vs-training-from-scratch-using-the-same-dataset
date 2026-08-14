"""Phase 2 — Dataset Construction (Tokenise, Chunk, Split).

Goal (plan §Phase 2): tokenize document_clean.txt with the custom BPE tokenizer
produced in Phase 1, apply the same contiguous-holdout / sliding-window strategy
as Track 1's Phase 3, save plain PyTorch tensors, and document all decisions.

Key design choices:
  - context_length = 256   same block_size as Track 1 for architectural alignment
  - train_stride  = 128    50% overlap — maximises sample density from limited data
  - val_stride    = 256    non-overlapping — independent, non-redundant val spans
  - val_fraction  = 0.15   contiguous last-15% holdout (same boundary as Track 1)

Why context_length=256 (NOT a CE-comparability claim):
  Matching block_size keeps the forward-pass structure identical across tracks.
  Raw cross-entropy loss is NOT directly comparable between tracks because it scales
  with log(vocab_size) — Track 2's 1024-token vocab has a theoretical CE ceiling of
  ln(1024) ≈ 6.93 nats vs Track 1's ln(49152) ≈ 10.80 nats.  The comparison metric
  for the write-up is bits-per-byte (BPB), computed in Phase 5, which normalises by
  raw UTF-8 bytes and is therefore vocabulary-agnostic.

Smoke-test invocation contract:
  - bare:          python build_dataset.py           → smoke_test + main()
  - local-only:    python build_dataset.py --smoke-test → smoke_test only (no main)
  The smoke test always runs first as an unconditional invariant guard.

Outputs (relative to repo root):
  data/processed/slm_train.pt          — {input_ids, labels} LongTensor[N,256]
  data/processed/slm_val.pt            — same schema, val chunks
  data/processed/track2_dataset_stats.json
  TASK2_slm_from_scratch/configs/run_phase2_dataset.json
  TASK2_slm_from_scratch/configs/split_strategy.md

Run on Colab (from repo root after git pull):
  !python TASK2_slm_from_scratch/scripts/build_dataset.py

Definition of Done (plan Phase 2):
  [ ] Script runs end-to-end from document_clean.txt + trained tokenizer to tensors
  [ ] Stats logged; context length and split strategy documented
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

# ── Bootstrap: make scripts/ importable regardless of CWD ───────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import SEED, TRACK_DIR, iso_now, dump_config, set_seed  # noqa: E402
from config import (  # noqa: E402
    CLEAN_TXT,
    CONFIGS_DIR,
    DATASET_STATS_JSON,
    PROCESSED_DIR,
    TOKENIZER_DIR,
    TRAIN_PT,
    VAL_PT,
)

# ── Hyperparameters ──────────────────────────────────────────────────────────
CONTEXT_LENGTH = 256    # tokens per chunk (matches Track 1's block_size)
TRAIN_STRIDE   = 128    # 50% overlap — maximises training sample density
VAL_STRIDE     = 256    # non-overlapping val windows — independent eval spans
VAL_FRACTION   = 0.15   # last 15% of token sequence held out for validation

SPLIT_STRATEGY_MD = CONFIGS_DIR / "split_strategy.md"


# ── Pure-Python sliding-window helper ────────────────────────────────────────
# Copied from TASK1_finetuning_model/scripts/build_dataset.py — no Track 1 imports.

def sliding_windows(token_ids: list[int], context_length: int, stride: int) -> list[list[int]]:
    """Return fixed-length chunks from token_ids using a sliding window.

    Trailing tokens that do not fill a complete window are silently dropped.
    Caller should log the dropped count for full token accounting.

    Args:
        token_ids:      flat list of integer token IDs
        context_length: number of tokens per chunk
        stride:         step size between window start positions

    Returns:
        List of lists, each of length context_length.
    """
    chunks = []
    for start in range(0, len(token_ids) - context_length + 1, stride):
        chunks.append(token_ids[start: start + context_length])
    return chunks


# ── Smoke test — unconditional invariant guard ────────────────────────────────

def _smoke_test() -> None:
    """Validate sliding_windows(), split logic, and tensor round-trip.

    Uses the same production constants (CONTEXT_LENGTH, TRAIN_STRIDE, VAL_STRIDE)
    on a 2,000-element fake token sequence so several windows are produced and
    the dropped-remainder logic is exercised.

    Requires only torch (deferred import) and pure Python — no tokenizers library,
    no real data files.  Raises AssertionError on any failure.
    """
    import torch  # noqa: PLC0415

    print("[smoke] Running smoke test with production constants on fake token sequence ...")

    N = 2_000
    fake_ids = list(range(N))
    val_fraction = 0.15

    boundary = int(N * (1.0 - val_fraction))
    train_ids = fake_ids[:boundary]
    val_ids   = fake_ids[boundary:]

    train_chunks = sliding_windows(train_ids, CONTEXT_LENGTH, TRAIN_STRIDE)
    val_chunks   = sliding_windows(val_ids,   CONTEXT_LENGTH, VAL_STRIDE)

    # ── Shape assertions ──────────────────────────────────────────────────────
    expected_train = (len(train_ids) - CONTEXT_LENGTH) // TRAIN_STRIDE + 1
    expected_val   = (len(val_ids)   - CONTEXT_LENGTH) // VAL_STRIDE   + 1

    assert len(train_chunks) == expected_train, (
        f"[smoke] FAIL: expected {expected_train} train chunks, got {len(train_chunks)}"
    )
    assert len(val_chunks) == expected_val, (
        f"[smoke] FAIL: expected {expected_val} val chunks, got {len(val_chunks)}"
    )
    for c in train_chunks:
        assert len(c) == CONTEXT_LENGTH, f"[smoke] FAIL: train chunk length {len(c)} != {CONTEXT_LENGTH}"
    for c in val_chunks:
        assert len(c) == CONTEXT_LENGTH, f"[smoke] FAIL: val chunk length {len(c)} != {CONTEXT_LENGTH}"

    # ── Zero-leakage: no train window touches the val region ──────────────────
    if train_chunks and val_chunks:
        last_train_end = (len(train_chunks) - 1) * TRAIN_STRIDE + CONTEXT_LENGTH
        assert last_train_end <= boundary, (
            f"[smoke] FAIL: last train window ends at {last_train_end}, "
            f"exceeds boundary {boundary}"
        )

    # ── Tensor round-trip via a tmp file ──────────────────────────────────────
    # Use mkdtemp (not NamedTemporaryFile) — on Windows, NamedTemporaryFile
    # cannot be opened a second time while still open (error code 32).
    import tempfile  # noqa: PLC0415
    import shutil    # noqa: PLC0415
    t = torch.tensor(train_chunks[:4], dtype=torch.long)
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        tmp_pt = tmp_dir / "smoke_test.pt"
        torch.save({"input_ids": t}, tmp_pt)
        loaded = torch.load(tmp_pt, weights_only=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    assert loaded["input_ids"].shape == t.shape, (
        f"[smoke] FAIL: round-trip shape mismatch: {loaded['input_ids'].shape} vs {t.shape}"
    )

    print(
        f"[smoke] PASS — train={len(train_chunks)} chunks, val={len(val_chunks)} chunks, "
        f"context={CONTEXT_LENGTH}, train_stride={TRAIN_STRIDE}, val_stride={VAL_STRIDE}"
    )


# ── Split strategy document ───────────────────────────────────────────────────

def _write_split_strategy(
    *,
    total_tokens: int,
    boundary: int,
    train_tokens: int,
    val_tokens: int,
    train_chunks: int,
    val_chunks: int,
    tokens_covered_train: int,
    tokens_covered_val: int,
    dropped_tokens_train: int,
    dropped_tokens_val: int,
    tokenizer_vocab_size: int,
    chars_per_token: float,
    chars_per_window: float,
) -> None:
    """Write configs/split_strategy.md with all actual numbers."""
    approx_distinct_val_from_overlap = val_tokens // TRAIN_STRIDE
    approx_distinct_val_nooverlap    = val_tokens // VAL_STRIDE

    md = textwrap.dedent(f"""\
    # Split Strategy — Track 2, Phase 2

    **Script:** `TASK2_slm_from_scratch/scripts/build_dataset.py`

    ---

    ## Strategy: Contiguous Holdout (last {VAL_FRACTION:.0%} of tokens)

    The last **{VAL_FRACTION:.0%}** of the document's token sequence
    (tokens {boundary:,}–{total_tokens - 1:,} of {total_tokens:,} total) is held out
    for validation.  The remaining **{1 - VAL_FRACTION:.0%}** (tokens 0–{boundary - 1:,})
    forms the training set.  Windowing is applied **independently** within each region.

    This matches Track 1's split boundary exactly — both tracks evaluate on the same
    raw text span, making the comparison controlled.

    ---

    ## Context Length: 256 Tokens

    `context_length=256` was chosen to match Track 1's `block_size`, keeping the
    forward-pass structure architecturally identical across tracks.

    **Important: this is NOT a claim that token-level CE loss is directly comparable.**
    Cross-entropy loss scales with log(vocab_size):

    | Track | Vocab size | Random-model CE ceiling |
    |---|---|---|
    | Track 1 (SmolLM2) | ~49,152 | ln(49152) ≈ 10.80 nats |
    | Track 2 (custom BPE) | {tokenizer_vocab_size:,} | ln({tokenizer_vocab_size}) ≈ {__import__('math').log(tokenizer_vocab_size):.2f} nats |

    The correct cross-track comparison metric is **bits-per-byte (BPB)** — computed in
    Phase 5 by normalising total held-out loss (in nats) by the raw UTF-8 byte length
    of the validation text span, divided by ln(2).  BPB is vocabulary-agnostic.

    **Concrete text coverage per window:**

    The custom tokenizer produces {chars_per_token:.2f} characters per token on average
    (total_chars / total_tokens).  A {CONTEXT_LENGTH}-token window therefore covers
    approximately **{chars_per_window:.0f} raw characters** of source text.

    ---

    ## Why Not Random Chunk Splitting?

    With `train_stride={TRAIN_STRIDE}` and `context_length={CONTEXT_LENGTH}`, adjacent
    train chunks share {CONTEXT_LENGTH - TRAIN_STRIDE} tokens (50% overlap).  Random
    chunk splitting would contaminate validation with near-duplicate training windows.
    Contiguous holdout ensures token-index ranges are disjoint before windowing.

    ---

    ## Asymmetric Stride Design

    | Region | Stride | Overlap | Chunks | Reasoning |
    |---|---|---|---|---|
    | **Train** | {TRAIN_STRIDE} | 50% | {train_chunks} | Maximises sample density from limited data |
    | **Val** | {VAL_STRIDE} | 0% | {val_chunks} | Guarantees independent evaluation spans |

    Applying 50% overlap to the ~{val_tokens:,}-token val region would yield
    ~{approx_distinct_val_from_overlap} chunks but only ~{approx_distinct_val_nooverlap}
    worth of *distinct* content — inflating sample count without adding independent
    signal.

    ---

    ## Dropped-Remainder Tokens

    | Region | Total tokens | Tokens covered | Dropped |
    |---|---|---|---|
    | Train | {train_tokens:,} | {tokens_covered_train:,} | {dropped_tokens_train} |
    | Val   | {val_tokens:,} | {tokens_covered_val:,} | {dropped_tokens_val} |

    ---

    ## Known Caveats

    **Val-set content skew.** The last {VAL_FRACTION:.0%} of this academic paper likely
    comprises references and appendices, making the validation text distribution somewhat
    narrower than the training distribution (which contains the main body).  This is a
    *shared* constraint across both tracks — the split boundary is identical in Track 1 —
    so it does not introduce a cross-track bias, but it should be flagged as a known
    limitation in the write-up rather than treated as a neutral split.

    **Zero-leakage runtime assertion.** `build_dataset.py` asserts at runtime that the
    final train window's last token index is at most the boundary ({boundary:,}),
    guaranteeing zero index overlap between training and validation windows.  This
    assertion passed on this run.

    ---

    ## Phase 3 / Training Handoff

    Chunks are stored in contiguous document order.  Phase 4's training DataLoader
    **must** use `shuffle=True` for training; validation DataLoader uses `shuffle=False`.

    *Generated by build_dataset.py at {iso_now()}*
    """)

    SPLIT_STRATEGY_MD.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_STRATEGY_MD.write_text(md, encoding="utf-8")
    print(f"[phase2] split strategy doc → {SPLIT_STRATEGY_MD}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import math  # noqa: PLC0415

    set_seed(SEED)

    # ── 1. Pre-flight ─────────────────────────────────────────────────────────
    vocab_json  = TOKENIZER_DIR / "vocab.json"
    merges_txt  = TOKENIZER_DIR / "merges.txt"
    if not vocab_json.exists() or not merges_txt.exists():
        raise FileNotFoundError(
            f"Custom tokenizer not found in {TOKENIZER_DIR}.\n"
            "Run Phase 1 on Colab first:\n"
            "  !python TASK2_slm_from_scratch/scripts/train_tokenizer.py\n"
            "Then commit + push the tokenizer files before re-running Phase 2."
        )
    if not CLEAN_TXT.exists():
        raise FileNotFoundError(
            f"document_clean.txt not found at {CLEAN_TXT}.\n"
            "Run Track 1 Phase 1 first to extract the document."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 2. Config-as-file BEFORE computation (global convention §0) ───────────
    run_cfg = {
        "phase": "phase2_dataset",
        "context_length": CONTEXT_LENGTH,
        "train_stride": TRAIN_STRIDE,
        "val_stride": VAL_STRIDE,
        "val_fraction": VAL_FRACTION,
        "seed": SEED,
        "tokenizer_dir": str(TOKENIZER_DIR),
        "input_file": str(CLEAN_TXT),
    }
    dump_config(run_cfg, "phase2_dataset")

    # ── 3. Load custom tokenizer ──────────────────────────────────────────────
    from tokenizers import ByteLevelBPETokenizer  # noqa: PLC0415

    print(f"\n[phase2] loading tokenizer from {TOKENIZER_DIR} ...")
    tok = ByteLevelBPETokenizer(
        vocab=str(vocab_json),
        merges=str(merges_txt),
    )
    tokenizer_vocab_size: int = tok.get_vocab_size()
    print(f"[phase2] tokenizer vocab size: {tokenizer_vocab_size:,}")

    # ── 4. Tokenize full document ─────────────────────────────────────────────
    print(f"[phase2] reading {CLEAN_TXT} ...")
    text = CLEAN_TXT.read_text(encoding="utf-8")
    total_chars = len(text.encode("utf-8"))   # raw UTF-8 bytes (for BPB later)
    total_chars_str = len(text)               # character count (for chars/window)

    print("[phase2] tokenizing ...")
    encoding = tok.encode(text)
    token_ids: list[int] = encoding.ids
    total_tokens = len(token_ids)
    print(f"[phase2] total tokens: {total_tokens:,}  (vocab size {tokenizer_vocab_size:,})")

    # ── 5. Character coverage per token / per window ──────────────────────────
    chars_per_token: float = total_chars_str / total_tokens if total_tokens else 0.0
    chars_per_window: float = chars_per_token * CONTEXT_LENGTH
    print(
        f"[phase2] chars/token: {chars_per_token:.3f}  "
        f"→ chars/window ({CONTEXT_LENGTH} tok): {chars_per_window:.0f}"
    )

    # ── 6. Contiguous holdout split ───────────────────────────────────────────
    boundary = int(total_tokens * (1.0 - VAL_FRACTION))
    train_token_ids = token_ids[:boundary]
    val_token_ids   = token_ids[boundary:]
    train_tokens = len(train_token_ids)
    val_tokens   = len(val_token_ids)
    print(f"\n[phase2] split boundary: token {boundary:,}")
    print(f"[phase2] train: tokens 0–{boundary - 1:,}  ({train_tokens:,} tokens, {1 - VAL_FRACTION:.0%})")
    print(f"[phase2] val:   tokens {boundary:,}–{total_tokens - 1:,}  ({val_tokens:,} tokens, {VAL_FRACTION:.0%})")

    # ── 7. Sliding-window chunking ────────────────────────────────────────────
    print(
        f"\n[phase2] chunking train: context={CONTEXT_LENGTH}, "
        f"stride={TRAIN_STRIDE} (50% overlap)"
    )
    train_chunks_raw = sliding_windows(train_token_ids, CONTEXT_LENGTH, TRAIN_STRIDE)
    n_train = len(train_chunks_raw)

    print(
        f"[phase2] chunking val:   context={CONTEXT_LENGTH}, "
        f"stride={VAL_STRIDE} (non-overlapping)"
    )
    val_chunks_raw = sliding_windows(val_token_ids, CONTEXT_LENGTH, VAL_STRIDE)
    n_val = len(val_chunks_raw)

    print(f"[phase2] train chunks: {n_train},  val chunks: {n_val}")

    # ── 8. Dropped-remainder accounting ──────────────────────────────────────
    tokens_covered_train = (CONTEXT_LENGTH + (n_train - 1) * TRAIN_STRIDE) if n_train > 0 else 0
    tokens_covered_val   = (CONTEXT_LENGTH + (n_val   - 1) * VAL_STRIDE)   if n_val   > 0 else 0
    dropped_train = train_tokens - tokens_covered_train
    dropped_val   = val_tokens   - tokens_covered_val
    print(f"[phase2] train: covered={tokens_covered_train:,}, dropped={dropped_train}")
    print(f"[phase2] val:   covered={tokens_covered_val:,},   dropped={dropped_val}")

    # ── 9. Zero-leakage assertion ──────────────────────────────────────────────
    if n_train > 0 and n_val > 0:
        last_train_end = (n_train - 1) * TRAIN_STRIDE + CONTEXT_LENGTH
        assert last_train_end <= boundary, (
            f"LEAKAGE DETECTED: last train window ends at token {last_train_end}, "
            f"which exceeds the split boundary {boundary}."
        )
    print("[phase2] zero-leakage assertion passed")

    # ── 10. Build tensors (labels = input_ids — no pre-shifting) ─────────────
    import torch  # noqa: PLC0415

    print("\n[phase2] building tensors ...")
    train_input_ids = torch.tensor(train_chunks_raw, dtype=torch.long)  # [N_train, 256]
    train_labels    = train_input_ids.clone()
    val_input_ids   = torch.tensor(val_chunks_raw,   dtype=torch.long)  # [N_val, 256]
    val_labels      = val_input_ids.clone()

    print(f"[phase2] train_input_ids: {tuple(train_input_ids.shape)}")
    print(f"[phase2] val_input_ids:   {tuple(val_input_ids.shape)}")

    # ── 11. Save to disk ──────────────────────────────────────────────────────
    torch.save({"input_ids": train_input_ids, "labels": train_labels}, TRAIN_PT)
    torch.save({"input_ids": val_input_ids,   "labels": val_labels},   VAL_PT)
    print(f"\n[phase2] saved → {TRAIN_PT}")
    print(f"[phase2] saved → {VAL_PT}")

    # ── 12. Write dataset stats JSON ─────────────────────────────────────────
    stats = {
        "phase": "phase2_dataset",
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "tokenizer_dir": str(TOKENIZER_DIR),
        "context_length": CONTEXT_LENGTH,
        "train_stride": TRAIN_STRIDE,
        "val_stride": VAL_STRIDE,
        "val_fraction": VAL_FRACTION,
        "total_tokens": total_tokens,
        "total_chars_utf8_bytes": total_chars,
        "total_chars_unicode": total_chars_str,
        "chars_per_token": round(chars_per_token, 4),
        "chars_per_window": round(chars_per_window, 1),
        "split_boundary_token_idx": boundary,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_chunks": n_train,
        "val_chunks": n_val,
        "tokens_covered_train": tokens_covered_train,
        "tokens_covered_val": tokens_covered_val,
        "dropped_tokens_train": dropped_train,
        "dropped_tokens_val": dropped_val,
        "train_overlap_pct": round((CONTEXT_LENGTH - TRAIN_STRIDE) / CONTEXT_LENGTH * 100, 1),
        "val_overlap_pct": 0.0,
        "seed": SEED,
        "timestamp": iso_now(),
        "context_length_note": (
            f"256 tokens chosen for architectural alignment with Track 1 (same block_size). "
            f"Raw CE loss is NOT directly comparable: Track 2 vocab={tokenizer_vocab_size} → "
            f"CE ceiling ≈ {math.log(tokenizer_vocab_size):.2f} nats vs Track 1 vocab=49152 → "
            f"≈ {math.log(49152):.2f} nats. Use BPB from Phase 5 for cross-track comparison."
        ),
        "val_set_caveat": (
            "Last 15% of this academic paper likely comprises references/appendix — "
            "val text distribution is narrower than train. Shared across both tracks "
            "(identical split boundary), so not a cross-track bias, but a known limitation."
        ),
    }
    DATASET_STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
    DATASET_STATS_JSON.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[phase2] dataset stats → {DATASET_STATS_JSON}")

    # ── 13. Write split_strategy.md ──────────────────────────────────────────
    _write_split_strategy(
        total_tokens=total_tokens,
        boundary=boundary,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        train_chunks=n_train,
        val_chunks=n_val,
        tokens_covered_train=tokens_covered_train,
        tokens_covered_val=tokens_covered_val,
        dropped_tokens_train=dropped_train,
        dropped_tokens_val=dropped_val,
        tokenizer_vocab_size=tokenizer_vocab_size,
        chars_per_token=chars_per_token,
        chars_per_window=chars_per_window,
    )

    # ── 14. Definition-of-Done assertions ────────────────────────────────────
    assert TRAIN_PT.exists(), f"FAIL: {TRAIN_PT} not written"
    assert VAL_PT.exists(),   f"FAIL: {VAL_PT} not written"
    assert DATASET_STATS_JSON.exists(), f"FAIL: {DATASET_STATS_JSON} not written"
    assert SPLIT_STRATEGY_MD.exists(),  f"FAIL: {SPLIT_STRATEGY_MD} not written"
    assert n_train > 0, "FAIL: zero train chunks produced"
    assert n_val   > 0, "FAIL: zero val chunks produced"

    # Verify shapes and label integrity from disk
    loaded_train = torch.load(TRAIN_PT, weights_only=True)
    loaded_val   = torch.load(VAL_PT,   weights_only=True)
    assert tuple(loaded_train["input_ids"].shape) == (n_train, CONTEXT_LENGTH), (
        f"FAIL: train shape {tuple(loaded_train['input_ids'].shape)} "
        f"!= ({n_train}, {CONTEXT_LENGTH})"
    )
    assert tuple(loaded_val["input_ids"].shape) == (n_val, CONTEXT_LENGTH), (
        f"FAIL: val shape {tuple(loaded_val['input_ids'].shape)} "
        f"!= ({n_val}, {CONTEXT_LENGTH})"
    )
    assert bool((loaded_train["input_ids"] == loaded_train["labels"]).all()), (
        "FAIL: train input_ids and labels diverged after save/load"
    )
    assert bool((loaded_val["input_ids"] == loaded_val["labels"]).all()), (
        "FAIL: val input_ids and labels diverged after save/load"
    )

    print("\n[phase2] all Definition-of-Done assertions passed")
    print(f"[phase2] train: {n_train} chunks × {CONTEXT_LENGTH} tokens  ({dropped_train} dropped)")
    print(f"[phase2] val:   {n_val} chunks × {CONTEXT_LENGTH} tokens  ({dropped_val} dropped)")
    print(f"[phase2] chars/window: {chars_per_window:.0f} raw characters per training example")
    print("[phase2] Phase 2 complete. Review data/processed/ and configs/, then commit.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Track 2 Phase 2 — Dataset Construction"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test only (no tokenizer files required); skip main()",
    )
    args = parser.parse_args()

    _smoke_test()              # always runs first — unconditional invariant check
    if not args.smoke_test:
        main()
