"""Phase 3 — Dataset Construction (Chunking & Splitting)."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

# Bootstrap: make scripts/ importable regardless of CWD
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import SEED, iso_now, dump_config, set_seed  # noqa: E402
from config import (  # noqa: E402
    CLEAN_TXT,
    DATASET_STATS_JSON,
    PROCESSED_DIR,
    SPLIT_STRATEGY_MD,
    TRAIN_PT,
    VAL_PT,
)

# Hyperparameters — change here, not buried in code
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
CONTEXT_LENGTH = 256        # tokens per chunk
TRAIN_STRIDE = 128          # 50% overlap for training — maximises sample density
VAL_STRIDE = 256            # non-overlapping for validation — independent eval spans
VAL_FRACTION = 0.15         # last 15% of tokens reserved for validation (contiguous)

# Sliding-window helper

def sliding_windows(token_ids: list, context_length: int, stride: int) -> list:
    """Return a list of fixed-length chunks from token_ids using a sliding window."""
    chunks = []
    for start in range(0, len(token_ids) - context_length + 1, stride):
        chunks.append(token_ids[start : start + context_length])
    return chunks

def main() -> None:
    import torch
    from transformers import AutoTokenizer

    seed = set_seed(SEED)
    print(f"[phase3] seed = {seed}")

    # Dump run config before any computation (config-as-file convention)
    run_config = {
        "phase": 3,
        "model_name": MODEL_NAME,
        "context_length": CONTEXT_LENGTH,
        "train_stride": TRAIN_STRIDE,
        "val_stride": VAL_STRIDE,
        "val_fraction": VAL_FRACTION,
        "seed": SEED,
    }
    cfg_path = dump_config(run_config, "phase3")
    print(f"[phase3] run config saved -> {cfg_path}")

    # Load tokenizer
    print(f"\n[phase3] loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Honour the Phase 2 pad-token decision: set pad_token to eos_token if missing.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[phase3] vocab size: {tokenizer.vocab_size:,}")

    # Tokenize the full document
    print(f"\n[phase3] reading {CLEAN_TXT} ...")
    text = CLEAN_TXT.read_text(encoding="utf-8")
    print("[phase3] tokenizing (this may take ~5-10 s on CPU) ...")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(token_ids)
    print(f"[phase3] total tokens: {total_tokens:,}")

    # Compute contiguous holdout split boundary
    boundary = int(total_tokens * (1.0 - VAL_FRACTION))
    train_token_ids = token_ids[:boundary]
    val_token_ids = token_ids[boundary:]
    train_tokens = len(train_token_ids)
    val_tokens = len(val_token_ids)
    print(f"\n[phase3] split boundary: token {boundary:,}")
    print(f"[phase3] train region: tokens 0 -> {boundary - 1:,}  ({train_tokens:,} tokens, {1 - VAL_FRACTION:.0%})")
    print(f"[phase3] val   region: tokens {boundary:,} -> {total_tokens - 1:,}  ({val_tokens:,} tokens, {VAL_FRACTION:.0%})")

    # Apply sliding-window chunking independently per region
    print(f"\n[phase3] chunking train region: context_length={CONTEXT_LENGTH}, stride={TRAIN_STRIDE} (50% overlap)")
    train_chunks_raw = sliding_windows(train_token_ids, CONTEXT_LENGTH, TRAIN_STRIDE)
    train_chunks = len(train_chunks_raw)

    print(f"[phase3] chunking val region:   context_length={CONTEXT_LENGTH}, stride={VAL_STRIDE} (non-overlapping)")
    val_chunks_raw = sliding_windows(val_token_ids, CONTEXT_LENGTH, VAL_STRIDE)
    val_chunks = len(val_chunks_raw)

    print(f"[phase3] train chunks: {train_chunks}")
    print(f"[phase3] val chunks:   {val_chunks}")

    # Dropped-remainder accounting
    # Tokens at the tail of each region that cannot fill a full context_length window.
    # Reported in stats for full token accounting (covered + dropped == region total).
    tokens_covered_train = (CONTEXT_LENGTH + (train_chunks - 1) * TRAIN_STRIDE) if train_chunks > 0 else 0
    tokens_covered_val = (CONTEXT_LENGTH + (val_chunks - 1) * VAL_STRIDE) if val_chunks > 0 else 0
    dropped_tokens_train = train_tokens - tokens_covered_train
    dropped_tokens_val = val_tokens - tokens_covered_val
    print(f"\n[phase3] tokens covered (train): {tokens_covered_train:,}  dropped: {dropped_tokens_train}")
    print(f"[phase3] tokens covered (val):   {tokens_covered_val:,}  dropped: {dropped_tokens_val}")

    # Zero-leakage assertion
    # train_token_ids is token_ids[0:boundary]
    if train_chunks > 0 and val_chunks > 0:
        last_train_window_end = (train_chunks - 1) * TRAIN_STRIDE + CONTEXT_LENGTH
        assert last_train_window_end <= boundary, (
            f"LEAKAGE DETECTED: last train window ends at global token index "
            f"{last_train_window_end}, which exceeds the boundary {boundary}."
        )
    print("[phase3] zero-leakage assertion passed (train and val windows do not overlap)")

    # Build pure tensor dictionaries
    # Storing raw tensor stacks (NOT a pickled custom Dataset class). This means:
    print("\n[phase3] building tensors ...")
    train_input_ids = torch.tensor(train_chunks_raw, dtype=torch.long)  # [N_train, 256]
    train_labels = train_input_ids.clone()

    val_input_ids = torch.tensor(val_chunks_raw, dtype=torch.long)      # [N_val, 256]
    val_labels = val_input_ids.clone()

    print(f"[phase3] train_input_ids shape: {tuple(train_input_ids.shape)}")
    print(f"[phase3] val_input_ids shape:   {tuple(val_input_ids.shape)}")

    # Save to disk
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_payload = {"input_ids": train_input_ids, "labels": train_labels}
    val_payload = {"input_ids": val_input_ids, "labels": val_labels}

    import torch as _torch
    _torch.save(train_payload, TRAIN_PT)
    _torch.save(val_payload, VAL_PT)
    print(f"\n[phase3] saved -> {TRAIN_PT}")
    print(f"[phase3] saved -> {VAL_PT}")

    # Write dataset_stats.json
    stats = {
        "model_name": MODEL_NAME,
        "context_length": CONTEXT_LENGTH,
        "train_stride": TRAIN_STRIDE,
        "val_stride": VAL_STRIDE,
        "val_fraction": VAL_FRACTION,
        "total_tokens": total_tokens,
        "split_boundary_token_idx": boundary,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_chunks": train_chunks,
        "val_chunks": val_chunks,
        "tokens_covered_train": tokens_covered_train,
        "tokens_covered_val": tokens_covered_val,
        "dropped_tokens_train": dropped_tokens_train,
        "dropped_tokens_val": dropped_tokens_val,
        "train_overlap_pct": round((CONTEXT_LENGTH - TRAIN_STRIDE) / CONTEXT_LENGTH * 100, 1),
        "val_overlap_pct": round((CONTEXT_LENGTH - VAL_STRIDE) / CONTEXT_LENGTH * 100, 1),
        "seed": SEED,
        "timestamp": iso_now(),
        "note": (
            "train/val chunked independently from contiguous token regions to prevent "
            "near-duplicate window leakage. train_stride < val_stride is intentional: "
            "denser training samples, independent (non-redundant) validation samples. "
            "Phase 5 DataLoader must use shuffle=True for train, shuffle=False for val."
        ),
    }
    DATASET_STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
    DATASET_STATS_JSON.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[phase3] dataset stats saved -> {DATASET_STATS_JSON}")

    # Write split_strategy.md
    approximate_distinct_val_from_overlap = val_tokens // TRAIN_STRIDE
    approximate_distinct_val_nooverlap = val_tokens // VAL_STRIDE

    split_md = textwrap.dedent(f"""\
    # Split Strategy — Track 1, Phase 3

    **Script:** `TASK1_finetuning_model/scripts/build_dataset.py`

    ---

    ## Strategy Chosen: Contiguous Holdout (last {VAL_FRACTION:.0%} of tokens)

    The last **{VAL_FRACTION:.0%}** of the document's token sequence (tokens
    {boundary:,}–{total_tokens - 1:,} of {total_tokens:,} total) is reserved as the
    validation set. The remaining **{1 - VAL_FRACTION:.0%}** (tokens 0–{boundary - 1:,})
    forms the training set. Windowing is applied **independently** within each region.

    ---

    ## Why Not Random Chunk Splitting?

    With a `train_stride` of {TRAIN_STRIDE} tokens and `context_length` of
    {CONTEXT_LENGTH} tokens, adjacent chunks share {CONTEXT_LENGTH - TRAIN_STRIDE}
    tokens of content (50% overlap). If we first windowed the full document and then
    split randomly, a train chunk starting at position *p* and a val chunk starting at
    position *p + {TRAIN_STRIDE}* would share {CONTEXT_LENGTH - TRAIN_STRIDE} tokens —
    contaminating the validation signal with near-duplicate training content. Contiguous
    holdout eliminates this by ensuring train and val token index ranges are disjoint
    before any windowing occurs.

    ---

    ## Asymmetric Stride Design

    | Region | Stride | Overlap | Chunks | Reasoning |
    |---|---|---|---|---|
    | **Train** | {TRAIN_STRIDE} | 50% | {train_chunks} | Maximises training sample density from limited data |
    | **Val** | {VAL_STRIDE} | 0% | {val_chunks} | Guarantees independent evaluation spans; no redundant signal |

    Applying 50% overlap to the ~{val_tokens:,}-token validation region would yield
    ~{approximate_distinct_val_from_overlap} chunks but only ~{approximate_distinct_val_nooverlap} worth of
    *distinct* content — artificially inflating validation sample count without adding
    independent signal. Non-overlapping windows ensure each reported val loss step
    reflects a unique {CONTEXT_LENGTH}-token span.

    ---

    ## Dropped-Remainder Tokens

    Tokens at the end of each region that don't fill a complete {CONTEXT_LENGTH}-token
    window are discarded rather than padded. These are logged for full accounting:

    | Region | Total tokens | Tokens covered | Dropped |
    |---|---|---|---|
    | Train | {train_tokens:,} | {tokens_covered_train:,} | {dropped_tokens_train} |
    | Val | {val_tokens:,} | {tokens_covered_val:,} | {dropped_tokens_val} |

    Coverage check (train): `{CONTEXT_LENGTH} + {train_chunks - 1}x{TRAIN_STRIDE} = {tokens_covered_train:,}`;
    `{train_tokens:,} - {tokens_covered_train:,} = {dropped_tokens_train}` dropped tokens. Verified.

    ---

    ## Zero-Leakage Runtime Assertion

    `build_dataset.py` asserts at runtime that the final train window's last token index
    is at most the boundary ({boundary:,}), guaranteeing zero index overlap between
    training and validation windows. This assertion passed on this run.

    ---

    ## Phase 5 DataLoader Handoff Note

    Chunks are stored in contiguous document order. Phase 5 **must** configure the
    training `DataLoader` with `shuffle=True` to break document ordering across batches.
    The validation `DataLoader` must use `shuffle=False` (or no shuffle) to keep
    evaluation order stable and reproducible across runs.
    """)
    SPLIT_STRATEGY_MD.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_STRATEGY_MD.write_text(split_md, encoding="utf-8")
    print(f"[phase3] split strategy doc saved -> {SPLIT_STRATEGY_MD}")

    # Definition-of-Done assertions
    assert TRAIN_PT.exists(), f"FAIL: {TRAIN_PT} not written"
    assert VAL_PT.exists(), f"FAIL: {VAL_PT} not written"
    assert DATASET_STATS_JSON.exists(), f"FAIL: {DATASET_STATS_JSON} not written"
    assert SPLIT_STRATEGY_MD.exists(), f"FAIL: {SPLIT_STRATEGY_MD} not written"
    assert train_chunks > 0, "FAIL: zero train chunks produced"
    assert val_chunks > 0, "FAIL: zero val chunks produced"

    # Verify tensor shapes and label integrity from disk
    loaded_train = torch.load(TRAIN_PT, weights_only=True)
    loaded_val = torch.load(VAL_PT, weights_only=True)
    assert tuple(loaded_train["input_ids"].shape) == (train_chunks, CONTEXT_LENGTH), (
        f"FAIL: train input_ids shape {tuple(loaded_train['input_ids'].shape)} "
        f"!= ({train_chunks}, {CONTEXT_LENGTH})"
    )
    assert tuple(loaded_val["input_ids"].shape) == (val_chunks, CONTEXT_LENGTH), (
        f"FAIL: val input_ids shape {tuple(loaded_val['input_ids'].shape)} "
        f"!= ({val_chunks}, {CONTEXT_LENGTH})"
    )
    assert bool((loaded_train["input_ids"] == loaded_train["labels"]).all()), (
        "FAIL: train input_ids and labels diverged after save/load"
    )
    assert bool((loaded_val["input_ids"] == loaded_val["labels"]).all()), (
        "FAIL: val input_ids and labels diverged after save/load"
    )

    print("\n[phase3] all Definition-of-Done assertions passed")
    print(
        f"[phase3] train: {train_chunks} chunks x {CONTEXT_LENGTH} tokens  "
        f"({dropped_tokens_train} tokens dropped)"
    )
    print(
        f"[phase3] val:   {val_chunks} chunks x {CONTEXT_LENGTH} tokens  "
        f"({dropped_tokens_val} tokens dropped)"
    )
    print("[phase3] Phase 3 complete. Review data/processed/ and configs/, then commit.")

if __name__ == "__main__":
    main()
