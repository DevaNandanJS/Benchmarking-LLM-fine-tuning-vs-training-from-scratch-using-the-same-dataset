# Split Strategy — Track 1, Phase 3

**Script:** `finetuning_model/scripts/build_dataset.py`

---

## Strategy Chosen: Contiguous Holdout (last 15% of tokens)

The last **15%** of the document's token sequence (tokens
55,287–65,043 of 65,044 total) is reserved as the
validation set. The remaining **85%** (tokens 0–55,286)
forms the training set. Windowing is applied **independently** within each region.

---

## Why Not Random Chunk Splitting?

With a `train_stride` of 128 tokens and `context_length` of
256 tokens, adjacent chunks share 128
tokens of content (50% overlap). If we first windowed the full document and then
split randomly, a train chunk starting at position *p* and a val chunk starting at
position *p + 128* would share 128 tokens —
contaminating the validation signal with near-duplicate training content. Contiguous
holdout eliminates this by ensuring train and val token index ranges are disjoint
before any windowing occurs.

---

## Asymmetric Stride Design

| Region | Stride | Overlap | Chunks | Reasoning |
|---|---|---|---|---|
| **Train** | 128 | 50% | 430 | Maximises training sample density from limited data |
| **Val** | 256 | 0% | 38 | Guarantees independent evaluation spans; no redundant signal |

Applying 50% overlap to the ~9,757-token validation region would yield
~76 chunks but only ~38 worth of
*distinct* content — artificially inflating validation sample count without adding
independent signal. Non-overlapping windows ensure each reported val loss step
reflects a unique 256-token span.

---

## Dropped-Remainder Tokens

Tokens at the end of each region that don't fill a complete 256-token
window are discarded rather than padded. These are logged for full accounting:

| Region | Total tokens | Tokens covered | Dropped |
|---|---|---|---|
| Train | 55,287 | 55,168 | 119 |
| Val | 9,757 | 9,728 | 29 |

Coverage check (train): `256 + 429x128 = 55,168`;
`55,287 - 55,168 = 119` dropped tokens. Verified.

---

## Zero-Leakage Runtime Assertion

`build_dataset.py` asserts at runtime that the final train window's last token index
is at most the boundary (55,287), guaranteeing zero index overlap between
training and validation windows. This assertion passed on this run.

---

## Phase 5 DataLoader Handoff Note

Chunks are stored in contiguous document order. Phase 5 **must** configure the
training `DataLoader` with `shuffle=True` to break document ordering across batches.
The validation `DataLoader` must use `shuffle=False` (or no shuffle) to keep
evaluation order stable and reproducible across runs.
