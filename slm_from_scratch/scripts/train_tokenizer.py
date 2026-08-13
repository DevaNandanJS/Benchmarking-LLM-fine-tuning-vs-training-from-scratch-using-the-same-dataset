"""Phase 1 — Custom Tokenizer Training with Vocabulary Sweep.

Goal (plan §Phase 1): train a ByteLevelBPE tokenizer on document_clean.txt,
sweep 3 vocabulary sizes (256 / 1024 / 4096), compute fertility for each,
select the best size using a principled diminishing-returns criterion, save
the final tokenizer files, and write a quantitative justification.

Why ByteLevelBPE:
  - Base units are raw bytes (256 possible), so any Unicode text (including
    LaTeX artifacts, special dashes, non-ASCII punctuation in the PDF) is
    representable with zero unknown tokens.
  - BPE is simple, auditable, and directly implementable — consistent with the
    plan's requirement that we understand every line.

Why the tokenizer sees the FULL document (not just the train split):
  - The tokenizer learns a COMPRESSION SCHEME (which byte-pairs to merge),
    not a predictive model. It gains no ability to predict the next token
    from seeing val-set text.
  - Training the tokenizer only on the train split would risk val text
    containing subword patterns outside the vocabulary — a worse problem
    than any theoretical "leakage."
  - This is industry standard: GPT-2's tokenizer was trained on WebText,
    which overlaps with every NLP benchmark ever built.

Vocabulary selection criterion:
  - Compute fertility = total_tokens / total_words for each size.
  - Use a DIMINISHING-RETURNS ELBOW, not an arbitrary threshold:
      ratio = Δfertility(1024→4096) / Δfertility(256→1024)
    If ratio < 0.30, the jump from 1024 to 4096 bought <30% of the
    compression gain we got from 256 to 1024 — not worth 4× the embedding
    table size. Choose 1024.
  - Also check for corpus plateau: if actual_vocab < 0.90 × requested,
    BPE exhausted the corpus's merge budget — flag and prefer the smaller size.

Outputs (relative to slm_from_scratch/):
  eval/vocab_sweep.csv          — requested|actual|fertility|total_tokens
  tokenizer/vocab.json          — final BPE vocabulary
  tokenizer/merges.txt          — final merge rules
  configs/tokenizer_choice.md   — written justification with actual numbers
  configs/run_phase1_tokenizer.json — config-as-file (before any training)

Run on Colab (from repo root after git pull):
  !python slm_from_scratch/scripts/train_tokenizer.py

Definition of Done (plan Phase 1):
  [x] vocab_sweep.csv with ≥3 candidate vocab sizes, fertility, token counts
  [x] Final tokenizer saved to slm_from_scratch/tokenizer/
  [x] Written justification referencing actual sweep numbers
"""
from __future__ import annotations

import csv
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
    EVAL_DIR,
    TOKENIZER_DIR,
    TOKENIZER_CHOICE_MD,
    VOCAB_SWEEP_CSV,
)

# ── Hyperparameters ──────────────────────────────────────────────────────────
SWEEP_VOCAB_SIZES = [256, 1024, 4096]   # candidates; see plan §Phase 1 step 2
MIN_FREQUENCY     = 2                   # BPE merge threshold
SPECIAL_TOKENS    = ["<|endoftext|>"]   # single EOS/padding token
# Diminishing-returns threshold: if the 1024→4096 fertility gain is <30% of
# the 256→1024 gain, declare 4096 to have diminishing returns.
DIMINISHING_RATIO_THRESHOLD = 0.30
# Plateau detection: if actual vocab < 90% of requested, corpus is exhausted.
PLATEAU_FRACTION = 0.90


def train_bpe(vocab_size: int) -> tuple[int, float, int]:
    """Train a ByteLevelBPETokenizer at the given vocab_size.

    Returns:
        actual_vocab_size: vocabulary entries actually produced (may be < requested)
        fertility:         average tokens per whitespace-split word
        total_tokens:      total token count for the full document
    """
    from tokenizers import ByteLevelBPETokenizer  # noqa: PLC0415

    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[str(CLEAN_TXT)],
        vocab_size=vocab_size,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    )

    # ── Encode the full document to get statistics ───────────────────────────
    text = CLEAN_TXT.read_text(encoding="utf-8")
    encoding = tok.encode(text)
    total_tokens: int = len(encoding.ids)

    # Whitespace-split word count (same denominator Track 1 stats.json uses)
    total_words: int = len(text.split())
    fertility: float = round(total_tokens / total_words, 4) if total_words else 0.0

    actual_vocab_size: int = tok.get_vocab_size()

    print(
        f"  vocab_size={vocab_size:>5}  actual={actual_vocab_size:>5}  "
        f"fertility={fertility:.4f}  total_tokens={total_tokens:,}"
    )

    # ── Save this candidate's tokenizer for inspection ───────────────────────
    cand_dir = TOKENIZER_DIR.parent / "tokenizer_candidates" / f"vocab{vocab_size}"
    cand_dir.mkdir(parents=True, exist_ok=True)
    tok.save_model(str(cand_dir))

    return actual_vocab_size, fertility, total_tokens


def select_vocab_size(
    sweep: dict[int, dict],
) -> tuple[int, str]:
    """Choose the best vocab size using a diminishing-returns elbow criterion.

    Args:
        sweep: {requested_vocab: {actual, fertility, total_tokens}}

    Returns:
        chosen_vocab: the chosen requested vocab size
        reasoning:    human-readable explanation referencing actual numbers
    """
    sizes = sorted(sweep.keys())   # [256, 1024, 4096]

    f256  = sweep[256]["fertility"]
    f1024 = sweep[1024]["fertility"]
    f4096 = sweep[4096]["fertility"]

    delta_low  = f256  - f1024   # fertility drop from 256→1024 (positive = improvement)
    delta_high = f1024 - f4096   # fertility drop from 1024→4096

    # Guard: if corpus is so small that fertility barely changes at all,
    # delta_low could be near-zero — avoid division by zero.
    if delta_low < 0.01:
        ratio = 0.0
        ratio_str = "N/A (Δ256→1024 is near-zero; corpus likely too small for any merges)"
    else:
        ratio = delta_high / delta_low
        ratio_str = f"{ratio:.3f}"

    plateau_4096 = sweep[4096]["actual"] < PLATEAU_FRACTION * 4096
    plateau_1024 = sweep[1024]["actual"] < PLATEAU_FRACTION * 1024

    lines: list[str] = [
        "## Vocabulary sweep results",
        "",
        f"| Requested | Actual | Fertility | Total tokens |",
        f"|-----------|--------|-----------|--------------|",
    ]
    for v in sizes:
        s = sweep[v]
        lines.append(
            f"| {v:>9} | {s['actual']:>6} | {s['fertility']:>9.4f} | {s['total_tokens']:>12,} |"
        )

    lines += [
        "",
        "## Selection criterion: diminishing-returns elbow",
        "",
        f"- Fertility drop 256→1024:  {delta_low:.4f}",
        f"- Fertility drop 1024→4096: {delta_high:.4f}",
        f"- Ratio (high/low):         {ratio_str}  (threshold: {DIMINISHING_RATIO_THRESHOLD})",
        f"- Corpus plateau at 4096:   {'YES' if plateau_4096 else 'NO'}  "
        f"(actual={sweep[4096]['actual']} vs requested=4096, "
        f"threshold={PLATEAU_FRACTION:.0%})",
        "",
    ]

    if plateau_4096 and ratio < DIMINISHING_RATIO_THRESHOLD:
        chosen = 1024
        reason = (
            f"Both signals point to 1024: the corpus exhausted BPE's merge budget at 4096 "
            f"(actual vocab {sweep[4096]['actual']} < {int(PLATEAU_FRACTION*4096)}), "
            f"AND the fertility gain from 1024→4096 was only {ratio:.1%} of the 256→1024 gain — "
            f"far below the {DIMINISHING_RATIO_THRESHOLD:.0%} diminishing-returns threshold. "
            f"Going to 4096 would expand the embedding table 4× for near-zero compression benefit."
        )
    elif plateau_4096:
        chosen = 1024
        reason = (
            f"4096-vocab training plateaued: only {sweep[4096]['actual']} actual tokens produced "
            f"(corpus too small to fill {int(PLATEAU_FRACTION*4096)}+ merge operations). "
            f"Using 1024 gives real merges with a vocabulary the corpus can actually support."
        )
    elif ratio < DIMINISHING_RATIO_THRESHOLD:
        chosen = 1024
        reason = (
            f"Diminishing returns: 1024→4096 improved fertility by only "
            f"{delta_high:.4f} ({ratio:.1%} of the 256→1024 gain of {delta_low:.4f}). "
            f"The 4× embedding table growth (1024×192=196k → 4096×192=786k parameters) "
            f"is not justified by this marginal compression improvement."
        )
    else:
        chosen = 4096
        reason = (
            f"The corpus supported 4096 merges without plateau AND the 1024→4096 fertility "
            f"improvement ({delta_high:.4f}) was {ratio:.1%} of the 256→1024 gain — "
            f"returns are still substantial. Selecting 4096."
        )

    lines += [
        f"## Chosen vocabulary size: {chosen}",
        "",
        reason,
        "",
        "## Note on tokenizer training with the full document",
        "",
        "The tokenizer sees the full corpus (train + val text combined) to learn merge rules.",
        "This is intentional and standard practice — the tokenizer learns a *compression scheme*,",
        "not a predictive model, so it gains no ability to predict held-out tokens.",
        "Training on the train split only would risk val text containing out-of-vocabulary",
        "subword patterns, which is a worse problem. (Reference: GPT-2's tokenizer was trained",
        "on WebText, which overlaps with every NLP benchmark.)",
        "",
        f"*Generated by train_tokenizer.py at {iso_now()}*",
    ]

    return chosen, "\n".join(lines)


def sanity_check(tokenizer_dir: Path) -> None:
    """Tokenize 5 sampled sentences and print token strings for visual inspection."""
    from tokenizers import ByteLevelBPETokenizer  # noqa: PLC0415

    tok = ByteLevelBPETokenizer(
        vocab=str(tokenizer_dir / "vocab.json"),
        merges=str(tokenizer_dir / "merges.txt"),
    )

    text = CLEAN_TXT.read_text(encoding="utf-8")
    # Sample 5 sentences from different thirds of the document
    sentences = [s.strip() for s in text.split(".") if 20 < len(s.strip()) < 200]
    n = max(len(sentences) // 5, 1)
    samples = [sentences[i * n] for i in range(5) if i * n < len(sentences)]

    print("\n── Sanity check: tokenization of 5 sample sentences ──")
    for i, sent in enumerate(samples[:5], 1):
        enc = tok.encode(sent)
        tokens = enc.tokens
        fragmentation = "OK" if len(tokens) / max(len(sent.split()), 1) < 8 else "⚠ HIGH FRAGMENTATION"
        print(f"\n[{i}] Input:  {sent[:80]}{'...' if len(sent) > 80 else ''}")
        print(f"    Tokens ({len(tokens)}): {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        print(f"    Fragmentation: {fragmentation}")


def main() -> None:
    set_seed(SEED)

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not CLEAN_TXT.exists():
        raise FileNotFoundError(
            f"document_clean.txt not found at {CLEAN_TXT}\n"
            "Run Track 1 Phase 1 first to extract and clean the document."
        )

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Config-as-file BEFORE training (global convention §0) ────────────────
    run_cfg = {
        "phase": "phase1_tokenizer",
        "sweep_vocab_sizes": SWEEP_VOCAB_SIZES,
        "min_frequency": MIN_FREQUENCY,
        "special_tokens": SPECIAL_TOKENS,
        "diminishing_ratio_threshold": DIMINISHING_RATIO_THRESHOLD,
        "plateau_fraction": PLATEAU_FRACTION,
        "input_file": str(CLEAN_TXT),
        "seed": SEED,
    }
    dump_config(run_cfg, "phase1_tokenizer")

    # ── Vocabulary sweep ──────────────────────────────────────────────────────
    print("\n═══ Track 2 Phase 1 — Vocabulary Sweep ═══")
    print(f"Input: {CLEAN_TXT}")
    print(f"Candidates: {SWEEP_VOCAB_SIZES}\n")

    sweep: dict[int, dict] = {}
    for vocab_size in SWEEP_VOCAB_SIZES:
        print(f"Training BPE vocab_size={vocab_size} ...")
        actual, fertility, total_tokens = train_bpe(vocab_size)
        sweep[vocab_size] = {
            "requested": vocab_size,
            "actual":    actual,
            "fertility": fertility,
            "total_tokens": total_tokens,
        }

    # ── Write vocab_sweep.csv ─────────────────────────────────────────────────
    with VOCAB_SWEEP_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["requested_vocab_size", "actual_vocab_size",
                        "fertility", "total_tokens"],
        )
        writer.writeheader()
        for v, s in sorted(sweep.items()):
            writer.writerow({
                "requested_vocab_size": s["requested"],
                "actual_vocab_size":    s["actual"],
                "fertility":            s["fertility"],
                "total_tokens":         s["total_tokens"],
            })
    print(f"\n[sweep] Written: {VOCAB_SWEEP_CSV}")

    # ── Select best vocabulary size ───────────────────────────────────────────
    chosen_vocab, justification_md = select_vocab_size(sweep)
    print(f"\n[selection] Chosen vocab size: {chosen_vocab}")
    print(textwrap.indent(justification_md, "  "))

    TOKENIZER_CHOICE_MD.write_text(justification_md + "\n", encoding="utf-8")
    print(f"[choice] Written: {TOKENIZER_CHOICE_MD}")

    # ── Re-train and save the FINAL chosen tokenizer ─────────────────────────
    print(f"\nRe-training final tokenizer at vocab_size={chosen_vocab} ...")
    from tokenizers import ByteLevelBPETokenizer  # noqa: PLC0415

    final_tok = ByteLevelBPETokenizer()
    final_tok.train(
        files=[str(CLEAN_TXT)],
        vocab_size=chosen_vocab,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    )
    final_tok.save_model(str(TOKENIZER_DIR))
    print(f"[tokenizer] Saved to: {TOKENIZER_DIR}")
    print(f"  Files: {list(TOKENIZER_DIR.iterdir())}")

    # ── Sanity check ──────────────────────────────────────────────────────────
    sanity_check(TOKENIZER_DIR)

    print("\n═══ Phase 1 complete ═══")
    print(f"  vocab_sweep.csv   → {VOCAB_SWEEP_CSV}")
    print(f"  tokenizer/        → {TOKENIZER_DIR}")
    print(f"  tokenizer_choice  → {TOKENIZER_CHOICE_MD}")


if __name__ == "__main__":
    main()
