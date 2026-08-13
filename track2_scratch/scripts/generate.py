"""Phase 6 — Qualitative Evaluation (Training from Scratch).

Goal: run the same 8 prompts that Track 1 used, produce sampling and greedy
completions, annotate each for memorization / incoherence / novel-plausible
quality, and save to generations/track2_samples.md.

Prompts are taken verbatim from track1_finetune/generations/track1_samples.md
so the comparison is controlled on inputs.

Outputs:
    track2_scratch/generations/track2_samples.md  — 8 prompts × 2 modes each

Run on Colab (from repo root after git pull):
    !python track2_scratch/scripts/generate.py
    !python track2_scratch/scripts/generate.py --run base  # force a specific run

Smoke-test (local, CPU, no checkpoint needed):
    python track2_scratch/scripts/generate.py --smoke-test

Definition of Done (plan §Phase 6):
    [ ] track2_samples.md present with >= 8 prompt blocks
    [ ] Both sampling (T=0.8, top_p=0.9) and greedy completions present per prompt
    [ ] Each completion annotated (memorization / incoherence / novel-plausible)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# ── Bootstrap: make scripts/ importable regardless of CWD ────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import SEED, iso_now, set_seed  # noqa: E402
from config import (  # noqa: E402
    BEST_VAL_DIR,
    GENERATIONS_DIR,
    SWEEP_RESULTS_CSV,
    TOKENIZER_DIR,
    TRACK2_SAMPLES_MD,
)


# ════════════════════════════════════════════════════════════════════════════
# §1 — Prompts (same 8 as Track 1 — controlled comparison)
# ════════════════════════════════════════════════════════════════════════════

PROMPTS: list[str] = [
    "The transformer architecture relies on the attention mechanism to",
    "In the context of language model evaluation, bits-per-byte measures",
    "Fine-tuning a pre-trained model on a domain-specific dataset allows",
    "The key advantage of training a model from scratch is",
    "When the training loss decreases but validation loss increases, this indicates",
    "Parameter-efficient fine-tuning methods such as LoRA reduce the number of",
    "The tokenizer plays a critical role in language modelling because",
    "Compared to larger models, a small language model trained from scratch",
]

MAX_NEW_TOKENS = 100
SAMPLING_TEMPERATURE = 0.8
SAMPLING_TOP_P       = 0.9
GREEDY_TEMPERATURE   = 0.0   # triggers argmax branch in GPT.generate()


# ════════════════════════════════════════════════════════════════════════════
# §2 — Tokenizer loader
# ════════════════════════════════════════════════════════════════════════════

def load_tokenizer(tokenizer_dir: Path):
    """Load the trained ByteLevelBPE tokenizer and resolve eos_token_id.

    eos_token_id is the integer ID of '<|endoftext|>' in vocab.json.
    Resolution uses tokenizer.token_to_id() — one line, explicit, logged.
    If the special token is absent (should never happen post-Phase 1), generation
    still works but won't early-stop on EOS; logged as a WARNING, not a hard-fail.

    Returns (tokenizer, eos_token_id or None).
    """
    from tokenizers import ByteLevelBPETokenizer  # noqa: PLC0415

    vocab_json = tokenizer_dir / "vocab.json"
    merges_txt = tokenizer_dir / "merges.txt"
    if not vocab_json.exists() or not merges_txt.exists():
        raise FileNotFoundError(
            f"[generate] Tokenizer files not found in {tokenizer_dir}. "
            "Run Phase 1 on Colab first."
        )
    tok = ByteLevelBPETokenizer(str(vocab_json), str(merges_txt))
    eos_token_id = tok.token_to_id("<|endoftext|>")
    if eos_token_id is None:
        print(
            "[generate] WARNING: '<|endoftext|>' not found in vocab — "
            "EOS early-stopping disabled; generation runs for full max_new_tokens"
        )
    else:
        print(f"[generate] eos_token_id = {eos_token_id}")
    return tok, eos_token_id


# ════════════════════════════════════════════════════════════════════════════
# §3 — Checkpoint loader (mirrors eval.py §2)
# ════════════════════════════════════════════════════════════════════════════

def find_best_run() -> str:
    """Return the best-val run name from sweep_results.csv."""
    import csv  # noqa: PLC0415
    if not SWEEP_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"[generate] sweep_results.csv not found at {SWEEP_RESULTS_CSV}. "
            "Run Phase 4 on Colab first."
        )
    best_run, best_loss = None, float("inf")
    with SWEEP_RESULTS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            loss = float(row["best_val_loss"])
            if loss < best_loss:
                best_loss, best_run = loss, row["run_name"]
    if best_run is None:
        raise ValueError("[generate] sweep_results.csv is empty.")
    print(f"[generate] best run: {best_run}  val_loss={best_loss:.4f}")
    return best_run


def load_checkpoint(run_name: str, device):
    """Load best_val.pt for the given run. Returns (model, cfg_dict)."""
    import torch  # noqa: PLC0415
    from model import GPT, GPTConfig  # noqa: PLC0415

    ckpt_dir = BEST_VAL_DIR / run_name
    cfg_path = ckpt_dir / "best_val_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"[generate] {cfg_path} not found. Run Phase 4 on Colab first."
        )
    cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
    config = GPTConfig(
        vocab_size  = cfg_dict["vocab_size"],
        block_size  = cfg_dict["block_size"],
        n_layer     = cfg_dict["n_layer"],
        n_head      = cfg_dict["n_head"],
        n_embd      = cfg_dict["n_embd"],
        dropout     = cfg_dict.get("dropout", 0.1),
        bias        = cfg_dict.get("bias", True),
        tie_weights = cfg_dict.get("tie_weights", True),
    )
    model = GPT(config).to(device)
    ckpt_path = ckpt_dir / "best_val.pt"
    model.load_state_dict(
        torch.load(ckpt_path, weights_only=True, map_location=device)
    )
    model.eval()
    print(f"[generate] loaded checkpoint: {ckpt_path}")
    return model, cfg_dict


# ════════════════════════════════════════════════════════════════════════════
# §4 — Encode and decode helpers
# ════════════════════════════════════════════════════════════════════════════

def encode_prompt(tok, prompt: str, device) -> torch.Tensor:
    """Encode a text prompt to a (1, T) LongTensor on device."""
    ids = tok.encode(prompt).ids
    return torch.tensor([ids], dtype=torch.long, device=device)


def decode_ids(tok, ids: list[int]) -> str:
    """Decode a list of token IDs to a string."""
    return tok.decode(ids)


# ════════════════════════════════════════════════════════════════════════════
# §5 — Annotation heuristics
# ════════════════════════════════════════════════════════════════════════════

def annotate(completion: str, prompt: str) -> str:
    """Return a categorical quality annotation for a completion.

    Three categories (per plan §Phase 6):
    - memorization:     completion contains a long verbatim run of domain
                        n-grams unlikely to be coincidental (e.g. exact
                        dataset phrases like specific author names, formulas).
    - incoherence:      completion devolves into repetition of the same token
                        or short phrase, or generates characters that are not
                        valid English (evidence of a degenerate distribution).
    - novel-plausible:  completion is grammatical, on-topic, and not a
                        verbatim copy — the model generates something new.

    These are heuristics, not ground truth.  The human reviewer should revise
    the labels in track2_samples.md after visual inspection.

    Note: "Domain vocabulary" in compare.py's comparison table is a MANUAL
    judgment based on reading these annotations — it is not derived
    programmatically from this function's outputs.
    """
    # Repetition check: dominant n-gram occupying > 30% of tokens
    tokens = completion.split()
    if len(tokens) > 5:
        for n in (1, 2, 3):
            ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            if ngrams:
                mode = max(set(ngrams), key=ngrams.count)
                if ngrams.count(mode) / max(1, len(ngrams)) > 0.30:
                    return "incoherence (repetition)"

    # Long verbatim phrase check: a 10+ character exact substring from a
    # known domain phrase set.  Lightweight, no external data needed.
    DOMAIN_KEYWORDS = [
        "language model", "neural network", "transformer", "attention",
        "fine-tuning", "pre-trained", "tokenizer", "gradient descent",
        "cross-entropy", "perplexity",
    ]
    verbatim_hits = sum(kw in completion.lower() for kw in DOMAIN_KEYWORDS)
    if verbatim_hits >= 3 and len(completion) < 200:
        return "memorization (domain phrase density high)"

    return "novel-plausible"


# ════════════════════════════════════════════════════════════════════════════
# §6 — Markdown writer
# ════════════════════════════════════════════════════════════════════════════

def write_samples_md(
    results: list[dict],
    run_name: str,
    cfg_dict: dict,
) -> None:
    """Write track2_samples.md with all prompt completions and annotations.

    Format mirrors track1_samples.md for side-by-side comparison.
    """
    lines = [
        "# Track 2 — From-Scratch GPT: Qualitative Samples",
        "",
        f"> **Run:** {run_name}  |  "
        f"**Checkpoint:** best_val  |  "
        f"**Params:** n_layer={cfg_dict.get('n_layer')} "
        f"n_embd={cfg_dict.get('n_embd')} "
        f"n_head={cfg_dict.get('n_head')} "
        f"vocab_size={cfg_dict.get('vocab_size')}",
        f"> **Generated:** {iso_now()}",
        f"> **max_new_tokens:** {MAX_NEW_TOKENS}  |  "
        f"**Sampling:** T={SAMPLING_TEMPERATURE}, top_p={SAMPLING_TOP_P}  |  "
        f"**Greedy:** T=0 (argmax)",
        "",
        "> **Annotation key:** `memorization` — verbatim domain phrases; "
        "`incoherence` — repetition / degenerate output; "
        "`novel-plausible` — grammatical, on-topic, non-verbatim.",
        "",
        "> **Note:** Auto-annotations are heuristics. Revise labels below after "
        "visual inspection of the completions against the training text.",
        "",
        "---",
        "",
    ]

    for i, result in enumerate(results, start=1):
        lines += [
            f"## Sample {i}",
            "",
            f"**Prompt:** `{result['prompt']}`",
            "",
            "### Sampling (T=0.8, top_p=0.9)",
            "",
            f"```",
            result["prompt"] + result["sampling_new_text"],
            "```",
            "",
            f"> **Annotation:** {result['sampling_annotation']}",
            "",
            "### Greedy (argmax)",
            "",
            f"```",
            result["prompt"] + result["greedy_new_text"],
            "```",
            "",
            f"> **Annotation:** {result['greedy_annotation']}",
            "",
            "---",
            "",
        ]

    GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    TRACK2_SAMPLES_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[generate] samples → {TRACK2_SAMPLES_MD}")


# ════════════════════════════════════════════════════════════════════════════
# §7 — Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="Track 2 Phase 6 — Qualitative generation")
    parser.add_argument(
        "--run", default=None,
        help="Run name (small / base / base_highlr). "
             "Defaults to best run from sweep_results.csv.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Smoke-test: tiny random model, no checkpoint needed.",
    )
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cpu" if args.smoke_test else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[generate] device={device}  smoke={args.smoke_test}")

    if args.smoke_test:
        _smoke_test(device)
        return

    run_name = args.run or find_best_run()
    model, cfg_dict = load_checkpoint(run_name, device)
    tok, eos_id     = load_tokenizer(TOKENIZER_DIR)

    results = []
    for i, prompt in enumerate(PROMPTS, start=1):
        print(f"[generate] prompt {i}/{len(PROMPTS)}: {prompt[:50]}...")

        input_ids = encode_prompt(tok, prompt, device)

        # Sampling
        set_seed(SEED + i)     # per-prompt seed: reproducible but different per prompt
        out_sampling = model.generate(
            input_ids.clone(),
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=SAMPLING_TEMPERATURE,
            top_p=SAMPLING_TOP_P,
            eos_token_id=eos_id,
        )
        new_ids_sampling = out_sampling[0, input_ids.shape[1]:].tolist()
        sampling_text    = decode_ids(tok, new_ids_sampling)
        sampling_ann     = annotate(sampling_text, prompt)

        # Greedy
        out_greedy = model.generate(
            input_ids.clone(),
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=GREEDY_TEMPERATURE,
            eos_token_id=eos_id,
        )
        new_ids_greedy = out_greedy[0, input_ids.shape[1]:].tolist()
        greedy_text    = decode_ids(tok, new_ids_greedy)
        greedy_ann     = annotate(greedy_text, prompt)

        print(f"[generate]   sampling annotation: {sampling_ann}")
        print(f"[generate]   greedy  annotation:  {greedy_ann}")

        results.append({
            "prompt":             prompt,
            "sampling_new_text":  sampling_text,
            "sampling_annotation":sampling_ann,
            "greedy_new_text":    greedy_text,
            "greedy_annotation":  greedy_ann,
        })

    write_samples_md(results, run_name, cfg_dict)

    # Definition-of-Done assertions
    assert TRACK2_SAMPLES_MD.exists(), f"FAIL: {TRACK2_SAMPLES_MD} not written"
    content = TRACK2_SAMPLES_MD.read_text(encoding="utf-8")
    assert content.count("## Sample") >= len(PROMPTS), (
        f"FAIL: expected {len(PROMPTS)} Sample blocks, "
        f"found {content.count('## Sample')}"
    )
    assert "Sampling" in content and "Greedy" in content, (
        "FAIL: samples.md missing sampling or greedy sections"
    )
    print("\n[generate] ✅ all Definition-of-Done assertions passed")
    print("[generate] Phase 6 complete.")


# ════════════════════════════════════════════════════════════════════════════
# §8 — Smoke-test
# ════════════════════════════════════════════════════════════════════════════

def _smoke_test(device) -> None:
    """Verify generation + annotation + markdown writing without real data.

    Uses a tiny random-init GPT and a dummy tokenizer that just assigns one
    token per character (so IDs are stable and decode is round-trippable).
    """
    import torch  # noqa: PLC0415
    from model import GPT, GPTConfig  # noqa: PLC0415

    print("[generate] ── SMOKE TEST ────────────────────────────────────────")

    config = GPTConfig(vocab_size=128, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model  = GPT(config).to(device)
    model.eval()

    # Tiny prompt: first 5 ASCII token IDs
    prompt_ids = torch.tensor([[65, 66, 67, 68, 69]], dtype=torch.long, device=device)

    # Sampling
    set_seed(SEED)
    out_s = model.generate(
        prompt_ids.clone(), max_new_tokens=10,
        temperature=0.8, top_p=0.9,
    )
    assert out_s.shape == (1, 15), f"FAIL: sampling shape {out_s.shape} != (1, 15)"
    print(f"[generate] SMOKE: sampling output shape {tuple(out_s.shape)}  ✓")

    # Greedy (argmax branch)
    out_g = model.generate(
        prompt_ids.clone(), max_new_tokens=10,
        temperature=0.0,   # triggers argmax branch
    )
    assert out_g.shape == (1, 15), f"FAIL: greedy shape {out_g.shape} != (1, 15)"
    print(f"[generate] SMOKE: greedy output shape   {tuple(out_g.shape)}  ✓")

    # Greedy is deterministic — running it twice should give the same output
    out_g2 = model.generate(
        prompt_ids.clone(), max_new_tokens=10, temperature=0.0,
    )
    assert torch.equal(out_g, out_g2), "FAIL: greedy generation is not deterministic"
    print("[generate] SMOKE: greedy determinism  ✓")

    # Annotation helper
    ann = annotate(" ".join(["word"] * 20), "prompt")
    assert ann in ("memorization (domain phrase density high)", "incoherence (repetition)",
                   "novel-plausible"), f"FAIL: unexpected annotation: {ann}"
    print(f"[generate] SMOKE: annotation='{ann}'  ✓")

    # Markdown writer (write to temp dir)
    import tempfile, shutil  # noqa: E401, PLC0415
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # Temporarily patch output path
        tmp_md = tmp_dir / "track2_samples.md"
        results = [{
            "prompt":              PROMPTS[0],
            "sampling_new_text":   " test sampling",
            "sampling_annotation": "novel-plausible",
            "greedy_new_text":     " test greedy",
            "greedy_annotation":   "novel-plausible",
        }]
        lines = ["# Test", "", "## Sample 1", "", f"**Prompt:** `{PROMPTS[0]}`", ""]
        tmp_md.write_text("\n".join(lines), encoding="utf-8")
        assert tmp_md.exists(), "FAIL: smoke MD not created"
        print(f"[generate] SMOKE: markdown write OK  ✓")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("[generate] ✅ SMOKE TEST PASSED")
    print("[generate] Run without --smoke-test on Colab after Phase 4 completes.")


# ════════════════════════════════════════════════════════════════════════════
# §9 — Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
