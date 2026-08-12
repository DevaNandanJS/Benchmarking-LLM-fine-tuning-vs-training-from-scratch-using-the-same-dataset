"""Phase 7 — Qualitative Evaluation (Generation).

Goal: demonstrate the fine-tuned model can plausibly complete domain-relevant
text from the held-out validation region of the source document.

Prompts are selected deterministically (SEED=42) from the validation text span
(boundary: onwards in document_clean.txt) by extracting sentence starts and
taking the first ~60 tokens as the prompt prefix.

Two decoding modes per prompt:
    - Sampling:  do_sample=True, temperature=0.8, top_p=0.9
    - Greedy:    do_sample=False (deterministic)
Both are saved so the write-up can cite which is more coherent.

Output:
    track1_finetune/generations/track1_samples.md  — ≥8 prompt/completion pairs
                                                     with per-sample annotation

Smoke-test mode (--smoke):
    Runs 2 prompts on CPU with max_new_tokens=10. Used to verify shapes, decode
    paths, and markdown formatting before burning Colab inference time.

Run on Colab (from repo root after git pull):
    !python track1_finetune/scripts/generate.py --run r8

Definition of Done (plan §Phase 7):
    [ ] generations/track1_samples.md contains >=5 prompt/completion pairs
        with brief one-sentence annotations
"""
from __future__ import annotations

import argparse
import json
import random
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
    GENERATIONS_DIR,
    SWEEP_RESULTS_CSV,
    TRACK1_SAMPLES_MD,
)

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
NUM_PROMPTS = 8             # number of prompts to generate completions for
PROMPT_MAX_TOKENS = 60      # tokens to use as the prompt prefix
MAX_NEW_TOKENS = 80         # tokens to generate per prompt


def pick_best_run() -> str:
    """Read sweep_results.csv and return the run with lowest best_val_loss."""
    import csv
    if not SWEEP_RESULTS_CSV.exists():
        return "r8"
    best_run, best_loss = "r8", float("inf")
    with SWEEP_RESULTS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            loss = float(row["best_val_loss"])
            if loss < best_loss:
                best_loss = loss
                best_run = row["run_name"]
    return best_run


def extract_prompts(val_text: str, tokenizer, n: int, max_tokens: int,
                    seed: int = SEED) -> list[str]:
    """Extract n prompts from val_text by splitting at sentence ends and
    taking the first max_tokens worth of each selected sentence start.

    Selection is deterministic via the fixed seed.
    """
    # Split at period followed by space/newline (rough sentence boundary).
    import re
    sentences = re.split(r"(?<=[.!?])\s+", val_text)
    # Filter to sentences long enough to truncate meaningfully.
    candidates = [s.strip() for s in sentences if len(s.split()) > 20]

    rng = random.Random(seed)
    selected = rng.sample(candidates, min(n, len(candidates)))

    prompts = []
    for sent in selected:
        token_ids = tokenizer.encode(sent, add_special_tokens=False)
        truncated_ids = token_ids[:max_tokens]
        prompts.append(tokenizer.decode(truncated_ids, skip_special_tokens=True))

    return prompts


def generate_completion(model, tokenizer, prompt: str, device,
                        do_sample: bool, max_new_tokens: int) -> str:
    """Generate a single completion for the given prompt text."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=0.8, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)   # greedy

    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)

    # Decode only the newly generated tokens (strip the prompt prefix)
    new_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def write_samples_md(prompts: list[str], sampled: list[str],
                     greedy: list[str], run_name: str, smoke: bool) -> None:
    """Write prompt/completion pairs to track1_samples.md."""
    lines = [
        f"# Track 1 — Generation Samples (LoRA Fine-Tuning, {run_name.upper()})\n",
        "\n",
        "> **Source:** Completions generated from `HuggingFaceTB/SmolLM2-135M` fine-tuned\n",
        "> with LoRA on *LLM4Log: A Systematic Review of Large Language Model-based Log Analysis*.\n",
        "> Prompts are truncated sentence starts from the held-out **validation region**\n",
        "> (last 15% of the document by token position — not seen during training).\n",
        f"> Generated: {iso_now()}\n",
        "\n",
        "---\n",
        "\n",
    ]

    for i, (prompt, samp, grd) in enumerate(zip(prompts, sampled, greedy), 1):
        lines += [
            f"## Sample {i}\n",
            "\n",
            "**Prompt:**\n",
            f"> {prompt}\n",
            "\n",
            "**Completion (sampling, temperature=0.8, top_p=0.9):**\n",
            f"> {samp}\n",
            "\n",
            "**Completion (greedy):**\n",
            f"> {grd}\n",
            "\n",
            # One-sentence annotation placeholder — keep brief, factual.
            "**Annotation:** "
            "[TODO: one sentence — does the completion stay on-topic/domain-relevant? "
            "Is it fluent? Does it reproduce or hallucinate content?]\n",
            "\n",
            "---\n",
            "\n",
        ]

    if smoke:
        lines.insert(2, "> **SMOKE TEST** — completions generated on CPU with max_new_tokens=10.\n")

    GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    TRACK1_SAMPLES_MD.write_text("".join(lines), encoding="utf-8")
    print(f"[phase7] samples saved → {TRACK1_SAMPLES_MD}")


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="Phase 7 — Generation")
    parser.add_argument(
        "--run", default=None,
        help="Run name (r4/r8/r16). Default: best from sweep_results.csv.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-test: 2 prompts, 10 tokens, CPU. Verifies shapes and markdown output.",
    )
    args = parser.parse_args()
    smoke = args.smoke

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run or pick_best_run()
    print(f"[phase7] device={device}  run={run_name}  smoke={smoke}")

    # ── 1. Load tokenizer ─────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Load best checkpoint ───────────────────────────────────────────
    ckpt_dir = BEST_VAL_DIR / run_name
    if not ckpt_dir.exists() and not smoke:
        print(f"[phase7] ERROR: checkpoint not found at {ckpt_dir}. Run train.py first.")
        sys.exit(1)

    if smoke:
        # Use the base model (no LoRA checkpoint needed) for smoke shape testing
        print("[phase7] SMOKE: loading base model (no checkpoint) for shape verification")
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    else:
        from peft import PeftModel
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        print(f"[phase7] loading checkpoint from {ckpt_dir} ...")
        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, str(ckpt_dir))

    model = model.to(device)
    model.eval()

    # ── 3. Extract prompts from the validation text region ────────────────
    stats = json.loads(DATASET_STATS_JSON.read_text(encoding="utf-8"))
    boundary = stats["split_boundary_token_idx"]

    text = CLEAN_TXT.read_text(encoding="utf-8")
    # Approximate character boundary: decode boundary tokens from the token stream.
    # We use the same offset-mapping approach as evaluate.py for consistency.
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"]
    actual_boundary = min(boundary, len(offsets) - 1)
    char_start = offsets[actual_boundary][0]
    val_text = text[char_start:]

    n_prompts = 2 if smoke else NUM_PROMPTS
    max_new = 10 if smoke else MAX_NEW_TOKENS

    prompts = extract_prompts(val_text, tokenizer, n=n_prompts,
                               max_tokens=PROMPT_MAX_TOKENS, seed=SEED)
    print(f"[phase7] extracted {len(prompts)} prompts from validation text region")

    # ── 4. Generate completions ───────────────────────────────────────────
    sampled_completions = []
    greedy_completions = []

    for i, prompt in enumerate(prompts):
        print(f"[phase7] generating for prompt {i+1}/{len(prompts)} ...")
        samp = generate_completion(model, tokenizer, prompt, device,
                                   do_sample=True, max_new_tokens=max_new)
        grd = generate_completion(model, tokenizer, prompt, device,
                                  do_sample=False, max_new_tokens=max_new)
        sampled_completions.append(samp)
        greedy_completions.append(grd)

    # ── 5. Write markdown output ──────────────────────────────────────────
    write_samples_md(prompts, sampled_completions, greedy_completions, run_name, smoke)

    # ── 6. Definition-of-Done assertion ──────────────────────────────────
    assert TRACK1_SAMPLES_MD.exists(), f"FAIL: {TRACK1_SAMPLES_MD} not written"
    content = TRACK1_SAMPLES_MD.read_text(encoding="utf-8")
    assert content.count("## Sample") >= (2 if smoke else 5), (
        "FAIL: fewer than 5 prompt/completion pairs in track1_samples.md"
    )

    print(f"\n[phase7] ✅ Definition-of-Done: {content.count('## Sample')} samples written")
    print("[phase7] Phase 7 complete. Fill in the [TODO] annotations in track1_samples.md.")


if __name__ == "__main__":
    main()
