"""Phase 4 — LoRA Configuration & Model Wrapping.

Goal: wrap SmolLM2-135M with a PEFT LoRA adapter, verify only the intended
parameters are trainable, and run a single-batch sanity forward/backward pass
before any real training (Phase 5).

Key design choices:
  - Dtype is CONDITIONAL on device:
      * CPU  → torch.float32   (fp16 addmm/matmul ops on CPU are unsupported
                                in many PyTorch builds; local dry-runs would
                                fail with a dtype error unrelated to code logic)
      * CUDA → torch.float16   (T4 is Turing-gen / SM 7.5; bfloat16 requires
                                Ampere / SM 8.0+ — bf16 on T4 falls back to
                                software emulation and is significantly slower;
                                fp16 with AMP is the correct choice for T4)

  - target_modules = ["q_proj", "v_proj"]  (plan §Phase 4 default)
      PEFT matches target_modules by SUBSTRING, not exact equality.
      "q_proj" matches "model.layers.0.self_attn.q_proj" because PEFT checks:
          any(target in name for target in target_modules)
      This means short strings like "proj" would inadvertently match many layers —
      always use the most specific suffix that uniquely identifies the layer type.

  - Three sweep configs authored up-front (r=4, r=8, r=16) for Phase 5 to consume.
      The sanity forward/backward uses r=8 (plan default). The sweep itself runs
      in Phase 5 — three separate training runs, each with its own metrics.jsonl.

⚠  FORWARD NOTE FOR PHASE 5 — fp16 TRAINING STABILITY:
    fp16 forward/backward WITHOUT loss scaling is a well-known source of NaN
    loss on T4, SEPARATE from the LR/clipping causes in Appendix A of the plan.
    Phase 5's training loop MUST:
      1.  Use  torch.cuda.amp.autocast(dtype=torch.float16)  around the forward
          pass (mixed-precision: compute in fp16, accumulate in fp32).
      2.  Use  torch.cuda.amp.GradScaler  to scale the loss before .backward()
          and unscale + clip + step via scaler.step(optimizer).
    Without GradScaler, fp16 gradients underflow to zero and training silently
    stalls or produces NaN. This is flagged here at Phase 4 so it is not
    discovered mid-training.

Outputs (relative to repo root):
  track1_finetune/configs/run_phase4_r4.json    — sweep config, r=4
  track1_finetune/configs/run_phase4_r8.json    — sweep config, r=8 (sanity run)
  track1_finetune/configs/run_phase4_r16.json   — sweep config, r=16
  track1_finetune/configs/trainable_params.json — required DoD deliverable
  (sanity_check_loss also recorded in trainable_params.json)

Run on Colab (from repo root after git pull):
  !python track1_finetune/scripts/wrap_lora.py

Definition of Done (plan §Phase 4):
  [x] target_modules confirmed correct for SmolLM2-135M architecture (validated
      at runtime against model.named_modules(), not guessed)
  [x] Trainable parameter count/percentage logged → configs/trainable_params.json
  [x] One successful forward/backward pass completed without error
"""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

# ── Bootstrap: make scripts/ importable regardless of CWD ──────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import SEED, MetricsLogger, dump_config, iso_now, set_seed  # noqa: E402
from config import (  # noqa: E402
    CONFIGS_DIR,
    MODEL_ARCH_JSON,
    TRAIN_PT,
    TRAINABLE_PARAMS_JSON,
)

# ── Hyperparameters ─────────────────────────────────────────────────────────
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"

# LoRA sweep candidates — Phase 5 will train all three as separate named runs.
# r=8 is used for the Phase 4 sanity forward/backward pass.
LORA_SWEEP = [
    {"r": 4,  "lora_alpha": 8,  "run_name": "phase4_r4"},
    {"r": 8,  "lora_alpha": 16, "run_name": "phase4_r8"},   # default / sanity check
    {"r": 16, "lora_alpha": 32, "run_name": "phase4_r16"},
]
LORA_DEFAULT_IDX = 1           # index into LORA_SWEEP to use for the sanity pass
LORA_DROPOUT = 0.05

# PEFT matches target_modules by SUBSTRING against module names (see module docstring).
# "q_proj" matches "model.layers.N.self_attn.q_proj" for any layer N.
# SmolLM2-135M is Llama-style with separate q_proj / k_proj / v_proj / o_proj.
# Plan default: start with q + v only; add k/o as a Phase 5 sweep dimension if needed.
#
# ⚠  GQA NOTE: SmolLM2-135M uses Grouped-Query Attention, so k_proj and v_proj
#    project to a SMALLER dimension than q_proj. The matrices are not equal-sized.
#    Do NOT assume equal per-module trainable-param contributions. The authoritative
#    count comes from print_trainable_parameters() output captured below.
TARGET_MODULES = ["q_proj", "v_proj"]

# Fallback module names if model_architecture.json is absent (offline / local run).
# These are the known SmolLM2-135M attention projection names, confirmed in
# configs/model_choice.md and cross-checked at runtime against named_modules().
SMOLLM2_KNOWN_ATTN_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",   # MLP projections (for reference)
]


# ── Helper: capture print_trainable_parameters() output ────────────────────

def capture_trainable_params(model) -> tuple[int, int, float]:
    """Return (trainable, total, pct) by capturing print_trainable_parameters().

    PEFT's print_trainable_parameters() prints a human-readable line but doesn't
    return structured data. We call it directly and also compute counts manually
    so we have machine-readable numbers for trainable_params.json.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = round(100.0 * trainable / total, 3) if total > 0 else 0.0

    # Also fire the official PEFT print (visible in Colab cell output).
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        model.print_trainable_parameters()
    finally:
        sys.stdout = old_stdout
    peft_line = buf.getvalue().strip()
    print(f"[phase4] PEFT: {peft_line}")

    return trainable, total, pct


# ── Helper: validate target_modules against actual model module names ───────

def validate_target_modules(model, target_modules: list[str]) -> list[str]:
    """Assert that every target string appears as a substring in at least one
    module name. Raises ValueError with the full module list if any miss.

    PEFT uses substring matching (not exact equality) — "q_proj" matches
    "model.layers.0.self_attn.q_proj". This function uses the same logic
    so the validation is consistent with what PEFT will actually do.
    """
    all_names = [name for name, _ in model.named_modules()]
    missing = []
    for t in target_modules:
        if not any(t in name for name in all_names):
            missing.append(t)

    if missing:
        attn_candidates = [
            n for n in all_names
            if any(kw in n for kw in ("proj", "attn", "mlp", "linear"))
        ][:30]
        raise ValueError(
            f"[phase4] target_modules validation FAILED.\n"
            f"  Not found (substring match) in named_modules(): {missing}\n"
            f"  Candidate module names (first 30 with 'proj'/'attn'/'mlp'):\n"
            + "\n".join(f"    {n}" for n in attn_candidates)
        )

    found_sample = {
        t: next(n for n in all_names if t in n)
        for t in target_modules
    }
    for t, example in found_sample.items():
        print(f"[phase4] ✓ target '{t}' → e.g. '{example}'")
    return all_names


def main() -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed = set_seed(SEED)
    print(f"[phase4] seed = {seed}")

    # ── 1. Device & dtype selection ────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Conditional dtype: fp32 on CPU (fp16 addmm is unsupported on CPU in many
    # PyTorch builds — would fail with a dtype error unrelated to code logic),
    # fp16 on CUDA (correct for T4/Turing; bf16 requires Ampere or newer).
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"[phase4] device = {device}  |  dtype = {dtype}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / 1e9, 2)
        print(f"[phase4] GPU: {props.name}  VRAM: {vram_gb} GB")

    # ── 2. Dump sweep configs (config-as-file before any compute) ─────────
    for cfg in LORA_SWEEP:
        run_cfg = {
            "phase": 4,
            "model_name": MODEL_NAME,
            "lora_r": cfg["r"],
            "lora_alpha": cfg["lora_alpha"],
            "lora_dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "seed": SEED,
            "timestamp": iso_now(),
            "note": (
                "Pre-sweep config authored at Phase 4. "
                "Training runs in Phase 5 consume these configs. "
                "See eval/sweep_results.csv for final comparison."
            ),
        }
        out = dump_config(run_cfg, cfg["run_name"])
        print(f"[phase4] sweep config saved → {out}")

    # ── 3. Read model_architecture.json for target module verification ─────
    if MODEL_ARCH_JSON.exists():
        arch = json.loads(MODEL_ARCH_JSON.read_text(encoding="utf-8"))
        arch_source = "configs/model_architecture.json"
        print(f"[phase4] loaded architecture from {arch_source}")
    else:
        print(
            f"[phase4] WARNING: {MODEL_ARCH_JSON} not found (not yet committed "
            "from Colab). Falling back to known SmolLM2-135M module names. "
            "Run Phase 2 (select_model.py) on Colab first to produce this file."
        )
        arch = {"all_module_names": SMOLLM2_KNOWN_ATTN_MODULES}
        arch_source = "fallback (known SmolLM2-135M names)"

    # ── 4. Load tokenizer ──────────────────────────────────────────────────
    print(f"\n[phase4] loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[phase4] pad_token set to eos_token (consistent with Phase 2 decision)")

    # ── 5. Load base model ─────────────────────────────────────────────────
    print(f"[phase4] loading base model in {dtype} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model = model.to(device)
    total_base_params = sum(p.numel() for p in model.parameters())
    print(f"[phase4] base model total params: {total_base_params:,}")

    # ── 6. Validate target_modules against real named_modules() ───────────
    print(f"\n[phase4] validating target_modules={TARGET_MODULES} ...")
    print(f"[phase4] architecture source: {arch_source}")
    # Validate against the ACTUAL loaded model — most reliable source.
    # (model_architecture.json is a backup reference, but runtime > JSON)
    try:
        validate_target_modules(model, TARGET_MODULES)
    except ValueError as e:
        print(str(e))
        print("\n[phase4] STOPPING. Fix target_modules before proceeding.")
        sys.exit(1)

    # ── 7. Apply LoRA adapter ──────────────────────────────────────────────
    default_cfg = LORA_SWEEP[LORA_DEFAULT_IDX]
    print(
        f"\n[phase4] applying LoRA: r={default_cfg['r']}, "
        f"alpha={default_cfg['lora_alpha']}, dropout={LORA_DROPOUT}, "
        f"target_modules={TARGET_MODULES}"
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=default_cfg["r"],
        lora_alpha=default_cfg["lora_alpha"],
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        # inference_mode=False is the default — explicit here for readability.
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)

    # ── 8. Capture & log trainable parameter counts ───────────────────────
    print("\n[phase4] trainable parameter counts:")
    trainable, total, pct = capture_trainable_params(model)
    print(f"[phase4] trainable: {trainable:,}  /  total: {total:,}  ({pct:.3f}%)")

    # ── 9. Sanity forward/backward pass ───────────────────────────────────
    print("\n[phase4] loading train tensors for sanity forward/backward pass ...")
    if not TRAIN_PT.exists():
        print(f"[phase4] ERROR: {TRAIN_PT} not found. Run Phase 3 first.")
        sys.exit(1)

    train_data = torch.load(TRAIN_PT, weights_only=True)
    dataset = TensorDataset(train_data["input_ids"], train_data["labels"])
    # batch_size=4 is enough for a sanity check — not tuned for Phase 5 throughput.
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    input_ids, labels = batch
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=2e-4
    )
    optimizer.zero_grad()

    # NOTE: On CUDA, the real Phase 5 training loop MUST use
    #   torch.cuda.amp.autocast(dtype=torch.float16)  +  GradScaler
    # to prevent fp16 gradient underflow / NaN loss. This sanity pass does
    # NOT use AMP because:
    #  (a) on CPU (local dry-run) AMP with fp16 doesn't apply, and
    #  (b) the goal here is only to confirm shapes and that backward() runs —
    #      not to simulate the full mixed-precision training stack.
    # Phase 5's training loop will add AMP. See module docstring for details.
    out = model(input_ids=input_ids, labels=labels)
    loss = out.loss
    print(f"[phase4] sanity forward pass — loss = {loss.item():.4f}")

    assert math.isfinite(loss.item()), (
        f"[phase4] FAIL: sanity loss is not finite ({loss.item()}) — "
        "check model loading, dtype, or label alignment."
    )

    loss.backward()
    print("[phase4] sanity backward pass — OK (no errors, gradients computed)")
    optimizer.zero_grad()  # clean up — we're not saving this state

    sanity_loss = round(loss.item(), 4)

    # ── 10. Write trainable_params.json (required DoD deliverable) ─────────
    trainable_params_record = {
        "model_name": MODEL_NAME,
        "lora_r": default_cfg["r"],
        "lora_alpha": default_cfg["lora_alpha"],
        "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
        "bias": "none",
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": pct,
        "sanity_check_loss": sanity_loss,
        "sanity_check_loss_finite": math.isfinite(sanity_loss),
        "device": str(device),
        "dtype": str(dtype),
        "architecture_source": arch_source,
        "seed": SEED,
        "timestamp": iso_now(),
        "note": (
            "trainable_parameters and trainable_percentage are authoritative — "
            "computed from sum(p.numel() for p in model.parameters() if p.requires_grad). "
            "GQA note: SmolLM2-135M uses Grouped-Query Attention; k_proj/v_proj project "
            "to a smaller dimension than q_proj — do not assume equal per-module "
            "contributions when reasoning about parameter counts. "
            "Phase 5 MUST use torch.cuda.amp.autocast + GradScaler for fp16 stability."
        ),
    }
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINABLE_PARAMS_JSON.write_text(
        json.dumps(trainable_params_record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n[phase4] trainable_params.json saved → {TRAINABLE_PARAMS_JSON}")

    # ── 11. Definition-of-Done assertions ──────────────────────────────────
    # Config files
    for cfg in LORA_SWEEP:
        cfg_path = CONFIGS_DIR / f"run_{cfg['run_name']}.json"
        assert cfg_path.exists(), f"FAIL: {cfg_path} not written"
    # Trainable params JSON
    assert TRAINABLE_PARAMS_JSON.exists(), f"FAIL: {TRAINABLE_PARAMS_JSON} not written"
    # Trainable % is in a sane LoRA range (should be <<10% for low-rank LoRA)
    assert 0 < pct < 10.0, (
        f"FAIL: trainable_percentage={pct:.3f}% is outside expected LoRA range (0–10%). "
        "Check that the base model was frozen and that target_modules took effect."
    )
    # Loss is finite
    assert math.isfinite(sanity_loss), "FAIL: sanity loss not finite"

    print("\n[phase4] ✅ all Definition-of-Done assertions passed")
    print(f"[phase4] trainable: {trainable:,} params  ({pct:.3f}%)")
    print(f"[phase4] sanity loss: {sanity_loss}")
    print(f"[phase4] sweep configs written: r=4, r=8, r=16  → configs/run_phase4_r*.json")
    print("[phase4] Phase 4 complete. Commit configs/ back to repo, then run Phase 5.")


if __name__ == "__main__":
    main()
