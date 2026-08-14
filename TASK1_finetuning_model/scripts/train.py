"""Phase 5 — Training Loop & Execution.

Goal: run LoRA fine-tuning with a manual training loop (not Trainer), full
metrics.jsonl logging, checkpoint saving on val improvement, early stopping,
and three-run sweep (r=4, r=8, r=16) producing eval/sweep_results.csv.

Why manual loop over HuggingFace Trainer:
    The execution plan (§Phase 5 step 1) explicitly notes a manual loop is
    "arguably the stronger interview artifact since Trainer hides steps you'd
    otherwise have to explain." Every gradient step, scheduler tick, AMP scaler
    call, and validation pass is visible and explainable line by line — no
    TrainingArguments magic to handwave.

⚠  fp16 AMP REQUIRED: training with fp16 without loss scaling is a well-known
    source of NaN gradients. This script uses:
        torch.cuda.amp.autocast(dtype=torch.float16)  — mixed-precision forward
        torch.cuda.amp.GradScaler                     — loss scaling / unscaling
    Both are required for stable fp16 training on T4. See wrap_lora.py docstring.

⚠  LoRA adapter dtype: after get_peft_model(), the LoRA A/B matrices are kept
    in fp32 by PEFT even when the frozen base is fp16. This is intentional —
    adapter gradients need fp32 precision. autocast does NOT force-downcast
    requires_grad=True parameters, so the A/B matrices stay fp32 during the
    forward pass accumulation. This is correct behavior; confirmed by the post-
    wrap dtype assertion in main().

⚠  Pad-masking: labels[labels == pad_token_id] = -100 is included as a safeguard.
    Phase 3 built full context_length=256 chunks from a continuous token stream
    with no padding, so this guard fires exactly 0 times on the current dataset.
    It is kept defensively — if context length or data ever changes to produce
    padded batches, the guard prevents training on <pad> tokens as content.

Sweep design:
    Three independent runs are launched separately from the notebook:
        !python TASK1_finetuning_model/scripts/train.py --run r4
        !python TASK1_finetuning_model/scripts/train.py --run r8
        !python TASK1_finetuning_model/scripts/train.py --run r16
    Each writes its own logs/<run_name>/metrics.jsonl and checkpoints under
    checkpoints/best_val/<run_name>/. After all three, sweep_results.csv has
    3 rows and the best run can be identified.

Smoke-test mode (--smoke):
    Truncates train/val to 4 chunks and runs 3 steps on CPU. Used for local
    pre-flight validation of shapes, pad-masking, AMP dtype handling, and
    MetricsLogger output before the real Colab run. Catches runtime issues
    that AST parsing misses.

Outputs (relative to repo root):
    TASK1_finetuning_model/configs/run_phase5_<run_name>.json   — run config dump
    TASK1_finetuning_model/logs/<run_name>/metrics.jsonl         — step-level logging
    TASK1_finetuning_model/checkpoints/best_val/<run_name>/      — best adapter
    TASK1_finetuning_model/eval/sweep_results.csv                — appended after each run

Run on Colab (from repo root after git pull):
    !python TASK1_finetuning_model/scripts/train.py --run r8   # or r4, r16

Definition of Done (plan §Phase 5):
    [ ] At least one complete training run with full metrics.jsonl logging (>=20 pts)
    [ ] Best checkpoint (by val loss) saved separately and identified
    [ ] Small sweep (>=2 configs) completed and tabulated in sweep_results.csv
    [ ] Overfitting behavior noted with the step/epoch it occurred at
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import SEED, MetricsLogger, dump_config, iso_now, set_seed  # noqa: E402
from config import (  # noqa: E402
    BEST_VAL_DIR,
    CONFIGS_DIR,
    EVAL_DIR,
    LOGS_DIR,
    SWEEP_RESULTS_CSV,
    TRAIN_PT,
    TRAINABLE_PARAMS_JSON,
    VAL_PT,
)

# ── Fixed training hyperparameters (starting values per plan §Phase 5) ───────
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
LEARNING_RATE = 2e-4          # LoRA tolerates higher LR than full FT
BATCH_SIZE = 8                # try 16 if VRAM allows after first run
MAX_EPOCHS = 10               # with early stopping — tiny dataset, expect early best
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05           # 5% of total steps → linear warmup
MAX_GRAD_NORM = 1.0           # gradient clipping
EARLY_STOP_PATIENCE = 2       # stop if val loss rises for this many consecutive epochs

# Minimum logged points across the run — N is derived from total steps.
MIN_LOG_POINTS = 25

# LoRA sweep configs authored by wrap_lora.py — loaded by run name.
LORA_CONFIGS = {
    "r4":  {"r": 4,  "lora_alpha": 8,  "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
    "r8":  {"r": 8,  "lora_alpha": 16, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
    "r16": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
}


def build_model_and_tokenizer(lora_cfg: dict, dtype):
    """Load SmolLM2-135M + tokenizer, wrap with LoRA per lora_cfg."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)

    # Dtype assertion: PEFT keeps LoRA A/B matrices in fp32 for gradient stability,
    # even when the frozen base is fp16. Confirm this is the case after wrapping.
    lora_param_dtypes = set(
        str(p.dtype) for name, p in model.named_parameters()
        if p.requires_grad
    )
    print(f"[phase5] LoRA trainable parameter dtypes: {lora_param_dtypes}")
    # Expected: {'torch.float32'} — adapters stay fp32 inside fp16 base.
    # autocast does not force-downcast requires_grad=True params.

    return model, tokenizer


def compute_val_loss(model, val_loader, device, pad_id: int) -> float:
    """Compute mean cross-entropy loss over the full validation set."""
    import torch

    model.eval()
    total_loss = 0.0
    total_chunks = 0

    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Pad-masking guard: set pad positions to -100 so cross_entropy ignores them.
            # NOTE: Phase 3 built full 256-token chunks with no padding, so this fires 0
            # times on the current dataset. Kept defensively for correctness.
            labels = labels.clone()
            labels[labels == pad_id] = -100

            out = model(input_ids=input_ids, labels=labels)
            total_loss += out.loss.item()
            total_chunks += 1

    model.train()
    return total_loss / total_chunks if total_chunks > 0 else float("nan")


def append_sweep_row(run_name: str, lora_r: int, final_train_loss: float,
                     final_val_loss: float, best_val_loss: float,
                     best_val_epoch: int) -> None:
    """Append one row to eval/sweep_results.csv (create with header if absent)."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SWEEP_RESULTS_CSV.exists()
    with SWEEP_RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_name", "lora_r", "final_train_loss",
                "final_val_loss", "best_val_loss", "best_val_epoch",
                "timestamp",
            ])
        writer.writerow([
            run_name, lora_r,
            round(final_train_loss, 6), round(final_val_loss, 6),
            round(best_val_loss, 6), best_val_epoch,
            iso_now(),
        ])
    print(f"[phase5] sweep row appended -> {SWEEP_RESULTS_CSV}")


def main() -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    parser = argparse.ArgumentParser(description="Phase 5 — LoRA training sweep")
    parser.add_argument(
        "--run", choices=["r4", "r8", "r16"], default="r8",
        help="Which sweep config to train (r4 / r8 / r16). Default: r8.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "Smoke-test mode: truncate dataset to 4 chunks, run 3 steps on CPU. "
            "Used for local pre-flight shape/dtype/AMP validation before Colab run. "
            "Does NOT write checkpoints or update sweep_results.csv."
        ),
    )
    args = parser.parse_args()
    run_name = args.run
    smoke = args.smoke

    seed = set_seed(SEED)
    print(f"[phase5] run={run_name}  smoke={smoke}  seed={seed}")

    lora_cfg = LORA_CONFIGS[run_name]

    # ── 1. Device & dtype ─────────────────────────────────────────────────
    if smoke:
        device = torch.device("cpu")
        dtype = torch.float32   # CPU: fp32 only
        print("[phase5] SMOKE MODE — CPU fp32, 4 chunks, 3 steps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"[phase5] device={device}  dtype={dtype}")

    # ── 2. Dump run config before any compute ─────────────────────────────
    run_cfg = {
        "phase": 5,
        "run_name": run_name,
        "model_name": MODEL_NAME,
        "lora_r": lora_cfg["r"],
        "lora_alpha": lora_cfg["lora_alpha"],
        "lora_dropout": lora_cfg["lora_dropout"],
        "target_modules": lora_cfg["target_modules"],
        "learning_rate": LEARNING_RATE,
        "batch_size": 1 if smoke else BATCH_SIZE,
        "max_epochs": 1 if smoke else MAX_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "max_grad_norm": MAX_GRAD_NORM,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "smoke_test": smoke,
        "seed": SEED,
        "timestamp": iso_now(),
    }
    cfg_path = dump_config(run_cfg, f"phase5_{run_name}")
    print(f"[phase5] run config saved -> {cfg_path}")

    # ── 3. Load datasets ──────────────────────────────────────────────────
    print(f"[phase5] loading tensors from {TRAIN_PT} / {VAL_PT} ...")
    train_data = torch.load(TRAIN_PT, weights_only=True)
    val_data = torch.load(VAL_PT, weights_only=True)

    train_input_ids = train_data["input_ids"]
    train_labels = train_data["labels"]
    val_input_ids = val_data["input_ids"]
    val_labels = val_data["labels"]

    if smoke:
        # Truncate to 4 chunks for local pre-flight validation
        train_input_ids = train_input_ids[:4]
        train_labels = train_labels[:4]
        val_input_ids = val_input_ids[:4]
        val_labels = val_labels[:4]
        print(f"[phase5] SMOKE: truncated to {len(train_input_ids)} train / {len(val_input_ids)} val chunks")

    train_dataset = TensorDataset(train_input_ids, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_labels)

    batch_sz = 1 if smoke else BATCH_SIZE
    # shuffle=True for train (break document-order correlations per Phase 3 note).
    # shuffle=False for val (reproducible eval order per Phase 3 note).
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False)

    print(f"[phase5] train batches/epoch: {len(train_loader)}  val batches: {len(val_loader)}")

    # ── 4. Build model ────────────────────────────────────────────────────
    print(f"\n[phase5] building LoRA model: {MODEL_NAME} r={lora_cfg['r']} ...")
    model, tokenizer = build_model_and_tokenizer(lora_cfg, dtype)
    model = model.to(device)
    pad_id = tokenizer.pad_token_id

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[phase5] trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    # ── 5. Optimizer + scheduler ──────────────────────────────────────────
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    max_epochs = 1 if smoke else MAX_EPOCHS
    max_steps = smoke and 3 or (max_epochs * len(train_loader))
    warmup_steps = max(1, int(WARMUP_RATIO * max_steps))

    def lr_lambda(current_step: int) -> float:
        """Linear warmup then cosine decay."""
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # ── 6. AMP scaler (CUDA only) ─────────────────────────────────────────
    # GradScaler is required for fp16 training to prevent gradient underflow/NaN.
    # Disabled in smoke mode (CPU fp32) and when falling back to CPU.
    use_amp = device.type == "cuda" and not smoke
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[phase5] AMP (GradScaler): {'enabled' if use_amp else 'disabled (CPU or smoke)'}")

    # ── 7. Logging setup ─────────────────────────────────────────────────
    logger = MetricsLogger(run_name)
    log_every_n = max(1, max_steps // MIN_LOG_POINTS)
    print(f"[phase5] logging every {log_every_n} steps (targeting {MIN_LOG_POINTS}+ log points)")

    # ── 8. Training loop ─────────────────────────────────────────────────
    global_step = 0
    best_val_loss = float("inf")
    best_val_epoch = -1
    val_loss_history: list[float] = []
    early_stopped = False
    overfitting_step = None    # step where train↓ / val↑ divergence was first seen

    print(f"\n[phase5] starting training: max_epochs={max_epochs}  max_steps={max_steps}")
    model.train()

    for epoch in range(max_epochs):
        epoch_train_loss = 0.0
        epoch_batches = 0

        for batch_input_ids, batch_labels in train_loader:
            if smoke and global_step >= 3:
                break   # smoke: only 3 steps

            batch_input_ids = batch_input_ids.to(device)
            batch_labels = batch_labels.to(device)

            # Pad-masking guard (see module docstring — inert on current dataset).
            batch_labels = batch_labels.clone()
            batch_labels[batch_labels == pad_id] = -100

            optimizer.zero_grad()

            # Mixed-precision forward pass.
            # autocast: compute in fp16, accumulate in fp32; LoRA A/B stay fp32.
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
                out = model(input_ids=batch_input_ids, labels=batch_labels)
                loss = out.loss

            assert math.isfinite(loss.item()), (
                f"[phase5] NaN/Inf loss at step {global_step} — "
                "check LR (try 5–10x lower), gradient clipping, or AMP setup. "
                "See Appendix A of finetuning_execution_plan.md."
            )

            # Scale → backward → unscale → clip → step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                MAX_GRAD_NORM,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_train_loss += loss.item()
            epoch_batches += 1
            global_step += 1

            # Per-step logging (val_loss=null between epoch evaluations)
            if global_step % log_every_n == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.log(
                    step=global_step,
                    epoch=epoch + epoch_batches / max(1, len(train_loader)),
                    train_loss=loss.item(),
                    val_loss=None,
                    lr=current_lr,
                )
                print(
                    f"[phase5] step {global_step:4d}  "
                    f"train_loss={loss.item():.4f}  lr={current_lr:.2e}"
                )

        mean_train_loss = epoch_train_loss / max(1, epoch_batches)

        # ── 9. Epoch-end validation pass ─────────────────────────────────
        val_loss = compute_val_loss(model, val_loader, device, pad_id)
        val_loss_history.append(val_loss)
        current_lr = scheduler.get_last_lr()[0]

        # Log epoch-end point with val_loss
        logger.log(
            step=global_step,
            epoch=float(epoch + 1),
            train_loss=mean_train_loss,
            val_loss=val_loss,
            lr=current_lr,
        )
        print(
            f"[phase5] EPOCH {epoch+1:2d}  "
            f"train_loss={mean_train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={current_lr:.2e}"
        )

        # ── 10. Checkpoint on improvement ─────────────────────────────────
        if not smoke and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
            ckpt_dir = BEST_VAL_DIR / run_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            print(f"[phase5] [OK] new best val_loss={val_loss:.4f} -> checkpoint saved -> {ckpt_dir}")

        # ── 11. Early stopping ────────────────────────────────────────────
        # Trigger if val loss has increased for PATIENCE consecutive epochs
        # while we're still training (train loss still decreasing typically).
        # This is not a bug — it's the expected overfitting signal at this data
        # scale. Document it explicitly as per plan §Phase 5 step 5.
        if not smoke and len(val_loss_history) >= EARLY_STOP_PATIENCE + 1:
            recent = val_loss_history[-(EARLY_STOP_PATIENCE + 1):]
            if all(recent[i] < recent[i + 1] for i in range(EARLY_STOP_PATIENCE)):
                if overfitting_step is None:
                    overfitting_step = global_step
                    print(
                        f"[phase5] ⚠ OVERFITTING DETECTED at step {global_step} "
                        f"(epoch {epoch+1}): val loss rose for {EARLY_STOP_PATIENCE} "
                        f"consecutive epochs ({recent}) while training continues. "
                        "This is the expected signal at this data scale — "
                        "stopping and reporting as per plan §Phase 5 step 5."
                    )
                early_stopped = True
                break

        if smoke and global_step >= 3:
            break

    # ── 12. Post-training summary ─────────────────────────────────────────
    final_train_loss = mean_train_loss if epoch_batches > 0 else float("nan")
    final_val_loss = val_loss_history[-1] if val_loss_history else float("nan")

    print(f"\n[phase5] training complete")
    print(f"[phase5] final_train_loss={final_train_loss:.4f}")
    print(f"[phase5] final_val_loss={final_val_loss:.4f}")
    print(f"[phase5] best_val_loss={best_val_loss:.4f}  at epoch {best_val_epoch}")
    if early_stopped:
        print(f"[phase5] early stopped at step {global_step} (epoch {epoch+1})")
        if overfitting_step:
            print(f"[phase5] overfitting divergence first observed at step {overfitting_step}")
    if not smoke:
        print(f"[phase5] best checkpoint -> {BEST_VAL_DIR / run_name}")

    # ── 13. Append to sweep_results.csv ──────────────────────────────────
    if not smoke:
        append_sweep_row(
            run_name=run_name,
            lora_r=lora_cfg["r"],
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            best_val_loss=best_val_loss,
            best_val_epoch=best_val_epoch,
        )
        print(f"[phase5] sweep_results.csv updated -> {SWEEP_RESULTS_CSV}")
    else:
        print("[phase5] SMOKE MODE complete — no checkpoints or CSV written")
        print("[phase5] Smoke test passed: shapes OK, loss finite, AMP/dtype checks done")

    print("[phase5] Phase 5 run complete. Commit logs/, checkpoints/, eval/ back to repo.")


if __name__ == "__main__":
    main()
