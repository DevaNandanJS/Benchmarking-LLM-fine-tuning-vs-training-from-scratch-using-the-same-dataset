# Model Choice — Track 1 Fine-Tuning (Phase 2)

**Script:** [`track1_finetune/scripts/select_model.py`](../scripts/select_model.py)  
**Status:** Decision locked. Run the script in Colab to produce `configs/run_phase2.json` and `configs/model_architecture.json`.

---

## Selected Model

**`HuggingFaceTB/SmolLM2-135M`**

---

## Why SmolLM2-135M (not GPT-2)

The plan (§Phase 2 step 1) lists `HuggingFaceTB/SmolLM2-135M` as the default recommendation and GPT-2 (124M) as the canonical alternative. SmolLM2-135M was chosen for the following reasons:

**1. Unambiguous LoRA target modules.** SmolLM2 uses a Llama-style architecture with *separate* attention projection layers: `q_proj`, `k_proj`, `v_proj`, and `o_proj`. Each is an independently addressable `nn.Linear`, which means `LoraConfig.target_modules=["q_proj", "v_proj"]` (or all four) maps to exactly the layers intended with no ambiguity.

GPT-2, by contrast, uses a fused `c_attn` projection that packs all three of Q, K, V into a single weight matrix. PEFT can target it, but the targeting is less selective and requires understanding the fused structure — more Phase 4 complexity for no modeling benefit.

**Verification:** the actual layer names were extracted by `select_model.py` during its run and saved to `configs/model_architecture.json`. Phase 4 **must consult that file** rather than relying on documentation or this reasoning table. Look for modules whose names contain `q_proj`, `k_proj`, `v_proj`, `o_proj`.

**2. Size and plan alignment.** At 135M parameters, SmolLM2 falls within the plan's recommended 125M–360M range. The memory math (see below) confirms it fits comfortably on the T4 without quantization.

**3. Plan recommendation.** The execution plan explicitly names SmolLM2-135M as the default (§Phase 2 step 1).

---

## Pad Token Decision

SmolLM2-135M's tokenizer may or may not ship with a `<pad>` token depending on the version. `select_model.py` checks `tokenizer.pad_token is None` and sets it to `tokenizer.eos_token` if missing. The exact outcome is logged in `configs/run_phase2.json` under `"pad_token_decision"`.

> **Downstream note for Phase 5 — collator must mask padding positions in labels.**
> Whether the pad token is native or aliased to `<eos>`, the data collator in Phase 5 **must set `labels = -100` at padding positions** before they reach the loss function. Without this, the model trains on `<eos>`/`<pad>` tokens as if they were real content — inflating loss and corrupting the gradient signal on padded sequences. Verify this is handled before the first training run.

---

## Quantization Decision

**No quantization — plain fp16/bf16 LoRA. `bitsandbytes` is not required.**

| Item | Value |
|---|---|
| Total parameters | 135M (confirmed at script run time — see `run_phase2.json`) |
| fp16 weight footprint | 135M × 2 bytes ≈ **270 MB** |
| T4 VRAM | 15 GB |
| Weight footprint / VRAM | ~1.8% |

**Important caveat — this is weights-only:** the 270 MB figure covers only the frozen base-model weights stored in fp16. Full training memory also includes:
- **Activations** (forward pass intermediate tensors, batch-size dependent)
- **Gradients and optimizer state** for the LoRA-trainable parameter slice

Because LoRA keeps only ~0.5–2% of parameters trainable, the additional memory for gradients and optimizer state is small in absolute terms. However, the total footprint should be confirmed empirically in Phase 5 by checking `torch.cuda.max_memory_allocated()` after the first training step and `nvidia-smi` during training. The quantization decision can be revisited at that point if VRAM pressure is unexpectedly high (e.g., due to a large batch size or long context length).

---

## Architecture Verification

Full output of `[n for n, _ in model.named_modules()]` is saved in `configs/model_architecture.json` by `select_model.py`. This is the authoritative source for Phase 4 — do not set `target_modules` in `LoraConfig` without cross-referencing it.
