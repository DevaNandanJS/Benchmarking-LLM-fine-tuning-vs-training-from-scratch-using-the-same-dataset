# Track 1, Phase 2 — Base Model & Tokenizer Selection

> **Phase goal:** Lock in the pretrained base model, confirm it loads on T4, and establish
> the exact token count of the document under the real model tokenizer.

**Status:** 🔲 Not started  
**Script (to be created):** `track1_finetune/scripts/load_model.py`

---

*This document will be filled in when Phase 2 is complete.*

## What Phase 2 Covers

- Choosing between `HuggingFaceTB/SmolLM2-135M` and `gpt2` (124M)
- Loading the model and tokenizer, printing parameter count
- Confirming the pad token situation (GPT-2-family tokenizers often have no pad token)
- Recomputing the document token count with the real tokenizer
- Making the quantization decision (fp16 vs. QLoRA 4-bit) based on T4 VRAM math
- Writing `configs/model_choice.md` with the reasoning

## Key Questions to Answer

- Which model was chosen and why?
- What is the exact parameter count?
- What is the exact token count under the real tokenizer (vs. the 60,590 GPT-2 proxy)?
- Was quantization needed? What was the VRAM calculation?
- Was a pad token added? How does this affect training?
