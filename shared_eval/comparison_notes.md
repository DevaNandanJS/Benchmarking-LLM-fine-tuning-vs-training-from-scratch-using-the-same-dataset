# Cross-Track Comparison Notes

> **Status:** Track 1 complete. Track 2 fields are TBD — fill in once Track 2 finishes.
> Last updated: 2026-08-12T10:03:27.129144+00:00

---

## Primary Comparison Metric: Bits-Per-Byte (BPB)

BPB is the only metric directly comparable across tracks. It normalises loss by
raw text bytes rather than token count, making it tokenizer-agnostic. Lower = better.

| Track | Model | Strategy | BPB | Perplexity | Best Val CE Loss |
|---|---|---|---|---|---|
| **Track 1** | SmolLM2-135M (LoRA) | Fine-tune pretrained | **1.309722** | 10.7794 | 2.377641 |
| **Track 2** | [TBD] (scratch) | Train from random init | **TBD** | TBD | TBD |

---

## Efficiency Comparison

| Metric | Track 1 (LoRA) | Track 2 (Scratch) |
|---|---|---|
| Trainable parameters | 460,800 (0.341%) | TBD (full model) |
| Total parameters | ~135M (frozen base + adapters) | TBD |
| Training time | TBD (from Phase 5 logs) | TBD |
| GPU memory peak | TBD (torch.cuda.max_memory_allocated) | TBD |

---

## Qualitative Generation Comparison

*(Fill in after both tracks have generated samples in Phase 7)*

| Aspect | Track 1 (LoRA) | Track 2 (Scratch) |
|---|---|---|
| Domain vocabulary | TBD | TBD |
| Fluency | TBD | TBD |
| Coherence over ~80 tokens | TBD | TBD |
| Hallucination tendency | TBD | TBD |

---

## One-Paragraph Summary

*(Write after both tracks are complete)*

Track 1 started with a model that already understands language (grammar, vocabulary,
common phrasings) and used LoRA to adapt only 0.341% of its parameters to
the domain document. Track 2 started from random weights and had to learn both
language fundamentals and domain content from the same ~65K-token document.
The BPB comparison (1.309722 vs TBD) quantifies which strategy
produced a model better at predicting the source text, controlling for vocabulary
size differences. The trainable-parameter count comparison (460,800 for Track 1
vs TBD for Track 2) quantifies the compute cost of each approach.

---

## Files in this directory

| File | Contents |
|---|---|
| `track1_final_metrics.json` | CE loss, perplexity, BPB for the best Track 1 checkpoint |
| `track1_loss_curve.png` | Train/val loss curve for the best Track 1 run |
| `comparison_notes.md` | This file |
| `track2_final_metrics.json` | *(TBD — Track 2)* |
| `track2_loss_curve.png` | *(TBD — Track 2)* |
