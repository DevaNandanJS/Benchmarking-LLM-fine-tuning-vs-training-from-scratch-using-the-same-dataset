# Project Progress — LLM Fine-Tuning vs. Training from Scratch

> **Purpose of this folder:** A human-readable, phase-by-phase record of every decision,
> implementation step, finding, and lesson learned while completing this project.
> Written to be studied later — technical but accessible, with enough context that
> you can re-read any section cold and understand exactly what was done and why.

---

## What This Project Is About

We are benchmarking two fundamentally different strategies for teaching a language model
about a specific document:

| Track | Strategy | Core idea |
|---|---|---|
| **Track 1** — LoRA Fine-Tuning | Take a *pretrained* model and adapt it with a tiny set of trainable parameters (LoRA) | Start smart, tune cheaply |
| **Track 2** — Training from Scratch | Build a small language model and train it entirely on the target document from random weights | Start from zero, no prior knowledge |

The experiment is controlled: both tracks use the **same source document**, the same
evaluation metrics, and the same train/validation split — so any difference in the
final results is attributable to the strategy, not the data.

**Source document:** `LLM4Log: A Systematic Review of Large Language Model-based Log Analysis`
(54-page academic paper, ~60,000 tokens)

---

## Repository Layout (Quick Reference)

```
llm_task/
  data/
    raw/                     ← original PDF lives here
    extracted/               ← cleaned .txt, stats, extraction audit trail
    processed/               ← tokenized datasets (built in Phase 3)
  track1_finetune/
    scripts/                 ← all Python logic (agent edits these)
    configs/                 ← per-run JSON configs
    logs/                    ← metrics.jsonl per training run
    checkpoints/             ← LoRA adapter weights
    eval/                    ← loss curves, metrics, sweep results
    generations/             ← model-generated text samples
    track1_run.ipynb         ← thin Colab orchestrator notebook
  track2_scratch/            ← same structure as track1_finetune/
  shared_eval/               ← cross-track comparison artifacts
  progress/                  ← THIS FOLDER — human-readable progress docs
    track1_finetune/
    track2_scratch/
  plan/                      ← machine-executable task plans (for the agent)
  writeup/                   ← final project write-up
  requirements.txt
```

---

## Track 1 — Fine-Tuning Progress

| Phase | Topic | Status | Document |
|---|---|---|---|
| 0 | Environment Setup | ✅ Complete | *(setup was done prior to this session)* |
| 1 | Data Extraction & Cleaning | ✅ Complete | [phase1_data_extraction.md](track1_finetune/phase1_data_extraction.md) |
| 2 | Base Model & Tokenizer Selection | 🔲 Not started | — |
| 3 | Dataset Construction (Chunking & Splitting) | 🔲 Not started | — |
| 4 | LoRA Configuration & Model Wrapping | 🔲 Not started | — |
| 5 | Training Loop & Execution | 🔲 Not started | — |
| 6 | Quantitative Evaluation | 🔲 Not started | — |
| 7 | Qualitative Evaluation (Generation) | 🔲 Not started | — |
| 8 | Cross-Track Comparison Prep | 🔲 Not started | — |

---

## Track 2 — Training from Scratch Progress

| Phase | Topic | Status | Document |
|---|---|---|---|
| 0 | Environment Setup | 🔲 Not started | — |
| 1 | Data Extraction & Cleaning | 🔲 Not started | [phase1_data_extraction.md](track2_scratch/phase1_data_extraction.md) |
| 2 | Architecture Design | 🔲 Not started | — |
| 3 | Dataset Construction | 🔲 Not started | — |
| 4 | Training Loop & Execution | 🔲 Not started | — |
| 5 | Evaluation | 🔲 Not started | — |
| 6 | Cross-Track Comparison | 🔲 Not started | — |

---

## Key Design Decisions (Global)

These apply to both tracks and were fixed before any coding started.

| Decision | Choice | Reason |
|---|---|---|
| Random seed | `42` | Reproducibility — logged in every run config |
| Logging format | `metrics.jsonl` (one JSON object per step) | Consistent, machine-readable, easy to plot |
| Config-as-file | Dump hyperparams to `configs/run_<name>.json` before training | Every result is traceable to an exact config |
| Hyperparameter sweeps | Always run ≥2 values; never pick blindly | Evidence-based selection, defensible in write-up |
| Token count proxy | `tiktoken` GPT-2 for Phase 1 stats | Fast, no model download; replaced by real tokenizer in Phase 2 |

---

*Last updated: Phase 1 complete (Track 1)*
