# Agent Conventions & Guidelines

These conventions and guardrails apply to **all** phases across both tracks in this repository.
Read and follow them before authoring or modifying any code.

---

## 1. Project Goal & Overview

**Core Goal:** Benchmark **LLM Fine-Tuning** (Parameter-Efficient Fine-Tuning / LoRA on pre-trained open-weights LLMs) versus **Training a Small Language Model (SLM) from Scratch** using the **exact same dataset** and preprocessing pipeline.

### The Dual-Track Architecture

1. **Track 1 (`TASK1_finetuning_model/`):**
   - Fine-tune a pre-trained open-weights language model (e.g., Qwen, LLaMA family) using LoRA / PEFT.
   - Focus on adaptation efficiency, parameter retention, convergence speed, and downstream task quality on domain-specific text extracted from the dataset.

2. **Track 2 (`TASK2_slm_from_scratch/`):**
   - Train a custom Small Language Model (custom Transformer architecture + custom BPE tokenizer) completely from scratch on the exact same dataset.
   - Focus on architectural sizing, tokenizer vocabulary optimization, training dynamics, loss scaling, and sample generation capability.

3. **Cross-Track Comparison (`shared_eval/`):**
   - Comprehensive side-by-side benchmarking evaluating loss curves, perplexity, task evaluation metrics, inference latency/throughput, VRAM consumption, and sample generation quality under identical test prompts and holdout splits.

---

## 2. Directory Layout

All paths in the project are organized as follows:

```
├── data/
│   ├── raw/                 # Original source documents (e.g., PDFs)
│   ├── extracted/           # Cleaned plain text and extracted statistics (stats.json)
│   └── processed/           # Tokenized/chunked dataset artifacts (per track)
├── TASK1_finetuning_model/
│   ├── configs/             # run_<name>.json, model_choice.md, trainable_params.json
│   ├── checkpoints/         # LoRA adapters, sweep checkpoints (gitignored)
│   ├── logs/                # metrics.jsonl per run, env_check.json, environment.txt
│   ├── eval/                # loss_curve.png, final_metrics.json, sweep_results.csv, *.md
│   ├── generations/         # Generation samples and qualitative outputs
│   ├── scripts/             # Modular Python logic (common.py, train.py, eval.py, etc.)
│   └── finetuning_run.ipynb # Thin Colab orchestrator notebook
├── TASK2_slm_from_scratch/
│   ├── configs/             # Run configs, architecture hyperparameters
│   ├── checkpoints/         # Model weights, optimizer states (gitignored)
│   ├── logs/                # metrics.jsonl per run, environment.txt
│   ├── eval/                # Evaluation reports, loss comparisons, metrics
│   ├── generations/         # SLM generation samples
│   ├── tokenizer/           # Trained BPE tokenizer files (vocab, merges)
│   ├── scripts/             # Modular Python logic (model.py, train.py, eval.py, etc.)
│   └── slm_run.ipynb        # Thin Colab orchestrator notebook
├── shared_eval/             # Cross-track benchmarking & side-by-side comparison artifacts
├── progress/                # Study-ready documentation (user-triggered)
│   ├── README.md            # Overall project index and status table
│   ├── TASK1_finetuning_model/
│   └── TASK2_slm_from_scratch/
└── requirements.txt         # Pinned project dependencies
```

---

## 3. Authoring vs. Execution Model (Colab & Local Environment)

- **Local Authoring:** All real logic lives in importable, modular `.py` files under `TASK1_finetuning_model/scripts/` and `TASK2_slm_from_scratch/scripts/`. The agent edits these `.py` files locally.
- **Thin Orchestrator Notebooks:** `finetuning_run.ipynb` and `slm_run.ipynb` are thin orchestrators. Notebook cells only sync code (`git pull`), install dependencies, and execute scripts via `!python TASK1_finetuning_model/scripts/<script>.py`.
- **Remote Kernel Filesystem:** The remote Google Colab kernel runs in its own environment (e.g., `/content`). Local file edits are **not** automatically mirrored to the remote kernel without syncing.
- **Human-in-the-Loop Sync Workflow:**
  1. Agent authors/updates `.py` modules locally.
  2. Code is committed and pushed to Git: `git add -A && git commit -m "..." && git push`.
  3. The user executes the sync cell in the notebook on Colab (`git pull`).
  4. The user runs the notebook execution cells on the Colab GPU.
  5. The agent inspects returned logs, metric files, and evaluation outputs to proceed.
- **Terminal vs. GPU Execution:** Do **not** run GPU-intensive training or inference scripts directly from the local terminal expecting GPU acceleration. Anything requiring a GPU must be executed via notebook cells on the remote GPU kernel.
- **Dependency Lock:** `environment.txt` (`pip freeze` from the remote Colab execution) should be committed to maintain an exact dependency record.

---

## 4. Global Conventions & Guardrails (Mandatory in Every Script)

Every script across both tracks must adhere to these conventions:

- **Determinism:** `SEED = 42`. Call `set_seed()` (from `scripts/common.py`) at the top of every script and record the seed in every run's config JSON.
- **Logging Format:** Every training run emits `logs/<run_name>/metrics.jsonl`, writing one JSON object per logged step:
  ```json
  {"step": 100, "epoch": 1.0, "train_loss": 1.8452, "val_loss": 1.9124, "lr": 0.0003, "timestamp": "2026-08-16T01:00:00Z"}
  ```
  Always use `MetricsLogger` from `scripts/common.py` — do not implement ad-hoc custom logging.
- **Config-as-File:** Dump every run's complete configuration to `configs/run_<name>.json` **before** training starts using `common.dump_config()`.
- **Empirical Hyperparameter Discipline:** Never pick hyperparameter values blindly. Run small sweeps (2–3 values per key parameter), log results under distinct run names, and tabulate comparisons into `eval/sweep_results.csv` before selecting final configurations.
- **Parameter Accounting:** For Track 1 (PEFT/LoRA), log `print_trainable_parameters()` output to `configs/trainable_params.json`. For Track 2 (SLM), log total model parameters, non-embedding parameters, and layer breakdown.
- **Track Isolation:** Keep Track 1 and Track 2 fully decoupled. Each track has its own `scripts/common.py` where `TRACK_DIR` resolves to that track's directory. Never import `TASK1` modules from `TASK2` or vice versa.
- **Cross-Platform & UTF-8 Safety:** Reconfigure standard output streams (`sys.stdout`, `sys.stderr`) to UTF-8 on initialization to prevent encoding errors across different OS environments.
- **Path Resolution:** Always resolve file paths using `pathlib.Path(__file__).resolve()` rather than hardcoded string paths or assumptions about the current working directory.

---

## 5. Definition-of-Done Discipline

A phase or milestone is complete **only** when its concrete deliverables are verifiably present on disk:
- Scripts are modular, executable, and free of syntax/runtime errors.
- Configuration JSON files (`configs/run_<name>.json`) are written.
- Training metric logs (`logs/<run_name>/metrics.jsonl`) are generated and populated.
- Evaluation metrics, loss curve figures, and generation artifacts exist in `eval/` and `generations/`.
- Checkpoints and adapters are properly saved in `checkpoints/`.

Do not mark any task or phase as complete based on intent alone.

---

## 6. Progress Documentation (User-Triggered)

The `progress/` directory houses human-readable documentation summarizing decisions, implementations, and empirical results.

- **Trigger:** Only create or update progress documentation when the user explicitly requests it (e.g., "document phase 1", "write up what we did", "update progress").
- **Documentation Structure:**
  - `progress/README.md`: Project status overview and tracking matrix.
  - `progress/TASK1_finetuning_model/phase<N>_<name>.md`: Track 1 phase reports.
  - `progress/TASK2_slm_from_scratch/phase<N>_<name>.md`: Track 2 phase reports.
- **Document Contents (when requested):**
  1. Goal and high-level summary.
  2. Walkthrough of implementation logic and architectural decisions.
  3. Key findings, edge cases, and solutions.
  4. Decisions table (options considered, choice made, rationale).
  5. Verification checklist with supporting evidence (metrics, logs, artifact paths).
