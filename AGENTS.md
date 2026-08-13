# Agent Conventions (llm_task)

These conventions come from `plan/finetuning_execution_plan.md` §0 ("Global conventions")
and §0.a ("Authoring vs. execution model"). They apply to **every** phase of both tracks.
Read them before writing any code.

## Repo root

`llm_task/`. All paths below are relative to this root.

## Directory layout

```
llm_task/
  data/
    raw/                 # original PDF  -> data/raw/document.pdf (Phase 1)
    extracted/           # cleaned .txt + stats.json
    processed/           # tokenized/chunked datasets (per track, prefixed)
  finetuning_model/
    configs/             # run_<name>.json, model_choice.md, split_strategy.md, trainable_params.json
    checkpoints/         # LoRA adapters, sweep_<name>/ subdirs (gitignored)
    logs/                # metrics.jsonl per run, env_check.json, environment.txt (committed)
    eval/                # loss_curve.png, final_metrics.json, sweep_results.csv, *.md
    generations/         # finetuning_samples.md
    scripts/             # importable .py modules — THIS is where the agent edits
    finetuning_run.ipynb # thin orchestrator notebook (cells sync + call into scripts/)
  slm_from_scratch/
    configs/ checkpoints/ logs/ eval/ generations/ scripts/ slm_run.ipynb
  shared_eval/           # cross-track comparison artifacts
  progress/              # human-readable phase-by-phase documentation (updated after EVERY phase)
    README.md            # project index + status table for both tracks
    finetuning_model/    # one .md file per completed phase
    slm_from_scratch/    # one .md file per completed phase
  writeup/
  requirements.txt       # pinned deps (installed on the Colab kernel)
```

## Global conventions (mandatory in every script)

- **Determinism:** `SEED = 42`. Call `set_seed()` (from `scripts/common.py`) at the top of
  every script and record the seed in every run's config JSON.
- **Logging format:** every training run emits `logs/<run_name>/metrics.jsonl`, one JSON
  object per logged step:
  `{"step": int, "epoch": float, "train_loss": float, "val_loss": float|null, "lr": float, "timestamp": iso8601}`.
  Use `MetricsLogger` from `scripts/common.py` — do not hand-roll it.
- **Config-as-file:** dump every run's hyperparameters to `configs/run_<name>.json` BEFORE
  training starts (use `common.dump_config`).
- **No silent numeric choices:** never pick one value from a hyperparameter range blindly.
  Run a small sweep (2–3 values), each under a distinct run name, then tabulate into
  `eval/sweep_results.csv` before locking a final config.
- **Trainable params:** log the LoRA `print_trainable_parameters()` output to
  `configs/trainable_params.json` (required deliverable).

## Authoring vs. execution model (VS Code + Google Colab extension)

- **The remote Colab kernel has its own filesystem (typically `/content`).** Local `.py`
  files are NOT visible to it. Files only reach the kernel via `git clone/pull` (the sync
  cell) or Drive.
- **Cell execution ≠ terminal execution.** Selecting "Colab" as the notebook kernel only
  makes *cells* run on the GPU. A `python foo.py` run from an agent's shell tool runs
  locally, with no GPU. Anything that needs the GPU must run as a notebook cell.
- **Pattern:** all real logic lives in importable `.py` modules under `finetuning_model/scripts/`
  (agent edits these). The notebook `finetuning_run.ipynb` is a thin orchestrator: its cells only
  sync code (`git clone/pull`), install deps, and call `!python finetuning_model/scripts/<script>.py`.
- **Sync workflow (critical):** after the agent edits local files, the user must
  `git add -A && git commit -m ... && git push` BEFORE re-running the notebook's sync cell,
  or the remote kernel will keep running stale code.

## Agent/terminal guidance

- Author any file locally, any time — no Colab connection needed.
- You cannot trigger Colab kernel execution from your own shell tool. Treat running the
  notebook as a **human-in-the-loop checkpoint**: agent finishes a phase's code → user runs
  the relevant notebook cells → agent reads back the resulting logs/checkpoints from disk
  (they'll be in the repo after the user copies/pushes, or under Drive) to continue.
- Do NOT run GPU-requiring scripts from the terminal expecting a GPU; you'll waste time on
  errors that only exist because of the missing GPU.
- `environment.txt` (`pip freeze` from the remote session) must be committed back to the
  repo — it is the reproducibility artifact that locks exact dependency versions.

## Definition-of-Done discipline

Complete a phase only when its "Definition of Done" checklist in
`plan/finetuning_execution_plan.md` is verifiably satisfied (files exist, logged, committed).
Do not mark a phase done on intent.

## Progress documentation (user-triggered, not automatic)

The `progress/` folder contains human-readable, study-ready documentation of every
decision and implementation step taken. The user reads these documents to understand
and revise the project.

**When:** The user explicitly asks — e.g. "document phase 1", "write up what we did",
"update progress". Do NOT write or update progress documents unless the user asks.

### Folder structure

```
progress/
  README.md                    # project index + status table for both tracks
  finetuning_model/
    phase1_data_extraction.md  # one file per phase
    phase2_model_selection.md
    ...
  slm_from_scratch/
    phase1_data_extraction.md  # stub exists
    ...
```

### How to document a phase when requested

1. Read all relevant scripts, configs, logs, and artifacts for the target phase.
2. Fill out or update the phase document (`progress/<track>/phase<N>_<name>.md`):
   - **Goal & summary** of the phase.
   - **Step-by-step walkthrough** of code logic and why choices were made.
   - **Key findings / bugs / edge cases** encountered and how they were solved.
   - **Decisions table** (choice + reasoning).
   - **Definition of Done checklist** marked with evidence.
3. Update the status table in `progress/README.md` (`🔲 Not started` → `✅ Complete`).
