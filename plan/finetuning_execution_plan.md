# Track 1 Execution Plan — Fine-Tuning a Pretrained Model (LoRA)

**Purpose of this document:** a phase-by-phase, agent-executable build plan. Each phase states its goal, exact inputs, ordered steps, concrete commands/API calls, the artifacts it must produce, and a "Definition of Done" checklist so completion is verifiable rather than assumed. This is meant to be handed directly to a coding agent (e.g., Claude Code) as a task list — each phase can be a separate work unit/commit.

**Companion document:** `TASK2_slm_from_scratch_slm_execution_plan.md`. The two pipelines deliberately share conventions (directory layout, chunking approach, logging format, evaluation metrics) so their outputs can be compared fairly at the end — see Phase 8 in both documents.

---

## 0. Global conventions (apply across all phases)

- **Repo root:** `llm_task/`
- **Shared directory layout** (create at the start of Phase 1):
  ```
  llm_task/
    data/
      raw/                 # original PDF
      extracted/            # cleaned .txt
      processed/             # tokenized/chunked datasets (per track, prefixed)
    TASK1_finetuning_model/
      configs/
      checkpoints/
      logs/
      eval/
      generations/
      scripts/               # importable .py modules — this is what the agent edits
      finetuning_run.ipynb        # thin orchestrator notebook — cells sync + call into scripts/
    TASK2_slm_from_scratch/
      ...
    shared_eval/            # cross-track comparison artifacts
    writeup/
  ```
- **If you're driving this from VS Code with the Google Colab extension:** the remote Colab kernel has its own filesystem, separate from your local VS Code workspace — files the agent writes locally aren't automatically visible to it. All "real" logic should live in `.py` modules under each track's `scripts/` folder (agent-editable, plain files); each track also gets one thin orchestrator notebook whose cells only sync code onto the remote kernel and call into those scripts. See Phase 0 for the exact pattern — read it before starting even if you're already comfortable with Colab, since the execution model here differs from both plain browser-Colab and a purely local setup.
- **Determinism:** set and log a fixed random seed (`SEED = 42` or your choice) for Python, NumPy, and PyTorch (`torch.manual_seed`, `torch.cuda.manual_seed_all`) at the top of every script. Log the seed value into every run's config JSON — reproducibility is an explicit grading criterion.
- **Logging format:** every training run must emit a `metrics.jsonl` file with one JSON object per logged step: `{"step": int, "epoch": float, "train_loss": float, "val_loss": float|null, "lr": float, "timestamp": iso8601}`. This is what Phase 6/7 plotting consumes, and it's what makes "training and validation loss curves" (an explicit deliverable requirement) trivial to produce later.
- **Config-as-file:** every run's hyperparameters must be dumped to `configs/run_<name>.json` before training starts, so every result is traceable to an exact, saved configuration — this is what you'll cite in the write-up and defend live.
- **No silent numeric choices:** wherever a hyperparameter range is given below (e.g., "LoRA rank 4–16"), the agent should run a *small* sweep (2–3 values) rather than pick one arbitrarily, log results for each under a distinct run name, and only then select a final config — record the sweep table itself as an artifact (`eval/sweep_results.csv`).

---

## Phase 0 — Environment Setup

**Goal:** a reproducible, verified Colab (or local GPU) environment before touching any data — and, since an agent is doing much of the authoring, a clear separation between "where code is written" (local, by the agent) and "where code actually executes on a GPU" (the remote Colab kernel, if using Colab).

### 0.a — Authoring vs. execution model (read before the steps below)

If you're using the official **Google Colab extension for VS Code**, three things change how the agent should work compared to a plain local setup:

- **The remote kernel's filesystem is separate from your local VS Code workspace.** A `.py` file the agent writes to your local project folder is not automatically visible to the remote Colab VM — it only sees what's actually on it (typically under `/content`) plus whatever you mount (Drive) or fetch into it (git clone/pull). Don't assume a locally-written script can just be `!python scripts/foo.py`'d remotely without first syncing it there.
- **Cell execution ≠ terminal execution.** Selecting "Colab" as the notebook kernel makes *cells* run on the remote GPU; it does not redirect your VS Code integrated terminal (or an agent's shell/bash tool) to that machine. A plain `python foo.py` run from a terminal still runs locally, with no GPU, regardless of what kernel a notebook nearby is connected to. **Anything that needs the GPU must run as a notebook cell against the connected Colab kernel.**
- **Practical implication for agent-driven execution:** the agent can reliably author every file here — the `.py` scripts and the orchestrator notebook's cells — directly on local disk, any time, independent of whether a Colab session is even connected. Whether the agent's *terminal* tool can itself trigger execution against this specific extension's live kernel session isn't something I can confirm — treat it as a manual VS Code UI action (connect kernel → run cell / Run All) and plan for a human-in-the-loop checkpoint: agent finishes editing a phase → you run the corresponding cells → agent reads the resulting logs/checkpoints back from disk to continue. If fully unattended agent execution matters more to you than free-tier convenience, an SSH-accessible cloud GPU box (where the agent's own shell runs directly on the GPU machine) sidesteps this limitation entirely.
- **If you're instead using plain browser Colab or a local/other GPU machine**, ignore the VS Code-specific bits below (kernel picker, extension install) and follow the equivalent generic step (mount Drive via the browser UI, or just run everything in a normal local terminal if working locally) — the invariant that matters is: setup happens in the place code will actually execute, not just where it's authored.

**Recommended pattern:** keep all real logic in importable `.py` modules under `TASK1_finetuning_model/scripts/` (as the rest of this plan already assumes) and add one thin orchestrator notebook, `TASK1_finetuning_model/finetuning_run.ipynb`, whose cells only sync code and call into those scripts:
```python
# Cell 1 — sync the latest local edits onto the remote kernel
!git clone https://github.com/<you>/llm_task.git || (cd llm_task && git pull)
%cd llm_task

# Cell 2 — install deps (see step 5 below)
!pip install -q -r requirements.txt

# Cell 3 — run a phase
!python TASK1_finetuning_model/scripts/extract_text.py
```
This means re-running the sync cell picks up whatever the agent just edited, without hand-copying code into cells — and it keeps the notebook itself tiny/diff-friendly, while the actual code a reviewer or interviewer reads is ordinary, readable `.py` files, which is also just a stronger artifact for the "explain every line" interview requirement than a wall of notebook cells.

### Steps

1. Install the extension: **Extensions view → search "Google Colab" → Install** the official Google-published one (it will pull in the VS Code Jupyter extension as a dependency if you don't already have it).
2. Open/create `TASK1_finetuning_model/finetuning_run.ipynb`, use the kernel picker (top-right of the notebook) → **Select Another Kernel → Colab → New Colab Server** (or **Auto Connect** to reattach to a recent session — note this silently starts a *new* server if the old one expired, so re-verify GPU/files afterward) → choose **GPU** → **T4** on the free tier, and sign in with your Google account when prompted.
3. Verify the connection and hardware in a cell:
   ```python
   import torch
   assert torch.cuda.is_available()
   print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
   ```
4. Set up persistence for anything that must survive session restarts (checkpoints, logs, processed data):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   Point `checkpoints/`, `logs/`, and `data/processed/` at a Drive-backed path if you want them to outlive a disconnected/expired session. The code itself doesn't strictly need this if you're re-syncing via git each session (previous step) — but large binary artifacts (checkpoints, datasets) are a poor fit for git, so Drive (or an equivalent object store) is the right place for those specifically.
5. Install pinned dependencies **inside a notebook cell**, not your local terminal — installing locally puts packages where nothing GPU-relevant runs:
   ```python
   !pip install -q transformers accelerate peft datasets
   !pip install -q pypdf pdfplumber
   !pip install -q matplotlib pandas
   !pip install -q bitsandbytes   # only if the quantization decision gate (Phase 3) says you need it
   ```
6. Record `!pip freeze > TASK1_finetuning_model/environment.txt` from the same session, then get this file back into the version-controlled repo (commit it from the remote via git, or copy it back) — it's a reproducibility artifact and should live with the code, not just on the ephemeral Colab VM.
7. Create the directory structure from Section 0 — this has no GPU dependency and can be done locally by the agent before any Colab connection exists at all.

**Definition of Done:**
- [ ] Colab extension installed, kernel connected, GPU detected/named, memory printed and logged.
- [ ] Orchestrator notebook (`finetuning_run.ipynb`) exists with a working sync cell, verified to pull in a test local edit.
- [ ] Drive mounted (if used) and confirmed writable.
- [ ] `environment.txt` produced on the remote session and copied back into the version-controlled repo.
- [ ] Full directory tree exists (empty subfolders are fine).

---

## Phase 1 — Data Extraction & Cleaning

**Goal:** a clean `.txt` file of the document's actual content, with an exact token count established before any modeling decision is finalized.

**Steps:**
1. Place the source PDF at `data/raw/document.pdf`.
2. Extract text using `pdfplumber` (preferred for layout fidelity) with a fallback to `pypdf` if extraction quality is poor (garbled text, missing sections). Write a small extraction script `scripts/extract_text.py` that:
   - Extracts page-by-page text.
   - Strips obvious non-content artifacts: repeated headers/footers, page numbers, isolated figure/table captions if they're not prose, excessive whitespace/line-break artifacts from PDF layout (e.g., mid-sentence line breaks that should be joined).
   - Concatenates into a single `data/extracted/document_clean.txt`.
3. **Manually spot-check** the output (open the file, read a few random sections) — automated extraction from PDFs frequently mangles tables, multi-column layouts, or hyphenation. This step cannot be fully automated away; a coding agent should flag any suspicious sections (e.g., garbled character runs, obvious column-merging) rather than silently proceeding.
4. Compute and log exact statistics into `data/extracted/stats.json`:
   - Character count, word count (whitespace-split).
   - Token count under the base model's *pretrained* tokenizer (see Phase 2) — this is the number that actually matters for every later sizing decision.
5. **Decision gate:** if the cleaned document yields fewer than roughly a few thousand tokens, flag this explicitly in the write-up as an even more extreme low-data regime than assumed and note it will affect epoch count / overfitting speed even more strongly.

**Definition of Done:**
- [ ] `data/extracted/document_clean.txt` exists and passes a manual spot-check (no garbled text, sentences intact).
- [ ] `data/extracted/stats.json` contains exact char/word/token counts.
- [ ] Extraction script is committed and re-runnable end-to-end from the raw PDF.

---

## Phase 2 — Base Model & Tokenizer Selection

**Goal:** lock in the pretrained base model and confirm it loads and runs on the T4 before building anything on top of it.

**Steps:**
1. Choose the base model. Default recommendation: `HuggingFaceTB/SmolLM2-135M` (or `-360M` if more headroom is wanted) as base, or `gpt2` (124M) as the more "canonical" alternative — pick one and record the reasoning in `configs/model_choice.md` (2–3 sentences: why this size, why this model family, referencing the blueprint document's Section 3.1 trade-off table).
2. Load and smoke-test:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model_name = "HuggingFaceTB/SmolLM2-135M"   # or "gpt2"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)
   print(sum(p.numel() for p in model.parameters()), "total params")
   ```
3. Confirm the tokenizer has a pad token set (GPT-2-family tokenizers often don't by default — set `tokenizer.pad_token = tokenizer.eos_token` if missing, and log this decision).
4. Re-tokenize `document_clean.txt` with *this exact* tokenizer and update `stats.json` with the precise token count under the model's own vocabulary (may differ slightly from Phase 1's estimate).
5. **Decision gate — quantization:** compute the base model's fp16 memory footprint (`num_params * 2 bytes`) and compare against available T4 VRAM. If the chosen model is in the 125M–360M range, it will fit comfortably in fp16 and **no quantization (QLoRA) is needed** — proceed with plain fp16/bf16 LoRA. Only fall back to 4-bit quantization via `bitsandbytes` if a larger base model is chosen and doesn't fit; if that happens, document why the model choice changed.

**Definition of Done:**
- [ ] Base model + tokenizer load successfully, parameter count printed and logged.
- [ ] Pad token confirmed/set, decision logged.
- [ ] `stats.json` updated with exact token count under the real tokenizer.
- [ ] Quantization decision (yes/no) recorded with the memory-math justification.

---

## Phase 3 — Dataset Construction (Chunking & Splitting)

**Goal:** a tokenized, chunked, train/validation-split dataset in Hugging Face `datasets` format, ready for causal LM training.

**Steps:**
1. Tokenize the full cleaned text into a single long sequence of token IDs.
2. Chunk with a sliding window: choose `context_length` (start at 128 or 256) and `stride` (e.g., `context_length // 2` for 50% overlap — record whatever ratio is chosen and why). Implement in `scripts/build_dataset.py`:
   ```python
   def sliding_windows(token_ids, context_length, stride):
       chunks = []
       for start in range(0, len(token_ids) - context_length, stride):
           chunks.append(token_ids[start:start + context_length])
       return chunks
   ```
3. Split chunks into train/validation. Two valid approaches — pick one and justify it in `configs/split_strategy.md`:
   - **Contiguous holdout** (recommended default): reserve the *last* ~10–15% of the document (by token position, before windowing) as validation, window each region independently. This avoids near-duplicate leakage between train and val that random chunk-splitting would cause when windows overlap.
   - **Random chunk holdout**: only appropriate if stride ≥ context_length (i.e., no overlap) — otherwise flag the leakage risk explicitly.
4. Wrap as a `datasets.Dataset` (or plain `torch.utils.data.Dataset`) yielding `input_ids`/`labels` (labels = input_ids for causal LM, shifted internally by the model/collator).
5. Save processed dataset to `data/processed/finetune_train.pt` / `finetune_val.pt` (or `datasets` `save_to_disk`).
6. Log final counts: number of train chunks, number of val chunks, tokens per chunk, total tokens covered (should approximately reconstruct Phase 1's total, adjusted for stride).

**Definition of Done:**
- [ ] `scripts/build_dataset.py` runs end-to-end from `document_clean.txt` to saved train/val tensors.
- [ ] Chunk counts and context length logged in `data/processed/dataset_stats.json`.
- [ ] Split strategy documented and justified.

---

## Phase 4 — LoRA Configuration & Model Wrapping

**Goal:** wrap the base model with LoRA, verify only the intended parameters are trainable, before any actual training.

**Steps:**
1. Define the LoRA config (start here; sweep in Phase 6 if time allows):
   ```python
   from peft import LoraConfig, get_peft_model, TaskType
   lora_config = LoraConfig(
       task_type=TaskType.CAUSAL_LM,
       r=8,                 # sweep candidate values: 4, 8, 16
       lora_alpha=16,        # convention: ~2x r as a starting point
       lora_dropout=0.05,
       target_modules=["q_proj", "v_proj"],   # or model-specific equivalents; inspect model.named_modules() first
       bias="none",
   )
   model = get_peft_model(model, lora_config)
   model.print_trainable_parameters()
   ```
   **Note:** `target_modules` names are architecture-specific — before setting this, print `[n for n, _ in model.named_modules()]` and identify the actual attention projection layer names for the chosen base model (GPT-2 uses a fused `c_attn`; Llama-style architectures like SmolLM2 use separate `q_proj`/`k_proj`/`v_proj`/`o_proj`). Do not assume the names above are correct without checking.
2. Confirm and log the exact trainable parameter count and percentage from `print_trainable_parameters()` output — this is a required, checkable deliverable number (Section 3.2 of the blueprint).
3. Sanity-check a single forward/backward pass on one batch before committing to a full run (catches shape/device errors early):
   ```python
   batch = next(iter(train_dataloader))
   out = model(**batch)
   out.loss.backward()
   print("sanity check loss:", out.loss.item())
   ```

**Definition of Done:**
- [ ] `target_modules` confirmed correct for the specific base model architecture (not guessed).
- [ ] Trainable parameter count/percentage logged (`configs/trainable_params.json`).
- [ ] One successful forward/backward pass completed without error.

---

## Phase 5 — Training Loop & Execution

**Goal:** run LoRA fine-tuning with full metric logging, checkpointing, and safeguards against silent overfitting.

**Steps:**
1. Use either Hugging Face `Trainer`/`TrainingArguments` (faster to set up correctly) or a manual loop (more transparent for the "explain every line" requirement — a manual loop is arguably the stronger interview artifact since `Trainer` hides steps you'd otherwise have to explain). If using `Trainer`, still be prepared to explain what it's doing internally (gradient accumulation, scheduler, checkpointing) since "I used a library default" is not a sufficient defense answer.
2. Core training hyperparameters — starting values (sweep per Section 0's rule, don't lock in blind):
   - Learning rate: start at `2e-4` (LoRA tolerates higher LR than full FT since so few parameters move); log the final chosen value and the sweep evidence.
   - Batch size: as large as comfortably fits in T4 memory at your context length (try 8–16 to start; increase if headroom allows, using `nvidia-smi` or `torch.cuda.max_memory_allocated()` to check).
   - Epochs: cap at a small number (e.g., 5–10) with **early stopping on validation loss** — given the tiny dataset, expect the best checkpoint to occur early. Do not train to a fixed high epoch count and just take the last checkpoint.
   - Weight decay: `0.01` starting point.
   - LR schedule: cosine or linear decay with a short warmup (e.g., 5–10% of total steps).
   - Gradient clipping: `max_grad_norm=1.0` as a standard safeguard.
3. Log every N steps (choose N so you get at least ~20–30 logged points across the whole run) to `logs/metrics.jsonl` per the Section 0 format, including a validation pass at the same cadence (or at minimum once per epoch).
4. Checkpoint the LoRA adapter (not the full base model — that's the point of PEFT) at each validation improvement:
   ```python
   model.save_pretrained(f"checkpoints/best_val")
   ```
5. **Explicit overfitting guard:** if validation loss increases for two consecutive logged evaluations while training loss keeps decreasing, that is a real, expected, and reportable signal at this data scale — do not treat it as a bug to eliminate; treat it as the stopping point and document it in the write-up as evidence you understand the overfitting dynamics discussed in the blueprint.
6. Run the small hyperparameter sweep promised in Section 0 (e.g., LoRA rank 4 vs 8 vs 16, or two learning rates) as separate named runs under `checkpoints/sweep_<name>/`, each with its own `metrics.jsonl`, and produce `eval/sweep_results.csv` summarizing final train/val loss per run.

**Definition of Done:**
- [ ] At least one complete training run with full `metrics.jsonl` logging.
- [ ] Best checkpoint (by validation loss) saved separately and identified.
- [ ] Small sweep (≥2 configurations) completed and tabulated.
- [ ] Overfitting behavior (if/when observed) explicitly noted with the step/epoch it occurred at.

---

## Phase 6 — Quantitative Evaluation

**Goal:** produce the numbers and plots that go directly into the write-up's "training and validation loss curves" and comparison sections.

**Steps:**
1. Plot train/val loss vs. step from `metrics.jsonl` using `matplotlib`; save to `eval/loss_curve.png`. Include both curves on one plot with a legend.
2. Compute final held-out metrics on the best checkpoint:
   - Token-level cross-entropy loss and perplexity on the validation split.
   - **Bits-per-byte (BPB)** on the same validation text — required for fair comparison against Track 2, which will likely use a different vocabulary. Compute as: `BPB = (total_loss_in_nats / total_bytes_of_val_text) / ln(2)`, where `total_loss_in_nats` is summed cross-entropy over all validation tokens and `total_bytes_of_val_text` is the raw UTF-8 byte length of the validation text span (not token count — this is what makes it tokenizer-agnostic).
   - Save all final numbers to `eval/final_metrics.json`.
3. Write a short (3–5 sentence) interpretation of the loss curve into `eval/loss_curve_interpretation.md`: does it show healthy convergence, early overfitting, underfitting, or something else — and what specifically in the curve's shape supports that reading (this written interpretation is an explicit deliverable requirement, not optional).

**Definition of Done:**
- [ ] `eval/loss_curve.png` produced and legible (labeled axes, legend).
- [ ] `eval/final_metrics.json` contains loss, perplexity, and BPB.
- [ ] Written interpretation of the curve exists and references specific features of the plot (not generic boilerplate).

---

## Phase 7 — Qualitative Evaluation (Generation)

**Goal:** demonstrate the model can plausibly complete domain-relevant text — the task's actual stated success criterion.

**Steps:**
1. Select 5–10 prompts by truncating held-out (validation-region) sentences/paragraphs from the source document at a natural break point.
2. Generate completions from the fine-tuned model (with reasonable decoding settings — e.g., `do_sample=True, temperature=0.7-1.0, max_new_tokens=50-100`, or greedy/beam for a more deterministic comparison; try both and pick what best demonstrates coherence).
3. Save prompts + completions to `generations/finetuning_samples.md` in a readable prompt→completion format.
4. Briefly annotate each sample (1 sentence) on whether it stayed plausible/coherent/on-domain — this feeds directly into the "clear analysis of what each approach can and cannot do" deliverable requirement.

**Definition of Done:**
- [ ] `generations/finetuning_samples.md` contains ≥5 prompt/completion pairs with brief annotations.

---

## Phase 8 — Cross-Track Comparison Prep

**Goal:** stage everything needed for the final comparison section without duplicating work already done in the Track 2 plan.

**Steps:**
1. Copy `eval/final_metrics.json` and `eval/loss_curve.png` into `shared_eval/track1_*` (namespaced) so both tracks' artifacts sit side by side for the write-up.
2. Note in `shared_eval/comparison_notes.md`: the BPB numbers for both tracks (once Track 2 is done), a one-paragraph qualitative comparison of the generation samples, and the trainable-parameter count comparison (LoRA adapter size vs. Track 2's full model size) as a concrete efficiency data point.

**Definition of Done:**
- [ ] Track 1 metrics/plots staged in `shared_eval/`.
- [ ] Comparison notes file started (to be completed once Track 2 finishes).

---

## Appendix A — Common failure modes to watch for (agent troubleshooting checklist)

- **Loss is `NaN` early in training:** almost always learning rate too high, or missing gradient clipping — reduce LR by 5–10x and re-check.
- **Validation loss identical to pretrained-model's initial loss and never moves:** LoRA update too weak — check `target_modules` are actually the right layer names (Phase 4), rank isn't accidentally 0/misconfigured, or LR is too low.
- **Tokenizer has no pad token, batching crashes:** set `tokenizer.pad_token = tokenizer.eos_token` (Phase 2) and ensure the data collator respects this.
- **GPU OOM:** reduce batch size first, then context length; a 125–360M model with LoRA should not OOM a T4 at reasonable settings, so an OOM here is a signal something else is wrong (e.g., accidentally training the full base model instead of the LoRA-wrapped one, or not using gradient checkpointing when unnecessary memory is held).
- **`print_trainable_parameters()` shows ~100% trainable:** the base model wasn't actually frozen before `get_peft_model` wrapping, or the config's `task_type`/target modules didn't take effect — investigate before training.
