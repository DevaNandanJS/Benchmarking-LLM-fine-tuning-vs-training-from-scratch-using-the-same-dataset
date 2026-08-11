# Track 2 Execution Plan — Training a Small Language Model From Scratch

**Purpose of this document:** a phase-by-phase, agent-executable build plan for a custom decoder-only Transformer trained from random initialization on the document. Same conventions and directory layout as `track1_finetuning_execution_plan.md` (see that document's Section 0) — read that section first if you haven't; it's not repeated in full here, only the parts that differ.

**Companion document:** `track1_finetuning_execution_plan.md`. Final comparison happens in both documents' Phase 8/9.

---

## 0. Track-2-specific conventions

- All artifacts live under `track2_scratch/` mirroring Track 1's subfolder structure: `configs/`, `checkpoints/`, `logs/`, `eval/`, `generations/`.
- Same `metrics.jsonl` logging format as Track 1 (Section 0 of that document) — this is what makes the two tracks' loss curves plottable on comparable axes later.
- Same seed discipline, same config-dump-before-training discipline, same "sweep, don't guess" discipline for hyperparameters.
- This track reuses `data/extracted/document_clean.txt` produced in Track 1's Phase 1 — **do not re-extract**; if Track 1 hasn't been run yet, run its Phase 1 first (it's track-agnostic data prep).

---

## Phase 0 — Environment Setup

**Goal:** minimal, from-scratch training environment — deliberately fewer dependencies than Track 1, since nothing here relies on a pretrained model hub.

**If you're using the VS Code + Google Colab extension workflow, read Track 1's Phase 0 §0.a first** — it covers the authoring-vs-execution split (agent edits `.py` files locally; anything GPU-bound must run as a notebook cell against the remote kernel, whose filesystem is separate from your local workspace) and the thin-orchestrator-notebook pattern this plan reuses. Not repeated in full here.

**Steps:**
1. Confirm the GPU/Colab connection exactly as in Track 1 Phase 0 steps 1–3. If you're working through both tracks in one sitting, you can **reuse the same running Colab session** rather than spinning up a second one — open `track2_scratch/track2_run.ipynb` and connect via **Auto Connect** to attach to the already-running server (saves reconnect time and avoids a second GPU-queue wait). If you're returning in a later session, connect fresh as in Track 1.
2. Sync code the same way as Track 1 (git clone/pull cell). This repo should already contain `data/extracted/document_clean.txt` from Track 1's Phase 1 — **do not re-extract it here**; if it's missing after syncing, run Track 1's Phase 1 first (it's track-agnostic data prep, not something to duplicate).
3. Install dependencies inside a notebook cell (not a local terminal — see Track 1 §0.a for why that distinction matters):
   ```python
   !pip install -q torch tokenizers matplotlib pandas numpy
   ```
   Note `tokenizers` (Hugging Face's Rust-backed tokenizer training library) is the only "external" dependency of substance — everything else (model, training loop) will be implemented directly, per the task's requirement for "your own architecture and training code."
4. Log `!pip freeze > track2_scratch/environment.txt` and copy it back into the version-controlled repo, same discipline as Track 1.
5. Create `track2_scratch/{configs,checkpoints,logs,eval,generations,scripts}/` (the `scripts/` folder is where the agent's actual model/training code lives) and the orchestrator notebook `track2_scratch/track2_run.ipynb` (same sync-then-call-into-scripts pattern as Track 1's `track1_run.ipynb`).

**Definition of Done:**
- [ ] Connected to a Colab GPU kernel (reused or fresh), confirmed via the same hardware-check cell as Track 1.
- [ ] `track2_run.ipynb` created with working sync/install cells.
- [ ] Environment installed and versions logged, copied back into the repo.
- [ ] Directory structure created, including `scripts/` for agent-authored modules.

---

## Phase 1 — Custom Tokenizer Training

**Goal:** a small-vocabulary byte-level BPE tokenizer trained only on this document, with an explicit, logged justification for the chosen vocabulary size.

**Steps:**
1. Train a `ByteLevelBPETokenizer` (handles arbitrary Unicode robustly without an explicit unknown-token problem) using the `tokenizers` library:
   ```python
   from tokenizers import ByteLevelBPETokenizer
   tokenizer = ByteLevelBPETokenizer()
   tokenizer.train(
       files=["data/extracted/document_clean.txt"],
       vocab_size=1024,          # sweep candidate; see step 2
       min_frequency=2,
       special_tokens=["<|endoftext|>"],
   )
   tokenizer.save_model("track2_scratch/tokenizer")
   ```
2. **Vocabulary size sweep (required, not optional):** train at minimum 3 candidate sizes — e.g., **256, 1024, 4096** — and for each, compute and log:
   - Resulting vocabulary size actually produced (BPE may plateau below the target if the corpus is too small to fill it — this itself is diagnostic and worth noting).
   - **Fertility**: average tokens per word on the source document (`total_tokens / total_words`) — lower is more compressive, but too-low fertility at tiny vocab sizes trades off against needing longer context windows for the same content.
   - Resulting total token count for the whole document under each vocab size (this changes how many training chunks Phase 2 produces).
   Save this comparison to `track2_scratch/eval/vocab_sweep.csv`. Pick a final vocabulary size and write 2–3 sentences in `configs/tokenizer_choice.md` justifying it against the trade-off described in the blueprint document (Section 4.2): embedding-table size relative to model size and to corpus size.
3. Sanity check: tokenize a few sentences from the document and print the resulting tokens — confirm no pathological over-fragmentation (near character-level for common words) or under-fragmentation (huge chunks that don't look like meaningful subwords) at the chosen size.
4. Add an explicit end-of-text special token and confirm it's correctly inserted between logical document sections if applicable (or omit if the whole document is treated as one continuous stream — decide and log which).

**Definition of Done:**
- [ ] `vocab_sweep.csv` with ≥3 candidate vocab sizes and their fertility/token-count stats.
- [ ] Final tokenizer files saved to `track2_scratch/tokenizer/` (`vocab.json`, `merges.txt`).
- [ ] Written justification for the chosen vocab size referencing the actual sweep numbers (not generic reasoning).

---

## Phase 2 — Dataset Construction

**Goal:** tokenized, windowed, train/val-split dataset under the *custom* tokenizer, using the same chunking philosophy as Track 1 for later comparability.

**Steps:**
1. Tokenize the full document with the chosen custom tokenizer into one long ID sequence.
2. Apply the same sliding-window chunking approach as Track 1 Phase 3 — **match the split strategy (contiguous holdout) and, where feasible, a comparable context length** so the two tracks' "training examples" are conceptually aligned for the write-up's comparison section. Context length here is likely to differ from Track 1's if the vocabulary is much smaller (smaller vocab → more tokens per unit of text → consider whether to match token-count-per-example or raw-text-length-per-example; document whichever you choose).
3. Save as plain `torch.Tensor` (a single long tensor of token IDs is sufficient at this data scale — no need for memory-mapped binary files as in large-corpus setups like nanoGPT's `prepare.py`, since the entire dataset comfortably fits in RAM).
4. Log chunk counts, context length, and train/val split sizes to `data/processed/track2_dataset_stats.json`.

**Definition of Done:**
- [ ] Script runs end-to-end from `document_clean.txt` + trained tokenizer to saved train/val tensors.
- [ ] Stats logged; context length and split strategy documented and consistent with Track 1's approach (or differences explicitly justified).

---

## Phase 3 — Model Architecture Implementation

**Goal:** a from-scratch, understood-line-by-line decoder-only Transformer, sized appropriately for the data scale.

**Steps:**
1. Define a config dataclass, e.g.:
   ```python
   @dataclass
   class GPTConfig:
       vocab_size: int         # from Phase 1's final tokenizer
       block_size: int         # = context length from Phase 2
       n_layer: int = 6         # sweep candidate: 4, 6, 8
       n_head: int = 4          # must evenly divide n_embd
       n_embd: int = 192        # sweep candidate: 128, 192, 256
       dropout: float = 0.1
       bias: bool = True
   ```
2. Implement the core blocks (this is the part you must be able to explain line-by-line in the interview — implement it yourself rather than importing a pre-built GPT class, even though the architecture pattern follows the standard GPT-2-style design):
   - **Token + positional embeddings**: learned embedding tables (`nn.Embedding` for tokens, `nn.Embedding` for absolute positions up to `block_size`), summed.
   - **Causal self-attention**: multi-head scaled dot-product attention with a causal mask (either an explicit lower-triangular mask + `masked_fill`, or `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)` for the built-in fused/efficient path — using the fused version is fine and still something you can explain conceptually, just be honest in the interview that you used the built-in causal-masking implementation rather than hand-writing the mask).
   - **MLP block**: two linear layers with a GELU (or ReLU) activation in between, typically expanding to 4x `n_embd` in the hidden layer.
   - **Transformer block**: pre-layernorm residual structure — `x = x + attn(ln1(x))`, `x = x + mlp(ln2(x))`.
   - **Final layernorm + LM head**: project back to vocabulary logits; consider **weight tying** between the token embedding and the LM head (`self.lm_head.weight = self.token_embedding.weight`) — a well-known parameter-saving technique worth naming and justifying (reduces parameter count meaningfully at this scale, and is standard practice in GPT-2-style implementations) even if you decide not to use it.
3. Implement a `forward()` that accepts `input_ids` (and optionally `labels`), returns `logits` and, if labels given, cross-entropy `loss` (shift logits/labels by one position for next-token prediction — get this indexing right and be ready to explain exactly why the shift is needed).
4. Implement weight initialization explicitly (don't rely purely on PyTorch defaults) — a common, defensible choice is small-std normal initialization (e.g., std=0.02) for linear/embedding layers, matching common GPT-2-style implementations; log whatever you choose.
5. Print and log total parameter count; sanity-check it's in the intended small range (low single-digit to a few tens of millions — not hundreds of millions) given the config chosen.
6. Unit-test the model in isolation before touching real data: feed a random `input_ids` tensor of the right shape, confirm output logits shape is `[batch, seq_len, vocab_size]`, confirm loss is a finite scalar and `loss.backward()` runs without error.

**Definition of Done:**
- [ ] Model implemented as original code (not an imported pretrained-model class), with each component (embeddings, attention, MLP, block, head) separately identifiable and commentable.
- [ ] Parameter count printed/logged and sanity-checked as "small" relative to the corpus.
- [ ] Isolated forward/backward unit test passes on random data before real training begins.

---

## Phase 4 — Training Loop Implementation

**Goal:** a from-scratch training loop (justified above: writing this yourself, not via `Trainer`, is appropriate here since the task explicitly asks for "your own... training code") with the same logging discipline as Track 1.

**Steps:**
1. Optimizer: `AdamW`, with weight decay applied selectively (common, defensible convention: apply weight decay to 2D+ parameters — linear/embedding weights — but not to biases/layernorm parameters; implement this split explicitly and be ready to explain why: biases/norm parameters don't benefit from and can be hurt by weight decay).
2. LR schedule: warmup (linear, over the first ~5–10% of total steps) followed by cosine decay to a small minimum LR — implement manually with `torch.optim.lr_scheduler.LambdaLR` or a hand-written schedule function; log the LR at every step alongside loss (already part of the `metrics.jsonl` schema).
3. Gradient clipping: clip global norm to e.g. `1.0` — from-scratch training from random init is more prone to early instability than fine-tuning, so this matters more here than in Track 1.
4. Batch construction: random sampling of chunk indices per batch (with replacement across epochs is fine at this scale) or a shuffled-epoch approach — either is defensible; log which.
5. Training loop skeleton:
   ```python
   for step in range(total_steps):
       batch = get_batch(train_data, batch_size, device)
       logits, loss = model(batch["input_ids"], labels=batch["labels"])
       loss.backward()
       torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
       optimizer.step()
       scheduler.step()
       optimizer.zero_grad()
       if step % log_every == 0:
           val_loss = evaluate(model, val_data)   # full or sampled val pass
           log_metrics(step, loss.item(), val_loss, scheduler.get_last_lr()[0])
       if step % save_every == 0 and val_loss < best_val_loss:
           torch.save(model.state_dict(), "checkpoints/best_val.pt")
   ```
6. **Explicit stopping discipline:** given the data scale, expect the useful training window to be short — set a generous `total_steps` upper bound but rely on tracking best-val-loss checkpointing (as above) rather than assuming the final step is the best one. Do not silently just use the last checkpoint.
7. Run the small architecture/hyperparameter sweep promised in Section 0's cross-document convention: at minimum, compare 2–3 combinations among `{n_layer, n_embd, learning_rate}` as separate named runs, each with full `metrics.jsonl` logging, summarized in `eval/sweep_results.csv`.

**Definition of Done:**
- [ ] Full training loop runs end-to-end with `metrics.jsonl` logging at the agreed cadence.
- [ ] Best-val-loss checkpoint saved and distinguishable from "last" checkpoint.
- [ ] Small sweep (≥2 configurations) completed and tabulated.

---

## Phase 5 — Quantitative Evaluation

**Goal:** mirror Track 1's Phase 6 exactly in structure, so the two tracks' numbers can sit side by side.

**Steps:**
1. Plot train/val loss vs. step → `eval/loss_curve.png`.
2. Compute final metrics on the best checkpoint: token-level cross-entropy loss, perplexity, and **bits-per-byte (BPB)** — using the *same BPB formula* as Track 1 (total held-out loss in nats, divided by raw UTF-8 byte length of the validation text span, divided by ln 2) so this specific number is directly comparable across tracks despite the different vocabularies. Save to `eval/final_metrics.json`.
3. Write a loss-curve interpretation (`eval/loss_curve_interpretation.md`) — for a from-scratch model on this little data, explicitly discuss whether you observe the "fast overfitting" signature the blueprint predicts (validation loss turning upward earlier/faster than in Track 1), and whether the model shows healthy initial descent at all (a flat, high loss from the start would indicate an optimization or architecture bug, not just data scarcity — distinguish between these two failure modes explicitly).

**Definition of Done:**
- [ ] `eval/loss_curve.png` and `eval/final_metrics.json` produced.
- [ ] Interpretation explicitly compares the *shape* of this curve's overfitting dynamics to what's expected relative to Track 1 (even before Track 1's exact numbers are in hand, the qualitative comparison can be pre-written).

---

## Phase 6 — Qualitative Evaluation (Generation)

**Goal:** demonstrate the from-scratch model can plausibly continue in-domain text — with realistic expectations set for what "plausible" means for a model with no general-language prior.

**Steps:**
1. Use the *same* held-out prompts as Track 1's Phase 7 where possible (same source sentences/paragraphs) — this makes the qualitative comparison direct rather than apples-to-oranges.
2. Generate completions via simple autoregressive sampling (implement `generate()` yourself: repeatedly forward-pass, take logits at the last position, sample/argmax, append, repeat up to `max_new_tokens` or until the end-of-text token) — again, writing this yourself (rather than relying on a library `generate()` method) matches the "explain every line" requirement well here since it's a short, self-contained loop.
3. Save to `generations/track2_samples.md` with the same prompt/completion/annotation format as Track 1.
4. In the annotations, be honest about the expected quality ceiling: look specifically at whether completions are near-verbatim reproductions of training chunks (memorization — expected and fine per the task's stated bar) vs. incoherent/off-domain (a real weakness to name) vs. genuinely novel-but-plausible in-domain phrasing (the best-case outcome, worth highlighting if observed).

**Definition of Done:**
- [ ] `generations/track2_samples.md` with ≥5 prompt/completion pairs (ideally the same prompts as Track 1), annotated for memorization vs. incoherence vs. novel-plausible phrasing.

---

## Phase 7 — Cross-Track Comparison Prep

**Goal:** finalize the comparison artifacts jointly with Track 1's Phase 8.

**Steps:**
1. Copy `eval/final_metrics.json` and `eval/loss_curve.png` into `shared_eval/track2_*`.
2. Complete `shared_eval/comparison_notes.md` (started in Track 1's Phase 8) with:
   - Side-by-side BPB numbers for both tracks (the tokenizer-agnostic metric that makes this comparison fair).
   - Side-by-side qualitative generation comparison using the shared prompt set.
   - Parameter-count comparison: Track 1's trainable LoRA parameters vs. Track 2's total model parameters — a concrete data point for the "parameter count vs. memory footprint" trade-off discussion.
   - A short paragraph explicitly stating which approach "won" on which axis (quality, data efficiency, compute cost, deployment footprint) rather than declaring one approach an unqualified winner — this maps directly onto the write-up's required production recommendation.

**Definition of Done:**
- [ ] Track 2 metrics/plots staged in `shared_eval/`.
- [ ] `comparison_notes.md` completed with both tracks' numbers and a stated, reasoned recommendation.

---

## Appendix A — Common failure modes to watch for (agent troubleshooting checklist)

- **Loss stuck near `ln(vocab_size)` (the loss of a uniform-random model) and never decreases:** likely a bug, not a data-scarcity issue — check the labels are correctly shifted by one position, check the causal mask is actually applied (a model that can see future tokens will trivially "solve" next-token prediction and show a suspiciously *low* loss instead — the opposite bug, also worth checking for), check gradients are actually flowing (`loss.backward()` called, `optimizer.step()` called, LR not accidentally 0).
- **Loss becomes `NaN`:** reduce LR, confirm gradient clipping is actually wired in before the optimizer step (not after), check initialization isn't producing extreme values.
- **Validation loss far higher than training loss from the very first evaluation:** check the train/val split doesn't have a bug (e.g., val set accidentally empty, or drawn from a wildly different part of the document with different vocabulary distribution than train).
- **Generation produces the same repeated token/phrase indefinitely:** classic small-model degenerate greedy-decoding failure mode — try sampling with temperature/top-k/top-p instead of pure argmax, and treat this as an expected limitation to name explicitly, not necessarily a bug to fully eliminate.
- **Tokenizer produces far fewer merges than the requested `vocab_size`:** expected when the corpus is small — the BPE algorithm simply runs out of frequent pairs to merge. Log the *actual* resulting vocab size, not the requested one, and note this as a direct, concrete symptom of the data-scarcity constraint discussed in the blueprint.
