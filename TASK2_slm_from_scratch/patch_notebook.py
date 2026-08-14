import json

with open('TASK2_slm_from_scratch/slm_run.ipynb', encoding='utf-8') as f:
    nb = json.load(f)


def md_cell(source):
    src = source if isinstance(source, list) else [source]
    return {
        'cell_type': 'markdown',
        'id': f'md-{abs(hash(str(src))) % 0xffffffff:08x}',
        'metadata': {},
        'source': src,
    }


def code_cell(source):
    src = source if isinstance(source, list) else [source]
    return {
        'cell_type': 'code',
        'execution_count': None,
        'id': f'cd-{abs(hash(str(src))) % 0xffffffff:08x}',
        'metadata': {},
        'outputs': [],
        'source': src,
    }


# ── Phase 5 cells ─────────────────────────────────────────────────────────────

phase5_md = [
    '---\n', '\n',
    '## Phase 5 — Quantitative Evaluation\n', '\n',
    '**Prerequisites:** Phase 4 (all 3 sweep runs) must be complete and committed.\n',
    'Specifically, these files must exist after `git pull`:\n',
    '- `TASK2_slm_from_scratch/eval/sweep_results.csv` (written by train.py)\n',
    '- `TASK2_slm_from_scratch/checkpoints/best_val/<run>/best_val.pt` (best checkpoint)\n',
    '- `data/processed/slm_val.pt` (val tensors from Phase 2)\n',
    '- `data/processed/slm_dataset_stats.json` (must contain `split_boundary_token_idx`)\n',
    '\n',
    '**Outputs produced:**\n',
    '- `TASK2_slm_from_scratch/eval/loss_curve.png` -- train/val curves for all 3 sweep runs\n',
    '- `TASK2_slm_from_scratch/eval/final_metrics.json` -- CE loss, perplexity, BPB\n',
    '- `TASK2_slm_from_scratch/eval/loss_curve_interpretation.md` -- templated interpretation\n',
    '\n',
    '**After this cell:** commit all three outputs, then run Phase 6.',
]

phase5_smoke_src = [
    '# Cell 5a -- Smoke-test (CPU, no checkpoint or real data needed)\n',
    '# Verifies the BPB accumulation logic, plotting, and JSON writing.\n',
    '!python TASK2_slm_from_scratch/scripts/eval.py --smoke-test',
]

phase5_run_src = [
    '# Cell 5b -- Run full quantitative evaluation (GPU required)\n',
    '# Reads sweep_results.csv to auto-detect the best run.\n',
    '# Pass --run <name> to override: !python ... --run base\n',
    '!python TASK2_slm_from_scratch/scripts/eval.py',
]

phase5_inspect_src = [
    '# Cell 5c -- Inspect evaluation outputs\n',
    'import json as _json\n',
    '\n',
    "metrics = _json.load(open('TASK2_slm_from_scratch/eval/final_metrics.json'))\n",
    "print('=== Track 2 Final Metrics ===')\n",
    'for k, v in metrics.items():\n',
    "    sv = str(v)\n",
    "    print(f'  {k}: ' + (sv[:100] + '...' if len(sv) > 100 else sv) + '')\n",
    '\n',
    "bpb_gap = metrics['bpb'] - 1.309722\n",
    "print(f\"\\nBPB = {metrics['bpb']}  (Track 1 = 1.309722  gap = {bpb_gap:+.6f})\")\n",
    "print('\\n=== Loss Curve Interpretation (excerpt) ===')\n",
    "print(open('TASK2_slm_from_scratch/eval/loss_curve_interpretation.md').read())\n",
]

# ── Phase 6 cells ─────────────────────────────────────────────────────────────

phase6_md = [
    '---\n', '\n',
    '## Phase 6 -- Qualitative Evaluation (Generation)\n', '\n',
    '**Prerequisites:** Phase 5 must have completed and its outputs committed.\n',
    'The best checkpoint must exist at:\n',
    '`TASK2_slm_from_scratch/checkpoints/best_val/<best_run>/best_val.pt`\n',
    '\n',
    '**Prompts:** same 8 prompts as Track 1 (controlled comparison).\n',
    '\n',
    '**Outputs produced:**\n',
    '- `TASK2_slm_from_scratch/generations/slm_samples.md` -- 8 prompts x 2 decoding modes\n',
    '\n',
    '**After this cell:** commit slm_samples.md, review annotations manually,\n',
    'revise labels where the heuristic is wrong, then run Phase 7.',
]

phase6_smoke_src = [
    '# Cell 6a -- Smoke-test (CPU, no checkpoint needed)\n',
    '# Verifies generate() determinism, annotation logic, markdown writing.\n',
    '!python TASK2_slm_from_scratch/scripts/generate.py --smoke-test',
]

phase6_run_src = [
    '# Cell 6b -- Run qualitative generation (GPU recommended)\n',
    '# Uses the same 8 prompts as Track 1 -- controlled comparison.\n',
    '# Sampling: T=0.8, top_p=0.9  |  Greedy: argmax (T=0)\n',
    '!python TASK2_slm_from_scratch/scripts/generate.py',
]

phase6_inspect_src = [
    '# Cell 6c -- Preview generated samples\n',
    "print(open('TASK2_slm_from_scratch/generations/slm_samples.md').read())\n",
]

# ── Phase 7 cells ─────────────────────────────────────────────────────────────

phase7_md = [
    '---\n', '\n',
    '## Phase 7 -- Cross-Track Comparison\n', '\n',
    '**Prerequisites:** Phase 5 AND Phase 6 must have completed and been committed.\n',
    'Also requires `shared_eval/finetuning_final_metrics.json` (already committed).\n',
    '\n',
    '**Outputs produced:**\n',
    '- `shared_eval/slm_final_metrics.json` -- copy of Phase 5 metrics\n',
    '- `shared_eval/slm_loss_curve.png` -- copy of Phase 5 curve\n',
    '- `shared_eval/comparison_notes.md` -- fully populated (no TBD)\n',
    '\n',
    '**DoD assertion:** the script hard-fails if any "TBD" remains in comparison_notes.md.\n',
    '\n',
    '**After this cell:** commit shared_eval/ outputs.\n',
    'Phase 7 is the final phase -- the project is complete.',
]

phase7_smoke_src = [
    '# Cell 7a -- Smoke-test\n',
    '!python TASK2_slm_from_scratch/scripts/compare.py --smoke-test',
]

phase7_run_src = [
    '# Cell 7b -- Run cross-track comparison (CPU, fast)\n',
    '# Copies artefacts to shared_eval/ and writes comparison_notes.md.\n',
    '!python TASK2_slm_from_scratch/scripts/compare.py',
]

phase7_inspect_src = [
    '# Cell 7c -- Preview comparison notes\n',
    "print(open('shared_eval/comparison_notes.md').read())\n",
]

# ── Replace stub cells 31, 32, 33 with the 12 real cells ─────────────────────

new_cells = [
    md_cell(phase5_md),
    code_cell(phase5_smoke_src),
    code_cell(phase5_run_src),
    code_cell(phase5_inspect_src),
    md_cell(phase6_md),
    code_cell(phase6_smoke_src),
    code_cell(phase6_run_src),
    code_cell(phase6_inspect_src),
    md_cell(phase7_md),
    code_cell(phase7_smoke_src),
    code_cell(phase7_run_src),
    code_cell(phase7_inspect_src),
]

# Cells 31, 32, 33 are the three stubs (0-indexed)
nb['cells'] = nb['cells'][:31] + new_cells

with open('TASK2_slm_from_scratch/slm_run.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Done. Notebook now has {len(nb["cells"])} cells (stubs at 31/32/33 replaced with 12 real cells).')
