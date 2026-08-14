# Loss Curve Interpretation — Track 1 (LoRA Fine-Tuning, R16)

> **Note:** This interpretation was templated by evaluate.py and populated with
> real statistics. Revise the phrasing to reflect your own reading of the curve.

## Curve Summary

The training loss decreased from **3.4239** at step 1 to
**3.0963** by the final logged step, indicating that the LoRA
adapter successfully updated its weights and the model learned from the training
signal rather than staying at its pretrained loss baseline.

The validation loss reached its minimum at **step 540** (epoch 10.0),
where val_loss = **2.3869** (perplexity ≈ 10.7794).
No clear overfitting divergence was observed within the training run.

## Metric Summary

| Metric | Value |
|---|---|
| Final mean CE loss (val) | 2.377641 |
| Perplexity | 10.7794 |
| Bits-per-byte (BPB) | 1.309722 |
| Best val checkpoint step | 540 |

## Interpretation

The curve shape is consistent with a **[healthy convergence / early overfitting / underfitting]**
pattern — revise this based on the actual shape:

- **Healthy convergence:** both curves fall together and plateau; val follows train closely.
- **Early overfitting (likely at this scale):** train continues falling after val bottoms out;
  the divergence point is the step reported above.
- **Underfitting:** both curves plateau at a high loss and neither falls significantly.

The BPB of **1.309722** is the primary cross-track comparison metric.
Lower BPB means the model assigns more probability mass to the actual text per byte,
regardless of vocabulary size. This value will be compared directly to Track 2's BPB
once that track completes.
