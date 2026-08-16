"""Phase 3 — Model Architecture Implementation (Training from Scratch)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Bootstrap: make scripts/ importable regardless of CWD
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import SEED, TRACK_DIR, REPO_ROOT, dump_config, iso_now, set_seed  # noqa: E402
from config import (  # noqa: E402
    CONFIGS_DIR,
    DATASET_STATS_JSON,
    MODEL_CONFIG_JSON,
    TOKENIZER_DIR,
    TRAINABLE_PARAMS_JSON,
)

# ════════════════════════════════════════════════════════════════════════════
# §1 — GPTConfig
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class GPTConfig:
    """Hyperparameter container for the from-scratch GPT model."""

    vocab_size: int                # set from Phase 1 tokenizer at runtime
    block_size: int = 256          # context length — matches Track 1 for comparison
    n_layer: int   = 6             # sweep candidate: 4, 6, 8  (Phase 4)
    n_head: int    = 4             # must evenly divide n_embd
    n_embd: int    = 192           # sweep candidate: 128, 192, 256 (Phase 4)
    dropout: float = 0.1           # applied in attention, MLP, and embedding
    bias: bool     = True          # use bias in Linear / LayerNorm layers
    tie_weights: bool = True       # share token-embedding and LM-head weights

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )

# ════════════════════════════════════════════════════════════════════════════
# §2 — CausalSelfAttention
# ════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product attention with a causal mask."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_dim = config.n_embd // config.n_head  # dimension per attention head
        self.dropout  = config.dropout  # stored as float for the SDPA conditional

        # Fused Q/K/V projection: one Linear, then split into three heads
        # Weight: (n_embd, 3*n_embd) | Bias: (3*n_embd,) if bias=True
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection: maps concatenated heads back to residual-stream dimension.
        # Named c_proj so the targeted residual-scaling init pass can find it by name.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Residual dropout applied after the output projection (nn.Dropout respects
        # self.training automatically — no manual guard needed here).
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim (C == n_embd)

        # Compute Q, K, V from the fused projection
        qkv = self.c_attn(x)                     # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # each (B, T, C)

        # Reshape to (B, n_head, T, head_dim) for multi-head attention
        # view+transpose rather than reshape so we avoid a copy when contiguous.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Causal scaled dot-product attention
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )  # (B, nh, T, hd)

        # Merge heads back and apply output projection
        # contiguous() is needed before view() when the tensor was transposed.
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.resid_drop(self.c_proj(y))
        return y

# ════════════════════════════════════════════════════════════════════════════
# §3 — MLP
# ════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Position-wise feed-forward network (two linear layers + GELU)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.act    = nn.GELU(approximate="tanh")  # tanh approx matches GPT-2
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.drop   = nn.Dropout(config.dropout)   # residual dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)    # (B, T, 4*C) — expand to hidden dimension
        x = self.act(x)     # GELU non-linearity
        x = self.c_proj(x)  # (B, T, C) — project back to residual-stream dim
        x = self.drop(x)    # regularisation before residual add
        return x

# ════════════════════════════════════════════════════════════════════════════
# §4 — TransformerBlock
# ════════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """One decoder layer: pre-LayerNorm attention + pre-LayerNorm MLP."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # pre-norm attention residual
        x = x + self.mlp(self.ln2(x))   # pre-norm MLP residual
        return x

# ════════════════════════════════════════════════════════════════════════════
# §5 — GPT (full model)
# ════════════════════════════════════════════════════════════════════════════

class GPT(nn.Module):
    """Decoder-only Transformer language model (GPT-2 style, from scratch)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # Build all sub-layers
        self.token_embedding    = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.emb_drop           = nn.Dropout(config.dropout)
        self.blocks             = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f    = nn.LayerNorm(config.n_embd, bias=config.bias)
        # LM head: no bias (standard); weight is optionally tied to token_embedding
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Base initialisation pass
        # Visits every sub-module in registration order and applies std=0.02
        # normal init to nn.Linear and nn.Embedding; LN to identity.
        self.apply(self._init_weights)

        # Residual-scaling second pass
        # The output projections (c_proj) in attention and MLP add to the
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)

        # Weight tying (after init — see class docstring)
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        """Base initialisation applied to every sub-module via self.apply()."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass. Args:."""
        device = input_ids.device
        B, T   = input_ids.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}. "
            f"Either reduce the sequence length or increase block_size in GPTConfig."
        )

        # Token + positional embeddings
        pos     = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)
        tok_emb = self.token_embedding(input_ids)   # (B, T, n_embd)
        pos_emb = self.position_embedding(pos)       # (T, n_embd) — broadcast over B
        x       = self.emb_drop(tok_emb + pos_emb)  # (B, T, n_embd)

        # N transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final LayerNorm + LM head
        x      = self.ln_f(x)       # (B, T, n_embd)
        logits = self.lm_head(x)    # (B, T, vocab_size)

        # Next-token prediction loss
        loss = None
        if labels is not None:
            # Shift: logits[i] predicts token[i+1]
            shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, vocab_size)
            shift_labels = labels[:, 1:].contiguous()        # (B, T-1)
            # F.cross_entropy expects (N, C) and (N,); flatten batch and seq dims.
            # Using the functional form (not manual softmax + NLL) for numerical
            # stability and efficiency.
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return logits, loss

    def count_parameters(self) -> dict:
        """Return total and per-component trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)

        def _count(module: nn.Module) -> int:
            return sum(
                p.numel()
                for p in module.parameters(recurse=True)
                if p.requires_grad
            )

        # When weights are tied, lm_head.parameters() returns an empty iterator
        # because lm_head.weight is already counted under token_embedding.
        return {
            "total":              total,
            "token_embedding":    _count(self.token_embedding),
            "position_embedding": _count(self.position_embedding),
            "transformer_blocks": _count(self.blocks),
            "final_layernorm":    _count(self.ln_f),
            "lm_head_tied":       0 if self.config.tie_weights else _count(self.lm_head),
            "weight_tying":       self.config.tie_weights,
            "timestamp":          iso_now(),
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive next-token generation loop. Args:."""
        was_training = self.training
        self.eval()

        try:
            for _ in range(max_new_tokens):
                # a. Crop context to block_size
                ctx = (
                    input_ids
                    if input_ids.shape[1] <= self.config.block_size
                    else input_ids[:, -self.config.block_size :]
                )

                # b. Forward — take only the last-position logits
                logits, _ = self(ctx)           # (1, T, vocab_size)
                logits = logits[:, -1, :]       # (1, vocab_size)

                # c. Greedy branch — bypass softmax entirely
                if temperature <= 0.0:
                    next_token = logits.argmax(dim=-1, keepdim=True)  # (1, 1)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    if eos_token_id is not None and next_token.item() == eos_token_id:
                        break
                    continue

                # d. Temperature scaling
                logits = logits / temperature

                # e. Top-k filtering
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    # Zero out all logits below the k-th largest value.
                    threshold = logits.topk(k).values[..., -1, None]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))

                # f. Top-p (nucleus) filtering
                if top_p is not None:
                    # Sort descending, compute cumulative softmax probs,
                    # zero out tokens once the cumulative mass exceeds top_p.
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    # Remove tokens with cumulative prob above top_p.
                    # Shift right by 1 to include the token that pushed us over.
                    remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    # Scatter back to original token ordering.
                    logits = torch.zeros_like(logits).scatter_(
                        dim=-1, index=sorted_idx, src=sorted_logits
                    )

                # g. Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

                # h. Append and check EOS
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break

        finally:
            # Always restore training state, even if an exception is raised.
            if was_training:
                self.train()

        return input_ids

# ════════════════════════════════════════════════════════════════════════════
# §6 — Runtime config loader
# ════════════════════════════════════════════════════════════════════════════

def _load_runtime_config() -> GPTConfig:
    """Build a GPTConfig from Phase 1/2 outputs, with safe defaults."""
    vocab_size  = 1024   # default — BPE sweep most likely selects this
    block_size  = 256    # matches Track 1 and build_dataset.py constant

    vocab_json = TOKENIZER_DIR / "vocab.json"
    if vocab_json.exists():
        with vocab_json.open(encoding="utf-8") as f:
            vocab_data  = json.load(f)
        vocab_size = len(vocab_data)
        print(f"[phase3] vocab_size={vocab_size} (read from {vocab_json})")
    else:
        print(
            f"[phase3] WARNING: tokenizer not found at {vocab_json}.\n"
            f"         Falling back to vocab_size={vocab_size}.\n"
            f"         Run Phase 1 on Colab first for the real value."
        )

    if DATASET_STATS_JSON.exists():
        with DATASET_STATS_JSON.open(encoding="utf-8") as f:
            stats = json.load(f)
        block_size = stats.get("context_length", block_size)
        print(f"[phase3] block_size={block_size} (read from {DATASET_STATS_JSON})")
    else:
        print(
            f"[phase3] WARNING: dataset stats not found at {DATASET_STATS_JSON}.\n"
            f"         Falling back to block_size={block_size}.\n"
            f"         Run Phase 2 on Colab first for the real value."
        )

    return GPTConfig(vocab_size=vocab_size, block_size=block_size)

# ════════════════════════════════════════════════════════════════════════════
# §7 — Unit tests (pure validation — no filesystem side effects)
# ════════════════════════════════════════════════════════════════════════════

def _unit_test(config: GPTConfig) -> None:
    """Validate the model in isolation on random data before any real training."""
    print("\n══ Phase 3 Unit Tests ══")
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[test] device={device}  vocab_size={config.vocab_size}  "
          f"block_size={config.block_size}  n_layer={config.n_layer}  "
          f"n_embd={config.n_embd}")

    model = GPT(config).to(device)
    B, T  = 2, config.block_size

    # Test 1: Output shape
    print("[test 1/8] Output shape ...")
    ids = torch.randint(0, config.vocab_size, (B, T), device=device)
    logits, _ = model(ids)
    assert logits.shape == (B, T, config.vocab_size), (
        f"FAIL: expected logits shape {(B, T, config.vocab_size)}, got {logits.shape}"
    )
    print(f"         PASS — logits shape: {tuple(logits.shape)}")

    # Test 2: Loss is finite scalar
    print("[test 2/8] Loss is a finite scalar ...")
    ids  = torch.randint(0, config.vocab_size, (B, T), device=device)
    _, loss = model(ids, labels=ids)
    assert loss is not None,              "FAIL: loss is None when labels provided"
    assert loss.ndim == 0,                f"FAIL: loss is not a scalar — shape {loss.shape}"
    assert torch.isfinite(loss).item(),   f"FAIL: loss is not finite — got {loss.item()}"
    print(f"         PASS — loss={loss.item():.4f}")

    # Test 3: Backward pass, gradients exist and are finite
    print("[test 3/8] Backward pass + finite gradients ...")
    model.zero_grad()
    ids  = torch.randint(0, config.vocab_size, (B, T), device=device)
    _, loss = model(ids, labels=ids)
    loss.backward()
    nan_params, inf_params = [], []
    for pn, p in model.named_parameters():
        if p.grad is None:
            continue
        if torch.isnan(p.grad).any():
            nan_params.append(pn)
        if torch.isinf(p.grad).any():
            inf_params.append(pn)
    assert not nan_params, f"FAIL: NaN gradients in: {nan_params}"
    assert not inf_params, f"FAIL: Inf gradients in: {inf_params}"
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "FAIL: no parameter received a gradient"
    print("         PASS — gradients exist and are finite")

    # Test 4: Parameter count in expected range
    print("[test 4/8] Parameter count ...")
    param_info = model.count_parameters()
    total = param_info["total"]
    assert 1_000_000 <= total <= 20_000_000, (
        f"FAIL: parameter count {total:,} is outside expected range [1M, 20M]. "
        f"Check GPTConfig defaults."
    )
    print(f"         PASS — total trainable params: {total:,}")

    # Test 5: Weight tying identity
    print("[test 5/8] Weight tying ...")
    if config.tie_weights:
        assert model.lm_head.weight is model.token_embedding.weight, (
            "FAIL: tie_weights=True but lm_head.weight is NOT the same object as "
            "token_embedding.weight.  Check __init__ ordering (tie must happen last)."
        )
        print("         PASS — lm_head.weight is token_embedding.weight (same object)")
    else:
        assert model.lm_head.weight is not model.token_embedding.weight, (
            "FAIL: tie_weights=False but lm_head.weight IS token_embedding.weight"
        )
        print("         PASS — weights are not tied (tie_weights=False)")

    # Test 6: Variable-length input (shorter than block_size)
    print("[test 6/8] Variable-length input ...")
    short_T = max(4, config.block_size // 4)  # well below block_size
    ids_short = torch.randint(0, config.vocab_size, (1, short_T), device=device)
    logits_short, _ = model(ids_short)
    assert logits_short.shape == (1, short_T, config.vocab_size), (
        f"FAIL: short-sequence logits shape {logits_short.shape} != "
        f"{(1, short_T, config.vocab_size)}"
    )
    print(f"         PASS — T={short_T} (< block_size={config.block_size}) works")

    # Test 7: Initial loss ≈ ln(vocab_size) (random-model sanity)
    print("[test 7/8] Initial loss near ln(vocab_size) ...")
    model_fresh = GPT(config).to(device)
    ids_rand    = torch.randint(0, config.vocab_size, (B, T), device=device)
    with torch.no_grad():
        _, loss_fresh = model_fresh(ids_rand, labels=ids_rand)
    expected   = math.log(config.vocab_size)
    tolerance  = 1.0
    actual_val = loss_fresh.item()
    assert abs(actual_val - expected) <= tolerance, (
        f"FAIL: initial loss {actual_val:.4f} is not near ln({config.vocab_size})={expected:.4f} "
        f"(tolerance ±{tolerance}). This likely indicates an init or masking bug."
    )
    print(f"         PASS — initial loss={actual_val:.4f}, ln(vocab_size)={expected:.4f}")

    # Test 8: Eval-mode dropout determinism
    print("[test 8/8] Eval-mode dropout determinism ...")
    model_eval = GPT(config).to(device)
    model_eval.eval()
    ids_eval = torch.randint(0, config.vocab_size, (B, T), device=device)
    with torch.no_grad():
        _, loss_eval_1 = model_eval(ids_eval, labels=ids_eval)
        _, loss_eval_2 = model_eval(ids_eval, labels=ids_eval)
    assert loss_eval_1.item() == loss_eval_2.item(), (
        f"FAIL: eval-mode forward passes gave different losses "
        f"({loss_eval_1.item():.6f} vs {loss_eval_2.item():.6f}). "
        f"Dropout is still firing in eval mode — check `dropout_p=self.dropout "
        f"if self.training else 0.0` in CausalSelfAttention.forward()."
    )
    print(f"         PASS — eval-mode loss is deterministic ({loss_eval_1.item():.6f})")

    print("\n[phase3] All 8 unit tests passed [OK]")

# ════════════════════════════════════════════════════════════════════════════
# §8 — Artifact writer (separate from unit tests — no side effects in tests)
# ════════════════════════════════════════════════════════════════════════════

def _dump_phase3_artifacts(config: GPTConfig, model: GPT) -> None:
    """Write run_phase3_model.json and trainable_params.json. Called by __main__ AFTER _unit_test() passes. Kept separate from."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Architecture config (global config-as-file convention §0)
    cfg_dict = asdict(config)
    cfg_dict["phase"] = "phase3_model"
    cfg_dict["seed"]  = SEED
    dump_config(cfg_dict, "phase3_model")  # writes configs/run_phase3_model.json

    # Trainable parameter breakdown
    param_info = model.count_parameters()
    TRAINABLE_PARAMS_JSON.parent.mkdir(parents=True, exist_ok=True)
    TRAINABLE_PARAMS_JSON.write_text(
        json.dumps(param_info, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[phase3] trainable params -> {TRAINABLE_PARAMS_JSON}")

    # Print summary to stdout (captured by Colab cell output)
    print(f"\n[phase3] Parameter summary:")
    for k, v in param_info.items():
        if isinstance(v, int):
            print(f"  {k}: {v:,}")
        else:
            print(f"  {k}: {v}")

# ════════════════════════════════════════════════════════════════════════════
# §9 — Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track 2 Phase 3 — Model Architecture"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run unit tests only; skip artifact writing. "
            "No Phase 1/2 outputs required (safe for local checks)."
        ),
    )
    args = parser.parse_args()

    # 1. Runtime config — reads from Phase 1/2 outputs if present, else defaults.
    config = _load_runtime_config()

    # 2. Unit tests — ALWAYS run, regardless of --smoke-test flag.
    #    Any AssertionError stops execution here.  No artifacts are ever
    #    written from a model that fails its own tests.
    _unit_test(config)

    # 3. Artifact dump — skipped in --smoke-test mode.
    #    Only reached if _unit_test() completed without AssertionError.
    if not args.smoke_test:
        final_model = GPT(config)
        _dump_phase3_artifacts(config, final_model)
        print("\n[phase3] Done. Ready for Phase 4.")
    else:
        print("\n[phase3] Smoke-test mode: all tests passed. No artifacts written.")
