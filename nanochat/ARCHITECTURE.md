# nanochat Architecture

> Generated from [`nanochat/gpt.py`](nanochat/gpt.py) — default config: `n_layer=12, n_embd=768, n_head=6, n_kv_head=6, sequence_len=2048, vocab_size=32768`

---

## High-Level Data Flow

```
idx ──► ① wte ──► ② norm ──► ③ smear ──┬── x0 ──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
                                         │        │  │  │  │  │  │  │  │  │  │  │  │  │
                                         │        ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼
                                         │       ④ h0→h1→h2→h3→h4→h5→h6→...→h11
                                         │                     │
                                         │                     └── x_backout (saved at layer 6)
                                         │                              │
                                         └──────────────────────────────┼── (x0 also feeds each layer)
                                                                        ▼
                                                             ⑤ backout ──► ⑥ norm ──► ⑦ lm_head
                                                                                          │
                                                                                    ⑧ softcap
                                                                                          │
                                                                                    ⑨ loss/logits
```

---

## Full Architecture

```
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                         ① TOKEN EMBEDDING (wte)                                     │
  │                                                                                                      │
  │  x = wte(idx)  →  (B, 2048, 768)                                                                     │
  │                                                                                                      │
  │  nn.Embedding(32768, 768) — looks up each token ID → 768-dim vector                                  │
  │  Weight init: N(0, 0.8). No position embedding added (RoPE handles it).                              │
  └──────────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                     │ x
                                                     ▼
  ┌──────────────────────────────────────────────────╧───────────────────────────────────────────────────┐
  │                                         ② INPUT RMS NORM                                            │
  │                                                                                                      │
  │  x = F.rms_norm(x, (768,))  →  (B, 2048, 768)                                                        │
  │                                                                                                      │
  │  Normalizes each token vector to unit RMS. No learnable parameters.                                  │
  └──────────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                     │ x
                                                     ▼
  ┌──────────────────────────────────────────────────╧───────────────────────────────────────────────────┐
  │                                         ③ SMEAR (Bigram Mixing)                                     │
  │                                                                                                      │
  │  gate = λ_smear · σ(Linear(x[:, 1:, :24]))  →  (B, 2047, 1)                                         │
  │  x[1:] += gate · x[:-1]                      →  (B, 2048, 768)                                      │
  │                                                                                                      │
  │  Each token gets a bit of its predecessor's embedding. Gate driven by 24 dims of hidden state.       │
  │  λ_smear init=0 → starts as no-op.                                                                   │
  │                                                                                                      │
  │  ★ x0 = x  (saved for per-layer residual blend)                                                      │
  └──────────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                     │ x, x0
                                                     ▼
  ┌──────────────────────────────────────────────────╧───────────────────────────────────────────────────┐
  │                                       ④ TRANSFORMER TRUNK (12 layers)                                │
  │                                                                                                      │
  │    x0 ───┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐       │
  │          │      │      │      │      │      │      │      │      │      │      │      │      │       │
  │          ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼       │
  │    x ──► h.0 ─► h.1 ─► h.2 ─► h.3 ─► h.4 ─► h.5 ─► h.6 ─► h.7 ─► h.8 ─► h.9 ─► h.10 ─► h.11     │
  │                                          ★ backout saved here                                       │
  └──────────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                     │ x, x_backout
                                                     ▼
  ┌──────────────────────────────────────────────────╧───────────────────────────────────────────────────┐
  │                                         ⑤ BACKOUT                                                    │
  │                                                                                                      │
  │  x = x - λ_backout · x_backout  →  (B, 2048, 768)                                                    │
  │                                                                                                      │
  │  Subtracts the layer-6 representation from the final hidden state.                                   │
  │  λ_backout init=0.2, learnable. Removes "low-level" features.                                       │
  └──────────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                     │ x
                                                     ▼
  ┌──────────────────────────────────────────────────╧───────────────────────────────────────────────────┐
  │                                         ⑥ FINAL RMS NORM                                            │
  │                                                                                                      │
  │  x = norm(x)  →  (B, 2048, 768)                                                                      │
  │                                                                                                      │
  └──────────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                     │ x
                                                     ▼
  ┌──────────────────────────────────────────────────╧───────────────────────────────────────────────────┐
  │                                         ⑦ LM HEAD (Logit Projection)                                 │
  │                                                                                                      │
  │  logits = Linear(768 → 32768)(x)  →  (B, 2048, 32768)                                                │
  │                                                                                                      │
  │  Untied from wte! Separate weight matrix. Init: N(0, 0.001).                                         │
  └──────────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                     │ logits
                                                     ▼
  ┌──────────────────────────────────────────────────╧───────────────────────────────────────────────────┐
  │                                         ⑧ LOGIT SOFTCAP                                             │
  │                                                                                                      │
  │  logits = 15 · tanh(logits / 15)  →  (B, 2048, 32768)                                                │
  │                                                                                                      │
  │  Caps logits to [-15, 15], ~identity for small values.                                               │
  │  Prevents extreme logits from dominating softmax early in training.                                   │
  └──────────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                     │ logits
                                                     ▼
  ┌──────────────────────────────────────────────────╧───────────────────────────────────────────────────┐
  │                                         ⑨ OUTPUT                                                     │
  │                                                                                                      │
  │  Training:   loss = CrossEntropyLoss(logits, targets)                                                │
  │  Inference:  return logits[:, -1, :]  →  next token prediction                                       │
  │                                                                                                      │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Single Layer Internals (h.0 ~ h.11)

Each layer follows the same structure. The only differences are window size, whether it has Value Embeddings, and the residual blend coefficients.

```
                        ┌──────────────────────────────────────────────────────┐
                        │               ENTRY: x_in  →  (B, 2048, 768)        │
                        └──────────────────────┬───────────────────────────────┘
                                               │
                        ┌──────────────────────▼───────────────────────────────┐
                        │                ① RESIDUAL BLEND                      │
                        │                                                      │
                        │  x = λ_resid[i] · x_in + λ_x0[i] · x0               │
                        │                                                      │
                        │  Blends the previous hidden state with the original   │
                        │  normalized embedding (x0). λ_x0 decays across depth. │
                        └──────────────────────┬───────────────────────────────┘
                                               │ x
                                               ▼
                        ┌──────────────────────────────────────────────────────┐
                        │  ② ATTENTION                                         │
                        │                                                      │
                        │   n  = norm(x)            →  (B, 2048, 768)          │
                        │                                                      │
                        │   Q  = Linear(768→768)(n) →  (B, T, 6, 128)         │
                        │   K  = Linear(768→768)(n) →  (B, T, 6, 128)         │
                        │   V  = Linear(768→768)(n) →  (B, T, 6, 128)         │
                        │                                                      │
                        │   ── Value Embedding (if has_ve) ──                  │
                        │   ve = value_embeds[i](idx)   →  (B, T, 768)         │
                        │   gate = 3·σ(Linear₁₂→₆(x[:12])) → (B, T, 6)        │
                        │   V += gate.unsqueeze(-1) · ve.view(B,T,6,128)       │
                        │                                                      │
                        │   Q = apply_rotary_emb(Q, cos, sin)                  │
                        │   K = apply_rotary_emb(K, cos, sin)                  │
                        │   Q = RMSNorm(Q) × 1.2                               │
                        │   K = RMSNorm(K) × 1.2                               │
                        │                                                      │
                        │   y = FlashAttn(Q, K, V, causal=True,                │
                        │                    window_size=S|L)                  │
                        │      →  (B, T, 6, 128)                               │
                        │                                                      │
                        │   y = c_proj(reshape) →  (B, T, 768)  ★ zero-init ★ │
                        │                                                      │
                        │  x = x + y  (residual)                               │
                        └──────────────────────┬───────────────────────────────┘
                                               │ x
                                               ▼
                        ┌──────────────────────────────────────────────────────┐
                        │  ③ MLP                                               │
                        │                                                      │
                        │   n  = norm(x)            →  (B, 2048, 768)          │
                        │   h  = c_fc(n)            →  (B, T, 3072)            │
                        │   h  = relu(h)²           →  (B, T, 3072)            │
                        │   y  = c_proj(h)          →  (B, T, 768)  ★ zero ★  │
                        │                                                      │
                        │  x = x + y  (residual)                               │
                        └──────────────────────┬───────────────────────────────┘
                                               │ x_out
                                               ▼
                        ┌──────────────────────────────────────────────────────┐
                        │  BACKOUT CHECKPOINT (layer 6 only)                   │
                        │                                                      │
                        │  if i == 6:  x_backout = x_out                       │
                        │                                                      │
                        │  x_out goes to next layer: h[i+1]                    │
                        └──────────────────────────────────────────────────────┘
```

---

## Per-Layer Configuration

```
         ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
         │ h.0  │ h.1  │ h.2  │ h.3  │ h.4  │ h.5  │ h.6  │ h.7  │ h.8  │ h.9  │ h.10 │ h.11 │
         ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
 Window  │  S   │  S   │  S   │  L   │  S   │  S   │  S   │  L   │  S   │  S   │  S   │  L   │
         ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
 VE      │  ✗   │  ✓   │  ✗   │  ✓   │  ✗   │  ✓   │  ✗   │  ✓   │  ✗   │  ✓   │  ✗   │  ✓   │
         ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
λ_resid  │1.150 │1.142 │1.133 │1.125 │1.117 │1.108 │1.100 │1.092 │1.083 │1.075 │1.067 │1.058 │
         ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
λ_x0     │0.200 │0.186 │0.173 │0.159 │0.145 │0.132 │0.118 │0.105 │0.091 │0.077 │0.064 │0.050 │
         ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
Backout  │      │      │      │      │      │      │  ★   │      │      │      │      │      │
         └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

Legend:
  S       = short window (~768 tokens, ~¼ context)
  L       = long window (2048 tokens, full context)
  VE ✓    = layer has Value Embedding
  ★       = backout checkpoint (layer n_layer//2 = 6)
```

### Window Pattern Details

The pattern `"SSSL"` is tiled across 12 layers, giving the repeating unit: **3 short, 1 long**.

- **Short layers** (S): attend to max 768 previous tokens. ~3× cheaper than full attention (`O(n·w)` vs `O(n²)`).
- **Long layers** (L): attend to all 2048 tokens. Every 4th layer + the final layer.
- **Rationale**: Local processing is cheap; every 4th layer propagates global information. From Longformer/Mistral-style sliding window attention.

### Value Embedding Details

Alternating layers (scoped by `has_ve(i, n_layer)` — matches last-layer parity, last always included):
- Each VE layer has its own `nn.Embedding(32768, 768)` = ~25M params per layer × 6 layers = ~151M total.
- The VE is gated per head: `gate = 3·σ(Linear₁₂→₆(x[:12]))`, range (0, 3).
- **Why**: Gives attention heads a direct "what token is this" signal alongside the contextual V.

### Residual Blend Details

- **λ_resid[i]**: Scales the previous hidden state. Decays from 1.15 → 1.05 (stronger residual at early layers).
- **λ_x0[i]**: Blends the original normalized embedding (x0) back in. Decays from 0.20 → 0.05.
- **Together**: `x = λ_resid · x + λ_x0 · x0` counteracts "residual stream collapse" and preserves token identity through depth.

### Backout Details

- At layer 6 (`n_layer // 2`), the hidden state is cached as `x_backout`.
- After layer 11: `x = x - λ_backout · x_backout`, with λ_backout init=0.2, learnable.
- **Intuition**: Subtract "low-level" features present at the midpoint, leaving only the high-level features added in layers 7-11.

---

## Parameter Summary

| Module              | Shape              | Parameters  | Init               | Notes                               |
|---------------------|--------------------|-------------|--------------------|-------------------------------------|
| `wte`               | [32768, 768]       |  25,165,824 | N(0, 0.8)          | Token embedding                     |
| `h[i].attn.c_q`    | [768, 768]         |     589,824 | U(-s, s)           | Query projection (×12 = 7.1M)       |
| `h[i].attn.c_k`    | [768, 768]         |     589,824 | U(-s, s)           | Key projection (×12)                |
| `h[i].attn.c_v`    | [768, 768]         |     589,824 | U(-s, s)           | Value projection (×12)              |
| `h[i].attn.c_proj` | [768, 768]         |     589,824 | **zeros**          | Attention output (×12)              |
| `h[i].attn.ve_gate`| [12, 6]            |          72 | U(0, 0.02)         | Only on VE layers (×6)              |
| `h[i].mlp.c_fc`    | [3072, 768]        |   2,359,296 | U(-s·0.4, s·0.4)   | MLP expand (×12 = 28.3M)            |
| `h[i].mlp.c_proj`  | [3072, 768]        |   2,359,296 | **zeros**          | MLP contract (×12)                  |
| `value_embeds[k]`  | [32768, 768]       |  25,165,824 | U(-s, s)           | ×6 layers with VE = ~151M           |
| `lm_head`          | [32768, 768]       |  25,165,824 | N(0, 0.001)        | Untied from wte                     |
| `resid_lambdas`    | [12]               |          12 | 1.15→1.05          | Per-layer residual scale            |
| `x0_lambdas`       | [12]               |          12 | 0.20→0.05          | Per-layer x0 blend                  |
| `smear_gate`       | [24, 1]            |          24 | U(0, 0.02)         |                                     |
| `smear_lambda`     | scalar             |           1 | 0                  | Start as no-op                      |
| `backout_lambda`   | scalar             |           1 | 0.2                |                                     |
| **Total**          |                    | **~263M**   |                    |                                     |

---

## Architectural Innovators (vs Standard GPT-2)

| Feature               | This Model                | Standard GPT-2        | Benefit                                         |
|-----------------------|---------------------------|-----------------------|-------------------------------------------------|
| Position encoding     | RoPE (rotary)             | Learned absolute      | Length generalization, relative bias            |
| Normalization         | RMS Norm (no params)      | LayerNorm (affine)    | Simpler, fewer params                           |
| Attention type        | GQA (n_kv_head ≤ n_head) | MHA (all heads)       | KV cache savings at inference                   |
| Activation            | ReLU²                     | GELU                  | Simpler, competitive quality                    |
| Window pattern        | Sliding (S=local, L=full) | Full context all      | Compute efficient (O(n·w) vs O(n²))            |
| Embedding/LM head     | Untied                    | Tied (shared weights) | Different roles → better training               |
| Output projections    | Zero init                 | Random init           | Blocks start as identity, stable early training |
| QK norm               | Yes                       | No                    | Prevents attention logit explosion              |
| Logit softcap         | Yes (15·tanh)             | No                    | Stable cross-entropy early on                   |
| Value embeddings      | Yes (ResFormer-style)     | No                    | Static token identity in V                      |
| Smear (bigram)        | Yes                       | No                    | Cheap local info at input                       |
| Backout               | Yes                       | No                    | Removes low-level features before lm_head       |
| x0 residual           | Yes (per-layer)           | No                    | Preserves token identity through depth          |
| Residual scalars      | Yes (per-layer)           | No                    | Controls residual stream growth                 |
