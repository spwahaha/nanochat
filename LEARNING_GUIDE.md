# Nanochat: A Beginner's to Advanced Learning Guide & Textbook

Welcome to the **nanochat** textbook! This guide is designed to serve as an end-to-end textbook for learning Large Language Models (LLMs) through the lens of a production-grade, highly optimized, yet minimal and hackable codebase.

The primary philosophy of [nanochat](./) is simplicity, speed, and mathematical elegance. Out of the box, it is configured around **one single dial of complexity: `--depth`** (the number of layers). When you adjust depth, all other hyperparameters (width, heads, learning rate schedules, decay horizons, etc.) are automatically calculated to be compute-optimal.

---

## 1. Directory and File Structure

Here is a folder-by-folder structure showing what each file does, its target responsibility, and how it achieves its goal.

```
.
├── LICENSE
├── README.md                           # Project homepage & speedrun leaderboard
├── pyproject.toml                      # uv dependencies, scripts, and extras
├── uv.lock                             # Locked dependencies
├── dev/
│   ├── gen_synthetic_data.py           # Generates identity conversation data
│   ├── repackage_data_reference.py     # reference code for DCLM dataset packaging
│   ├── estimate_gpt3_core.ipynb        # Jupyter notebook estimating parameter scaling
│   └── LEADERBOARD.md                  # Leaderboard info for speedruns
├── runs/
│   ├── speedrun.sh                     # Reference script for the ~$100 H100 pretraining speedrun
│   ├── miniseries.sh                   # Sweat depth sweep to train compute-optimal models
│   ├── scaling_laws.sh                 # Sweep script to analyze loss/FLOP scaling
│   └── runcpu.sh                       # Minimal CPU/MPS verification script
├── nanochat/                           # Core Library
│   ├── __init__.py                     # Package init
│   ├── common.py                       # Distributed setup, types, print utils, device detection
│   ├── gpt.py                          # The core Transformer Model and hyperparameter recipes
│   ├── dataloader.py                   # Pretraining data loading with BOS-aligned Best-Fit Cropping
│   ├── dataset.py                      # Dataset downloader, extractor, and verification
│   ├── optim.py                        # Fused AdamW and Muon / DistMuon optimizers
│   ├── engine.py                       # Autoregressive generation engine with KV cache & tool logic
│   ├── execution.py                    # Tool calling Python executor helpers
│   ├── tokenizer.py                    # Byte Pair Encoding (BPE) wrapper matching GPT-4
│   ├── checkpoint_manager.py           # Clean checkpoint saving/loading for model + optimizer
│   ├── loss_eval.py                    # Validation bits-per-byte (bpb) calculation
│   ├── core_eval.py                    # DCLM-like CORE zero-shot benchmark runner
│   ├── flash_attention.py              # FlashAttention-3 integration and SDPA fallbacks
│   ├── fp8.py                          # Float8 conversion for nn.Linear modules
│   ├── report.py                       # Code to generate local performance reports
│   └── ui.html                         # The static HTML/CSS/JS frontend for the web demo
├── scripts/                            # Main Executables
│   ├── base_train.py                   # Pretraining entry point (Phase 1)
│   ├── base_eval.py                    # Standalone base model evaluator
│   ├── chat_sft.py                     # Supervised Fine-Tuning entry point (Phase 2)
│   ├── chat_rl.py                      # Reinforcement Learning via GRPO entry point (Phase 3)
│   ├── chat_cli.py                     # Terminal-based chat interface
│   ├── chat_web.py                     # Web interface server
│   ├── tok_train.py                    # Tokenizer training script
│   └── tok_eval.py                     # Tokenizer compression evaluator
├── tasks/                              # Task Definitions for Evaluations & SFT
│   ├── common.py                       # Task base classes and mixtures
│   ├── customjson.py                   # Custom task loading from jsonl
│   ├── arc.py                          # ARC Easy/Challenge science QA
│   ├── mmlu.py                         # MMLU multiple choice questions
│   ├── gsm8k.py                        # GSM8k grade school math
│   ├── humaneval.py                    # HumanEval Python code generation
│   └── spellingbee.py                  # Spelling & character counting tasks
└── tests/
    └── test_engine.py                  # Unit tests for the generation engine
```

---

## 2. Core Model Architecture: Mathematical and Code Details

The heart of the repository is in [`nanochat/gpt.py`](./nanochat/gpt.py). The model deviates from standard GPT-2 in several critical ways to maximize throughput, stability, and representational capacity. Below are the key architectural innovations and their mathematics.

### A. Smear (Bigram Mixing)
During pretraining, it is highly beneficial if the model has a very fast, cheap way to capture bigram features (tokens immediately next to each other) without wasting expensive attention steps.
* **Math**:
  Given an embedding sequence $x = [x_0, x_1, \ldots, x_{T-1}]$ where each $x_i \in \mathbb{R}^{d}$, the Smear operation modifies the representations for all positions $p \geq 1$ by blending in a small, gated portion of the predecessor's embedding:
  $$\text{gate}_p = \lambda_{\text{smear}} \cdot \sigma\left(W_{\text{smear}} x_{p, :24}\right)$$
  $$x_p \leftarrow x_p + \text{gate}_p \cdot x_{p-1}$$
  Here, $\sigma$ is the sigmoid function, $\lambda_{\text{smear}}$ is a learnable scalar initialized to $0.0$, and $W_{\text{smear}} \in \mathbb{R}^{1 \times 24}$ is a linear mapping operating on the first $24$ dimensions of the token vector.
* **Code Implementation**:
  ```python
  # In forward() of GPT class:
  gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
  x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
  ```

### B. Rotary Position Embeddings (RoPE)
Instead of adding absolute positional embeddings to the token vectors, nanochat uses RoPE, which encodes relative position by rotating pairs of dimensions in the query and key vectors.
* **Math**:
  For head dimension $d$, we pair up dimensions index $2i$ and $2i+1$. Let $\theta_i = b^{-2i/d}$ where $b = 100,000$ is the base frequency. For a vector $q_t$ at time step $t$, the rotation for a paired dimension is:
  $$\begin{pmatrix} q_{t, 2i}' \\ q_{t, 2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(t\theta_i) & -\sin(t\theta_i) \\ \sin(t\theta_i) & \cos(t\theta_i) \end{pmatrix} \begin{pmatrix} q_{t, 2i} \\ q_{t, 2i+1} \end{pmatrix}$$
* **Code Implementation**:
  ```python
  def apply_rotary_emb(x, cos, sin):
      d = x.shape[3] // 2
      x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
      y1 = x1 * cos + x2 * sin        # rotate pairs of dims
      y2 = x1 * (-sin) + x2 * cos
      return torch.cat([y1, y2], 3)
  ```

### C. QK Normalization (QK Norm)
In deep transformers, the dot-products of queries and keys can grow very large, causing the softmax function to saturate (the "logit explosion" problem).
* **Math**:
  Queries ($Q$) and keys ($K$) are normalized along their head dimension to have unit root-mean-square:
  $$Q' = \text{RMSNorm}(Q) \cdot \gamma$$
  $$K' = \text{RMSNorm}(K) \cdot \gamma$$
  In nanochat, the scale factor $\gamma$ is set to $1.2$.
* **Code Implementation**:
  ```python
  # In CausalSelfAttention:
  q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
  q, k = norm(q), norm(k)  # QK Norm (F.rms_norm)
  q = q * 1.2
  k = k * 1.2
  ```

### D. Sliding Window Attention (SSSL Pattern)
To save compute at long sequence lengths, nanochat uses sliding window attention (inspired by Mistral/Longformer).
* **Details**:
  Instead of attending to all $2048$ historical tokens, certain layers are configured to use a short window (e.g., $768$ tokens), while others use the full long window ($2048$ tokens).
  The window pattern string (configured via `--window-pattern`, default `"SSSL"`) tiles across layers: $3$ short layers, then $1$ long layer.
  * **Short Window (S)**: Memory and compute complexity is $O(T \cdot W_{\text{short}})$, which is linear.
  * **Long Window (L)**: Propagates information globally. The final layer always uses a long window.

### E. Value Embeddings (ResFormer-style)
To prevent the fading of static token identity over deep layers, nanochat introduces value embeddings directly into value projections.
* **Math**:
  Alternating layers contain their own embedding table `value_embeds[i]`. During forward, a static representation of the token index $idx$ is fetched and mixed into the projection value $V$:
  $$\text{gate} = 3 \cdot \sigma\left(W_{\text{gate}} x_{:12}\right)$$
  $$V \leftarrow V + \text{gate} \cdot \text{value\_embed}(idx)$$
* **Code**:
  ```python
  if ve is not None:
      ve = ve.view(B, T, self.n_kv_head, self.head_dim)
      gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head)
      v = v + gate.unsqueeze(-1) * ve
  ```

### F. Residual Blending ($\lambda_{\text{resid}}$ and $\lambda_{x_0}$)
Traditional residual connections can cause "residual stream collapse" where token-specific identity is drowned out by accumulated noise.
* **Math**:
  Before feeding the layer input to attention/MLP block, we mix in the raw normalized token embeddings $x_0$:
  $$x \leftarrow \lambda_{\text{resid}}[l] \cdot x_{\text{in}} + \lambda_{x_0}[l] \cdot x_0$$
  * $\lambda_{\text{resid}}$ decays linearly across layers (e.g., $1.15 \to 1.05$) to reduce variance growth.
  * $\lambda_{x_0}$ decays linearly across layers (e.g., $0.20 \to 0.05$) to inject embedding bias early on.

### G. Backout (Mid-layer subtraction)
* **Math**:
  At the halfway layer (e.g., layer $6$ for a $12$-layer model), we cache the representation $x_{\text{backout}} = x_{L/2}$. Before projecting to logits, we subtract it:
  $$x_{\text{final}} = x_{\text{final}} - \lambda_{\text{backout}} \cdot x_{\text{backout}}$$
  Here, $\lambda_{\text{backout}}$ is a learnable scalar initialized to $0.2$. This forces the final logits to be driven purely by "high-level features" computed in the second half of the model by subtracting lower-level syntax features.

### H. Logit Softcapping
To prevent extreme logit values from causing loss spikes early in training:
$$\text{logits} = 15 \cdot \tanh\left(\frac{\text{logits}}{15}\right)$$
This squashes all logits into the range $[-15, 15]$, acting as an identity mapping for small values while clipping extreme outliers.

---

## 3. The Optimizers: AdamW vs. Muon

One of the defining performance drivers of nanochat is the combination of **AdamW** and **Muon** optimizers inside [`nanochat/optim.py`](./nanochat/optim.py).

| Parameter Group | Optimizer | Tuning Role |
| :--- | :--- | :--- |
| **Embeddings & head** (`wte`, `lm_head`) | **AdamW** | High-dimensional lookup parameters |
| **Scalars & gates** (Lambdas, gate weights) | **AdamW** | Coordinate weights, scalar multipliers |
| **Matrix Weights** (Linear layer weights in blocks) | **Muon** | Large 2D transformations (attention, MLPs) |

### What is Muon?
Muon (Momentum Orthogonalized by Newton-Schulz) is a modern optimization technique that replaces standard gradients with their **nearest orthogonal matrices**.

* **The Core Mathematical Intuition**:
  Standard optimization steps (like SGD or Adam) change both the *direction* and the *scale* of weight matrices. For 2D weight matrices (which act as linear mappings in attention and MLP layers), training is fastest when the weights remain close to orthogonal.
  If a matrix $W$ is orthogonal, it preserves the norm of the vectors it multiplies: $\|W x\| = \|x\|$. This stabilizes activation distributions and gradient propagation.

* **Polar Express Sign Method**:
  Instead of computing the exact Singular Value Decomposition ($W = U S V^T$) to extract the orthogonal components $U V^T$, Muon uses iterative approximations. nanochat implements the **Polar Express Sign Method**, which runs efficiently in `bfloat16` on GPUs:
  For a gradient matrix $X$, we scale it by its norm and repeatedly perform the polynomial update:
  $$A = X^T X \quad (\text{or } X X^T)$$
  $$X \leftarrow a X + X (b A + c A^2)$$
  By performing this iteration $5$ times using precalculated coefficients, $X$ converges to an orthogonal matrix.

* **NorMuon Variance Normalization**:
  After orthogonalization, the update values can have non-uniform variance across neurons (columns). NorMuon tracks the column-wise variance using an EMA of squared values and scales the update back to maintain stable magnitude distributions.

### Distributed Optimization (ZeRO-2 Style)
[`DistMuonAdamW`](./nanochat/optim.py#L299) shards optimizer states to handle multi-GPU scaling without PyTorch's native DDP:
1. **AdamW Sharding**: Large parameters (like `wte`) are sharded using `reduce_scatter` on gradients, updating only the local shard, and then combining via `all_gather`.
2. **Muon Stack and Chunk**: All parameters in a Muon group are stacked into a large tensor. The ranks split the parameters: if there are $K$ matrices and $N$ ranks, each rank owns $K/N$ matrices, performs local updates, and syncs via a single collective `all_gather`.
3. **Async Communication (3-Phase Overlap)**:
   * **Phase 1**: Launch async `reduce_scatter` or `all_reduce` on gradients.
   * **Phase 2**: Wait for reduces, compute update, and launch async `all_gather` for parameters.
   * **Phase 3**: Wait for all gathers to complete and write back to parameters.

---

## 4. Scaling Laws & Automatic Configuration

In [`scripts/base_train.py`](./scripts/base_train.py#L249), nanochat computes the optimal training horizon and hyperparameters dynamically based on depth.

1. **Optimal Tokens**: Given depth $D$, the total parameters excluding scalars is computed. Using Chinchilla-like rules (specifically targeting `--target-param-data-ratio`), the optimal token horizon is calculated:
   $$\text{Target Tokens} = \text{Data:Param Ratio} \times \text{Scaling Params}$$
2. **Optimal Batch Size ($B$)**: Follows the *Power Lines* paper:
   $$B \propto \text{Tokens}^{0.383}$$
   We scale the batch size from a known reference batch size ($2^{19}$ tokens for depth 12):
   $$B_{\text{opt}} = B_{\text{ref}} \times \left(\frac{\text{Target Tokens}}{\text{Reference Tokens}}\right)^{0.383}$$
3. **Learning Rate Scaling**:
   $$LR \propto \sqrt{B}$$
4. **Weight Decay Scaling**:
   $$\text{Weight Decay} = \text{WD}_{\text{ref}} \times \sqrt{\frac{B}{B_{\text{ref}}}} \times \frac{\text{Reference Tokens}}{\text{Target Tokens}}$$

---

## 5. The Three Training Phases: Pretraining, SFT, and RL

Training a conversational reasoning agent involves three distinct phases, each using a unique algorithm, loss function, and data packing strategy.

### Comparison Matrix: Pretraining vs. SFT vs. RL

| Feature | Phase 1: Pretraining (Base Model Train) | Phase 2: Supervised Fine-Tuning (SFT) | Phase 3: Reinforcement Learning (RL / GRPO) |
| :--- | :--- | :--- | :--- |
| **Core Executable** | [`scripts/base_train.py`](./scripts/base_train.py) | [`scripts/chat_sft.py`](./scripts/chat_sft.py) | [`scripts/chat_rl.py`](./scripts/chat_rl.py) |
| **Primary Goal** | General language capability, vocabulary acquisition, syntactic understanding. | Dialogue format adaptation, conversational alignment, basic task skills. | Precise task optimization (e.g. math reasoning), search optimization. |
| **Dataset Source** | Raw unstructured web text shards (e.g. DCLM/ClimbMix). | Curated conversation lists (`User` and `Assistant` roles). | Input questions/prompts paired with programmatic reward functions (e.g. GSM8K). |
| **Loss Calculation** | Autoregressive cross-entropy on **every** token in the sequence. | Autoregressive cross-entropy **only on Assistant response tokens**. | Policy Gradient (REINFORCE) weighted by group-relative advantages. |
| **Data Packing Strategy** | **Best-Fit Cropping**: Packs whole files and crops at boundary; no padding tokens. | **Best-Fit-Pad Packing**: Packs whole dialogs, pads rest with BOS; never crops messages. | **Rollout Batching**: Generates completions in groups, pads to match maximum rollout length. |
| **Loss Masking** | None (targets $\ge 0$ everywhere). | Prompts, tools, and padding tokens mapped to `ignore_index = -1`. | Prompts and forced python execution tokens mapped to `ignore_index = -1`. |
| **Weight Decay ($\lambda$)** | Scaled decaying value ($\lambda \to 0$ over cosine warmdown). | Reinitialized and kept at **zero** to avoid parameter decay. | Set to **zero** (policy update focuses purely on reinforcement paths). |
| **Loop Mode** | On-device forward/backward over pre-tokenized file streams. | On-device forward/backward over padded/tokenized multi-turn dialogues. | **Online Rollout Loop**: Generates candidate responses on the fly, scores them, and optimizes. |

---

### Phase 1: Pretraining Details ([`scripts/base_train.py`](./scripts/base_train.py))

* **Goal**: Provide the model with broad-world knowledge. By reading a massive portion of the web, the model builds a rich semantic representation of words, syntax, logic, and general facts.
* **Loss Function**: For sequence length $T$ and predictions $\hat{y}_t = P(x_t \mid x_{<t})$:
  $$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{<t})$$
* **Dataloader Mechanics (BOS-Aligned Best-Fit)**:
  Implemented in [`tokenizing_distributed_data_loader_bos_bestfit`](./nanochat/dataloader.py#L74).
  * Every sequence block of length $T$ starts with a `<|bos|>` token.
  * The loader reads a buffer of raw texts, fits as many entire documents as possible, and crops only the last document to fill the remaining capacity.
  * This guarantees **100% token utilization** (no padding) while ensuring documents have proper start anchors.

---

### Phase 2: Supervised Fine-Tuning (SFT) Details ([`scripts/chat_sft.py`](./scripts/chat_sft.py))

* **Goal**: Format the base model into a helpful, non-evasive assistant. It teaches the model to recognize conversational boundaries, stop generating when its turn is over, and format its responses.
* **Data Packing (Best-Fit-Pad)**:
  SFT dialogues must never be cropped (cutting a dialogue mid-response ruins conversational logic).
  * Conversations are packed using a best-fit algorithm.
  * If the remaining space in a row of length $T$ cannot fit another conversation, it is filled with padding tokens (`<|bos|>`).
* **Loss Masking**:
  We want to optimize what the model says, not what the user says. During SFT training, the full dialogue (including user prompts and assistant responses) is fed into the model. However, we only compute loss on the tokens belonging to the Assistant's role (the target output), masking out the user's prompts, special format tags, and padding:
  $$\mathcal{L}_{\text{SFT}} = -\frac{1}{\sum_t M_t} \sum_{t=1}^{T} M_t \log P(x_t \mid x_{<t})$$
  Where $M_t = 1$ if $x_t$ is a token belonging to the Assistant's response in the dataset, and $0$ otherwise (mapped to target index `-1` in PyTorch's `cross_entropy`).


---

### Phase 3: Reinforcement Learning (RL) Details ([`scripts/chat_rl.py`](./scripts/chat_rl.py))

* **Goal**: Maximize performance on complex target tasks (like math or coding) that require multi-step reasoning. Instead of learning to mimic demonstrations, the model learns by exploring paths and receiving reward signals.
* **GRPO-style Policy Gradient Math**:
  Instead of updating the policy with a separate critic network (which requires a secondary model), nanochat uses a variant of **Group Relative Policy Optimization (GRPO)** which is mathematically close to REINFORCE:
  1. For a question prompt, sample $N$ rollouts (e.g. $N=16$).
  2. Programmatically evaluate the rewards $R_1, \ldots, R_N \in \{0, 1\}$ (1.0 for correct final answer, 0.0 otherwise).
  3. Compute the **relative advantage** for each rollout $i$ by subtracting the group average reward:
     $$A_i = R_i - \mu_R \quad \text{where} \quad \mu_R = \frac{1}{N}\sum_{j=1}^{N} R_j$$
  4. The loss is the negative policy gradient:
     $$\mathcal{L}_{\text{PG}} = -\frac{1}{T_{\text{valid}}} \sum_{t \in \text{valid}} \log P(\text{token}_t \mid \text{history}_{<t}) \cdot A_i$$
     Only tokens generated by the assistant during its answer trace (excluding prompt and forced tools outputs) participate in $\mathcal{L}_{\text{PG}}$.
     *If a completion performs better than the average of the group ($A_i > 0$), we increase the log-probability of the tokens that generated it. If it performs worse ($A_i < 0$), we suppress them.*

---

## 6. The Generation Engine: KV Cache and Tool State Machine

Generating tokens autoregressively is highly resource-intensive if done naively. [`nanochat/engine.py`](./nanochat/engine.py) implements a fast inference loops utilizing a key-value (KV) cache and coordinates tool execution.

### KV Cache Mechanics
During decoding, the model generates one token at a time. To predict token $t+1$, it needs to compute the attention queries, keys, and values.
* **Problem**: Re-evaluating the full sequence $1 \ldots t$ at every step is $O(T^2)$ and wastes redundant computations since the past keys ($K$) and values ($V$) do not change.
* **Solution**: The [`KVCache`](./nanochat/engine.py#L82) allocates a static tensor of shape `(n_layers, B, T, H, D)`.
  1. During **prefill**, the prompt tokens are evaluated in one forward pass, and their $K, V$ values are written to the cache.
  2. During **decode**, only the single newly generated token is passed through the network. Its $K, V$ projections are written in-place, and attention is computed against the cached history.

### Tool Call State Machine
The generation loop acts as a state machine that handles Python execution to solve math problems.

```
       ┌───────────────────────────────┐
       │   Normal text generation      │
       └──────────────┬────────────────┘
                      │
           Senses <|python_start|>
                      ▼
       ┌───────────────────────────────┐
       │   Buffering python expression  │
       └──────────────┬────────────────┘
                      │
            Senses <|python_end|>
                      ▼
       ┌───────────────────────────────┐
       │  Run python code in Sandbox   │
       └──────────────┬────────────────┘
                      │
           Generates output string
                      ▼
       ┌───────────────────────────────┐
       │ Force inject:                 │
       │ <|output_start|> + output     │
       │   + <|output_end|>            │
       └───────────────────────────────┘
```

1. **Detection**: If the engine samples `<|python_start|>`, it sets `in_python_block = True` and buffers subsequent tokens.
2. **Sandboxed Evaluation**: On `<|python_end|>`, the engine extracts the string representation of the buffered tokens and passes them to `use_calculator()`. This safe utility restricts Python calls to mathematical operators and string operations like `.count()` (e.g. to solve counting challenges).
3. **Forced Injection**: The evaluation result is tokenized, wrapped in `<|output_start|>` and `<|output_end|>` boundaries, and appended to a `forced_tokens` deque. During the next few generation loops, the engine ignores model logits and injects these tokens directly.
