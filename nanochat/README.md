# NanoChat Core Library (`nanochat/`)

This directory contains the fundamental backbone and defining components of the Nanochat language model.

## Key Files:
- **`gpt.py`** ([`nanochat/gpt.py`](./gpt.py)): The main Transformer model (`nn.Module`). Defines self-attention, blocks, and network structure. See the [architecture diagram](./ARCHITECTURE.md) for a visual walkthrough.
- **`engine.py`** ([`nanochat/engine.py`](./engine.py)): Generation logic and caching infrastructure used during inference.
- **`tokenizer.py`** ([`nanochat/tokenizer.py`](./tokenizer.py)): Byte Pair Encoding (BPE) tokenization wrappers.
- **`dataloader.py`** ([`nanochat/dataloader.py`](./dataloader.py)): Shuffling and loading of datasets during training.
- **`optim.py`** ([`nanochat/optim.py`](./optim.py)): AdamW and Muon optimizers.
- **`ARCHITECTURE.md`** ([`nanochat/ARCHITECTURE.md`](./ARCHITECTURE.md)): Full model architecture diagram with dimensions, data flow, and component explanations.
