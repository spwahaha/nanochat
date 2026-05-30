# NanoChat Core Library (`nanochat/`)

This directory contains the fundamental backbone and defining components of the Nanochat language model.

## Key Files:
- **`gpt.py`**: The main Transformer model (`nn.Module`). Defines self-attention, blocks, and network structure.
- **`engine.py`**: Generation logic and caching infrastructure used during inference.
- **`tokenizer.py`**: Byte Pair Encoding (BPE) tokenization wrappers.
- **`dataloader.py`**: Shuffling and loading of datasets during training.
- **`optim.py`**: AdamW and Muon optimizers.
