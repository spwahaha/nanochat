# Nanochat: A Beginner's Learning Guide

Welcome to the world of Large Language Models! `nanochat` is actually one of the best possible repositories to learn from because its primary design goal is simplicity and hackability.

If you are a complete beginner, diving into a full deep learning repository can be overwhelming. Let’s break it down into a logical reading order. 

Here is your step-by-step curriculum for learning the `nanochat` codebase:

### 1. The Core Architecture (Start Here)
Before you look at training loops or servers, you need to understand the mathematical brain of the model.
- **[`nanochat/gpt.py`](./nanochat/gpt.py)**: This is the most important file in the entire repository. This contains the `nn.Module` definition for the Transformer model. 
  - *What to look for:* Read the `CausalSelfAttention` class and the `Block` class. See how they stack together inside the `GPT` class. Trace the `forward()` pass to see how input tokens turn into output predictions.

### 2. Tokenization (How the Model Reads)
LLMs don't read English text; they read arrays of integers ("tokens"). 
- **[`nanochat/tokenizer.py`](./nanochat/tokenizer.py)**: This file wraps the BPE (Byte Pair Encoding) logic.
  - *What to look for:* Understand the `encode()` function (which turns strings into lists of numbers) and the `decode()` function (which turns numbers back into strings). 
- **[`nanochat/dataloader.py`](./nanochat/dataloader.py)**: See how raw text from datasets is loaded, shuffled, and handed to the model in "batches" during training.

### 3. Model Inference (How the Model Thinks)
Once a model is trained, how does it actually generate text?
- **[`nanochat/engine.py`](./nanochat/engine.py)**: This file contains the logic for generating text (inference). 
  - *What to look for:* Look at the KV Cache setup. Generating one token at a time requires keeping track of past tokens quickly. See how `generate()` is implemented as a loop taking the most probable next token and feeding it back into the model.

### 4. The Training Loops (How the Model Learns)
Once you understand the model structure and how it generates text, see how it is trained.
- **[`scripts/base_train.py`](./scripts/base_train.py)**: This is the script used for the main pretraining phase.
  - *What to look for:* Look for the main `while` loop. You will see the standard PyTorch lifecycle: grabbing a batch of data, calculating the loss (`model(X, Y)`), computing gradients (`loss.backward()`), and taking an optimization step (`optimizer.step()`).
- **[`nanochat/optim.py`](./nanochat/optim.py)**: Here you will find the Optimizer logic (AdamW). Don't worry too much about the heavy math here, just know this is what updates the weights.

### 5. Interacting with the Model
After all that hard work, how do users actually talk to it?
- **[`scripts/chat_cli.py`](./scripts/chat_cli.py)**: A very straightforward script that loads the model weights and lets you type in a terminal to talk to it.
- **[`scripts/chat_web.py`](./scripts/chat_web.py)** & **[`nanochat/ui.html`](./nanochat/ui.html)**: A lightweight server and frontend combination that gives you the visual ChatGPT-like interface.

### 💡 General Tips for Beginners:
1. **Insert Prints:** The easiest way to learn is to insert `print(x.shape)` throughout [`nanochat/gpt.py`](./nanochat/gpt.py) during your CPU run. Seeing how the tensor shapes transform from `(batch, sequence_length)` to `(batch, sequence_length, embedding_dim)` will give you that "Aha!" moment.
2. **Follow the Data:** Instead of reading top-to-bottom, pick a script like [`base_train.py`](./scripts/base_train.py) and strictly follow what happens to the variables `x` and `y` from the moment they are created until the `loss` is calculated.
3. **Don't Get Bogged Down:** Skip past things like Distributed Data Parallel (`DDP`) logic, mixed-precision `dtype` casting, or Multi-GPU orchestration at first. Focus solely on the pure standard PyTorch components.
