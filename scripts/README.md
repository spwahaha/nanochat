# Scripts (`scripts/`)

This directory holds the executable entry points for the application. You call these directly using `python -m scripts.<script_name>`.

## Key Files:
- **`base_train.py`** ([`scripts/base_train.py`](./base_train.py)) / **`base_eval.py`** ([`scripts/base_eval.py`](./base_eval.py)): Pre-training the neural network from scratch and assessing base performance.
- **`chat_sft.py`** ([`scripts/chat_sft.py`](./chat_sft.py)) / **`chat_rl.py`** ([`scripts/chat_rl.py`](./chat_rl.py)): Fine-tuning the base model on chat interactions (SFT) and alignment (RL).
- **`chat_cli.py`** ([`scripts/chat_cli.py`](./chat_cli.py)) / **`chat_web.py`** ([`scripts/chat_web.py`](./chat_web.py)): Interfaces to converse with your trained models either via terminal or a local web server.
