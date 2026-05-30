# Orchestration Runs (`runs/`)

This folder contains Bash shell scripts (`*.sh`) used to coordinate end-to-end runs. They execute commands from `scripts/` with proper hyperparameters.

## Key Files:
- **`speedrun.sh`** ([`runs/speedrun.sh`](./speedrun.sh)): The official script to replicate the rapid GPT-2 compute capabilities on multiple H100 GPUs.
- **`runcpu.sh`** ([`runs/runcpu.sh`](./runcpu.sh)): An educational script configured for low-memory CPU environments (e.g. Apple Silicon M-series Macs).
- **`miniseries.sh`** ([`runs/miniseries.sh`](./miniseries.sh)) / **`scaling_laws.sh`** ([`runs/scaling_laws.sh`](./scaling_laws.sh)): Tools useful for scaling research and sweeping hyperparameters.
