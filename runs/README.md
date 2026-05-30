# Orchestration Runs (`runs/`)

This folder contains Bash shell scripts (`*.sh`) used to coordinate end-to-end runs. They execute commands from `scripts/` with proper hyperparameters.

## Key Files:
- **`speedrun.sh`**: The official script to replicate the rapid GPT-2 compute capabilities on multiple H100 GPUs.
- **`runcpu.sh`**: An educational/educational script configured for low-memory CPU environments (e.g. Apple Silicon M-series Macs).
- **`miniseries.sh` / `scaling_laws.sh`**: Tools useful for scaling research and sweeping hyperparameters.
