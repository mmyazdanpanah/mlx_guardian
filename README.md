# MLX Guardian

A tiny, passive terminal dashboard for monitoring MLX training jobs and Apple Silicon health.

## Features

- Live Rich TUI
- Training PID, CPU, process RAM and runtime
- Step / epoch / loss / learning rate / throughput / ETA when telemetry is available
- Loss, CPU and swap sparklines
- macOS memory, compression, swap, thermal status and free disk
- No daemon, database, server or network service
- Passive by default: Guardian does not start, stop, pause or modify your training job

## Quick start

```bash
git clone https://github.com/mmyazdanpanah/mlx_guardian.git
cd mlx_guardian
python3 -m pip install rich
./mlx_guardian.py <PID>
```

Optional telemetry file:

```
~/.mlx-guardian/training.jsonl
```

Each line is a JSON object. A trainer can emit fields such as `step`, `max_steps`, `epoch`, `loss`, `learning_rate`, `throughput`, and `eta`.

## Example

```text
🧠 MLX GUARDIAN   ● TRAINING

TRAINING                    APPLE SILICON
PID        1412             CPU          24.0%
CPU        24.0%            MEMORY       512 MB free
PROCESS RAM 2.0%            COMPRESSED    1.4 GB
STEP       120 / 500        SWAP         0
LOSS       1.23456          THERMAL      🟢 NORMAL
THROUGHPUT 2.1 step/s       DISK FREE    12 GB
```

## Philosophy

MLX Guardian is intentionally small. It observes an existing training process instead of becoming part of the training stack.

## License

MIT
