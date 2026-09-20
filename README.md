# MLX Guardian

A tiny, safe terminal guardian for MLX training on Apple Silicon.

MLX Guardian can supervise a training process it launches or passively monitor an existing PID. It captures native MLX training output, shows progress, loss, throughput and ETA when available, watches macOS memory, swap, thermal and disk state, and keeps a plain-text run log.

## Why this exists

Long MLX runs need visibility. Process-level CPU and RAM information alone cannot answer the useful questions:

- Is the trainer actually still running?
- What iteration is it on?
- How much is left?
- Is loss moving?
- Is swap or thermal pressure appearing?
- What happened if the terminal disappeared?

The first release focuses on those questions without a daemon, database, network service, MLX dependency, or training-framework lock-in.

## Features

- Rich live TUI
- Supervisor mode: Guardian launches the trainer
- Passive PID monitoring
- Captures stdout and stderr into timestamped logs
- Parses native MLX-style iteration output
- Step, loss, learning rate, iteration speed, tokens/sec and peak memory
- Optional total-step calculation from samples, batch size, accumulation and epochs
- Optional checkpoint discovery from numbered adapter/checkpoint files
- ETA from observed training speed
- CPU, process RAM, macOS memory/compression, swap, thermal status and disk free space
- Safe process-group shutdown on Ctrl-C/SIGTERM
- Unbuffered trainer launch for timely telemetry
- No MLX installation required by Guardian itself
- No network service and no daemon

## Install

Python 3.11+ is required.

~~~bash
git clone https://github.com/mmyazdanpanah/mlx_guardian.git
cd mlx_guardian
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

Run directly:

~~~bash
./mlx_guardian.py --version
~~~

Or install the command:

~~~bash
python -m pip install .
mlx-guardian --version
~~~

## Supervise a training run

~~~bash
mlx-guardian train \
  --script ~/path/to/train.py \
  --cwd ~/path/to/project
~~~

Additional trainer arguments go after the separator:

~~~bash
mlx-guardian train \
  --script ~/path/to/train.py \
  --cwd ~/path/to/project \
  -- --config config.yaml
~~~

For exact progress and ETA, give Guardian the total steps:

~~~bash
mlx-guardian train \
  --script ~/path/to/train.py \
  --cwd ~/path/to/project \
  --total-steps 22500
~~~

For epoch-based jobs, Guardian can calculate the target:

~~~bash
mlx-guardian train \
  --script ~/path/to/train.py \
  --cwd ~/path/to/project \
  --train-samples 90000 \
  --batch-size 2 \
  --grad-accumulation 4 \
  --epochs 2
~~~

This calculates:

ceil(samples / (batch x accumulation)) x epochs

For checkpoint-aware monitoring:

~~~bash
mlx-guardian train \
  --script ~/path/to/train.py \
  --cwd ~/path/to/project \
  --total-steps 22500 \
  --checkpoint-dir ~/path/to/adapters
~~~

Numbered files such as 00004800_adapters.safetensors are recognized automatically.

## Personal setup

Avoid putting machine-specific paths in the repository. Environment variables can provide trainer defaults:

~~~bash
export MLX_GUARDIAN_TRAINER="$HOME/path/to/train.py"
export MLX_GUARDIAN_PYTHON="$HOME/path/to/venv/bin/python3"
export MLX_GUARDIAN_CWD="$HOME/path/to/project"
~~~

Then:

~~~bash
mlx-guardian train --total-steps 22500
~~~

## Passive monitoring

Monitor an already-running process:

~~~bash
mlx-guardian monitor 1412
~~~

Passive mode does not start, stop, pause, or modify the process.

## Training telemetry

Guardian recognizes the native MLX-style line used by many trainers, for example:

~~~text
Iter 4800: Train loss 1.234, Learning Rate 2.000e-04, It/sec 0.658, Tokens/sec 78.322, Trained Tokens 385, Peak mem 1.518 GB
~~~

When a trainer emits that information, Guardian displays it directly.

Supervisor mode also saves captured output to:

~~~text
~/.mlx-guardian/training-YYYYMMDD-HHMMSS.log
~~~

## Safety model

Supervisor mode starts the trainer in a dedicated process group. Ctrl-C first sends SIGTERM to that group and waits up to five seconds; if the group does not exit, Guardian sends SIGKILL.

This prevents a launcher from leaving a hidden worker behind.

Passive mode is intentionally non-invasive.

## macOS scope

The Apple Silicon panel uses native macOS commands such as ps, memory_pressure, vm_stat, pmset, and df. Guardian does not require root privileges.

GPU utilization is deliberately not presented as a fabricated percentage. MLX/Metal activity can be observed indirectly, but a reliable cross-machine GPU utilization metric is outside this lightweight first release.

## Design principles

**Small.** One Python module plus Rich.

**Local.** No server, database, cloud account, or network telemetry.

**Safe.** Passive monitoring never controls the training job. Supervisor mode owns only the process group it created.

**Framework-light.** Guardian does not depend on MLX, mlx-lm, mlx-tune, Transformers, or a particular trainer API.

**Observable.** If a trainer prints a useful metric, Guardian should make it visible without requiring an integration SDK.

## License

MIT.
