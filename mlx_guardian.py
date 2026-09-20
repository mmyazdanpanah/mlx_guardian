#!/usr/bin/env python3
"""MLX Guardian: a small, safe TUI supervisor for MLX training on macOS."""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

APP_VERSION = "1.0.0"
HISTORY = 36
DEFAULT_LOG_DIR = Path.home() / ".mlx-guardian"
ITER_RE = re.compile(
    r"\bIter\s+(?P<step>\d+)\s*:\s*"
    r"Train loss\s+(?P<loss>[-+0-9.eE]+).*?"
    r"Learning Rate\s+(?P<lr>[-+0-9.eE]+).*?"
    r"It/sec\s+(?P<itsec>[-+0-9.eE]+).*?"
    r"Tokens/sec\s+(?P<toksec>[-+0-9.eE]+).*?"
    r"Trained Tokens\s+(?P<tokens>\d+).*?"
    r"Peak mem\s+(?P<peakmem>[-+0-9.eE]+)\s+GB",
    re.I,
)
CHECKPOINT_RE = re.compile(r"(?<!\d)(\d{3,})(?!\d)")


@dataclass
class Metrics:
    step: int | None = None
    loss: float | None = None
    learning_rate: float | None = None
    it_per_sec: float | None = None
    tokens_per_sec: float | None = None
    trained_tokens: int | None = None
    peak_mem_gb: float | None = None
    last_step_at: float | None = None

    @property
    def step_rate(self) -> float | None:
        return self.it_per_sec

    def eta(self, total_steps: int | None) -> str:
        if not total_steps or self.step is None:
            return "—"
        rate = self.step_rate
        if not rate or rate <= 0:
            return "—"
        return format_duration(max(0, total_steps - self.step) / rate)


@dataclass
class MonitorState:
    metrics: Metrics = field(default_factory=Metrics)
    loss_history: deque = field(default_factory=lambda: deque(maxlen=HISTORY))
    cpu_history: deque = field(default_factory=lambda: deque(maxlen=HISTORY))
    swap_history: deque = field(default_factory=lambda: deque(maxlen=HISTORY))
    last_output: str = ""
    latest_checkpoint: int | None = None
    returncode: int | None = None
    mode: str = "passive"
    log_path: Path | None = None


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds:02d}s"


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def process_info(pid: int) -> dict[str, str] | None:
    out = run(["ps", "-p", str(pid), "-o", "etime=,pcpu=,pmem=,state="])
    if not out:
        return None
    parts = out.split()
    if len(parts) < 4:
        return None
    return {"runtime": parts[0], "cpu": parts[1], "mem": parts[2], "state": parts[3]}


def memory_info() -> tuple[str, str]:
    raw = run(["memory_pressure"])
    free = compressed = "—"
    for line in raw.splitlines():
        low = line.lower()
        if "free" in low and ":" in line:
            free = line.split(":", 1)[1].strip()
        if "compressed" in low and ":" in line:
            compressed = line.split(":", 1)[1].strip()
    return free, compressed


def swap_pages() -> int:
    raw = run(["vm_stat"])
    for line in raw.splitlines():
        if "Pages swapped out" in line:
            try:
                return int(line.split(":")[-1].strip().rstrip("."))
            except ValueError:
                return 0
    return 0


def thermal_status() -> str:
    raw = run(["pmset", "-g", "therm"])
    return "NORMAL" if "No thermal warning level has been recorded" in raw else "WARNING"


def disk_free() -> str:
    raw = run(["df", "-h", str(Path.home())])
    lines = raw.splitlines()
    if len(lines) > 1:
        parts = lines[-1].split()
        if len(parts) >= 4:
            return parts[3]
    return "—"


def parse_metrics(line: str) -> dict | None:
    match = ITER_RE.search(line)
    if not match:
        return None
    d = match.groupdict()
    return {
        "step": int(d["step"]),
        "loss": float(d["loss"]),
        "learning_rate": float(d["lr"]),
        "it_per_sec": float(d["itsec"]),
        "tokens_per_sec": float(d["toksec"]),
        "trained_tokens": int(d["tokens"]),
        "peak_mem_gb": float(d["peakmem"]),
    }


def checkpoint_step(path: Path) -> int | None:
    matches = CHECKPOINT_RE.findall(path.name)
    return int(matches[-1]) if matches else None


def latest_checkpoint(directory: Path | None) -> int | None:
    if not directory or not directory.exists():
        return None
    best = None
    try:
        for item in directory.iterdir():
            step = checkpoint_step(item)
            if step is not None:
                best = step if best is None else max(best, step)
    except OSError:
        pass
    return best


def spark(values: deque | list, width: int = 30) -> str:
    if not values:
        return "—"
    vals = list(values)[-width:]
    chars = "▁▂▃▄▅▆▇█"
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return chars[3] * len(vals)
    return "".join(chars[min(7, int((v - lo) / (hi - lo) * 7))] for v in vals)


def pctbar(value: float, width: int = 18) -> str:
    x = max(0.0, min(100.0, value))
    n = round(x / 100 * width)
    return "█" * n + "░" * (width - n)


def progress_text(step: int | None, total: int | None) -> str:
    if step is None:
        return "—"
    if not total:
        return f"{step:,}"
    pct = max(0.0, min(100.0, step / total * 100))
    return f"{step:,} / {total:,}  ({pct:.1f}%)"


def compute_total_steps(args: argparse.Namespace) -> int | None:
    if args.total_steps:
        return args.total_steps
    if args.train_samples and args.batch_size and args.epochs:
        effective_batch = args.batch_size * max(1, args.grad_accumulation)
        steps_per_epoch = (args.train_samples + effective_batch - 1) // effective_batch
        return steps_per_epoch * args.epochs
    return None


def build_command(args: argparse.Namespace) -> list[str]:
    if not args.script:
        raise SystemExit("No trainer script. Use --script PATH or set MLX_GUARDIAN_TRAINER.")
    python = args.python or os.environ.get("MLX_GUARDIAN_PYTHON", sys.executable)
    script = Path(args.script).expanduser()
    if not script.exists():
        raise SystemExit(f"Trainer script not found: {script}")
    return [python, "-u", str(script), *args.trainer_args]


def start_training(args: argparse.Namespace) -> tuple[subprocess.Popen, MonitorState]:
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = DEFAULT_LOG_DIR / f"training-{timestamp}.log"
    cwd = Path(
        args.cwd
        or os.environ.get("MLX_GUARDIAN_CWD")
        or Path(args.script).expanduser().parent
    ).expanduser()
    if not cwd.exists():
        raise SystemExit(f"Working directory not found: {cwd}")

    proc = subprocess.Popen(
        build_command(args),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    return proc, MonitorState(mode="supervisor", log_path=log_path)


def consume_output(proc: subprocess.Popen, state: MonitorState) -> None:
    assert proc.stdout is not None
    log = state.log_path.open("a", encoding="utf-8") if state.log_path else open(os.devnull, "w")
    try:
        for raw in iter(proc.stdout.readline, ""):
            if not raw:
                break
            line = raw.rstrip()
            state.last_output = line
            event = parse_metrics(line)
            if event:
                m = state.metrics
                now = time.monotonic()
                if m.step is None or event["step"] != m.step:
                    m.last_step_at = now
                    m.step = event["step"]
                    m.loss = event["loss"]
                    m.learning_rate = event["learning_rate"]
                    m.it_per_sec = event["it_per_sec"]
                    m.tokens_per_sec = event["tokens_per_sec"]
                    m.trained_tokens = event["trained_tokens"]
                    m.peak_mem_gb = event["peak_mem_gb"]
                    state.loss_history.append(m.loss)
            log.write(line + "\n")
            log.flush()
    finally:
        log.close()
        proc.stdout.close()


def render(pid: int, state: MonitorState, total_steps: int | None, checkpoint_dir: Path | None):
    p = process_info(pid)
    if p is None:
        msg = (
            "[yellow]Training process exited or disappeared.[/yellow]"
            if state.mode == "supervisor"
            else "[red]Training process not found.[/red]"
        )
        return Panel(msg, title="MLX GUARDIAN")

    cpu = float(p["cpu"])
    free, compressed = memory_info()
    swap = swap_pages()
    thermal = thermal_status()
    state.cpu_history.append(cpu)
    state.swap_history.append(swap)

    if checkpoint_dir:
        state.latest_checkpoint = latest_checkpoint(checkpoint_dir)
    m = state.metrics

    train = Table.grid(padding=(0, 2))
    train.add_row("PID", str(pid))
    train.add_row("CPU", f"{cpu:.1f}%  {pctbar(cpu)}")
    train.add_row("PROCESS RAM", f"{p['mem']}%")
    train.add_row("RUNTIME", p["runtime"])
    train.add_row("STATE", p["state"])
    train.add_row("STEP", progress_text(m.step, total_steps))
    if m.loss is not None:
        train.add_row("LOSS", f"{m.loss:.5f}")
        train.add_row("LR", f"{m.learning_rate:.3e}")
        train.add_row("ITER/S", f"{m.it_per_sec:.3f}")
        train.add_row("TOKENS/S", f"{m.tokens_per_sec:.2f}")
        train.add_row("TRAINED TOKENS", f"{m.trained_tokens:,}")
        train.add_row("PEAK MEM", f"{m.peak_mem_gb:.3f} GB")
        train.add_row("ETA", m.eta(total_steps))
    if state.latest_checkpoint is not None:
        train.add_row("CHECKPOINT", f"{state.latest_checkpoint:,}")
    train.add_row("LOSS TREND", spark(state.loss_history))

    apple = Table.grid(padding=(0, 2))
    apple.add_row("CPU", f"{cpu:.1f}%")
    apple.add_row("MEMORY", free)
    apple.add_row("COMPRESSED", compressed)
    apple.add_row("SWAP PAGES", str(swap))
    apple.add_row(
        "THERMAL",
        "[green]🟢 NORMAL[/green]" if thermal == "NORMAL" else "[red]🔴 WARNING[/red]",
    )
    apple.add_row("DISK FREE", disk_free())
    apple.add_row("CPU ACTIVITY", spark(state.cpu_history))
    apple.add_row("SWAP TREND", spark(state.swap_history))

    signals = Table.grid()
    signals.add_row("CPU", Text(spark(state.cpu_history)))
    signals.add_row("LOSS", Text(spark(state.loss_history)))
    signals.add_row("SWAP", Text(spark(state.swap_history)))

    if thermal != "NORMAL":
        status = "[red]🔴 Thermal warning reported.[/red]"
    elif swap:
        status = "[yellow]🟡 Swap is active; memory pressure may be elevated.[/yellow]"
    elif state.mode == "supervisor":
        status = "[green]🟢 Guardian is supervising this training process.[/green]"
    else:
        status = "[green]🟢 Guardian is monitoring passively.[/green]"

    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(Panel(train, title="TRAINING"), Panel(apple, title="APPLE SILICON"))
    grid.add_row(Panel(signals, title="LIVE SIGNALS"))
    if state.log_path:
        grid.add_row(Panel(f"[dim]{state.log_path}[/dim]", title="LOG"))
    grid.add_row(Panel(status, title="GUARDIAN"))
    title = Text.from_markup(
        f"[bold cyan]🧠 MLX GUARDIAN[/bold cyan]   [green]● {APP_VERSION}[/green]"
    )
    return Group(title, grid)


def supervise(args: argparse.Namespace) -> int:
    total_steps = compute_total_steps(args)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser() if args.checkpoint_dir else None
    proc, state = start_training(args)
    reader = threading.Thread(target=consume_output, args=(proc, state), daemon=True)
    reader.start()

    def stop(_signum=None, _frame=None):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    console = Console()

    with Live(
        console=console,
        refresh_per_second=2,
        screen=True,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as live:
        while proc.poll() is None:
            live.update(render(proc.pid, state, total_steps, checkpoint_dir))
            time.sleep(0.5)
        state.returncode = proc.returncode
        live.update(render(proc.pid, state, total_steps, checkpoint_dir))

    reader.join(timeout=1)
    print()
    if state.returncode == 0:
        print("🟢 Training finished successfully.")
    else:
        print(f"🔴 Training exited with code {state.returncode}.")
    if state.log_path:
        print(f"📄 Log: {state.log_path}")
    return state.returncode or 0


def monitor(pid: int) -> int:
    state = MonitorState(mode="passive")
    console = Console()
    with Live(
        console=console,
        refresh_per_second=2,
        screen=True,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as live:
        while True:
            live.update(render(pid, state, None, None))
            if process_info(pid) is None:
                break
            time.sleep(0.5)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlx-guardian",
        description="Safe TUI supervisor and monitor for MLX training on macOS.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Launch and supervise an MLX trainer.")
    train.add_argument("--script", default=os.environ.get("MLX_GUARDIAN_TRAINER"))
    train.add_argument("--python", default=None, help="Python executable.")
    train.add_argument("--cwd", default=None, help="Trainer working directory.")
    train.add_argument("--total-steps", type=int, default=None)
    train.add_argument("--train-samples", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--grad-accumulation", type=int, default=1)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--checkpoint-dir", default=None)
    train.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the trainer after --.",
    )

    mon = sub.add_parser("monitor", help="Passively monitor an existing PID.")
    mon.add_argument("pid", type=int)

    return p


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        sys.argv.insert(1, "monitor")
    args = parser().parse_args()
    if args.command == "train":
        if args.trainer_args and args.trainer_args[0] == "--":
            args.trainer_args = args.trainer_args[1:]
        return supervise(args)
    return monitor(args.pid)


if __name__ == "__main__":
    raise SystemExit(main())
