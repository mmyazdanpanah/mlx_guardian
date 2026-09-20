#!/usr/bin/env python3
"""MLX Guardian — lightweight TUI for training + Apple Silicon health."""

import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    from rich import box
except ImportError:
    box = None

GUARDIAN_DIR = Path.home() / ".mlx-guardian"
DEFAULT_TELEMETRY = GUARDIAN_DIR / "training.jsonl"
HISTORY = 24


def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def process_info(pid):
    out = run(["ps", "-p", str(pid), "-o", "etime=,pcpu=,pmem=,state="])
    if not out:
        return None
    p = out.split()
    return {"runtime": p[0], "cpu": p[1], "mem": p[2], "state": p[3]}


def memory_info():
    raw = run(["memory_pressure"])
    free = compressed = "—"
    for line in raw.splitlines():
        low = line.lower()
        if "free" in low and ":" in line:
            free = line.split(":", 1)[1].strip()
        if "compressed" in low and ":" in line:
            compressed = line.split(":", 1)[1].strip()
    return free, compressed


def swap_pages():
    raw = run(["vm_stat"])
    page = 0
    for line in raw.splitlines():
        if "page size" in line.lower():
            try:
                page = int(line.split(":")[-1].strip())
            except Exception:
                pass
    if not page:
        page = 4096
    for line in raw.splitlines():
        if "Pages swapped out" in line:
            try:
                return int(line.split(":")[-1].strip().rstrip("."))
            except Exception:
                return 0
    return 0


def thermal():
    raw = run(["pmset", "-g", "therm"])
    return ("NORMAL" if "No thermal warning level has been recorded" in raw
            else "WARNING")


def disk_free():
    raw = run(["df", "-h", str(Path.home())])
    lines = raw.splitlines()
    if len(lines) > 1:
        p = lines[-1].split()
        if len(p) >= 4:
            return p[3]
    return "—"


def read_telemetry(path, last_event):
    if not path.exists():
        return last_event
    try:
        with path.open() as f:
            lines = f.readlines()[-20:]
        for line in lines:
            try:
                last_event = json.loads(line)
            except Exception:
                pass
    except Exception:
        pass
    return last_event


def spark(values, width=24):
    if not values:
        return "—"
    vals = list(values)[-width:]
    chars = "▁▂▃▄▅▆▇█"
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return chars[3] * len(vals)
    return "".join(chars[min(7, int((v-lo)/(hi-lo)*7))] for v in vals)


def pctbar(value, width=18):
    try:
        x = max(0, min(100, float(value)))
    except Exception:
        x = 0
    n = round(x / 100 * width)
    return "█" * n + "░" * (width - n)


def render(pid, event, cpu_hist, mem_hist, swap_hist, loss_hist):
    p = process_info(pid)
    if not p:
        return Panel("[red]Training process not found[/red]", title="MLX GUARDIAN")
    cpu = float(p["cpu"])
    free, compressed = memory_info()
    sw = swap_pages()
    cpu_hist.append(cpu)
    mem_hist.append(len(str(compressed)) if compressed != "—" else 0)
    swap_hist.append(sw)

    loss = event.get("loss", "—") if event else "—"
    if isinstance(loss, (int, float)):
        loss_hist.append(float(loss))

    title = Text.from_markup("[bold cyan]🧠 MLX GUARDIAN[/bold cyan]   [green]● TRAINING[/green]")
    train = Table.grid(padding=(0, 2))
    train.add_row("PID", str(pid))
    train.add_row("CPU", f"{cpu:.1f}%  {pctbar(cpu)}")
    train.add_row("PROCESS RAM", f"{p['mem']}%")
    train.add_row("RUNTIME", p["runtime"])
    train.add_row("STATE", p["state"])
    if event:
        step = event.get("step", "—")
        total = event.get("max_steps", "—")
        train.add_row("STEP", f"{step} / {total}")
        train.add_row("EPOCH", str(event.get("epoch", "—")))
        train.add_row("LOSS", f"{loss:.5f}" if isinstance(loss, float) else str(loss))
        train.add_row("LR", str(event.get("learning_rate", "—")))
        train.add_row("THROUGHPUT", f"{event.get('throughput', 0):.2f} step/s" if event.get("throughput") else "—")
        train.add_row("ETA", str(event.get("eta", "—")))
        train.add_row("LOSS TREND", spark(loss_hist))

    apple = Table.grid(padding=(0, 2))
    apple.add_row("CPU", f"{cpu:.1f}%")
    apple.add_row("MEMORY", free)
    apple.add_row("COMPRESSED", compressed)
    apple.add_row("SWAP", str(sw))
    apple.add_row("THERMAL", "[green]🟢 NORMAL[/green]" if thermal()=="NORMAL" else "[red]🔴 WARNING[/red]")
    apple.add_row("DISK FREE", disk_free())
    apple.add_row("ACTIVITY", spark(cpu_hist))
    apple.add_row("SWAP TREND", spark(swap_hist))

    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(Panel(train, title="TRAINING"), Panel(apple, title="APPLE SILICON"))
    grid.add_row(Panel(
        Group(Text("CPU ACTIVITY"), Text(spark(cpu_hist)), Text("LOSS"), Text(spark(loss_hist)), Text("SWAP"), Text(spark(swap_hist))),
        title="LIVE SIGNALS"
    ))

    status = "[green]🟢 Guardian is monitoring passively.[/green]"
    if thermal() != "NORMAL":
        status = "[red]🔴 Thermal warning reported.[/red]"
    elif sw:
        status = "[yellow]🟡 Swap is active; memory pressure may be elevated.[/yellow]"
    grid.add_row(Panel(status, title="GUARDIAN"))
    return Group(title, grid)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {Path(sys.argv[0]).name} <PID> [telemetry.jsonl]")
        raise SystemExit(2)
    pid = int(sys.argv[1])
    telemetry = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else DEFAULT_TELEMETRY
    cpu_hist, mem_hist, swap_hist, loss_hist = (deque(maxlen=HISTORY) for _ in range(4))
    event = {}
    with Live(console=None, refresh_per_second=2, screen=True) as live:
        while True:
            event = read_telemetry(telemetry, event)
            live.update(render(pid, event, cpu_hist, mem_hist, swap_hist, loss_hist))
            time.sleep(0.5)


if __name__ == "__main__":
    main()
