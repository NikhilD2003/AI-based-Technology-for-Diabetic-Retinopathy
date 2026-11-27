"""Live progress bar for src/train.py. Read-only; Ctrl+C closes it without affecting training.

    py -3.10 src/progress.py
"""
import time

from train import EXPERIMENTS, PREDS, SEEDS

runs = [f"{name}_s{seed}" for seed in SEEDS for name in EXPERIMENTS]  # same order as train.py
while True:
    todo = [r for r in runs if not (PREDS / f"{r}.npz").exists()]
    done = len(runs) - len(todo)
    bar = "#" * (30 * done // len(runs))
    times = [(PREDS / f"{r}.npz").stat().st_mtime for r in runs if r not in todo]
    since = f"{(time.time() - max(times)) / 60:.1f} min since last finish" if times else ""
    line = f"[{bar:<30}] {done}/{len(runs)}  " + (f"next: {todo[0]}  {since}" if todo else "ALL DONE")
    print(f"\r{line:<100}", end="", flush=True)
    if not todo:
        break
    time.sleep(5)
print()
