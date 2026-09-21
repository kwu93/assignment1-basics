"""Run cs336_basics/train.py on Modal GPUs.

One-time setup:
    modal volume create cs336-data
    modal volume put cs336-data data/TinyStoriesV2-GPT4-train.bin TinyStoriesV2-GPT4-train.bin
    modal volume put cs336-data data/TinyStoriesV2-GPT4-valid.bin TinyStoriesV2-GPT4-valid.bin
    (wandb API key comes from the existing Modal secret named "wandb", key WANDB_API_KEY)

Single run (everything after --extra is passed straight to train.py):
    modal run modal_train.py --run-name lr_1e-3 --extra "--lrmax 1e-3 --train-iters 10000 --batch-size 128 --dtype bfloat16"

Learning-rate sweep, one GPU per rate, all in parallel:
    modal run modal_train.py --lrs 1e-4,3e-4,1e-3,3e-3 --extra "--train-iters 10000 --batch-size 128 --dtype bfloat16"

Warmup, cosine length, and the final learning rate are derived by train.py from train-iters and lrmax
unless passed explicitly in --extra.

Checkpoints and metrics.jsonl for each run land in the cs336-runs volume under <run_name>/.
    modal volume get cs336-runs lr_1e-3/metrics.jsonl runs/lr_1e-3/metrics.jsonl

Pick the GPU with GPU=A100 modal run ... (default H100).
"""

import json
import os
import shlex
import subprocess
import sys
import time

import modal

APP_NAME = "cs336-basics"
DATA_DIR = "/data"
RUNS_DIR = "/runs"
GPU = os.environ.get("GPU", "H100")

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync()
    .add_local_python_source("cs336_basics")
)

data_volume = modal.Volume.from_name("cs336-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("cs336-runs", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])


@app.function(
    image=image,
    gpu=GPU,
    volumes={DATA_DIR: data_volume, RUNS_DIR: runs_volume},
    secrets=[wandb_secret],
    timeout=4 * 60 * 60,
)
def train(run_name: str, extra_args: list[str], train_file: str, val_file: str) -> dict:
    """Run one training job and return a summary of its final eval row."""
    cmd = [
        sys.executable,
        "/root/cs336_basics/train.py",
        "--train-path", os.path.join(DATA_DIR, train_file),
        "--val-path", os.path.join(DATA_DIR, val_file),
        "--chkpt-dir", RUNS_DIR,
        "--run_name", run_name,
        "--device", "cuda",
        *extra_args,
    ]
    print(f"[{run_name}] {shlex.join(cmd)}", flush=True)

    start = time.time()
    result = subprocess.run(cmd, cwd="/root")
    elapsed = time.time() - start
    runs_volume.commit()

    summary = {"run_name": run_name, "returncode": result.returncode, "wall_clock_s": round(elapsed)}
    metrics_path = os.path.join(RUNS_DIR, run_name, "metrics.jsonl")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        evals = [r for r in rows if "eval/val_loss" in r]
        if evals:
            last = evals[-1]
            summary["final_val_loss"] = last["eval/val_loss"]
            summary["final_step"] = last["step"]
            summary["min_val_loss"] = min(r["eval/val_loss"] for r in evals)
    return summary


@app.local_entrypoint()
def main(
    run_name: str = "",
    lrs: str = "",
    extra: str = "",
    prefix: str = "",
    train_file: str = "TinyStoriesV2-GPT4-train.bin",
    val_file: str = "TinyStoriesV2-GPT4-valid.bin",
):
    """Launch a single run (--run-name) or a learning-rate sweep (--lrs a,b,c)."""
    extra_args = shlex.split(extra)

    if lrs:
        jobs = []
        for lr_str in lrs.split(","):
            lr = float(lr_str)
            name = f"{prefix}lr_{lr_str.strip()}"
            args = [*extra_args, "--lrmax", str(lr)]
            jobs.append((name, args, train_file, val_file))
        print(f"Launching {len(jobs)} runs on {GPU}: {[j[0] for j in jobs]}")
        summaries = list(train.starmap(jobs))
    elif run_name:
        summaries = [train.remote(run_name, extra_args, train_file, val_file)]
    else:
        raise SystemExit("Pass --run-name for a single run or --lrs a,b,c for a sweep. See module docstring.")

    print("\nrun_name\tfinal_val_loss\tmin_val_loss\tfinal_step\twall_clock_s\treturncode")
    for s in summaries:
        print(
            f"{s['run_name']}\t{s.get('final_val_loss', 'n/a')}\t{s.get('min_val_loss', 'n/a')}\t"
            f"{s.get('final_step', 'n/a')}\t{s['wall_clock_s']}\t{s['returncode']}"
        )
