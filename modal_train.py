"""Run cs336_basics/train.py on Modal GPUs.

One-time setup (already done):
    modal volume create cs336-data
    modal volume put cs336-data data/TinyStoriesV2-GPT4-train.bin TinyStoriesV2-GPT4-train.bin
    modal volume put cs336-data data/TinyStoriesV2-GPT4-valid.bin TinyStoriesV2-GPT4-valid.bin
    (wandb API key comes from the existing Modal secret named "wandb", key WANDB_API_KEY)

Single run (everything after --extra is passed straight to train.py):
    modal run modal_train.py --run-name lr_2e-3 --extra "--lrmax 2e-3 --train-iters 10000 --batch-size 128 --dtype bfloat16"

Sweep any train.py argument, one GPU per value, all in parallel. Runs are named <prefix><short>_<value>:
    modal run modal_train.py --sweep lrmax=1e-3,2e-3,3e-3 --extra "--train-iters 10000 --batch-size 128 --dtype bfloat16"
    modal run modal_train.py --sweep batch-size=32,64,256,512 --fixed-tokens 327680000 --extra "--lrmax 2e-3 --dtype bfloat16"
    (--lrs a,b,c is shorthand for --sweep lrmax=a,b,c)

--fixed-tokens N sets --train-iters per run to N / (batch-size * context-length), so every run sees the same data.
--evals-per-run K sets --eval-interval per run to train-iters / K, so every run logs the same number of eval points.

Probe mode runs 50 steps per value with wandb off, reports ms/step, tokens/s and peak GPU memory, and deletes its outputs:
    GPU=B200 modal run modal_train.py --probe --sweep batch-size=256,512,1024,2048 --extra "--lrmax 2e-3 --dtype bfloat16"

Checkpoints, config.json and metrics.jsonl for each run land in the cs336-runs volume under <run_name>/.
A run whose directory already exists on the volume is refused (metrics.jsonl opens in append mode); delete it first:
    modal volume rm -r cs336-runs <run_name>

Warmup, cosine length and the final learning rate are derived by train.py from train-iters and lrmax unless passed in --extra.
Pick the GPU with GPU=B200 modal run ... (default H100).
"""

import collections
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time

import modal

APP_NAME = "cs336-basics"
DATA_DIR = "/data"
RUNS_DIR = "/runs"
GPU = os.environ.get("GPU", "H100")

# Short names used when building run names for swept arguments.
SHORT = {
    "lrmax": "lr",
    "batch-size": "bs",
    "context-length": "ctx",
    "num-layers": "L",
    "d-model": "d",
    "d-ff": "dff",
    "num-heads": "h",
    "vocab-size": "v",
    "weight-decay": "wd",
}
PROBE_ARGS = ["--train-iters", "50", "--eval-interval", "25", "--eval-iters", "2", "--log-interval", "10", "--no-wandb"]

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync()
    .add_local_python_source("cs336_basics")
)

data_volume = modal.Volume.from_name("cs336-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("cs336-runs", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])


def arg_value(args: list[str], name: str, default):
    """Return the last value passed for --<name> in an argv list, cast to default's type."""
    flag = f"--{name}"
    value = default
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            value = type(default)(args[i + 1])
    return value


def _poll_gpu_memory(stop: threading.Event, peak: list[int]) -> None:
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            peak[0] = max(peak[0], int(out.strip().splitlines()[0]))
        except Exception:
            pass
        stop.wait(1.0)


@app.function(
    image=image,
    gpu=GPU,
    volumes={DATA_DIR: data_volume, RUNS_DIR: runs_volume},
    secrets=[wandb_secret],
    timeout=4 * 60 * 60,
)
def train(run_name: str, extra_args: list[str], train_file: str, val_file: str, probe: bool = False) -> dict:
    """Run one training job and return a summary: final/min val loss, ms per step, tokens/s, peak GPU memory."""
    run_dir = os.path.join(RUNS_DIR, run_name)
    if os.path.exists(run_dir) and not probe:
        return {"run_name": run_name, "returncode": -1,
                "error": f"{run_dir} already exists on the volume; delete it or choose another name"}

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

    peak = [0]
    stop = threading.Event()
    threading.Thread(target=_poll_gpu_memory, args=(stop, peak), daemon=True).start()

    start = time.time()
    proc = subprocess.Popen(cmd, cwd="/root", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = collections.deque(maxlen=80)
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
    returncode = proc.wait()
    elapsed = time.time() - start
    stop.set()

    summary = {
        "run_name": run_name,
        "returncode": returncode,
        "wall_clock_s": round(elapsed),
        "peak_mem_gb": round(peak[0] / 1024, 1),
    }
    if returncode != 0 and any("out of memory" in l.lower() for l in tail):
        summary["error"] = "CUDA out of memory"
    elif returncode != 0:
        summary["error"] = f"exit {returncode}"

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        evals = [r for r in rows if "eval/val_loss" in r]
        if evals:
            summary["final_val_loss"] = evals[-1]["eval/val_loss"]
            summary["final_step"] = evals[-1]["step"]
            summary["min_val_loss"] = min(r["eval/val_loss"] for r in evals)
        # Steady-state step time from train rows after the first few steps (skips warmup/compile effects).
        train_rows = [r for r in rows if "train/loss" in r and r["step"] >= 20]
        if len(train_rows) >= 2:
            first, last = train_rows[0], train_rows[-1]
            per_step = (last["wall_clock"] - first["wall_clock"]) / (last["step"] - first["step"])
            bs = arg_value(extra_args, "batch-size", 32)
            ctx = arg_value(extra_args, "context-length", 256)
            summary["ms_per_step"] = round(per_step * 1000, 1)
            summary["tokens_per_s"] = round(bs * ctx / per_step)

    if probe:
        shutil.rmtree(run_dir, ignore_errors=True)
    runs_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    run_name: str = "",
    lrs: str = "",
    sweep: str = "",
    extra: str = "",
    prefix: str = "",
    fixed_tokens: int = 0,
    evals_per_run: int = 0,
    probe: bool = False,
    train_file: str = "TinyStoriesV2-GPT4-train.bin",
    val_file: str = "TinyStoriesV2-GPT4-valid.bin",
):
    """Launch a single run (--run-name), or a sweep (--sweep arg=v1,v2,... or --lrs v1,v2,...)."""
    extra_args = shlex.split(extra)
    if lrs:
        sweep = f"lrmax={lrs}"

    jobs: list[tuple[str, list[str]]] = []
    if sweep:
        arg, _, values = sweep.partition("=")
        arg = arg.strip().lstrip("-")
        if not values:
            raise SystemExit("--sweep expects arg=v1,v2,... (for example batch-size=32,64,128)")
        short = SHORT.get(arg, arg)
        for v in values.split(","):
            v = v.strip()
            jobs.append((f"{prefix}{short}_{v}", [*extra_args, f"--{arg}", v]))
    elif run_name:
        jobs.append((run_name, extra_args))
    else:
        raise SystemExit("Pass --run-name for a single run or --sweep arg=v1,v2,... for a sweep. See module docstring.")

    if probe:
        jobs = [(f"probe_{name}", [*args, *PROBE_ARGS]) for name, args in jobs]
    elif fixed_tokens:
        resolved = []
        for name, args in jobs:
            bs = arg_value(args, "batch-size", 32)
            ctx = arg_value(args, "context-length", 256)
            iters = fixed_tokens // (bs * ctx)
            resolved.append((name, [*args, "--train-iters", str(iters)]))
        jobs = resolved

    if evals_per_run and not probe:
        # Same number of evals per run regardless of step count, so curves are comparable across batch sizes.
        jobs = [(name, [*args, "--eval-interval", str(max(1, arg_value(args, "train-iters", 5000) // evals_per_run))])
                for name, args in jobs]

    mode = "probe" if probe else f"fixed {fixed_tokens:,} tokens" if fixed_tokens else "run"
    print(f"Launching {len(jobs)} {mode} job(s) on {GPU}: {[j[0] for j in jobs]}")
    summaries = list(train.starmap([(name, args, train_file, val_file, probe) for name, args in jobs]))

    cols = ["run_name", "final_val_loss", "min_val_loss", "final_step", "ms_per_step", "tokens_per_s", "peak_mem_gb", "wall_clock_s", "error"]
    print("\n" + "\t".join(cols))
    for s in summaries:
        print("\t".join(str(s.get(c, "")) for c in cols))
