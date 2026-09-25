"""Evaluate a checkpoint on an entire tokenized validation file, deterministically.

Walks the file in non-overlapping windows of context_length + 1 tokens (no random sampling),
so every run is scored on exactly the same tokens and the number is reproducible.

    uv run python cs336_basics/eval_full.py --run lr_2e-3 --val-path data/TinyStoriesV2-GPT4-valid.bin
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from transformer import TransformerLM, cross_entropy_with_logits


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--step", type=int, default=10000)
    p.add_argument("--chkpt-dir", default="checkpoints")
    p.add_argument("--val-path", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    run_dir = os.path.join(args.chkpt_dir, args.run)
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    model = TransformerLM(
        vocab_size=cfg["vocab_size"], context_length=cfg["context_length"], num_layers=cfg["num_layers"],
        d_model=cfg["d_model"], num_heads=cfg["num_heads"], d_ff=cfg["d_ff"], rope_theta=cfg["rope_theta"],
        norm=cfg.get("norm", "pre"), pos_emb=cfg.get("pos_emb", "rope"), ffn=cfg.get("ffn", "swiglu"),
        device=args.device, dtype=torch.float32,
    )
    ckpt = torch.load(os.path.join(run_dir, f"iteration_{args.step}"), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    data = np.memmap(args.val_path, dtype=np.uint16, mode="r")
    ctx = cfg["context_length"]
    n_windows = (len(data) - 1) // ctx
    print(f"{args.run} @ {args.step}: {len(data):,} val tokens, {n_windows:,} windows of {ctx}, batch {args.batch_size}, device {args.device}")

    total_loss, total_tokens, t0 = 0.0, 0, time.time()
    with torch.no_grad():
        for start in range(0, n_windows, args.batch_size):
            idx = np.arange(start, min(start + args.batch_size, n_windows))[:, None] * ctx + np.arange(ctx)[None, :]
            X = torch.tensor(data[idx], dtype=torch.int64, device=args.device)
            y = torch.tensor(data[idx + 1], dtype=torch.int64, device=args.device)
            loss = cross_entropy_with_logits(model(X), y)  # mean over the batch's tokens
            total_loss += loss.item() * X.numel()
            total_tokens += X.numel()
    print(f"full-validation loss: {total_loss / total_tokens:.4f} over {total_tokens:,} tokens ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
