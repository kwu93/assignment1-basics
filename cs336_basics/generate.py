"""Generate text from a trained checkpoint with several sampling settings.

Pull the final checkpoint of a run from the cs336-runs Modal volume and sample from it:
    uv run python cs336_basics/generate.py --run lr_2e-3 --pull

Reuse an already-downloaded checkpoint:
    uv run python cs336_basics/generate.py --run lr_2e-3 --prompt "Once upon a time" --max-new-tokens 256

Runs trained on OpenWebText use the OWT vocab:
    uv run python cs336_basics/generate.py --run owt_lr_2e-3 --pull --tokenizer owt --prompt "The"

Each sampling setting produces exactly --max-new-tokens new tokens (no early stop on <|endoftext|>),
so story boundaries show up inline as the literal <|endoftext|> string.
"""

import argparse
import json
import os
import subprocess
import time

import torch

from tokenizer import Tokenizer
from transformer import TransformerLM, decode

MODAL = os.path.expanduser("~/miniforge3/bin/modal")
TOKENIZERS = {
    "tinystories": ("data/bpe/TinyStoriesV2-GPT4-train-vocab.pickle", "data/bpe/TinyStoriesV2-GPT4-train-merges.pickle"),
    "owt": ("data/bpe/owt_vocab.pickle", "data/bpe/owt_merges.pickle"),
}

# (label, kwargs for decode)
SAMPLERS = [
    ("greedy", dict(method="greedy")),
    ("sample, temperature 1.0", dict(method="default", temperature=1.0)),
    ("sample, temperature 0.7", dict(method="default", temperature=0.7)),
    ("nucleus, top-p 0.9, temperature 1.0", dict(method="nucleus", top_p=0.9, temperature=1.0)),
    ("nucleus, top-p 0.9, temperature 0.8", dict(method="nucleus", top_p=0.9, temperature=0.8)),
]


def pull(run: str, step: int, run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    for name in (f"iteration_{step}", "config.json"):
        subprocess.run([MODAL, "volume", "get", "cs336-runs", f"{run}/{name}", os.path.join(run_dir, name), "--force"], check=True)


def load_model(run_dir: str, step: int) -> TransformerLM:
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    model = TransformerLM(
        vocab_size=cfg["vocab_size"], context_length=cfg["context_length"], num_layers=cfg["num_layers"],
        d_model=cfg["d_model"], num_heads=cfg["num_heads"], d_ff=cfg["d_ff"], rope_theta=cfg["rope_theta"],
        device="cpu", dtype=torch.float32,
    )
    ckpt = torch.load(os.path.join(run_dir, f"iteration_{step}"), map_location="cpu")
    model.load_state_dict(ckpt["model"])  # bf16 checkpoint tensors are upcast into the fp32 params
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="lr_2e-3")
    p.add_argument("--step", type=int, default=10000)
    p.add_argument("--chkpt-dir", default="checkpoints")
    p.add_argument("--pull", action="store_true", help="download the checkpoint and config from the cs336-runs volume first")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer", choices=sorted(TOKENIZERS), default="tinystories", help="which BPE vocab the run was trained with")
    args = p.parse_args()

    run_dir = os.path.join(args.chkpt_dir, args.run)
    if args.pull:
        pull(args.run, args.step, run_dir)

    model = load_model(run_dir, args.step)
    # No special tokens, so decode() never stops early and every sample has exactly max_new_tokens new tokens.
    vocab, merges = TOKENIZERS[args.tokenizer]
    tokenizer = Tokenizer.from_files(vocab, merges, special_tokens=None)
    print(f"run {args.run} @ step {args.step}, prompt {args.prompt!r}, {args.max_new_tokens} new tokens per sample\n")

    for label, kwargs in SAMPLERS:
        torch.manual_seed(args.seed)
        t0 = time.time()
        print(f"=== {label} ===")
        decode(args.prompt, tokenizer, model, args.max_new_tokens, **kwargs)
        print(f"--- {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
