"""Encode a large text file to uint16 token ids using every core.

Splits the file into byte ranges that begin right after a <|endoftext|> token, so no document is cut,
encodes each range in a worker with Tokenizer.encode on ~16 MB pieces, writes one temp file per range,
then concatenates the temp files in order. Output is identical to encode_dataset.py, just faster.

    uv run python cs336_basics/encode_parallel.py --data_filepath data/owt_train.txt \\
        --vocab_filepath data/bpe/owt_vocab.pickle --merges_filepath data/bpe/owt_merges.pickle \\
        --out_filepath data/owt_train.bin
"""

import argparse
import os
import shutil
import time
from multiprocessing import Pool

import numpy as np

from tokenizer import Tokenizer

SPECIAL = "<|endoftext|>"
PIECE = 16 * 1024 * 1024  # bytes of text encoded per Tokenizer.encode call


def find_boundaries(path: str, n: int) -> list[int]:
    """Byte offsets splitting the file into n ranges, each starting right after a special token (or at 0)."""
    size = os.path.getsize(path)
    marker = SPECIAL.encode()
    bounds = [0]
    with open(path, "rb") as f:
        for k in range(1, n):
            guess = size * k // n
            f.seek(guess)
            window = f.read(4 * 1024 * 1024)
            j = window.find(marker)
            if j < 0:
                continue
            bounds.append(guess + j + len(marker))
    bounds.append(size)
    return sorted(set(bounds))


def encode_range(job: tuple) -> tuple[int, int, float]:
    idx, path, start, end, vocab, merges, tmp = job
    tok = Tokenizer.from_files(vocab, merges, special_tokens=[SPECIAL])
    marker = SPECIAL.encode()
    n_tokens, t0 = 0, time.time()
    with open(path, "rb") as f, open(tmp, "wb") as out:
        f.seek(start)
        pos = start
        carry = b""
        while pos < end:
            chunk = carry + f.read(min(PIECE, end - pos))
            pos += len(chunk) - len(carry)
            if pos < end:
                # cut at the last document boundary so a special token is never split across pieces
                cut = chunk.rfind(marker)
                if cut >= 0:
                    cut += len(marker)
                    chunk, carry = chunk[:cut], chunk[cut:]
                else:
                    carry = b""
            else:
                carry = b""
            ids = tok.encode(chunk.decode("utf-8", errors="replace"))
            np.array(ids, dtype=np.uint16).tofile(out)
            n_tokens += len(ids)
    return idx, n_tokens, time.time() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_filepath", required=True)
    p.add_argument("--vocab_filepath", required=True)
    p.add_argument("--merges_filepath", required=True)
    p.add_argument("--out_filepath", required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()

    bounds = find_boundaries(args.data_filepath, args.workers)
    tmp_dir = args.out_filepath + ".parts"
    os.makedirs(tmp_dir, exist_ok=True)
    jobs = [
        (i, args.data_filepath, bounds[i], bounds[i + 1], args.vocab_filepath, args.merges_filepath, os.path.join(tmp_dir, f"{i:03d}.bin"))
        for i in range(len(bounds) - 1)
    ]
    print(f"{os.path.getsize(args.data_filepath) / 1e9:.2f} GB in {len(jobs)} ranges on {args.workers} workers", flush=True)

    t0 = time.time()
    total = 0
    with Pool(args.workers) as pool:
        for idx, n, dt in pool.imap_unordered(encode_range, jobs):
            total += n
            print(f"range {idx:3d}: {n:,} tokens in {dt:.0f}s", flush=True)

    with open(args.out_filepath, "wb") as out:
        for i in range(len(jobs)):
            with open(os.path.join(tmp_dir, f"{i:03d}.bin"), "rb") as part:
                shutil.copyfileobj(part, out)
    shutil.rmtree(tmp_dir)
    print(f"Done. {total:,} tokens -> {args.out_filepath} ({os.path.getsize(args.out_filepath) / 1e9:.2f} GB) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
