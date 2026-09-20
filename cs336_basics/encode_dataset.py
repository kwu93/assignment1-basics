from tokenizer import Tokenizer
import argparse
import numpy as np
import itertools



def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_filepath", type=str, required=True)
    p.add_argument("--merges_filepath", type=str, required=True)
    p.add_argument("--vocab_filepath", type=str, required=True)
    p.add_argument("--out_filepath", type=str, required=True)
    return p


if __name__ == "__main__":
    p = build_parser()
    args = p.parse_args()

    print(args)
    print("Building tokenizer")

    tokenizer = Tokenizer.from_files(
        args.vocab_filepath, 
        args.merges_filepath, 
        special_tokens = ["<|endoftext|>"]
    )

    print("Encoding...")

    outputs = []
    with open(args.data_filepath, encoding='utf-8') as f:
        CHUNK = 1_000_000
        with open(args.out_filepath, "wb") as outfile:
            ids = tokenizer.encode_iterable(f)
            while chunk := list(itertools.islice(ids, CHUNK)):
                np.array(chunk, dtype=np.uint16).tofile(outfile)
    print("Done.")



