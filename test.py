"""Quick manual test harness for the Tokenizer.

Loads the trained vocab/merges and runs encode() on a sample string.
Run from the repo root:  uv run python test.py --text 'some text here'
"""

import argparse

from cs336_basics.tokenizer import Tokenizer

VOCAB_PATH = "data/bpe/vocab.pickle"
MERGES_PATH = "data/bpe/merges.pickle"


def main():
    parser = argparse.ArgumentParser(description="Run the Tokenizer on a sample string.")
    parser.add_argument("--text", default="this is a test", help="text to encode")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH)

    encoded = tokenizer.encode(args.text)

    print(f"input:   {args.text!r}")
    print(f"encoded: {encoded}")


if __name__ == "__main__":
    main()
