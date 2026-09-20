from tests.adapters import run_train_bpe
import argparse
import pickle
import os

def parse_args():
    p = argparse.ArgumentParser(description="...")
    p.add_argument("--data-path", type=str, 
                   help="path to training data",
                   default="data/owt_train.txt"
                   )
    p.add_argument("--vocab-size", type=int, default=32000, help="tokenizer vocab size")
    p.add_argument("--output-dir", type=str, default="data/bpe/", help="output directory")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    special_tokens = ["<|endoftext|>"]
    print("Running BPE...")

    vocab, merges = run_train_bpe(args.data_path, args.vocab_size, special_tokens=special_tokens)

    assert len(vocab) == args.vocab_size
    assert len(merges) == args.vocab_size - 256 - len(special_tokens)

    print("Writing...")
    with open(os.path.join(args.output_dir, 'owt_vocab.pickle'), 'wb') as f:
        pickle.dump(vocab, f)

    with open(os.path.join(args.output_dir, 'owt_merges.pickle'), 'wb') as f:
        pickle.dump(merges, f)




