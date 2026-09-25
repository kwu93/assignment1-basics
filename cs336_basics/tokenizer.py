import argparse
import pickle
import os
import numpy as np
from typing import Iterable
import regex as re

INF = float('inf')

class Tokenizer:
    MAX_PRETOKEN_BYTES = 2048  # pre-tokens longer than this are merged in slices (see encode)
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.mt = self._construct_merge_table()
        self.word_encodings = {}
        # Sort special tokens by length
        if special_tokens is not None:
            vocab_id = len(self.vocab)
            for token in special_tokens:
                self.vocab[vocab_id] = token.encode('utf-8')
                vocab_id += 1
            self.special_tokens = sorted(special_tokens, key = lambda k: -len(k))
        else:
            self.special_tokens = None
        self.PRETOK_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def _construct_merge_table(self):
        merges, inv_vocab = self.merges, self.inv_vocab
        merge_table = {}
        for i in range(len(merges)):
            a, b = merges[i]
            vocab_id = inv_vocab[a + b]
            merge_table[(inv_vocab[a], inv_vocab[b])] = (i, vocab_id)
        return merge_table


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)

        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[str]:
        if self.special_tokens is not None:
            special_token_delimiter = '|'.join([re.escape(st) for st in self.special_tokens])
            chunks = re.split(rf'({special_token_delimiter})', text)
        else:
            chunks = [text]

        self.word_encodings = {}

        output = []
        for chunk in chunks:
            if self.special_tokens is not None and chunk in self.special_tokens:
                output.append(self.inv_vocab[chunk.encode('utf-8')])
                continue

            for m in re.finditer(self.PRETOK_PATTERN, chunk):
                word = m.group()
                if word not in self.word_encodings: 
                    ids = [self.inv_vocab[bytes([t])] for t in list(word.encode('utf-8'))]
                    # Merging is quadratic in pre-token length; web text has rare pathological "words"
                    # (100 KB runs of mojibake). Merge those in fixed-size slices instead of stalling on them.
                    if len(ids) > self.MAX_PRETOKEN_BYTES:
                        merged = []
                        for k in range(0, len(ids), self.MAX_PRETOKEN_BYTES):
                            merged.extend(self._apply_merge(ids[k:k + self.MAX_PRETOKEN_BYTES]))
                        self.word_encodings[word] = merged
                    else:
                        self.word_encodings[word] = self._apply_merge(ids)
                output.extend(self.word_encodings[word])
        return output


    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[str]: 
        for text in iterable: 
            output = self.encode(text)
            yield from output

    def decode(self, ids: list[int]) -> str:
        output = []
        for vid in ids:
            byte = self.vocab.get(vid, b'\xef\xbf\xbd')
            output.append(byte)
        result = b"".join(output).decode('utf-8', errors='replace')
        return result


    def _apply_merge(self, tokens):
        # Iterative rather than recursive: one merge per pass, so long pre-tokens (URLs, runs of punctuation)
        # no longer exhaust the recursion limit. Semantics are unchanged.
        while len(tokens) > 1:
            min_rank = INF
            merge_id = None
            apply_idx = None

            for i, pair in enumerate(zip(tokens[:-1], tokens[1:])):
                rank, vocab_id = self.mt.get(pair, (INF, INF))
                if rank < min_rank:
                    min_rank = rank
                    merge_id = vocab_id
                    apply_idx = i

            if min_rank == INF:
                break

            out = []
            i = 0
            while i < len(tokens):
                if i == apply_idx:
                    out.append(merge_id)
                    i += 1
                else:
                    out.append(tokens[i])
                i += 1
            tokens = out
        return tokens

