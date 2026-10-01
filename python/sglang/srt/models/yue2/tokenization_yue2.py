# SPDX-License-Identifier: Apache-2.0
"""The frozen text/ABC BPE. This is not the audio audio tokenizer."""
from __future__ import annotations
import base64
import unicodedata
from pathlib import Path

# TODO (yiakwy) : move to sglang tokenizer
class YuE2TextTokenizer:
    def __init__(self, merge_file):
        import tiktoken
        self.merge_file = Path(merge_file)
        ranks = {base64.b64decode(t): int(r) for t, r in
                 (line.split() for line in self.merge_file.read_bytes().splitlines() if line)}
        if len(ranks) != 151643:
            raise ValueError("Expected checkpoint-native qwen.tiktoken (151643 ordinary tokens)")
        specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<R>", "<S>", "<X>", "<mask>", "<sep>"]
        specials += [f"<extra_{i}>" for i in range(200)]
        specials[204:206] = ["<abc>", "</abc>"]
        pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        self._enc = tiktoken.Encoding("YuE2", pat_str=pattern, mergeable_ranks=ranks,
                                     special_tokens={s: i + len(ranks) for i, s in enumerate(specials)})

    def encode(self, text):
        return self._enc.encode_ordinary(unicodedata.normalize("NFC", text))

    def decode(self, ids):
        return self._enc.decode([int(i) for i in ids if 0 <= i < self._enc.n_vocab], errors="replace")

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        from .storage import resolve_model
        return cls(resolve_model(path, **kwargs) / "qwen.tiktoken")

    def save_pretrained(self, directory):
        import shutil
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "qwen.tiktoken"
        if target.resolve() != self.merge_file.resolve():
            shutil.copyfile(self.merge_file, target)
        return (str(target),)
