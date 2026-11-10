"""CPU hash golden: SGLang Engram ids match HF DeepSeek-V4.1-Flash inference/engram.py.

HF file used (fetched, that path only, no checkpoint):
  /tmp/dsv41-hf/inference/engram.py
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/engram.py

The short token string is the characters of "Hello" over a tiny fake vocab so the
test does not need the 476 GiB tokenizer. Compression is the real HF
build_compressed_token_map (NFKC/NFD/accents/lowercase), so "Hello" and "hello"
share hash ids.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.layers.engram import (
    EngramLayout,
    build_compressed_token_map,
    compute_engram_hash_ids,
    compute_hash_multipliers,
    engram_lookup_dtype,
)
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

torch.set_num_threads(2)

HF_ENGRAM_CANDIDATES = (
    os.path.expanduser("~/models/DeepSeek-V4.1-Flash/inference/engram.py"),
    "/tmp/dsv41-hf/inference/engram.py",
)
HF_SOURCE = (
    "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/engram.py"
)

# Tiny char vocab. Index 0 is the pad token (HF engram_pad_id default 0).
VOCAB_TEXTS = ["<pad>", "<unk>", " ", "h", "e", "l", "o", "H", "E", "L", "O"]
# "Hello" as one token per character.
HELLO_TEXT = "Hello"
HELLO_IDS = [VOCAB_TEXTS.index(ch) for ch in HELLO_TEXT]  # H,e,l,l,o
HELLO_LOWER_IDS = [VOCAB_TEXTS.index(ch) for ch in HELLO_TEXT.lower()]

LAYER_IDS = (1,)
MAX_NGRAM = 4
N_HEADS = 2
HEAD_DIM = 32
ENGRAM_VOCAB_SIZE = 32  # prime search starts above this - 1
NUM_EMBEDDINGS = (64,)


class _Backend:
    def __init__(self, texts: list[str]):
        self.texts = texts

    def decode(self, ids, skip_special_tokens=False):
        return self.texts[ids[0]]

    def id_to_token(self, token_id):
        return self.texts[token_id]


class _FakeTokenizer:
    def __init__(self, texts: list[str]):
        self.backend_tokenizer = _Backend(texts)
        self._n = len(texts)

    def __len__(self):
        return self._n


def _load_hf_engram():
    hf_path = next((p for p in HF_ENGRAM_CANDIDATES if os.path.isfile(p)), None)
    if hf_path is None:
        raise unittest.SkipTest(
            f"HF inference/engram.py not at {HF_ENGRAM_CANDIDATES} (source {HF_SOURCE})"
        )
    spec = importlib.util.spec_from_file_location(
        "dsv41_hf_inference_engram", hf_path
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _hf_args(compressed_vocab_size: int):
    return SimpleNamespace(
        engram_layer_ids=LAYER_IDS,
        engram_num_embeddings=NUM_EMBEDDINGS,
        engram_max_ngram_size=MAX_NGRAM,
        engram_n_heads=N_HEADS,
        engram_head_dim=HEAD_DIM,
        engram_vocab_size=ENGRAM_VOCAB_SIZE,
        engram_compressed_vocab_size=compressed_vocab_size,
        engram_pad_id=0,
        max_batch_size=1,
        max_seq_len=16,
    )


def _predecessor_table(input_ids: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """tokens [T, n] column 0 = token, column s = predecessor; blocked if pos < s."""
    t = input_ids.numel()
    tokens = input_ids.new_zeros(t, n)
    blocked = torch.zeros(t, n, dtype=torch.bool)
    ids = input_ids.reshape(-1)
    for pos in range(t):
        for shift in range(n):
            if pos < shift:
                blocked[pos, shift] = True
            else:
                tokens[pos, shift] = ids[pos - shift]
    return tokens, blocked


class TestDsv41EngramHashGolden(CustomTestCase):
    def test_hf_file_is_the_documented_source(self):
        hf_path = next((p for p in HF_ENGRAM_CANDIDATES if os.path.isfile(p)), None)
        self.assertIsNotNone(
            hf_path,
            f"expected fetched HF file in {HF_ENGRAM_CANDIDATES} from {HF_SOURCE}",
        )
        with open(hf_path) as f:
            text = f.read()
        self.assertIn("class NgramHashState", text)
        self.assertIn("compute_hash_multipliers", text)
        self.assertIn("10007 * layer_id", text)

    def test_multipliers_and_primes_match_hf(self):
        hf = _load_hf_engram()
        tokenizer = _FakeTokenizer(VOCAB_TEXTS)
        token_map, compressed = build_compressed_token_map(tokenizer)
        hf_map, hf_compressed = hf.build_compressed_token_map(tokenizer)
        self.assertEqual(token_map, hf_map)
        self.assertEqual(compressed, hf_compressed)

        ours = compute_hash_multipliers(LAYER_IDS, MAX_NGRAM, compressed)
        theirs = hf.compute_hash_multipliers(LAYER_IDS, MAX_NGRAM, compressed)
        self.assertTrue(torch.equal(ours, theirs), (ours, theirs))

        layout = EngramLayout.build(
            layer_ids=LAYER_IDS,
            num_embeddings=NUM_EMBEDDINGS,
            max_ngram_size=MAX_NGRAM,
            n_heads=N_HEADS,
            head_dim=HEAD_DIM,
            vocab_size=ENGRAM_VOCAB_SIZE,
        )
        hf_layout = hf.EngramLayout.from_args(_hf_args(compressed))
        self.assertEqual(layout.primes, hf_layout.primes)

    def test_hello_hash_ids_bit_identical_to_hf(self):
        hf = _load_hf_engram()
        tokenizer = _FakeTokenizer(VOCAB_TEXTS)
        token_map_list, compressed = hf.build_compressed_token_map(tokenizer)
        args = _hf_args(compressed)
        layout = hf.EngramLayout.from_args(args)
        state = hf.NgramHashState(args, layout, tokenizer)

        hello = torch.tensor([HELLO_IDS], dtype=torch.int64)
        hf_ids = state(hello, start_pos=0)
        self.assertEqual(hf_ids.dtype, torch.int64)

        tokens, blocked = _predecessor_table(hello, MAX_NGRAM)
        sgl_ids = compute_engram_hash_ids(
            tokens,
            blocked,
            pad_id=int(state.pad_id),
            token_map=state.token_map,
            multipliers=state.multipliers,
            primes=state.primes,
            offsets=state.offsets,
        )
        self.assertTrue(
            torch.equal(sgl_ids, hf_ids.squeeze(0)),
            (sgl_ids, hf_ids.squeeze(0)),
        )

        # Lowercase of the same string must hash identically (HF normalizer).
        hello_l = torch.tensor([HELLO_LOWER_IDS], dtype=torch.int64)
        hf_lower = state(hello_l, start_pos=0)
        self.assertTrue(torch.equal(hf_ids, hf_lower))
        self.assertEqual(token_map_list[HELLO_IDS[0]], token_map_list[HELLO_LOWER_IDS[0]])

    def test_sglang_layout_hash_matches_hf_state(self):
        """Rebuild multipliers/primes on the SGLang side; still bit-identical."""
        hf = _load_hf_engram()
        tokenizer = _FakeTokenizer(VOCAB_TEXTS)
        token_map_list, compressed = build_compressed_token_map(tokenizer)
        args = _hf_args(compressed)
        hf_state = hf.NgramHashState(args, hf.EngramLayout.from_args(args), tokenizer)

        layout = EngramLayout.build(
            layer_ids=LAYER_IDS,
            num_embeddings=NUM_EMBEDDINGS,
            max_ngram_size=MAX_NGRAM,
            n_heads=N_HEADS,
            head_dim=HEAD_DIM,
            vocab_size=ENGRAM_VOCAB_SIZE,
        )
        multipliers = compute_hash_multipliers(LAYER_IDS, MAX_NGRAM, compressed)
        primes = torch.tensor(layout.primes)
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
        offsets = torch.tensor(np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat]))
        token_map = torch.tensor(token_map_list)

        hello = torch.tensor([HELLO_IDS], dtype=torch.int64)
        tokens, blocked = _predecessor_table(hello, MAX_NGRAM)
        sgl_ids = compute_engram_hash_ids(
            tokens,
            blocked,
            pad_id=token_map_list[0],
            token_map=token_map,
            multipliers=multipliers,
            primes=primes,
            offsets=offsets,
        )
        hf_ids = hf_state(hello, start_pos=0).squeeze(0)
        self.assertTrue(torch.equal(sgl_ids, hf_ids), (sgl_ids, hf_ids))

    def test_lookup_dtype_sm70_is_fp16_hopper_is_bf16(self):
        with override_platform(is_sm70=True, is_sm90=False):
            self.assertEqual(engram_lookup_dtype(), torch.float16)
        with override_platform(is_sm70=False, is_sm90=True):
            self.assertEqual(engram_lookup_dtype(), torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
