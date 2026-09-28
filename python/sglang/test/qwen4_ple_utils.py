"""Shared fixtures for the Qwen4 PLE host-hash tests."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sglang.srt.models.qwen4_exp import Qwen4ExpNGramEmbedding
from sglang.srt.models.qwen4_exp_ple_rows import hash_contexts_numpy


def make_embedding(ngram_size=3, device="cuda", fused=False):
    embedding = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding.ngram_size = ngram_size
    embedding.ngram_heads = 8 * (ngram_size - 1)
    embedding.heads_per_ngram = 8
    embedding.eos_token_id = 0
    embedding.enable_ple_fusion = fused
    embedding.config = SimpleNamespace(seed=1234)
    embedding.unigram_vocab_size = 200000
    embedding.ple_layer_index = 0
    embedding.ngram_vocab_size_base = 20000000
    sizes, offsets, _ = embedding._build_head_vocab_and_offsets()
    embedding.layer_multipliers = embedding._build_layer_multipliers(ngram_size).to(
        device
    )
    embedding.ngram_heads_vocab_sizes = torch.tensor(sizes, device=device)
    embedding.ngram_heads_offsets = torch.tensor(offsets, device=device)
    return embedding


def assert_host_hash_matches(embedding):
    rng = np.random.default_rng(17)
    contexts = rng.integers(0, 200000, (4096, embedding.ngram_size), dtype=np.int64)
    contexts[rng.random(contexts.shape) < 0.2] = 0
    with mock.patch(
        "sglang.srt.models.qwen4_exp.get_req_to_token_pool",
        return_value=SimpleNamespace(ple_window_cache=None),
    ):
        actual = embedding._hash_contexts(
            torch.tensor(contexts, device=embedding.layer_multipliers.device),
            decode_sized=True,
        )
    expected = hash_contexts_numpy(
        contexts,
        embedding.layer_multipliers.cpu().numpy(),
        embedding.ngram_heads_vocab_sizes.cpu().numpy(),
        embedding.ngram_heads_offsets.cpu().numpy(),
        0,
    )
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)
