"""CPU regression coverage for Gemma4's quantization-aware embedding."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.gemma4_causal import Gemma4TextScaledWordEmbedding
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_unquantized_gemma4_embedding_preserves_lookup_and_scale():
    parallel = SimpleNamespace(tp_rank=0, tp_size=1, tp_group=None)
    with patch(
        "sglang.srt.layers.vocab_parallel_embedding.get_parallel",
        return_value=parallel,
    ):
        embedding = Gemma4TextScaledWordEmbedding(
            num_embeddings=5,
            embedding_dim=3,
            padding_idx=0,
            embed_scale=2.0,
            quant_config=None,
            enable_tp=False,
        )
        values = torch.arange(15, dtype=embedding.weight.dtype).reshape(5, 3)
        with torch.no_grad():
            embedding.weight.zero_()
            embedding.weight[:5].copy_(values)

        actual = embedding(torch.tensor([1, 3]))

    torch.testing.assert_close(actual, values[[1, 3]] * 2.0)
    assert embedding.padding_idx == 0
