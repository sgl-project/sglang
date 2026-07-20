import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import sglang.srt.layers.vocab_parallel_embedding as vocab_embedding
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def test_vocab_padding_is_platform_independent():
    parallel = SimpleNamespace(tp_rank=0, tp_size=24)

    with (
        patch.object(vocab_embedding, "_is_cpu", False),
        patch.object(vocab_embedding, "get_parallel", return_value=parallel),
    ):
        layer = vocab_embedding.VocabParallelEmbedding(129_280, 1)

    assert layer.padding_size == 1_536
    assert layer.num_embeddings_padded == 130_560
    assert layer.weight.shape == (5_440, 1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
