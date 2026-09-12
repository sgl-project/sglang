import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import sglang.srt.layers.vocab_parallel_embedding as vocab_embedding
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    ("tp_size", "expected_padding_size", "expected_padded_size"),
    [
        (1, 64, 129_280),
        (2, 64, 129_280),
        (4, 64, 129_280),
        (8, 64, 129_280),
        (16, 64, 129_280),
        (24, 1_536, 130_560),
        (32, 64, 129_280),
        (40, 64, 129_280),
        (48, 3_072, 132_096),
    ],
)
def test_vocab_padding_is_platform_independent(
    tp_size, expected_padding_size, expected_padded_size
):
    parallel = SimpleNamespace(tp_rank=0, tp_size=tp_size)

    with (
        patch.object(vocab_embedding, "_is_cpu", False),
        patch.object(vocab_embedding, "get_parallel", return_value=parallel),
    ):
        layer = vocab_embedding.VocabParallelEmbedding(129_280, 1)

    assert layer.padding_size == expected_padding_size
    assert layer.num_embeddings_padded == expected_padded_size
    assert layer.weight.shape == (expected_padded_size // tp_size, 1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
