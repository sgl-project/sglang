import pytest
import torch

from sglang.srt.layers.attention.vision import (
    SingletonCache,
    prepare_vision_attention_metadata,
    resolve_ascend_actual_seq_lengths,
    resolve_max_seqlen,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_resolve_max_seqlen_accepts_raw_tensor_without_attribute_cache():
    cu_seqlens = torch.tensor([0, 2, 7], dtype=torch.int32)

    assert resolve_max_seqlen(cu_seqlens, cu_seqlens) == 5


def test_resolve_max_seqlen_caches_on_singleton_carrier():
    source = SingletonCache()
    cu_seqlens = torch.tensor([0, 2, 7], dtype=torch.int32)

    assert resolve_max_seqlen(source, cu_seqlens) == 5
    assert source._max_seqlen == 5


def test_resolve_ascend_actual_seq_lengths_uses_cumulative_ends():
    metadata = prepare_vision_attention_metadata(
        torch.tensor([0, 3, 8, 10], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    # Per-image lengths are not valid actual_seq_lengths for the TND layout.
    assert metadata.seq_lens.tolist() == [3, 5, 2]
    assert resolve_ascend_actual_seq_lengths(metadata).tolist() == [3, 8, 10]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
