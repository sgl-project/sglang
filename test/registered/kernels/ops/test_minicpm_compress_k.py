import pytest
import torch

from sglang.srt.layers.attention.minicpm.fuse_kernel import (
    fused_attn_pooling_online_topk_decode,
)
from sglang.srt.layers.attention.minicpm.sparse_utils import compress_k_core_new
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_compress_k_writes_each_head_once():
    """Each compressed output must be produced once even with multiple KV heads."""
    key_cache = torch.arange(
        9 * 2 * 6,
        dtype=torch.float32,
        device="cuda",
    ).reshape(9, 2, 6)
    original = key_cache.clone()
    token_table = torch.tensor([[0, 0, 0, 1, 2, 3]], dtype=torch.int32, device="cuda")
    compressed_table = torch.tensor([[6, 7, 8]], dtype=torch.int32, device="cuda")
    full_compressed = torch.empty((3, 2, 6), device="cuda")

    compress_k_core_new(
        full_compressed,
        1,
        key_cache,
        token_table,
        compressed_table,
        torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
        torch.tensor([1], dtype=torch.int32, device="cuda"),
        torch.tensor([0, 3], dtype=torch.int32, device="cuda"),
        2,
        2,
        6,
    )

    expected = torch.stack(
        (
            original[6],
            original[0:2].mean(dim=0),
            original[2:4].mean(dim=0),
        )
    )
    torch.testing.assert_close(full_compressed, expected)
    torch.testing.assert_close(key_cache[7:9], expected[1:])


def test_fused_decode_topk_skips_dense_rows():
    kernel = fused_attn_pooling_online_topk_decode(
        batch_size=2,
        groups=16,
        heads=16,
        dim=128,
        topk=8,
        pooled_k_len=8,
        dense_len=5,
        dtype_str="bfloat16",
    )
    topk_indices = torch.full((1, 2, 8), -1, dtype=torch.int32, device="cuda")
    topk_values = torch.full(
        (1, 2, 8), float("-inf"), dtype=torch.float32, device="cuda"
    )

    kernel(
        torch.randn(32, 1, 128, dtype=torch.bfloat16, device="cuda"),
        torch.randn(4, 1, 128, dtype=torch.bfloat16, device="cuda"),
        torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"),
        torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda"),
        torch.tensor([3, 7], dtype=torch.int32, device="cuda"),
        topk_indices,
        topk_values,
    )

    assert torch.all(topk_indices[:, 0] == -1)
    assert torch.any(topk_indices[:, 1] >= 0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
