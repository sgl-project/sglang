# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.pi05_split_kv import split_kv_attention
from sglang.multimodal_gen.test.quality_tier_admission import (
    assert_error_no_worse_than_reference,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "batch,queries,prefix,kv_heads,width,dtype,mask_kind,num_splits",
    [
        (1, 50, 968, 1, 256, torch.bfloat16, "none", None),
        (2, 50, 800, 4, 256, torch.float16, "padding", None),
        (1, 7, 0, 1, 64, torch.bfloat16, "all", None),
        (1, 64, 968, 1, 128, torch.bfloat16, "padding", None),
        (1, 65, 968, 1, 128, torch.float16, "none", None),
        (1, 50, 800, 4, 64, torch.float16, "padding", 1),
        (1, 65, 32, 1, 128, torch.bfloat16, "all", 16),
    ],
)
def test_split_kv_matches_sdpa(
    dtype, batch, queries, prefix, kv_heads, width, num_splits, mask_kind
):
    torch.manual_seed(7)
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    q = torch.randn(batch, queries, 8, width, device=device, dtype=dtype).transpose(
        1, 2
    )
    pk, pv = [
        torch.randn(batch, kv_heads, prefix, width, device=device, dtype=dtype)
        for _ in range(2)
    ]
    sk, sv = [
        torch.randn(batch, kv_heads, queries, width, device=device, dtype=dtype)
        for _ in range(2)
    ]
    mask = None
    if mask_kind != "none":
        mask = torch.zeros(
            batch, 1, 1, prefix + queries, device=device, dtype=torch.float32
        )
        mask[..., ::3] = -float("inf")
        if mask_kind == "all":
            mask.fill_(-float("inf"))
    with torch.cuda.device(0), torch.no_grad():
        expected = F.scaled_dot_product_attention(
            q,
            torch.cat((pk, sk), dim=2).repeat_interleave(8 // kv_heads, dim=1),
            torch.cat((pv, sv), dim=2).repeat_interleave(8 // kv_heads, dim=1),
            attn_mask=None if mask is None else mask.to(dtype),
            scale=width**-0.5,
        )
        actual = split_kv_attention(
            q, pk, pv, sk, sv, mask, width**-0.5, num_splits=num_splits
        )
        assert torch.cuda.current_device() == 0
        reference = F.scaled_dot_product_attention(
            q.double(),
            torch.cat((pk, sk), dim=2).repeat_interleave(8 // kv_heads, dim=1).double(),
            torch.cat((pv, sv), dim=2).repeat_interleave(8 // kv_heads, dim=1).double(),
            attn_mask=None if mask is None else mask.double(),
            scale=width**-0.5,
        )
    assert torch.isfinite(actual).all()
    assert_error_no_worse_than_reference(
        reference_fp64=reference, baseline=expected, candidate=actual
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("queries", [50, 65])
def test_split_kv_graph_replay_uses_current_inputs(queries):
    tensors = [
        torch.randn(1, heads, length, 64, device="cuda", dtype=torch.bfloat16)
        for heads, length in [
            (4, queries),
            (1, 40),
            (1, 40),
            (1, queries),
            (1, queries),
        ]
    ]
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        split_kv_attention(*tensors, None, 0.125)
    stream.synchronize()
    with torch.no_grad(), torch.cuda.graph(graph, stream=stream):
        captured = split_kv_attention(*tensors, None, 0.125)
    tensors[0].add_(0.5)
    graph.replay()
    with torch.no_grad():
        expected = split_kv_attention(*tensors, None, 0.125)
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    tensors[0].requires_grad_(True)
    with pytest.raises(RuntimeError, match="inference-only"):
        split_kv_attention(*tensors, None, 0.125)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
