import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import mega_moe_pre_dispatch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _shared_sf_row(token_id: int, block_m: int) -> int:
    """DeepGEMM's MN-major row within the shared-L1 scale buffer."""
    aligned_block_m = (block_m + 127) // 128 * 128
    block_idx, m = divmod(token_id, block_m)
    transposed_m = (m // 128) * 128 + (m % 32) * 4 + (m % 128) // 32
    return block_idx * aligned_block_m + transposed_m


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Requires Blackwell GPU (sm_100+)",
)
def test_mxfp8_scale_output_uses_padded_row_stride() -> None:
    """Write each MXFP8 scale row without overwriting its physical padding."""
    torch.manual_seed(42)
    num_tokens, padded_max, hidden, top_k = 5, 8, 2304, 8
    num_groups = hidden // 32
    logical_scale_int32 = num_groups // 4
    scale_stride_int32 = 20
    marker = 0xA5

    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    topk_idx = (
        torch.arange(num_tokens * top_k, device="cuda", dtype=torch.int32)
        .reshape(num_tokens, top_k)
        .remainder(256)
    )
    topk_weights = torch.rand(num_tokens, top_k, device="cuda", dtype=torch.float32)

    buf_x = torch.empty(padded_max, hidden, device="cuda", dtype=torch.float8_e4m3fn)
    scale_bytes = torch.full(
        (padded_max, scale_stride_int32 * 4),
        marker,
        device="cuda",
        dtype=torch.uint8,
    )
    buf_x_sf = scale_bytes.view(torch.int32)[:, :logical_scale_int32]
    buf_topk_idx = torch.empty(padded_max, top_k, device="cuda", dtype=torch.int64)
    buf_topk_weights = torch.empty(
        padded_max, top_k, device="cuda", dtype=torch.float32
    )

    assert buf_x_sf.shape == (padded_max, logical_scale_int32)
    assert buf_x_sf.stride() == (scale_stride_int32, 1)
    mega_moe_pre_dispatch(
        x,
        topk_idx,
        topk_weights,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
    )
    torch.cuda.synchronize()

    logical_scale_bytes = logical_scale_int32 * 4
    assert torch.all(scale_bytes[:num_tokens, :logical_scale_bytes] != marker)
    assert torch.all(scale_bytes[:num_tokens, logical_scale_bytes:] == marker)
    assert torch.all(scale_bytes[num_tokens:] == marker)
    torch.testing.assert_close(buf_topk_idx[:num_tokens], topk_idx.to(torch.int64))
    torch.testing.assert_close(buf_topk_weights[:num_tokens], topk_weights)
    assert torch.all(buf_topk_idx[num_tokens:] == -1)
    assert torch.all(buf_topk_weights[num_tokens:] == 0)


@pytest.mark.parametrize("num_tokens,block_m", [(31, 32), (65, 64), (159, 128)])
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Requires Blackwell GPU (sm_100+)",
)
def test_mxfp8_scale_output_writes_deepgemm_shared_layout(
    num_tokens: int, block_m: int
) -> None:
    """Emit row-major and native shared-expert SF layouts in one quant pass."""
    torch.manual_seed(42)
    padded_max, hidden, top_k = 192, 2304, 8
    num_groups = hidden // 32
    packed_sf_k = num_groups // 4
    packed_sf_stride = (packed_sf_k + 3) // 4 * 4
    aligned_block_m = (block_m + 127) // 128 * 128
    num_sf_rows = ((padded_max + block_m - 1) // block_m) * aligned_block_m
    marker = -1515870811  # 0xA5A5A5A5 interpreted as signed int32

    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    topk_idx = torch.zeros(num_tokens, top_k, device="cuda", dtype=torch.int32)
    topk_weights = torch.ones(num_tokens, top_k, device="cuda", dtype=torch.float32)
    buf_x = torch.empty(padded_max, hidden, device="cuda", dtype=torch.float8_e4m3fn)
    buf_x_sf = torch.empty_strided(
        (padded_max, packed_sf_k),
        (packed_sf_stride, 1),
        device="cuda",
        dtype=torch.int32,
    )
    shared_sf = torch.empty_strided(
        (num_sf_rows, packed_sf_k),
        (1, num_sf_rows),
        device="cuda",
        dtype=torch.int32,
    )
    shared_sf.fill_(marker)
    buf_topk_idx = torch.empty(padded_max, top_k, device="cuda", dtype=torch.int64)
    buf_topk_weights = torch.empty(
        padded_max, top_k, device="cuda", dtype=torch.float32
    )

    mega_moe_pre_dispatch(
        x,
        topk_idx,
        topk_weights,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
        shared_l1_acts_sf=shared_sf,
        shared_block_m=block_m,
    )
    torch.cuda.synchronize()

    expected = torch.full_like(shared_sf, marker)
    for token_id in range(num_tokens):
        expected[_shared_sf_row(token_id, block_m)].copy_(buf_x_sf[token_id])
    torch.testing.assert_close(shared_sf, expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
