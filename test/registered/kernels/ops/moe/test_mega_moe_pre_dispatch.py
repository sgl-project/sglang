import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import mega_moe_pre_dispatch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
