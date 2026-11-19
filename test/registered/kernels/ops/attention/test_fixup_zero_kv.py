"""Empty-KV rows and the flat launch grid must survive long prefill batches."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize(
    ("batch_size", "queries_per_seq"),
    [
        pytest.param(0, 1, id="empty-batch"),
        pytest.param(65535, 1, id="old-grid-y-limit"),
        pytest.param(65536, 1, id="past-grid-y-limit"),
        pytest.param(70000, 1, id="beyond-grid-y-limit"),
        pytest.param(65536, 3, id="multiple-query-tokens"),
    ],
)
def test_fixup_rows_across_grid_y_limit(batch_size, queries_per_seq):
    kv_lens = torch.arange(batch_size, device="cuda", dtype=torch.int32) % 3
    cu_seqlens_q = (
        torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * queries_per_seq
    )
    zero_rows = (kv_lens == 0).repeat_interleave(queries_per_seq)

    num_tokens = batch_size * queries_per_seq
    out = (
        torch.arange(num_tokens * 64, device="cuda")
        .view(num_tokens, 4, 16)
        .to(torch.bfloat16)
    )
    lse = torch.arange(num_tokens * 4, device="cuda", dtype=torch.float32).view(
        num_tokens, 4
    )
    expected_out = out.clone()
    expected_lse = lse.clone()
    expected_out[zero_rows] = 0
    expected_lse[zero_rows] = -torch.inf

    fixup_zero_kv_rows(out, lse, kv_lens, cu_seqlens_q, queries_per_seq)

    # Checks the tail sequence beyond 65,535 and leaves positive-KV rows intact.
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(lse, expected_lse, rtol=0, atol=0)


def test_fixup_mla_width_offset_boundary():
    """MLA-wide zero-KV rows must not corrupt large flat output offsets."""
    num_tokens = 65538
    rows = [65534, 65535, 65536, 65537]
    kv_lens = torch.ones(num_tokens, device="cuda", dtype=torch.int32)
    kv_lens[65535:65537] = 0
    cu_seqlens_q = torch.arange(num_tokens + 1, device="cuda", dtype=torch.int32)

    out = torch.empty((num_tokens, 64, 512), device="cuda", dtype=torch.bfloat16)
    lse = torch.empty((num_tokens, 64), device="cuda", dtype=torch.float32)
    initial_out = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    initial_lse = torch.tensor([5.5, 6.5, 7.5, 8.5], dtype=torch.float32)
    out[rows] = initial_out[:, None, None].cuda()
    lse[rows] = initial_lse[:, None].cuda()

    # The second zero-KV row begins at 65536 * 64 * 512 == 2**31 elements.
    fixup_zero_kv_rows(out, lse, kv_lens, cu_seqlens_q, 1)
    torch.cuda.synchronize()

    expected_out = initial_out.clone()
    expected_out[1:3] = 0
    expected_lse = initial_lse.clone()
    expected_lse[1:3] = -torch.inf
    torch.testing.assert_close(
        out[rows].cpu(), expected_out[:, None, None].expand(4, 64, 512), rtol=0, atol=0
    )
    torch.testing.assert_close(
        lse[rows].cpu(), expected_lse[:, None].expand(4, 64), rtol=0, atol=0
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
