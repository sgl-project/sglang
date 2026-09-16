"""C2 graph-padding rows must leave every destination untouched."""

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.c2 import c2_decode_norm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_padded_even_and_odd_rows_are_inert() -> None:
    num_tokens, head_dim, ring_size = 2, 512, 2
    kv_input = torch.randn(num_tokens, 2 * head_dim, device="cuda")
    kv_state = torch.randn(num_tokens * ring_size, 2 * head_dim, device="cuda")
    state_before = kv_state.clone()
    norm_weight = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda")
    positions = torch.tensor([2, 3], dtype=torch.int64, device="cuda")
    req = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    raw_out_loc = torch.zeros(num_tokens, dtype=torch.int32, device="cuda")
    out = torch.full((num_tokens, head_dim), 123, dtype=torch.bfloat16, device="cuda")
    out_before = out.clone()

    result = c2_decode_norm(
        kv_input,
        kv_state,
        norm_weight,
        positions,
        req,
        raw_out_loc,
        1e-6,
        ring_size=ring_size,
        out=out,
    )

    assert torch.equal(result, out_before)
    assert torch.equal(kv_state, state_before)
