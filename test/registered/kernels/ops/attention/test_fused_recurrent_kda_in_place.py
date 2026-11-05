"""``fused_recurrent_kda`` with ``ssm_state_indices`` must update the state pool in place
and match the same kernel run on a gathered copy of the state."""

import pytest
import torch

from sglang.kernels.ops.attention.fla.kda import fused_recurrent_kda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@pytest.mark.parametrize("d", [64, 128])
def test_fused_recurrent_kda_in_place_matches_packed_path(d: int):
    torch.manual_seed(0)
    num_slots, h = 6, 3
    # Served pools interleave other state per slot, so the slot stride is not h * d * d.
    raw = torch.randn(num_slots, 2 * h * d * d, device="cuda")
    pool = raw[:, : h * d * d].view(num_slots, h, d, d)
    # A strided view of the index buffer; ignoring the stride would read the decoy slot 5.
    slots = torch.tensor([4, 5, -1, 5, 0, 5], device="cuda", dtype=torch.int32)[::2]
    n = slots.numel()
    q = torch.nn.functional.normalize(
        torch.randn(n, 1, h, d, device="cuda"), dim=-1
    ).bfloat16()
    k = torch.nn.functional.normalize(
        torch.randn(n, 1, h, d, device="cuda"), dim=-1
    ).bfloat16()
    v = torch.randn(n, 1, h, d, device="cuda").bfloat16()
    g = -torch.rand(n, 1, h, d, device="cuda")
    beta = torch.rand(n, 1, h, device="cuda")

    gathered = pool[slots.clamp_min(0)].masked_fill((slots < 0).view(n, 1, 1, 1), 0)
    expected_out, expected_state = fused_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        scale=d**-0.5,
        initial_state=gathered,
        inplace_final_state=False,
    )
    before = raw.clone()
    out, _ = fused_recurrent_kda(
        q, k, v, g, beta, scale=d**-0.5, initial_state=pool, ssm_state_indices=slots
    )
    assert torch.equal(out, expected_out)
    assert torch.equal(pool[[4, 0]], expected_state[[0, 2]])
    untouched = torch.ones(num_slots, dtype=torch.bool, device="cuda")
    untouched[[4, 0]] = False
    assert torch.equal(raw[untouched], before[untouched])
    assert torch.equal(raw[:, h * d * d :], before[:, h * d * d :])


if __name__ == "__main__":
    pytest.main([__file__])
