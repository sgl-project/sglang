"""Math behind --dcp-replicate-q-proj on the owned DSA DCP route (ROCm).

Project-before-merge: V-projecting every partial and then LSE-merging equals
merging the partials and then V-projecting, because the merge is linear in the
partial output.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")

if not torch.cuda.is_available():
    pytest.skip("GPU required", allow_module_level=True)

H = 64
KV_LORA_RANK = 512
V_HEAD_DIM = 256


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("N", [2, 4])
@pytest.mark.parametrize("B", [3, 144])
def test_project_before_merge_matches_merge_then_project(N, B):
    from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton

    torch.manual_seed(2)
    H_local = H // N
    partials = torch.randn(N, B, H_local, KV_LORA_RANK, device="cuda").bfloat16()
    lses = torch.randn(N, B, H_local, device="cuda") * 3
    w_vc = (
        torch.randn(H_local, KV_LORA_RANK, V_HEAD_DIM, device="cuda") * 0.04
    ).bfloat16()

    merged, _ = dcp_lse_combine_triton(partials, lses, is_lse_base_on_e=False)
    ref = torch.bmm(merged.transpose(0, 1).float(), w_vc.float()).transpose(0, 1)

    projected = partials.new_empty(N, B, H_local, V_HEAD_DIM)
    for i in range(N):
        # Same expression as _dcp_project_partials: token-major out view.
        torch.bmm(partials[i].transpose(0, 1), w_vc, out=projected[i].transpose(0, 1))
    pbm, _ = dcp_lse_combine_triton(projected, lses, is_lse_base_on_e=False)

    assert pbm.shape == (B, H_local, V_HEAD_DIM)
    assert _rel_err(pbm, ref) < 1e-2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
