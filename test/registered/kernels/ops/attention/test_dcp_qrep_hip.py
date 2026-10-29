"""Math behind --dcp-replicate-q-proj on the owned DSA DCP route (ROCm).

Covers:
1. The token-major absorb / V-projection: ``torch.bmm`` written through a
   transposed ``out`` view equals a head-major bmm, also into a strided slice.
2. Project-before-merge: V-projecting every partial and then LSE-merging equals
   merging the partials and then V-projecting.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")

if not torch.cuda.is_available():
    pytest.skip("GPU required", allow_module_level=True)

H = 64
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 192
V_HEAD_DIM = 256


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _token_major_bmm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    torch.bmm(x.transpose(0, 1), w, out=out.transpose(0, 1))


@pytest.mark.parametrize("T", [1, 24, 144, 1024])
@pytest.mark.parametrize(
    "K,N", [(QK_NOPE_HEAD_DIM, KV_LORA_RANK), (KV_LORA_RANK, V_HEAD_DIM)]
)
def test_token_major_bmm_matches_head_major(T, K, N):
    torch.manual_seed(0)
    x = torch.randn(T, H, K, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(H, K, N, device="cuda") * K**-0.5).bfloat16()
    ref = torch.bmm(x.transpose(0, 1).float(), w.float()).transpose(0, 1)

    out = x.new_empty(T, H, N)
    _token_major_bmm(x, w, out)

    assert _rel_err(out, ref) < 1e-2


def test_token_major_bmm_into_strided_slice():
    torch.manual_seed(1)
    T = 24
    x = torch.randn(T, H, QK_NOPE_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    w = (
        torch.randn(H, QK_NOPE_HEAD_DIM, KV_LORA_RANK, device="cuda") * 0.07
    ).bfloat16()
    # The absorb output can be the nope slice of a [T, H, 576] query buffer.
    buf = torch.zeros(T, H, KV_LORA_RANK + 64, device="cuda", dtype=torch.bfloat16)
    _token_major_bmm(x, w, buf[..., :KV_LORA_RANK])

    ref = torch.bmm(x.transpose(0, 1), w).transpose(0, 1)
    assert _rel_err(buf[..., :KV_LORA_RANK], ref) < 1e-2
    assert buf[..., KV_LORA_RANK:].abs().max().item() == 0


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
        _token_major_bmm(partials[i], w_vc, projected[i])
    pbm, _ = dcp_lse_combine_triton(projected, lses, is_lse_base_on_e=False)

    assert pbm.shape == (B, H_local, V_HEAD_DIM)
    assert _rel_err(pbm, ref) < 1e-2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
