"""Execution test for the CUDA raw-fp8 tilelang DSA sparse path.

Validated on sm_121 (DGX Spark GB10); requires SM89+ for fp8 tensor-core MMA.
One-hot construction makes the sparse attention output exactly recoverable, so
the fp8 path must match a bf16 reference to tight tolerance; a scrambled-index
negative control must NOT match (guards against the test passing vacuously).
"""

import pytest
import torch

tilelang_kernel = pytest.importorskip(
    "sglang.kernels.ops.attention.dsa.tilelang_kernel"
)

requires_fp8_cuda = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() < (8, 9),
    reason="needs CUDA SM89+ for fp8 tensor-core MMA",
)

H, DV, TOPK, SKV = 32, 512, 128, 4096  # GLM-5.3-Flash per-rank geometry, NoPE


def _one_hot_case(seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(1, 1, H, DV, device="cuda", generator=g, dtype=torch.bfloat16)
    kv = torch.randn(1, SKV, 1, DV, device="cuda", generator=g, dtype=torch.bfloat16)
    # One hot: q row i attends overwhelmingly to kv row target[i] by scaling.
    target = torch.randint(0, TOPK, (H,), device="cuda", generator=g)
    indices = torch.randperm(SKV, device="cuda", generator=g)[:TOPK]
    indices = indices.view(1, 1, TOPK).int()
    for h in range(H):
        q[0, 0, h] = 100.0 * kv[0, indices[0, 0, target[h]].long(), 0]
    expected = kv[0, indices[0, 0, target.long()].long(), 0].float()
    return q, kv, indices, expected


@requires_fp8_cuda
def test_fp8_one_hot_matches_reference():
    q, kv, indices, expected = _one_hot_case(0)
    out = tilelang_kernel.tilelang_sparse_fwd(
        q.to(torch.float8_e4m3fn), kv.to(torch.float8_e4m3fn), indices
    )
    got = out.view(H, DV).float()
    rel = (got - expected).abs().max() / expected.abs().max()
    assert rel < 0.08, f"fp8 one-hot rel err {rel:.4f} exceeds tolerance"


@requires_fp8_cuda
def test_fp8_negative_control_scrambled_indices_fails():
    q, kv, indices, expected = _one_hot_case(1)
    scrambled = indices.flip(-1).contiguous()
    out = tilelang_kernel.tilelang_sparse_fwd(
        q.to(torch.float8_e4m3fn), kv.to(torch.float8_e4m3fn), scrambled
    )
    got = out.view(H, DV).float()
    rel = (got - expected).abs().max() / expected.abs().max()
    assert rel > 0.5, (
        "scrambled indices still matched the reference — the harness is not "
        f"actually testing the sparse path (rel err {rel:.4f})"
    )
