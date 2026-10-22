"""Small numerical probes, no import-only coverage or print-only failures."""

import sys
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_ppu_ci

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from registered.ppu import load_legacy_ppu_module  # noqa: E402

test_topk_alignment = load_legacy_ppu_module(
    "unit/test_topk_prefill_bf16_alignment.py"
).test_topk_alignment

register_ppu_ci(est_time=90, suite="stage-b-kernel-test-1-gpu-ppu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm(dtype):
    from sgl_kernel import rmsnorm

    torch.manual_seed(42)
    x = torch.randn(4, 256).to(dtype)
    w = torch.randn(256).to(dtype)
    ref = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        * w.float()
    )
    actual = rmsnorm(x.cuda(), w.cuda(), eps=1e-6, enable_pdl=False)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual.cpu().float(),
        ref,
        atol=0.025 if dtype == torch.bfloat16 else 1e-5,
        rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary(dtype):
    from sglang.kernels.ops.attention.rotary_triton import triton_mrope_fused

    torch.manual_seed(43)
    q, k = torch.randn(4, 128).to(dtype), torch.randn(4, 64).to(dtype)
    angles = torch.arange(8 * 32).reshape(8, 32).float() / 257
    cache = torch.cat([angles.cos(), angles.sin()], dim=1).to(dtype)
    positions = torch.tensor([0, 1, 3, 7])
    cos, sin = cache[positions].float().chunk(2, dim=-1)

    def reference(x):
        left, right = x.reshape(4, -1, 64).float().chunk(2, dim=-1)
        return torch.cat(
            [
                left * cos[:, None] - right * sin[:, None],
                right * cos[:, None] + left * sin[:, None],
            ],
            dim=-1,
        ).reshape_as(x)

    qd, kd = q.cuda(), k.cuda()
    triton_mrope_fused(
        qd,
        kd,
        cache.cuda(),
        positions.repeat(3, 1).cuda(),
        [16, 8, 8],
        64,
        64,
        False,
        False,
        True,
        None,
    )
    torch.cuda.synchronize()
    for actual, expected in [(qd, reference(q)), (kd, reference(k))]:
        torch.testing.assert_close(
            actual.cpu().float(),
            expected,
            atol=0.025 if dtype == torch.bfloat16 else 1e-5,
            rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sampling(dtype):
    from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch

    # Different known winners per row; top-k=1 has an exact, nonrandom reference.
    probs = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]], dtype=dtype, device="cuda")
    result = top_k_top_p_min_p_sampling_from_probs_torch(
        probs,
        torch.ones(2, dtype=torch.int32, device="cuda"),
        torch.ones(2, device="cuda"),
        torch.zeros(2, device="cuda"),
        False,
        None,
        torch.zeros(2, dtype=torch.int64, device="cuda"),
    )
    torch.cuda.synchronize()
    assert result.cpu().tolist() == [1, 0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
