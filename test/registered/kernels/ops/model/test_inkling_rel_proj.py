"""rel_proj_small_t: the single-launch small-t rel projection (optional tau
prescale folded in registers) must match the reference chains it replaces --
{r*tau -> bf16 round -> projection} -- within bf16 GEMM rounding, on both the
production strided-r layout and contiguous inputs."""

import pytest
import torch

from sglang.kernels.ops.model.inkling.inkling_rel_proj import rel_proj_small_t
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

H, K, E, ROW = 16, 16, 1024, 2816


def _make_r(t, strided):
    torch.manual_seed(t + int(strided))
    if strided:
        qkvr = torch.randn(t, ROW, device="cuda", dtype=torch.bfloat16)
        return qkvr[:, ROW - H * K :].view(t, H, K)
    return torch.randn(t, H, K, device="cuda", dtype=torch.bfloat16)


def _ref(r, proj, tau):
    rf = r.float()
    if tau is not None:
        # The prescale contract: r*tau rounds to bf16 BEFORE the dot.
        rf = (rf * tau.view(-1, 1, 1)).bfloat16().float()
    return torch.einsum("thd,de->the", rf, proj.float())


@pytest.mark.parametrize("t", [1, 2, 5, 16, 32])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("with_tau", [False, True])
def test_rel_proj_small_t(t, strided, with_tau):
    r = _make_r(t, strided)
    proj = torch.randn(K, E, device="cuda", dtype=torch.bfloat16) * 0.1
    tau = (
        1.0 + 0.1 * torch.rand(t, device="cuda", dtype=torch.float32)
        if with_tau
        else None
    )
    out = rel_proj_small_t(r, proj, tau)
    assert out.is_contiguous() and out.shape == (t, H, E)
    ref = _ref(r, proj, tau)
    # fp32 accumulation, one bf16 round -- match the fp32 reference to
    # 2 bf16 ulp (the GEMM reduction-order slack vs cuBLAS is within this).
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_rel_proj_tau_isolation():
    """tau must only scale: kernel(tau) == kernel(no tau) computed on the
    pre-rounded r*tau operand -- guards the round-before-dot placement (a
    fold AFTER the dot would diverge at large |logits|)."""
    t = 8
    r = _make_r(t, True)
    proj = torch.randn(K, E, device="cuda", dtype=torch.bfloat16) * 0.1
    tau = 1.0 + 0.5 * torch.rand(t, device="cuda", dtype=torch.float32)
    out = rel_proj_small_t(r, proj, tau)
    r_pre = (r.float() * tau.view(-1, 1, 1)).bfloat16()
    assert torch.equal(out, rel_proj_small_t(r_pre.contiguous(), proj))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
