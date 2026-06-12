from __future__ import annotations

import pytest
import torch

from sglang.jit_kernel.tests.deepseek_v4.common import make_paged_context, to_seq_extend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=20, suite="nightly-kernel-1-gpu", nightly=True)

HEAD_DIM = 512
PAGE = 256
W_R = 256

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="compress-plan JIT kernel requires CUDA"
)


def _count_valid(plan: torch.Tensor) -> int:
    """Count non-invalid PlanC/PlanW rows."""
    assert plan.dim() == 2 and plan.dtype == torch.uint8
    head = plan[:, :4].contiguous().cpu().view(torch.int32).flatten()
    return int((head != -1).sum().item())


@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_default_boundary_is_byte_identical(compress_ratio: int) -> None:
    seq_extend = [(2 * PAGE, PAGE), (3 * PAGE, PAGE)]
    ctx = make_paged_context(
        bs=len(seq_extend), compress_ratio=compress_ratio, head_dim=HEAD_DIM
    )
    seq_lens_cpu, extend_lens_cpu, num_q = to_seq_extend(seq_extend)

    plan_a = ctx.make_prefill_plan(seq_lens_cpu, extend_lens_cpu, num_q)
    plan_b = ctx.make_prefill_plan(
        seq_lens_cpu, extend_lens_cpu, num_q, recompute_boundary=None
    )

    assert torch.equal(plan_a.plan_c, plan_b.plan_c)
    assert torch.equal(plan_a.plan_w, plan_b.plan_w)


@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_boundary_equal_to_prefix_len_suppresses_nothing(compress_ratio: int) -> None:
    seq_extend = [(2 * PAGE, PAGE), (3 * PAGE, PAGE)]
    ctx = make_paged_context(
        bs=len(seq_extend), compress_ratio=compress_ratio, head_dim=HEAD_DIM
    )
    seq_lens_cpu, extend_lens_cpu, num_q = to_seq_extend(seq_extend)
    prefix_lens = [s - e for s, e in seq_extend]
    boundary = torch.tensor(prefix_lens, dtype=torch.int64)

    base = ctx.make_prefill_plan(seq_lens_cpu, extend_lens_cpu, num_q)
    gated = ctx.make_prefill_plan(
        seq_lens_cpu, extend_lens_cpu, num_q, recompute_boundary=boundary
    )

    assert _count_valid(gated.plan_c) == _count_valid(base.plan_c)
    assert _count_valid(gated.plan_w) == _count_valid(base.plan_w)


@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_boundary_suppresses_window_compress_but_keeps_writes(
    compress_ratio: int,
) -> None:
    P = 4 * PAGE
    new = PAGE
    R = W_R
    prefix = P - R
    seq = P + new
    ext = seq - prefix
    ctx = make_paged_context(bs=1, compress_ratio=compress_ratio, head_dim=HEAD_DIM)
    seq_lens_cpu, extend_lens_cpu, num_q = to_seq_extend([(seq, ext)])

    base = ctx.make_prefill_plan(seq_lens_cpu, extend_lens_cpu, num_q)
    boundary = torch.tensor([P], dtype=torch.int64)
    gated = ctx.make_prefill_plan(
        seq_lens_cpu, extend_lens_cpu, num_q, recompute_boundary=boundary
    )

    base_c, gated_c = _count_valid(base.plan_c), _count_valid(gated.plan_c)
    base_w, gated_w = _count_valid(base.plan_w), _count_valid(gated.plan_w)

    assert gated_c < base_c
    assert gated_w == base_w


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
