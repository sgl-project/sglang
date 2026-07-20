from __future__ import annotations

import pytest
import torch

from sglang.jit_kernel.tests.deepseek_v4.common import make_paged_context, to_seq_extend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
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


def _valid_plan_rows(plan: torch.Tensor) -> torch.Tensor:
    rows = plan.contiguous().cpu().view(torch.int32)
    return rows[rows[:, 0] != -1]


def _compute_state_loc(swa_loc: int, *, ring_size: int) -> int:
    return (swa_loc // PAGE) * ring_size + swa_loc % ring_size


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


@pytest.mark.parametrize(
    "boundary",
    [
        pytest.param(torch.tensor([PAGE], dtype=torch.int64), id="wrong-length"),
        pytest.param(torch.tensor([PAGE, PAGE], dtype=torch.int32), id="wrong-dtype"),
        pytest.param(
            torch.tensor([[PAGE], [PAGE]], dtype=torch.int64), id="wrong-rank"
        ),
    ],
)
def test_boundary_metadata_is_validated(boundary: torch.Tensor) -> None:
    seq_extend = [(2 * PAGE, PAGE), (3 * PAGE, PAGE)]
    ctx = make_paged_context(bs=len(seq_extend), compress_ratio=4, head_dim=HEAD_DIM)
    seq_lens_cpu, extend_lens_cpu, num_q = to_seq_extend(seq_extend)

    with pytest.raises(RuntimeError):
        ctx.make_prefill_plan(
            seq_lens_cpu,
            extend_lens_cpu,
            num_q,
            recompute_boundary=boundary,
        )


def test_boundary_device_must_match_seq_lens() -> None:
    seq_extend = [(2 * PAGE, PAGE), (3 * PAGE, PAGE)]
    ctx = make_paged_context(bs=len(seq_extend), compress_ratio=4, head_dim=HEAD_DIM)
    seq_lens_cpu, extend_lens_cpu, num_q = to_seq_extend(seq_extend)
    boundary = torch.tensor([PAGE, PAGE], dtype=torch.int64, device="cuda")

    with pytest.raises(RuntimeError):
        ctx.make_prefill_plan(
            seq_lens_cpu,
            extend_lens_cpu,
            num_q,
            recompute_boundary=boundary,
        )


def test_prefill1_uses_override_for_current_extend_reads_and_writes() -> None:
    compress_ratio = 4
    seq_extend = [(16, 8), (19, 8)]
    ctx = make_paged_context(
        bs=len(seq_extend), compress_ratio=compress_ratio, head_dim=HEAD_DIM
    )
    seq_lens_cpu, extend_lens_cpu, num_q = to_seq_extend(seq_extend)
    extend_start_loc = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    override = 300 + 9 * torch.arange(num_q, dtype=torch.int32, device="cuda")

    plan = ctx.make_prefill_plan(
        seq_lens_cpu,
        extend_lens_cpu,
        num_q,
        swa_out_cache_loc_override=override,
        extend_start_loc=extend_start_loc,
    )

    plan_c_by_ragged = {
        int(row[1].item()) & 0xFFFF: row for row in _valid_plan_rows(plan.plan_c)
    }

    # Request 0 compresses at local_j == cr - 1, so read_page_0 still comes
    # from the persistent mapping while read_page_1 uses the current override.
    before_boundary = plan_c_by_ragged[compress_ratio - 1]
    assert int(before_boundary[2].item()) == ctx.state_loc(0, 7) // compress_ratio
    assert (
        int(before_boundary[3].item())
        == _compute_state_loc(
            int(override[compress_ratio - 1].item()), ring_size=ctx.ring_size
        )
        // compress_ratio
    )

    # Request 1 compresses at local_j == cr, so both endpoints of the read
    # window are already in its private override range.
    at_boundary_ragged = int(extend_start_loc[1].item()) + compress_ratio
    at_boundary = plan_c_by_ragged[at_boundary_ragged]
    assert (
        int(at_boundary[2].item())
        == _compute_state_loc(
            int(override[extend_start_loc[1]].item()), ring_size=ctx.ring_size
        )
        // compress_ratio
    )
    assert (
        int(at_boundary[3].item())
        == _compute_state_loc(
            int(override[at_boundary_ragged].item()), ring_size=ctx.ring_size
        )
        // compress_ratio
    )

    for ragged_id, write_loc in _valid_plan_rows(plan.plan_w).tolist():
        assert write_loc == _compute_state_loc(
            int(override[ragged_id].item()), ring_size=ctx.ring_size
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
