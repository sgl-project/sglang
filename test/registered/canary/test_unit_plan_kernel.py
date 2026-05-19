"""Differential test for the canary plan Triton kernel.

For each scenario, the Python reference (``plan_batch_from_forward_batch``
in ``kv_cache_canary_plan_ref``) and the Triton wrapper
(``plan_batch_from_forward_batch_triton`` in ``kv_cache_canary_plan``)
are both driven against the same synthetic ``ForwardBatch``. The ref's
host-side ``BatchPlan`` is projected into a ``BatchPlanGpu`` via the
shared ``fill_batch_plan_gpu_from_plan`` helper so the two outputs are
directly comparable. Every field of the two ``BatchPlanGpu`` instances
is then asserted byte-equal with ``torch.equal`` (no atol).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pytest
import torch

from sglang.jit_kernel.kv_cache_canary_plan import (
    plan_batch_from_forward_batch_triton,
)
from sglang.jit_kernel.kv_cache_canary_plan_ref import (
    BatchPlanGpu,
    allocate_batch_plan_gpu,
    fill_batch_plan_gpu_from_plan,
    plan_batch_from_forward_batch,
    reset_batch_plan_gpu_to_skip_sentinel,
)
from sglang.srt.kv_cache_canary.config import CanaryConfig, CanaryMode
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="extra-a", runner_config="1-gpu-small")


_DEVICE = "cuda"

_VERIFY_CAPACITY: int = 4096
_WRITE_CAPACITY: int = 256
_WRITE_REQ_CAPACITY: int = 32

_PLAN_FIELDS = (
    "verify_slot_indices",
    "verify_positions",
    "verify_prev_slot_indices",
    "write_slot_indices",
    "write_token_ids",
    "write_positions",
    "expected_write_token_ids",
    "expected_write_positions",
    "write_req_seed_slot_indices",
    "write_req_entry_starts",
    "write_req_entry_counts",
    "verify_num_valid",
    "write_req_num_valid",
)


@dataclass
class _FakeReqToTokenPool:
    """Minimal stand-in exposing the ``req_to_token`` attribute the planner reads."""

    req_to_token: torch.Tensor


@dataclass
class _FakeForwardBatch:
    """Duck-typed ``ForwardBatch`` with only the fields the plan code touches."""

    req_pool_indices: torch.Tensor
    input_ids: torch.Tensor
    positions: torch.Tensor
    out_cache_loc: torch.Tensor
    seq_lens: torch.Tensor
    forward_mode: ForwardMode
    req_to_token_pool: _FakeReqToTokenPool
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_prefix_lens: Optional[torch.Tensor] = None
    num_token_non_padded_cpu: Optional[int] = None


@dataclass
class _ScenarioSpec:
    """Recipe for synthesising one differential-test scenario.

    Fields are deliberately concrete (int / list[int]) so each scenario reads
    like a tiny hand-traced example; the helper builds the corresponding
    tensors + ``ForwardBatch`` stub from them.
    """

    name: str
    req_pool_indices: List[int]
    prefix_lens: List[int]
    extend_seq_lens: List[int]
    forward_mode: ForwardMode
    swa_window_size: Optional[int] = None
    has_swa_lut: bool = False
    swa_window_full_indices: List[int] = field(default_factory=list)
    num_token_non_padded_cpu: Optional[int] = None
    max_seq_len: int = 128
    full_pool_size: int = 256


def _build_req_to_token(
    *,
    bs: int,
    max_seq_len: int,
    req_pool_indices: List[int],
    prefix_lens: List[int],
    extend_seq_lens: List[int],
    base_slot: int = 100,
) -> torch.Tensor:
    """Deterministic ``req_to_token`` table covering each req's full footprint.

    Row 0 is the padding row (kept zero). Other rows hold a strictly increasing
    band of slot indices so per-position lookups are easy to reason about.
    """
    table = torch.zeros((bs + 4, max_seq_len), dtype=torch.int32)
    cursor = base_slot
    for r_idx, k_req, n in zip(req_pool_indices, prefix_lens, extend_seq_lens):
        if r_idx == 0:
            continue
        total = k_req + n
        for j in range(total):
            table[r_idx, j] = cursor
            cursor += 1
    return table


def _build_forward_batch(spec: _ScenarioSpec) -> _FakeForwardBatch:
    bs = len(spec.req_pool_indices)
    req_pool_indices = torch.tensor(
        spec.req_pool_indices, dtype=torch.int64, device=_DEVICE
    )
    if spec.forward_mode.is_extend() or spec.forward_mode.is_mixed():
        extend_seq_lens = torch.tensor(
            spec.extend_seq_lens, dtype=torch.int64, device=_DEVICE
        )
        extend_prefix_lens = torch.tensor(
            spec.prefix_lens, dtype=torch.int64, device=_DEVICE
        )
        seq_lens = torch.tensor(
            [p + n for p, n in zip(spec.prefix_lens, spec.extend_seq_lens)],
            dtype=torch.int64,
            device=_DEVICE,
        )
    else:
        extend_seq_lens = None
        extend_prefix_lens = None
        seq_lens = torch.tensor(
            [p + 1 for p in spec.prefix_lens], dtype=torch.int64, device=_DEVICE
        )

    total_tokens = sum(spec.extend_seq_lens)
    input_ids = torch.arange(
        1000, 1000 + total_tokens, dtype=torch.int64, device=_DEVICE
    )
    positions = torch.empty(total_tokens, dtype=torch.int64, device=_DEVICE)
    out_cache_loc = torch.empty(total_tokens, dtype=torch.int64, device=_DEVICE)
    cursor = 0
    out_slot_base = 500
    for r_idx, k_req, n in zip(
        spec.req_pool_indices, spec.prefix_lens, spec.extend_seq_lens
    ):
        for off in range(n):
            positions[cursor + off] = k_req + off
            out_cache_loc[cursor + off] = out_slot_base
            out_slot_base += 1
        cursor += n

    req_to_token = _build_req_to_token(
        bs=bs,
        max_seq_len=spec.max_seq_len,
        req_pool_indices=spec.req_pool_indices,
        prefix_lens=spec.prefix_lens,
        extend_seq_lens=spec.extend_seq_lens,
    ).to(_DEVICE)

    return _FakeForwardBatch(
        req_pool_indices=req_pool_indices,
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=out_cache_loc,
        seq_lens=seq_lens,
        forward_mode=spec.forward_mode,
        req_to_token_pool=_FakeReqToTokenPool(req_to_token=req_to_token),
        extend_seq_lens=extend_seq_lens,
        extend_prefix_lens=extend_prefix_lens,
        num_token_non_padded_cpu=spec.num_token_non_padded_cpu,
    )


def _build_swa_lut(*, full_size: int, window_full_indices: List[int]) -> torch.Tensor:
    """Build a ``[full_size + 1]`` LUT mapping in-window full slots to swa slots,
    everything else (including the trailing sentinel row) to ``-1``."""
    lut = torch.full((full_size + 1,), -1, dtype=torch.int64, device=_DEVICE)
    for swa_idx, full_idx in enumerate(window_full_indices):
        lut[full_idx] = swa_idx
    return lut


def _scenarios() -> List[_ScenarioSpec]:
    return [
        _ScenarioSpec(
            name="decode_basic",
            req_pool_indices=[1, 2, 3],
            prefix_lens=[4, 7, 1],
            extend_seq_lens=[1, 1, 1],
            forward_mode=ForwardMode.DECODE,
        ),
        _ScenarioSpec(
            name="extend_basic",
            req_pool_indices=[1, 2],
            prefix_lens=[0, 3],
            extend_seq_lens=[5, 4],
            forward_mode=ForwardMode.EXTEND,
        ),
        _ScenarioSpec(
            name="mixed_extend_decode",
            req_pool_indices=[1, 2, 3],
            prefix_lens=[6, 0, 9],
            extend_seq_lens=[1, 4, 1],
            forward_mode=ForwardMode.MIXED,
        ),
        _ScenarioSpec(
            name="swa_clipped_verify",
            req_pool_indices=[1, 2],
            prefix_lens=[20, 12],
            extend_seq_lens=[1, 1],
            forward_mode=ForwardMode.DECODE,
            swa_window_size=8,
            max_seq_len=64,
        ),
        _ScenarioSpec(
            name="swa_translate_full_to_swa",
            req_pool_indices=[1],
            prefix_lens=[4],
            extend_seq_lens=[2],
            forward_mode=ForwardMode.EXTEND,
            swa_window_size=4,
            has_swa_lut=True,
            # The four full-pool slots filled by _build_req_to_token start at
            # ``base_slot = 100`` and run [100, 101, 102, 103] for req_pool_idx 1.
            # The two new write slots come from out_cache_loc base 500.
            swa_window_full_indices=[100, 101, 102, 103, 500, 501],
            max_seq_len=32,
            full_pool_size=512,
        ),
        _ScenarioSpec(
            name="cuda_graph_padding_row",
            req_pool_indices=[1, 0, 2],
            prefix_lens=[3, 0, 5],
            extend_seq_lens=[1, 0, 1],
            forward_mode=ForwardMode.DECODE,
        ),
    ]


def _materialise_plan_ref(
    *,
    forward_batch: _FakeForwardBatch,
    config: CanaryConfig,
    swa_index_lut: Optional[torch.Tensor],
) -> BatchPlanGpu:
    """Run the Python ref then project into a ``BatchPlanGpu`` for comparison."""
    plan_gpu = allocate_batch_plan_gpu(
        device=torch.device(_DEVICE),
        verify_capacity=_VERIFY_CAPACITY,
        write_capacity=_WRITE_CAPACITY,
        write_req_capacity=_WRITE_REQ_CAPACITY,
    )
    plan_host = plan_batch_from_forward_batch(
        forward_batch=forward_batch, config=config, swa_index_lut=swa_index_lut
    )
    if plan_host is None:
        reset_batch_plan_gpu_to_skip_sentinel(plan_gpu)
        # Write-tile tail policy parity: triton wrapper also zeros the write
        # tile + sentinel-fills expected_write_*, so we mirror that here for
        # an apples-to-apples compare.
        plan_gpu.write_slot_indices.zero_()
        plan_gpu.write_token_ids.zero_()
        plan_gpu.write_positions.zero_()
        return plan_gpu
    fill_batch_plan_gpu_from_plan(launch=plan_gpu, plan=plan_host)
    return plan_gpu


def _materialise_plan_triton(
    *,
    forward_batch: _FakeForwardBatch,
    config: CanaryConfig,
    swa_index_lut: Optional[torch.Tensor],
) -> BatchPlanGpu:
    plan_gpu = allocate_batch_plan_gpu(
        device=torch.device(_DEVICE),
        verify_capacity=_VERIFY_CAPACITY,
        write_capacity=_WRITE_CAPACITY,
        write_req_capacity=_WRITE_REQ_CAPACITY,
    )
    plan_batch_from_forward_batch_triton(
        forward_batch=forward_batch,
        config=config,
        plan_out=plan_gpu,
        swa_index_lut=swa_index_lut,
    )
    return plan_gpu


def _assert_byte_equal(
    *, ref_plan: BatchPlanGpu, triton_plan: BatchPlanGpu, scenario: str
) -> None:
    nv = int(ref_plan.verify_num_valid.item())
    nwr = int(ref_plan.write_req_num_valid.item())
    nv_t = int(triton_plan.verify_num_valid.item())
    nwr_t = int(triton_plan.write_req_num_valid.item())
    assert nv == nv_t, f"{scenario}: verify_num_valid diverges ref={nv} triton={nv_t}"
    assert (
        nwr == nwr_t
    ), f"{scenario}: write_req_num_valid diverges ref={nwr} triton={nwr_t}"

    # Per-verify tile: compare only the active prefix; tail is unspecified.
    for field_name in (
        "verify_slot_indices",
        "verify_positions",
        "verify_prev_slot_indices",
    ):
        ref_tile = getattr(ref_plan, field_name)[:nv].cpu()
        triton_tile = getattr(triton_plan, field_name)[:nv].cpu()
        assert torch.equal(ref_tile, triton_tile), (
            f"{scenario}: field {field_name!r} active prefix diverges\n"
            f"  ref={ref_tile.tolist()}\n  triton={triton_tile.tolist()}"
        )

    # Per-write-req tile: compare only the active prefix.
    for field_name in (
        "write_req_seed_slot_indices",
        "write_req_entry_starts",
        "write_req_entry_counts",
    ):
        ref_tile = getattr(ref_plan, field_name)[:nwr].cpu()
        triton_tile = getattr(triton_plan, field_name)[:nwr].cpu()
        assert torch.equal(ref_tile, triton_tile), (
            f"{scenario}: field {field_name!r} active prefix diverges\n"
            f"  ref={ref_tile.tolist()}\n  triton={triton_tile.tolist()}"
        )

    # Per-write-entry tile: compare the full capacity. Both ref and triton
    # zero the tail past num_write and sentinel-fill expected_write_*.
    for field_name in (
        "write_slot_indices",
        "write_token_ids",
        "write_positions",
        "expected_write_token_ids",
        "expected_write_positions",
    ):
        ref_tile = getattr(ref_plan, field_name).cpu()
        triton_tile = getattr(triton_plan, field_name).cpu()
        assert torch.equal(ref_tile, triton_tile), (
            f"{scenario}: field {field_name!r} full-capacity tile diverges\n"
            f"  ref[:8]={ref_tile[:8].tolist()}\n  triton[:8]={triton_tile[:8].tolist()}"
        )


@pytest.fixture(scope="module")
def _cuda_required() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the plan kernel differential test")


@pytest.mark.parametrize("spec", _scenarios(), ids=lambda s: s.name)
def test_plan_triton_matches_ref(spec: _ScenarioSpec, _cuda_required: None) -> None:
    forward_batch = _build_forward_batch(spec)
    config = CanaryConfig(mode=CanaryMode.LOG, swa_window_size=spec.swa_window_size)
    swa_index_lut: Optional[torch.Tensor] = None
    if spec.has_swa_lut:
        swa_index_lut = _build_swa_lut(
            full_size=spec.full_pool_size,
            window_full_indices=spec.swa_window_full_indices,
        )

    ref_plan = _materialise_plan_ref(
        forward_batch=forward_batch, config=config, swa_index_lut=swa_index_lut
    )
    triton_plan = _materialise_plan_triton(
        forward_batch=forward_batch, config=config, swa_index_lut=swa_index_lut
    )

    _assert_byte_equal(ref_plan=ref_plan, triton_plan=triton_plan, scenario=spec.name)


def test_plan_triton_writes_int32_counters(_cuda_required: None) -> None:
    """``verify_num_valid`` and ``write_req_num_valid`` must remain int32 scalars
    so the downstream ``canary_step`` kernel reads them as int32 active counts."""
    spec = _ScenarioSpec(
        name="counter_dtype",
        req_pool_indices=[1, 2],
        prefix_lens=[3, 5],
        extend_seq_lens=[1, 1],
        forward_mode=ForwardMode.DECODE,
    )
    forward_batch = _build_forward_batch(spec)
    config = CanaryConfig(mode=CanaryMode.LOG)
    plan = _materialise_plan_triton(
        forward_batch=forward_batch, config=config, swa_index_lut=None
    )
    assert plan.verify_num_valid.dtype == torch.int32
    assert plan.write_req_num_valid.dtype == torch.int32
