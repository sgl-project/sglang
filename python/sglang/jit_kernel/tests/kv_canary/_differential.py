"""Differential (CUDA/Triton vs torch reference) run + byte-equal-assert helpers.

Three independent kernel-specific pairs live here — they are NOT unified because each kernel's input
surface and equality contract differs. Each pair is a verbatim move from the legacy per-kernel test
file (``test_plan.py`` / ``test_verify.py`` / ``test_write.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator, Optional

import torch

from sglang.jit_kernel.kv_canary.plan import canary_plan_step
from sglang.jit_kernel.kv_canary.plan_ref import (
    canary_plan_step_torch_reference,
)
from sglang.jit_kernel.kv_canary.verify import (
    CanaryLaunchTag,
    RealKvHashMode,
    RealKvSource,
    VerifyPlan,
    canary_verify_step,
)
from sglang.jit_kernel.kv_canary.verify_ref import (
    canary_verify_step_torch_reference,
)
from sglang.jit_kernel.kv_canary.write import (
    CanaryPseudoMode,
    WritePlan,
    canary_write_step,
)
from sglang.jit_kernel.kv_canary.write_ref import (
    canary_write_step_torch_reference,
)
from sglang.jit_kernel.tests.kv_canary.canary_helpers import (
    FakeViolationLog,
    assert_canary_buf_equal,
    assert_canary_state_equal,
)


def _run_both_plan(
    *,
    triton_verify: VerifyPlan,
    triton_write: WritePlan,
    ref_verify: VerifyPlan,
    ref_write: WritePlan,
    fb_req_pool_indices: torch.Tensor,
    fb_prefix_lens: torch.Tensor,
    fb_extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    extras: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
) -> None:
    extra_slots, extra_positions, extra_prev_slots, extra_num_valid = extras
    canary_plan_step(
        verify_plan_out=triton_verify,
        write_plan_out=triton_write,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extra_verify_slot_indices=extra_slots,
        extra_verify_positions=extra_positions,
        extra_verify_prev_slot_indices=extra_prev_slots,
        extra_verify_num_valid=extra_num_valid,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
    )
    canary_plan_step_torch_reference(
        verify_plan_out=ref_verify,
        write_plan_out=ref_write,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extra_verify_slot_indices=extra_slots,
        extra_verify_positions=extra_positions,
        extra_verify_prev_slot_indices=extra_prev_slots,
        extra_verify_num_valid=extra_num_valid,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
    )
    torch.cuda.synchronize()


def _run_both_and_assert_plan_byte_equal(
    *,
    triton_verify: VerifyPlan,
    triton_write: WritePlan,
    ref_verify: VerifyPlan,
    ref_write: WritePlan,
    fb_req_pool_indices: torch.Tensor,
    fb_prefix_lens: torch.Tensor,
    fb_extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    extras: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    active_verify_entries: Optional[int] = None,
    active_write_reqs: Optional[int] = None,
) -> None:
    _run_both_plan(
        triton_verify=triton_verify,
        triton_write=triton_write,
        ref_verify=ref_verify,
        ref_write=ref_write,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=extras,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
    )
    _assert_plans_byte_equal(
        triton_verify=triton_verify,
        triton_write=triton_write,
        ref_verify=ref_verify,
        ref_write=ref_write,
        active_verify_entries=active_verify_entries,
        active_write_reqs=active_write_reqs,
    )


def _assert_plans_byte_equal(
    *,
    triton_verify: VerifyPlan,
    triton_write: WritePlan,
    ref_verify: VerifyPlan,
    ref_write: WritePlan,
    active_verify_entries: Optional[int] = None,
    active_write_reqs: Optional[int] = None,
) -> None:
    """Byte-equal check on (Triton vs ref) plan outputs.

    Optional ``active_verify_entries`` / ``active_write_reqs`` truncate the comparison to the meaningful
    prefix; tail entries past the active count are kernel-undefined and need not match byte-equal.
    """
    n_verify = (
        active_verify_entries
        if active_verify_entries is not None
        else int(triton_verify.verify_num_valid[0].item())
    )
    n_verify_ref = int(ref_verify.verify_num_valid[0].item())
    assert (
        n_verify == n_verify_ref
    ), f"verify_num_valid diverged: triton={n_verify} ref={n_verify_ref}"
    if n_verify > 0:
        assert torch.equal(
            triton_verify.verify_slot_indices[:n_verify],
            ref_verify.verify_slot_indices[:n_verify],
        )
        assert torch.equal(
            triton_verify.verify_positions[:n_verify],
            ref_verify.verify_positions[:n_verify],
        )
        assert torch.equal(
            triton_verify.verify_prev_slot_indices[:n_verify],
            ref_verify.verify_prev_slot_indices[:n_verify],
        )

    n_write = (
        active_write_reqs
        if active_write_reqs is not None
        else int(triton_write.write_num_valid_reqs[0].item())
    )
    n_write_ref = int(ref_write.write_num_valid_reqs[0].item())
    assert (
        n_write == n_write_ref
    ), f"write_num_valid_reqs diverged: triton={n_write} ref={n_write_ref}"
    assert torch.equal(
        triton_write.write_offsets[: n_write + 1],
        ref_write.write_offsets[: n_write + 1],
    )
    if n_write > 0:
        assert torch.equal(
            triton_write.write_seed_slot_indices[:n_write],
            ref_write.write_seed_slot_indices[:n_write],
        )


def _run_both_verify(
    *,
    cuda_canary_buf: torch.Tensor,
    ref_canary_buf: torch.Tensor,
    plan_cuda,
    plan_ref,
    cuda_log: FakeViolationLog,
    ref_log: FakeViolationLog,
    real_kv_sources_cuda: tuple[RealKvSource, ...],
    real_kv_sources_ref: tuple[RealKvSource, ...],
    real_kv_hash_mode: RealKvHashMode,
    kernel_kind: CanaryLaunchTag = CanaryLaunchTag.HEAD_K_FULL,
) -> None:
    canary_verify_step(
        canary_buf=cuda_canary_buf,
        plan=plan_cuda,
        kernel_kind=kernel_kind,
        violation_ring=cuda_log.ring,
        violation_write_index=cuda_log.write_index,
        slot_run_counter=cuda_log.slot_run_counter,
        kernel_run_counter=cuda_log.kernel_run_counter,
        real_kv_sources=real_kv_sources_cuda,
        real_kv_hash_mode=real_kv_hash_mode,
    )
    canary_verify_step_torch_reference(
        canary_buf=ref_canary_buf,
        plan=plan_ref,
        kernel_kind=kernel_kind,
        violation_ring=ref_log.ring,
        violation_write_index=ref_log.write_index,
        slot_run_counter=ref_log.slot_run_counter,
        kernel_run_counter=ref_log.kernel_run_counter,
        real_kv_sources=real_kv_sources_ref,
        real_kv_hash_mode=real_kv_hash_mode,
    )
    torch.cuda.synchronize()


def _run_both_and_assert_verify_state_equal(
    *,
    cuda_canary_buf: torch.Tensor,
    ref_canary_buf: torch.Tensor,
    plan_cuda,
    plan_ref,
    cuda_log: FakeViolationLog,
    ref_log: FakeViolationLog,
    real_kv_sources_cuda: tuple[RealKvSource, ...],
    real_kv_sources_ref: tuple[RealKvSource, ...],
    real_kv_hash_mode: RealKvHashMode,
    kernel_kind: CanaryLaunchTag = CanaryLaunchTag.HEAD_K_FULL,
) -> None:
    _run_both_verify(
        cuda_canary_buf=cuda_canary_buf,
        ref_canary_buf=ref_canary_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=real_kv_sources_cuda,
        real_kv_sources_ref=real_kv_sources_ref,
        real_kv_hash_mode=real_kv_hash_mode,
        kernel_kind=kernel_kind,
    )
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def _run_both_write(
    *,
    cuda_canary_buf: torch.Tensor,
    ref_canary_buf: torch.Tensor,
    plan_cuda,
    plan_ref,
    fb_input_ids: torch.Tensor,
    fb_positions: torch.Tensor,
    fb_out_cache_loc: torch.Tensor,
    pseudo_mode: CanaryPseudoMode,
    pseudo_expected_tokens: torch.Tensor,
    pseudo_expected_positions: torch.Tensor,
    cuda_log: FakeViolationLog,
    ref_log: FakeViolationLog,
    real_kv_sources_cuda: tuple[RealKvSource, ...],
    real_kv_sources_ref: tuple[RealKvSource, ...],
    real_kv_hash_mode: RealKvHashMode,
    kernel_kind: CanaryLaunchTag = CanaryLaunchTag.HEAD_K_FULL,
) -> None:
    canary_write_step(
        canary_buf=cuda_canary_buf,
        plan=plan_cuda,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        kernel_kind=kernel_kind,
        pseudo_mode=pseudo_mode,
        pseudo_expected_tokens=pseudo_expected_tokens,
        pseudo_expected_positions=pseudo_expected_positions,
        violation_ring=cuda_log.ring,
        violation_write_index=cuda_log.write_index,
        slot_run_counter=cuda_log.slot_run_counter,
        kernel_run_counter=cuda_log.kernel_run_counter,
        real_kv_sources=real_kv_sources_cuda,
        real_kv_hash_mode=real_kv_hash_mode,
    )
    canary_write_step_torch_reference(
        canary_buf=ref_canary_buf,
        plan=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        kernel_kind=kernel_kind,
        pseudo_mode=pseudo_mode,
        pseudo_expected_tokens=pseudo_expected_tokens,
        pseudo_expected_positions=pseudo_expected_positions,
        violation_ring=ref_log.ring,
        violation_write_index=ref_log.write_index,
        slot_run_counter=ref_log.slot_run_counter,
        kernel_run_counter=ref_log.kernel_run_counter,
        real_kv_sources=real_kv_sources_ref,
        real_kv_hash_mode=real_kv_hash_mode,
    )
    torch.cuda.synchronize()


def _run_both_and_assert_write_buf_and_state_equal(
    *,
    cuda_canary_buf: torch.Tensor,
    ref_canary_buf: torch.Tensor,
    plan_cuda,
    plan_ref,
    fb_input_ids: torch.Tensor,
    fb_positions: torch.Tensor,
    fb_out_cache_loc: torch.Tensor,
    pseudo_mode: CanaryPseudoMode,
    pseudo_expected_tokens: torch.Tensor,
    pseudo_expected_positions: torch.Tensor,
    cuda_log: FakeViolationLog,
    ref_log: FakeViolationLog,
    real_kv_sources_cuda: tuple[RealKvSource, ...],
    real_kv_sources_ref: tuple[RealKvSource, ...],
    real_kv_hash_mode: RealKvHashMode,
    kernel_kind: CanaryLaunchTag = CanaryLaunchTag.HEAD_K_FULL,
) -> None:
    _run_both_write(
        cuda_canary_buf=cuda_canary_buf,
        ref_canary_buf=ref_canary_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=pseudo_mode,
        pseudo_expected_tokens=pseudo_expected_tokens,
        pseudo_expected_positions=pseudo_expected_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=real_kv_sources_cuda,
        real_kv_sources_ref=real_kv_sources_ref,
        real_kv_hash_mode=real_kv_hash_mode,
        kernel_kind=kernel_kind,
    )
    assert_canary_buf_equal(buf_a=cuda_canary_buf, buf_b=ref_canary_buf)
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


@dataclass(frozen=True, slots=True, kw_only=True)
class ShrinkResult:
    """Outcome of ``shrink_inputs``: minified fuzz inputs + audit trail of accepted mutations."""

    inputs: Any
    mutations_applied: list[str]


def shrink_inputs(
    inputs: Any,
    *,
    check_fn: Callable[[Any], bool],
    max_iterations: int = 50,
) -> ShrinkResult:
    """Greedy 1-step minify for a fuzz inputs dataclass.

    ``check_fn(candidate)`` returns True when ``candidate`` still reproduces the failure. Each round
    yields candidate-simpler-than-current mutations through ``_yield_simpler``; the first accepted
    candidate becomes the new current. Iteration stops when no mutation is accepted or ``max_iterations``
    is reached.
    """
    current = inputs
    applied: list[str] = []
    for _ in range(max_iterations):
        improved = False
        for label, candidate in _yield_simpler(current):
            try:
                still_fails = check_fn(candidate)
            except Exception:
                still_fails = False
            if still_fails:
                current = candidate
                applied.append(label)
                improved = True
                break
        if not improved:
            break
    return ShrinkResult(inputs=current, mutations_applied=applied)


def _yield_simpler(inputs: Any) -> Iterator[tuple[str, Any]]:
    """Yield (label, simpler_candidate) tuples for generic fuzz-input minifiers.

    The candidates touch only well-known field names; an inputs dataclass that lacks a field will simply
    have that mutation skipped. No kernel-specific knowledge is encoded here so the same shrinker drives
    Plan / Verify / Write fuzz failures uniformly.
    """
    fields = {
        f: getattr(inputs, f) for f in inputs.__dataclass_fields__  # type: ignore[attr-defined]
    }

    def emit(label: str, **overrides: Any) -> Iterator[tuple[str, Any]]:
        candidate = replace(inputs, **overrides)
        yield label, candidate

    bs_field = (
        "fb_req_pool_indices"
        if "fb_req_pool_indices" in fields
        else ("fb_input_ids" if "fb_input_ids" in fields else None)
    )
    if bs_field is not None and isinstance(fields[bs_field], torch.Tensor):
        tensor = fields[bs_field]
        if tensor.numel() > 1:
            new_len = tensor.numel() - 1
            related_tensors_overrides: dict[str, Any] = {}
            for name in (
                "fb_req_pool_indices",
                "fb_prefix_lens",
                "fb_extend_seq_lens",
                "fb_input_ids",
                "fb_positions",
                "fb_out_cache_loc",
                "pseudo_expected_tokens",
                "pseudo_expected_positions",
            ):
                t = fields.get(name)
                if (
                    isinstance(t, torch.Tensor)
                    and t.numel() >= new_len
                    and t.dim() == 1
                ):
                    related_tensors_overrides[name] = t[:new_len].contiguous()
            if related_tensors_overrides:
                yield from emit("drop_last_row", **related_tensors_overrides)

    if "swa_window_size" in fields and isinstance(fields["swa_window_size"], int):
        if fields["swa_window_size"] != 0:
            yield from emit(
                "swa_off", swa_window_size=0, full_to_swa_index_mapping=None
            )

    if "extras_count" in fields and isinstance(fields["extras_count"], int):
        if fields["extras_count"] > 0:
            yield from emit("extras_zero", extras_count=0)

    if "real_kv_hash_mode" in fields:
        cur = fields["real_kv_hash_mode"]
        if hasattr(cur, "value"):
            cls = cur.__class__
            if int(cur) == 2:
                yield from emit("hash_mode_bit", real_kv_hash_mode=cls(1))
            elif int(cur) == 1:
                yield from emit("hash_mode_off", real_kv_hash_mode=cls(0))

    if "real_kv_sources" in fields:
        srcs = fields["real_kv_sources"]
        if isinstance(srcs, tuple) and len(srcs) > 1:
            yield from emit("sources_to_one", real_kv_sources=srcs[:1])

    if "pseudo_mode" in fields:
        cur = fields["pseudo_mode"]
        if hasattr(cur, "value") and int(cur) != 0:
            cls = cur.__class__
            yield from emit("pseudo_off", pseudo_mode=cls(0))

    for name in ("verify_capacity", "write_req_capacity"):
        if name in fields and isinstance(fields[name], int):
            current_value = fields[name]
            if current_value > 8:
                yield from emit(f"shrink_{name}", **{name: max(8, current_value // 2)})
