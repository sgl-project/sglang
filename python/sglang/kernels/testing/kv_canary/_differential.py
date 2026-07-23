from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator, Optional

import torch

from sglang.kernels.ops.kv_canary import consts
from sglang.kernels.ops.kv_canary.plan import launch_canary_plan_kernels
from sglang.kernels.ops.kv_canary.plan_ref import (
    launch_canary_plan_kernels_torch_reference,
)
from sglang.kernels.ops.kv_canary.verify import (
    CanaryLaunchTag,
    RealKvSource,
    VerifyOrWriteContext,
    VerifyPlan,
    launch_canary_verify_kernel,
)
from sglang.kernels.ops.kv_canary.verify_ref import (
    launch_canary_verify_kernel_torch_reference,
)
from sglang.kernels.ops.kv_canary.write import WritePlan, launch_canary_write_kernel
from sglang.kernels.ops.kv_canary.write_ref import (
    launch_canary_write_kernel_torch_reference,
)
from sglang.kernels.testing.kv_canary._canary_helpers import (
    FakeViolationLog,
    assert_canary_buf_equal,
    assert_canary_state_equal,
    make_log_pair,
)

_DEVICE = torch.device("cuda")


def _run_both_plan(
    *,
    triton_verify: VerifyPlan,
    triton_write: WritePlan,
    ref_verify: VerifyPlan,
    ref_write: WritePlan,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    extras: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    assert_equal: bool = True,
    active_verify_entries: Optional[int] = None,
    active_write_reqs: Optional[int] = None,
    req_to_verify_expected_tokens: Optional[torch.Tensor] = None,
    req_to_verify_expected_tokens_valid_lens: Optional[torch.Tensor] = None,
    kv_token_id_vs_position_offset: int = 0,
) -> None:
    _ = extras
    verify_capacity = int(triton_verify.verify_slot_indices.shape[0])
    # Default lens to "no tighter bound than pool width" so existing kernel tests that
    # only care about gather wiring keep their old semantics without each call site
    # explicitly building a per-req lens tensor.
    if (
        req_to_verify_expected_tokens is not None
        and req_to_verify_expected_tokens_valid_lens is None
    ):
        req_to_verify_expected_tokens_valid_lens = torch.full(
            (int(req_pool_indices.shape[0]),),
            int(req_to_verify_expected_tokens.shape[1]),
            dtype=torch.int64,
            device=req_pool_indices.device,
        )
    launch_canary_plan_kernels(
        verify_plan_out=triton_verify,
        write_plan_out=triton_write,
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        extend_seq_lens=extend_seq_lens,
        req_to_token=req_to_token,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        verify_capacity=verify_capacity,
        req_to_verify_expected_tokens=req_to_verify_expected_tokens,
        req_to_verify_expected_tokens_valid_lens=req_to_verify_expected_tokens_valid_lens,
        kv_token_id_vs_position_offset=kv_token_id_vs_position_offset,
    )
    launch_canary_plan_kernels_torch_reference(
        verify_plan_out=ref_verify,
        write_plan_out=ref_write,
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        extend_seq_lens=extend_seq_lens,
        req_to_token=req_to_token,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        verify_capacity=int(ref_verify.verify_slot_indices.shape[0]),
        req_to_verify_expected_tokens=req_to_verify_expected_tokens,
        req_to_verify_expected_tokens_valid_lens=req_to_verify_expected_tokens_valid_lens,
        kv_token_id_vs_position_offset=kv_token_id_vs_position_offset,
    )
    torch.cuda.synchronize()

    if assert_equal:
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
    # When total_verify > VERIFY_CAPACITY the offsets kernel clears verify_enable and
    # plan_entries skips its scatter — leaving verify_slot_indices/positions/prev_slot_indices
    # as whatever the (torch.empty) allocation contained. Skip the byte-equal probe in that
    # case; verify_num_valid being clamped + verify_enable=0 is the contract here.
    triton_enable = int(triton_verify.enable[0].item())
    ref_enable = int(ref_verify.enable[0].item())
    assert (
        triton_enable == ref_enable
    ), f"verify_enable diverged: triton={triton_enable} ref={ref_enable}"
    if n_verify > 0 and triton_enable != 0:
        assert torch.equal(
            triton_verify.verify_slot_indices[:n_verify],
            ref_verify.verify_slot_indices[:n_verify],
        )
        assert torch.equal(
            triton_verify.verify_expected_tokens[:n_verify],
            ref_verify.verify_expected_tokens[:n_verify],
        )
        assert torch.equal(
            triton_verify.verify_expected_positions[:n_verify],
            ref_verify.verify_expected_positions[:n_verify],
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
    real_kv_hash_mode: consts.RealKvHashMode,
    kernel_kind: CanaryLaunchTag = CanaryLaunchTag.HEAD_K_FULL,
    assert_equal: bool = True,
    check_verify_expected_token: bool = True,
) -> None:
    launch_canary_verify_kernel(
        context=VerifyOrWriteContext(
            canary_buf=cuda_canary_buf,
            kernel_kind=kernel_kind,
            violation_ring=cuda_log.ring,
            violation_write_index=cuda_log.write_index,
            slot_run_counter=cuda_log.slot_run_counter,
            kernel_run_counter=cuda_log.kernel_run_counter,
            enable_chain_position_assert=cuda_log.enable_chain_position_assert,
            real_kv_sources=real_kv_sources_cuda,
            real_kv_hash_mode=real_kv_hash_mode,
        ),
        plan=plan_cuda,
        check_verify_expected_token=check_verify_expected_token,
    )
    launch_canary_verify_kernel_torch_reference(
        context=VerifyOrWriteContext(
            canary_buf=ref_canary_buf,
            kernel_kind=kernel_kind,
            violation_ring=ref_log.ring,
            violation_write_index=ref_log.write_index,
            slot_run_counter=ref_log.slot_run_counter,
            kernel_run_counter=ref_log.kernel_run_counter,
            enable_chain_position_assert=ref_log.enable_chain_position_assert,
            real_kv_sources=real_kv_sources_ref,
            real_kv_hash_mode=real_kv_hash_mode,
        ),
        plan=plan_ref,
        check_verify_expected_token=check_verify_expected_token,
    )
    torch.cuda.synchronize()

    if assert_equal:
        assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def _run_both_write(
    *,
    cuda_canary_buf: torch.Tensor,
    ref_canary_buf: torch.Tensor,
    plan_cuda,
    plan_ref,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    enable_write_verify_inputs: bool,
    expected_input_tokens: torch.Tensor,
    expected_input_positions: torch.Tensor,
    cuda_log: FakeViolationLog,
    ref_log: FakeViolationLog,
    real_kv_sources_cuda: tuple[RealKvSource, ...],
    real_kv_sources_ref: tuple[RealKvSource, ...],
    real_kv_hash_mode: consts.RealKvHashMode,
    kernel_kind: CanaryLaunchTag = CanaryLaunchTag.HEAD_K_FULL,
    assert_equal: bool = True,
) -> None:
    expected_tokens_for_launch = (
        expected_input_tokens if enable_write_verify_inputs else None
    )
    expected_positions_for_launch = (
        expected_input_positions if enable_write_verify_inputs else None
    )
    launch_canary_write_kernel(
        context=VerifyOrWriteContext(
            canary_buf=cuda_canary_buf,
            kernel_kind=kernel_kind,
            violation_ring=cuda_log.ring,
            violation_write_index=cuda_log.write_index,
            slot_run_counter=cuda_log.slot_run_counter,
            kernel_run_counter=cuda_log.kernel_run_counter,
            enable_chain_position_assert=cuda_log.enable_chain_position_assert,
            real_kv_sources=real_kv_sources_cuda,
            real_kv_hash_mode=real_kv_hash_mode,
        ),
        plan=plan_cuda,
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=out_cache_loc,
        enable_write_input_assert=enable_write_verify_inputs,
        expected_input_tokens=expected_tokens_for_launch,
        expected_input_positions=expected_positions_for_launch,
    )
    launch_canary_write_kernel_torch_reference(
        context=VerifyOrWriteContext(
            canary_buf=ref_canary_buf,
            kernel_kind=kernel_kind,
            violation_ring=ref_log.ring,
            violation_write_index=ref_log.write_index,
            slot_run_counter=ref_log.slot_run_counter,
            kernel_run_counter=ref_log.kernel_run_counter,
            enable_chain_position_assert=ref_log.enable_chain_position_assert,
            real_kv_sources=real_kv_sources_ref,
            real_kv_hash_mode=real_kv_hash_mode,
        ),
        plan=plan_ref,
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=out_cache_loc,
        enable_write_input_assert=enable_write_verify_inputs,
        expected_input_tokens=expected_tokens_for_launch,
        expected_input_positions=expected_positions_for_launch,
    )
    torch.cuda.synchronize()

    if assert_equal:
        assert_canary_buf_equal(buf_a=cuda_canary_buf, buf_b=ref_canary_buf)
        assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


@dataclass(frozen=True, slots=True, kw_only=True)
class ShrinkResult:
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
        "req_pool_indices"
        if "req_pool_indices" in fields
        else ("input_ids" if "input_ids" in fields else None)
    )
    if bs_field is not None and isinstance(fields[bs_field], torch.Tensor):
        tensor = fields[bs_field]
        if tensor.numel() > 1:
            new_len = tensor.numel() - 1
            related_tensors_overrides: dict[str, Any] = {}
            for name in (
                "req_pool_indices",
                "prefix_lens",
                "extend_seq_lens",
                "input_ids",
                "positions",
                "out_cache_loc",
                "expected_input_tokens",
                "expected_input_positions",
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

    if "enable_write_verify_inputs" in fields:
        cur = fields["enable_write_verify_inputs"]
        if hasattr(cur, "value") and int(cur) != 0:
            cls = cur.__class__
            yield from emit("pseudo_off", enable_write_verify_inputs=cls(0))

    for name in ("verify_capacity", "write_req_capacity"):
        if name in fields and isinstance(fields[name], int):
            current_value = fields[name]
            if current_value > 8:
                yield from emit(f"shrink_{name}", **{name: max(8, current_value // 2)})


def run_verify_diff(
    *,
    buf_pair: tuple[torch.Tensor, torch.Tensor],
    plan_pair: tuple[VerifyPlan, VerifyPlan],
    real_kv_sources_pair: tuple[tuple[RealKvSource, ...], tuple[RealKvSource, ...]] = (
        (),
        (),
    ),
    real_kv_hash_mode: consts.RealKvHashMode = consts.RealKvHashMode.NONE,
    kernel_kind: CanaryLaunchTag = CanaryLaunchTag.HEAD_K_FULL,
    device: torch.device = _DEVICE,
    assert_equal: bool = True,
    check_verify_expected_token: bool = True,
) -> tuple[FakeViolationLog, FakeViolationLog]:
    """Thin wrapper around ``_run_both_verify`` that creates a fresh log pair and packs (cuda, ref)
    buf/plan/source arguments into 2-tuples to drop ~8 lines of boilerplate per call site.
    """
    cuda_log, ref_log = make_log_pair(device=device)
    _run_both_verify(
        cuda_canary_buf=buf_pair[0],
        ref_canary_buf=buf_pair[1],
        plan_cuda=plan_pair[0],
        plan_ref=plan_pair[1],
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=real_kv_sources_pair[0],
        real_kv_sources_ref=real_kv_sources_pair[1],
        real_kv_hash_mode=real_kv_hash_mode,
        kernel_kind=kernel_kind,
        assert_equal=assert_equal,
        check_verify_expected_token=check_verify_expected_token,
    )
    return cuda_log, ref_log


def run_write_diff(
    *,
    buf_pair: tuple[torch.Tensor, torch.Tensor],
    plan_pair: tuple[WritePlan, WritePlan],
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    expected_input_tokens: torch.Tensor,
    expected_input_positions: torch.Tensor,
    enable_write_verify_inputs: bool = False,
    real_kv_sources_pair: tuple[tuple[RealKvSource, ...], tuple[RealKvSource, ...]] = (
        (),
        (),
    ),
    real_kv_hash_mode: consts.RealKvHashMode = consts.RealKvHashMode.NONE,
    kernel_kind: CanaryLaunchTag = CanaryLaunchTag.HEAD_K_FULL,
    device: torch.device = _DEVICE,
    assert_equal: bool = True,
) -> tuple[FakeViolationLog, FakeViolationLog]:
    """Thin wrapper around ``_run_both_write`` that creates a fresh log pair and packs (cuda, ref)
    buf/plan/source arguments into 2-tuples to drop ~10 lines of boilerplate per call site.
    """
    cuda_log, ref_log = make_log_pair(device=device)
    _run_both_write(
        cuda_canary_buf=buf_pair[0],
        ref_canary_buf=buf_pair[1],
        plan_cuda=plan_pair[0],
        plan_ref=plan_pair[1],
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=out_cache_loc,
        enable_write_verify_inputs=enable_write_verify_inputs,
        expected_input_tokens=expected_input_tokens,
        expected_input_positions=expected_input_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=real_kv_sources_pair[0],
        real_kv_sources_ref=real_kv_sources_pair[1],
        real_kv_hash_mode=real_kv_hash_mode,
        kernel_kind=kernel_kind,
        assert_equal=assert_equal,
    )
    return cuda_log, ref_log


def run_plan_diff(
    *,
    plan_pair: tuple[tuple[VerifyPlan, WritePlan], tuple[VerifyPlan, WritePlan]],
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    extras: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    swa_window_size: int = 0,
    full_to_swa_index_mapping: Optional[torch.Tensor] = None,
    assert_equal: bool = True,
    active_verify_entries: Optional[int] = None,
    active_write_reqs: Optional[int] = None,
    req_to_verify_expected_tokens: Optional[torch.Tensor] = None,
    req_to_verify_expected_tokens_valid_lens: Optional[torch.Tensor] = None,
    kv_token_id_vs_position_offset: int = 0,
) -> None:
    """Thin wrapper around ``_run_both_plan`` that unpacks ``((triton_v, triton_w), (ref_v, ref_w))``
    plan pairs to drop the per-call-site ``triton_verify=.../triton_write=.../ref_verify=...`` block.
    """
    (triton_verify, triton_write), (ref_verify, ref_write) = plan_pair
    _run_both_plan(
        triton_verify=triton_verify,
        triton_write=triton_write,
        ref_verify=ref_verify,
        ref_write=ref_write,
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        extend_seq_lens=extend_seq_lens,
        req_to_token=req_to_token,
        extras=extras,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        assert_equal=assert_equal,
        active_verify_entries=active_verify_entries,
        active_write_reqs=active_write_reqs,
        req_to_verify_expected_tokens=req_to_verify_expected_tokens,
        req_to_verify_expected_tokens_valid_lens=req_to_verify_expected_tokens_valid_lens,
        kv_token_id_vs_position_offset=kv_token_id_vs_position_offset,
    )
