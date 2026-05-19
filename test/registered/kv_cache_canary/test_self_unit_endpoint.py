from __future__ import annotations

from types import SimpleNamespace
from typing import List

import torch

from sglang.jit_kernel.kv_cache_canary_verify import (
    CANARY_SLOT_BYTES,
    CanaryLaunchTag,
    RealKvHashMode,
    VerifyPlan,
)
from sglang.jit_kernel.kv_cache_canary_write import CanaryPseudoMode, WritePlan
from sglang.srt.kv_cache_canary import endpoint as endpoint_module
from sglang.srt.kv_cache_canary.endpoint import CanaryEndpoint
from sglang.srt.kv_cache_canary.violation_state import ViolationLog
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-large")


def _record_call(call_log: List[str], name: str):
    def _stub(**kwargs):
        call_log.append(name)
        call_log.append(kwargs)

    return _stub


def _make_endpoint(*, device, kernel_kind=CanaryLaunchTag.HEAD_K_FULL, swa_lut=None):
    canary_buf = torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
    return CanaryEndpoint(
        canary_buf=canary_buf,
        kernel_kind=kernel_kind,
        real_kv_sources=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
        full_to_swa_index_mapping=swa_lut,
    )


def _make_kernel_args(device):
    verify_plan = VerifyPlan.allocate(verify_capacity=1, device=device)
    write_plan = WritePlan.allocate(write_req_capacity=1, device=device)
    log = ViolationLog.allocate(ring_capacity=2, device=device)
    return SimpleNamespace(
        verify_plan=verify_plan,
        write_plan=write_plan,
        violation_log=log,
        fb_input_ids=torch.zeros(1, dtype=torch.int32, device=device),
        fb_positions=torch.zeros(1, dtype=torch.int32, device=device),
        fb_out_cache_loc=torch.zeros(1, dtype=torch.int32, device=device),
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=torch.zeros(1, dtype=torch.int32, device=device),
        pseudo_expected_positions=torch.zeros(1, dtype=torch.int32, device=device),
        slot_run_counter=torch.zeros(1, dtype=torch.int64, device=device),
        kernel_run_counter=torch.zeros(1, dtype=torch.int64, device=device),
    )


def test_launch_per_forward_calls_verify_then_write(device, monkeypatch):
    calls: List = []
    monkeypatch.setattr(
        endpoint_module, "canary_verify_step", _record_call(calls, "verify")
    )
    monkeypatch.setattr(
        endpoint_module, "canary_write_step", _record_call(calls, "write")
    )
    ep = _make_endpoint(device=device)
    args = _make_kernel_args(device)
    ep.launch_per_forward(
        verify_plan=args.verify_plan,
        write_plan=args.write_plan,
        fb_input_ids=args.fb_input_ids,
        fb_positions=args.fb_positions,
        fb_out_cache_loc=args.fb_out_cache_loc,
        pseudo_mode=args.pseudo_mode,
        pseudo_expected_tokens=args.pseudo_expected_tokens,
        pseudo_expected_positions=args.pseudo_expected_positions,
        violation_log=args.violation_log,
        slot_run_counter=args.slot_run_counter,
        kernel_run_counter=args.kernel_run_counter,
    )
    assert calls[0] == "verify"
    assert calls[2] == "write"


def test_launch_per_forward_passes_kernel_kind(device, monkeypatch):
    captured: List = []
    monkeypatch.setattr(
        endpoint_module,
        "canary_verify_step",
        lambda **kwargs: captured.append(("verify", kwargs["kernel_kind"])),
    )
    monkeypatch.setattr(
        endpoint_module,
        "canary_write_step",
        lambda **kwargs: captured.append(("write", kwargs["kernel_kind"])),
    )
    ep = _make_endpoint(device=device, kernel_kind=CanaryLaunchTag.TAIL_V_SWA)
    args = _make_kernel_args(device)
    ep.launch_per_forward(
        verify_plan=args.verify_plan,
        write_plan=args.write_plan,
        fb_input_ids=args.fb_input_ids,
        fb_positions=args.fb_positions,
        fb_out_cache_loc=args.fb_out_cache_loc,
        pseudo_mode=args.pseudo_mode,
        pseudo_expected_tokens=args.pseudo_expected_tokens,
        pseudo_expected_positions=args.pseudo_expected_positions,
        violation_log=args.violation_log,
        slot_run_counter=args.slot_run_counter,
        kernel_run_counter=args.kernel_run_counter,
    )
    assert ("verify", CanaryLaunchTag.TAIL_V_SWA) in captured
    assert ("write", CanaryLaunchTag.TAIL_V_SWA) in captured


def test_launch_sweep_only_calls_verify(device, monkeypatch):
    calls: List[str] = []
    monkeypatch.setattr(
        endpoint_module, "canary_verify_step", lambda **kwargs: calls.append("verify")
    )
    monkeypatch.setattr(
        endpoint_module, "canary_write_step", lambda **kwargs: calls.append("write")
    )
    ep = _make_endpoint(device=device)
    args = _make_kernel_args(device)
    ep.launch_sweep(
        verify_plan=args.verify_plan,
        violation_log=args.violation_log,
        slot_run_counter=args.slot_run_counter,
        kernel_run_counter=args.kernel_run_counter,
    )
    assert calls == ["verify"]


def test_endpoint_shares_violation_log_across_launches(device, monkeypatch):
    captured_rings: List[int] = []
    monkeypatch.setattr(
        endpoint_module,
        "canary_verify_step",
        lambda **kwargs: captured_rings.append(kwargs["violation_ring"].data_ptr()),
    )
    monkeypatch.setattr(endpoint_module, "canary_write_step", lambda **kwargs: None)

    shared_log = ViolationLog.allocate(ring_capacity=2, device=device)
    ep_a = _make_endpoint(device=device, kernel_kind=CanaryLaunchTag.HEAD_K_FULL)
    ep_b = _make_endpoint(device=device, kernel_kind=CanaryLaunchTag.HEAD_V_FULL)

    plan = VerifyPlan.allocate(verify_capacity=1, device=device)
    sl = torch.zeros(1, dtype=torch.int64, device=device)
    kl = torch.zeros(1, dtype=torch.int64, device=device)
    ep_a.launch_sweep(
        verify_plan=plan,
        violation_log=shared_log,
        slot_run_counter=sl,
        kernel_run_counter=kl,
    )
    ep_b.launch_sweep(
        verify_plan=plan,
        violation_log=shared_log,
        slot_run_counter=sl,
        kernel_run_counter=kl,
    )
    assert (
        captured_rings[0] == captured_rings[1] == shared_log.violation_ring.data_ptr()
    )


def test_swa_endpoint_passes_lut(device, monkeypatch):
    captured: List = []
    monkeypatch.setattr(endpoint_module, "canary_verify_step", lambda **kwargs: None)
    monkeypatch.setattr(
        endpoint_module,
        "canary_write_step",
        lambda **kwargs: captured.append(kwargs["full_to_swa_index_mapping"]),
    )

    lut = torch.arange(8, dtype=torch.int32, device=device)
    swa_ep = _make_endpoint(
        device=device, kernel_kind=CanaryLaunchTag.HEAD_K_SWA, swa_lut=lut
    )
    full_ep = _make_endpoint(
        device=device, kernel_kind=CanaryLaunchTag.HEAD_K_FULL, swa_lut=None
    )
    args = _make_kernel_args(device)

    swa_ep.launch_per_forward(
        verify_plan=args.verify_plan,
        write_plan=args.write_plan,
        fb_input_ids=args.fb_input_ids,
        fb_positions=args.fb_positions,
        fb_out_cache_loc=args.fb_out_cache_loc,
        pseudo_mode=args.pseudo_mode,
        pseudo_expected_tokens=args.pseudo_expected_tokens,
        pseudo_expected_positions=args.pseudo_expected_positions,
        violation_log=args.violation_log,
        slot_run_counter=args.slot_run_counter,
        kernel_run_counter=args.kernel_run_counter,
    )
    full_ep.launch_per_forward(
        verify_plan=args.verify_plan,
        write_plan=args.write_plan,
        fb_input_ids=args.fb_input_ids,
        fb_positions=args.fb_positions,
        fb_out_cache_loc=args.fb_out_cache_loc,
        pseudo_mode=args.pseudo_mode,
        pseudo_expected_tokens=args.pseudo_expected_tokens,
        pseudo_expected_positions=args.pseudo_expected_positions,
        violation_log=args.violation_log,
        slot_run_counter=args.slot_run_counter,
        kernel_run_counter=args.kernel_run_counter,
    )
    assert captured[0] is lut
    assert captured[1] is None


def test_head_tail_share_class(device):
    from sglang.srt.kv_cache_canary.buffer_group import CanaryBufferGroup, PoolKind
    from sglang.srt.kv_cache_canary.config import CanaryConfig, CanaryMode
    from sglang.srt.kv_cache_canary.runner import _build_endpoints_for_phase

    config = CanaryConfig(mode=CanaryMode.RAISE)
    k_head = torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
    k_tail = torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
    v_head = torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
    v_tail = torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
    group = CanaryBufferGroup(
        kind=PoolKind.FULL,
        k_head=k_head,
        k_tail=k_tail,
        v_head=v_head,
        v_tail=v_tail,
        real_kv_sources_k=(),
        real_kv_sources_v=(),
        swa_index_lut=None,
    )
    heads = _build_endpoints_for_phase(buffer_group=group, config=config, phase="head")
    tails = _build_endpoints_for_phase(buffer_group=group, config=config, phase="tail")

    head, tail = heads[0], tails[0]
    assert type(head) is type(tail)
    assert head.launch_per_forward.__func__ is tail.launch_per_forward.__func__
    assert head.__class__.__module__ == tail.__class__.__module__
