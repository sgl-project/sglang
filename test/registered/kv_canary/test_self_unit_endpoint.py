from __future__ import annotations

from types import SimpleNamespace
from typing import List

import torch

from sglang.jit_kernel.kv_canary_verify import (
    CANARY_SLOT_BYTES,
    CanaryLaunchTag,
    RealKvHashMode,
    VerifyPlan,
)
from sglang.jit_kernel.kv_canary_write import CanaryPseudoMode, WritePlan
from sglang.srt.kv_canary import endpoint as endpoint_module
from sglang.srt.kv_canary.endpoint import (
    CanaryEndpoint,
    build_endpoints_from_group,
)
from sglang.srt.kv_canary.violation_state import (
    CanaryDeviceState,
    ViolationLog,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-large")


def _record_call(call_log: List, name: str):
    def _stub(**kwargs):
        call_log.append(name)
        call_log.append(kwargs)

    return _stub


def _make_endpoint(*, device, kernel_kind=CanaryLaunchTag.HEAD_K_FULL, swa_lut=None):
    canary_buf = torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
    slot_view = torch.zeros(1, dtype=torch.int64, device=device)
    kernel_view = torch.zeros(1, dtype=torch.int64, device=device)
    return CanaryEndpoint(
        kernel_kind=kernel_kind,
        canary_buf=canary_buf,
        full_to_swa_index_mapping=swa_lut,
        real_kv_sources=(),
        slot_run_counter_view=slot_view,
        kernel_run_counter_view=kernel_view,
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
        input_check_mode=CanaryPseudoMode.OFF,
        expected_input_tokens=torch.zeros(1, dtype=torch.int32, device=device),
        expected_input_positions=torch.zeros(1, dtype=torch.int32, device=device),
        real_kv_hash_mode=RealKvHashMode.OFF,
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
        input_check_mode=args.input_check_mode,
        expected_input_tokens=args.expected_input_tokens,
        expected_input_positions=args.expected_input_positions,
        violation_log=args.violation_log,
        real_kv_hash_mode=args.real_kv_hash_mode,
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
        input_check_mode=args.input_check_mode,
        expected_input_tokens=args.expected_input_tokens,
        expected_input_positions=args.expected_input_positions,
        violation_log=args.violation_log,
        real_kv_hash_mode=args.real_kv_hash_mode,
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
    ep = _make_endpoint(device=device, kernel_kind=CanaryLaunchTag.SWEEP_K_FULL)
    args = _make_kernel_args(device)
    ep.launch_sweep(
        verify_plan=args.verify_plan,
        violation_log=args.violation_log,
        real_kv_hash_mode=args.real_kv_hash_mode,
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
    ep_a = _make_endpoint(device=device, kernel_kind=CanaryLaunchTag.SWEEP_K_FULL)
    ep_b = _make_endpoint(device=device, kernel_kind=CanaryLaunchTag.SWEEP_V_FULL)

    plan = VerifyPlan.allocate(verify_capacity=1, device=device)
    ep_a.launch_sweep(
        verify_plan=plan,
        violation_log=shared_log,
        real_kv_hash_mode=RealKvHashMode.OFF,
    )
    ep_b.launch_sweep(
        verify_plan=plan,
        violation_log=shared_log,
        real_kv_hash_mode=RealKvHashMode.OFF,
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
        input_check_mode=args.input_check_mode,
        expected_input_tokens=args.expected_input_tokens,
        expected_input_positions=args.expected_input_positions,
        violation_log=args.violation_log,
        real_kv_hash_mode=args.real_kv_hash_mode,
    )
    full_ep.launch_per_forward(
        verify_plan=args.verify_plan,
        write_plan=args.write_plan,
        fb_input_ids=args.fb_input_ids,
        fb_positions=args.fb_positions,
        fb_out_cache_loc=args.fb_out_cache_loc,
        input_check_mode=args.input_check_mode,
        expected_input_tokens=args.expected_input_tokens,
        expected_input_positions=args.expected_input_positions,
        violation_log=args.violation_log,
        real_kv_hash_mode=args.real_kv_hash_mode,
    )
    assert captured[0] is lut
    assert captured[1] is None


def test_head_tail_share_class(device, make_buffer_group, base_config):
    group = make_buffer_group()
    device_state = CanaryDeviceState.allocate(
        config=base_config, device=device, num_tags=len(CanaryLaunchTag)
    )
    endpoints = build_endpoints_from_group(group=group, device_state=device_state)

    head = next(ep for ep in endpoints if ep.kernel_kind == CanaryLaunchTag.HEAD_K_FULL)
    tail = next(ep for ep in endpoints if ep.kernel_kind == CanaryLaunchTag.TAIL_K_FULL)

    assert type(head) is type(tail)
    assert head.launch_per_forward.__func__ is tail.launch_per_forward.__func__
    assert head.__class__.__module__ == tail.__class__.__module__
