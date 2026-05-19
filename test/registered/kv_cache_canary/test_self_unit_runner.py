from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import List

import pytest
import torch

from sglang.jit_kernel.kv_cache_canary_verify import CANARY_SLOT_BYTES, RealKvHashMode
from sglang.srt.kv_cache_canary import runner as runner_module
from sglang.srt.kv_cache_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_cache_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_cache_canary.runner import CanaryRunner
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-large")


def _make_group(*, device, has_v: bool = True, kind: PoolKind = PoolKind.FULL):
    return CanaryBufferGroup(
        kind=kind,
        k_head=torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device),
        k_tail=torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device),
        v_head=(
            torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
            if has_v
            else None
        ),
        v_tail=(
            torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
            if has_v
            else None
        ),
        real_kv_sources_k=(),
        real_kv_sources_v=(),
        swa_index_lut=None,
    )


def _make_pool(device, max_reqs: int = 4, max_seq: int = 8):
    table = torch.zeros(max_reqs, max_seq, dtype=torch.int32, device=device)
    return SimpleNamespace(req_to_token=table, size=max_reqs)


def _make_forward_batch(device, bs: int = 2):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: False, is_mixed=lambda: False),
        req_pool_indices=torch.tensor([1, 2][:bs], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([3, 4][:bs], dtype=torch.int32, device=device),
        extend_prefix_lens=None,
        extend_seq_lens=None,
        input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
        positions=torch.zeros(bs, dtype=torch.int32, device=device),
        out_cache_loc=torch.zeros(bs, dtype=torch.int32, device=device),
    )


def _make_runner(*, device, config=None, group=None, req_pool=None, **overrides):
    if config is None:
        config = CanaryConfig(
            mode=CanaryMode.RAISE, real_kv_hash_mode=RealKvHashMode.OFF
        )
    if group is None:
        group = _make_group(device=device)
    if req_pool is None:
        req_pool = _make_pool(device)
    capacities = dict(
        per_forward_verify_capacity=4,
        per_forward_write_req_capacity=2,
        running_sweep_verify_capacity=4,
        radix_sweep_verify_capacity=4,
        radix_sweep_extras_capacity=4,
    )
    capacities.update(overrides)
    return CanaryRunner(
        config=config,
        buffer_group=group,
        device=device,
        req_to_token_pool=req_pool,
        **capacities,
    )


def _stub_plan_and_kernels(monkeypatch):
    """Make plan / verify / write kernels no-op so CPU runs without CUDA jit."""
    monkeypatch.setattr(runner_module, "canary_plan_step", lambda **kwargs: None)

    from sglang.srt.kv_cache_canary import endpoint as endpoint_module

    monkeypatch.setattr(endpoint_module, "canary_verify_step", lambda **kwargs: None)
    monkeypatch.setattr(endpoint_module, "canary_write_step", lambda **kwargs: None)


def test_per_forward_orchestrates_plan_head_tail(device, monkeypatch):
    calls: List[str] = []
    _stub_plan_and_kernels(monkeypatch)
    monkeypatch.setattr(
        runner_module,
        "canary_plan_step",
        lambda **kwargs: calls.append("plan"),
    )
    from sglang.srt.kv_cache_canary import endpoint as endpoint_module

    monkeypatch.setattr(
        endpoint_module,
        "canary_verify_step",
        lambda **kwargs: calls.append(("verify", kwargs["kernel_kind"].name)),
    )
    monkeypatch.setattr(
        endpoint_module,
        "canary_write_step",
        lambda **kwargs: calls.append(("write", kwargs["kernel_kind"].name)),
    )

    runner = _make_runner(device=device)
    fb = _make_forward_batch(device)
    runner.run_head(forward_batch=fb)
    runner.run_tail(forward_batch=fb)

    assert calls[0] == "plan"
    assert any(
        c[0] == "verify" and "HEAD" in c[1] for c in calls[1:] if isinstance(c, tuple)
    )
    assert any(
        c[0] == "verify" and "TAIL" in c[1] for c in calls[1:] if isinstance(c, tuple)
    )


def test_sweep_every_n_cadence(device, monkeypatch):
    _stub_plan_and_kernels(monkeypatch)
    config = CanaryConfig(
        mode=CanaryMode.RAISE,
        real_kv_hash_mode=RealKvHashMode.OFF,
        real_data_sweep_every_n_steps=4,
    )
    runner = _make_runner(device=device, config=config)
    fb = _make_forward_batch(device)
    runner.set_last_forward_batch(fb)

    sweep_calls: List[int] = []
    real_run_sweep = runner.run_sweep
    monkeypatch.setattr(
        runner,
        "run_sweep",
        lambda: sweep_calls.append(runner._forward_step) or real_run_sweep(),
    )

    for _ in range(12):
        runner.end_of_forward()
    assert sweep_calls == [4, 8, 12]


def test_sweep_runs_both_running_and_radix_orphan(
    device, monkeypatch, make_radix_cache
):
    _stub_plan_and_kernels(monkeypatch)
    runner = _make_runner(device=device)
    runner.set_last_forward_batch(_make_forward_batch(device))
    runner.attach_radix_cache(make_radix_cache([[], [10, 11]]))

    invoked: List[str] = []
    original = runner._invoke_plan_kernel
    monkeypatch.setattr(
        runner,
        "_invoke_plan_kernel",
        lambda *, plan_input, plans: invoked.append("plan")
        or original(plan_input=plan_input, plans=plans),
    )
    launched: List[str] = []
    monkeypatch.setattr(
        runner, "_launch_sweep_endpoints", lambda **kwargs: launched.append("sweep")
    )
    runner.run_sweep()
    assert invoked.count("plan") == 2
    assert launched.count("sweep") == 2


def test_violation_pump_d2h_detects_errored(device, monkeypatch):
    _stub_plan_and_kernels(monkeypatch)
    runner = _make_runner(device=device)
    assert runner.is_errored() is False
    runner._latest_violation_write_index = 1
    assert runner.is_errored() is True


def test_cross_rank_allreduce_lockstep_raise(device, monkeypatch):
    _stub_plan_and_kernels(monkeypatch)
    runner = _make_runner(device=device)

    fake_state = {"local_flag": 0}

    def fake_is_initialized():
        return True

    def fake_get_tp_group():
        return SimpleNamespace(device_group=object())

    def fake_all_reduce(tensor, op, group):
        tensor.fill_(max(int(tensor.item()), fake_state["local_flag"]))

    monkeypatch.setattr(torch.distributed, "is_initialized", fake_is_initialized)
    monkeypatch.setattr(
        "sglang.srt.distributed.parallel_state.get_tp_group", fake_get_tp_group
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    fake_state["local_flag"] = 1
    assert runner._cross_rank_max(local_flag=0) == 1


def test_kernel_run_counter_watchdog_raises_on_zero(device, monkeypatch):
    _stub_plan_and_kernels(monkeypatch)
    config = CanaryConfig(
        mode=CanaryMode.RAISE,
        real_kv_hash_mode=RealKvHashMode.OFF,
        counter_zero_warmup_forwards=2,
    )
    runner = _make_runner(device=device, config=config)
    runner._forward_step = 5
    runner._latest_counters = (0, 0, 0, 0, 0, 0)
    monkeypatch.setattr(runner, "_cross_rank_max", lambda local_flag: local_flag)
    with pytest.raises(RuntimeError):
        runner._maybe_health_check()


def test_runner_disabled_short_circuits(device, monkeypatch):
    _stub_plan_and_kernels(monkeypatch)
    config = CanaryConfig(mode=CanaryMode.OFF)
    runner = _make_runner(device=device, config=config)

    plan_calls: List[str] = []
    monkeypatch.setattr(
        runner_module, "canary_plan_step", lambda **kwargs: plan_calls.append("plan")
    )
    fb = _make_forward_batch(device)
    runner.run_head(forward_batch=fb)
    runner.run_tail(forward_batch=fb)
    runner.run_sweep()
    runner.end_of_forward()
    assert plan_calls == []


def test_periodic_stats_log_every_n_step(device, monkeypatch, caplog):
    _stub_plan_and_kernels(monkeypatch)
    config = CanaryConfig(
        mode=CanaryMode.RAISE,
        real_kv_hash_mode=RealKvHashMode.OFF,
        health_print_every_n_forwards=5,
        counter_zero_warmup_forwards=1_000_000,
    )
    runner = _make_runner(device=device, config=config)
    runner._latest_counters = (7, 11, 21, 33, 0, 0)
    monkeypatch.setattr(runner, "_cross_rank_max", lambda local_flag: local_flag)

    with caplog.at_level(logging.INFO, logger=runner_module.logger.name):
        for step in range(1, 11):
            runner._forward_step = step
            runner._maybe_health_check()

    log_text = caplog.text
    assert "protected 5 forwards" in log_text or "protected 10 forwards" in log_text
    assert "head=7" in log_text
    assert "tail=11" in log_text


def test_per_forward_launches_both_head_and_tail(device, monkeypatch):
    _stub_plan_and_kernels(monkeypatch)
    runner = _make_runner(device=device)
    assert len(runner._head_endpoints) >= 1
    assert len(runner._tail_endpoints) >= 1

    counters: List[str] = []
    from sglang.srt.kv_cache_canary import endpoint as endpoint_module

    monkeypatch.setattr(
        endpoint_module,
        "canary_verify_step",
        lambda **kwargs: counters.append(kwargs["kernel_kind"].name),
    )
    monkeypatch.setattr(endpoint_module, "canary_write_step", lambda **kwargs: None)

    fb = _make_forward_batch(device)
    runner.run_head(forward_batch=fb)
    runner.run_tail(forward_batch=fb)
    assert any("HEAD" in name for name in counters)
    assert any("TAIL" in name for name in counters)


def test_sweep_path_detects_chain_mismatch(device, monkeypatch, make_radix_cache):
    _stub_plan_and_kernels(monkeypatch)
    runner = _make_runner(device=device)
    runner.set_last_forward_batch(_make_forward_batch(device))
    runner.attach_radix_cache(make_radix_cache([[], [10, 11, 12]]))

    sweep_kernel_kinds: List[str] = []
    from sglang.srt.kv_cache_canary import endpoint as endpoint_module

    monkeypatch.setattr(
        endpoint_module,
        "canary_verify_step",
        lambda **kwargs: sweep_kernel_kinds.append(kwargs["kernel_kind"].name),
    )
    runner.run_sweep()
    assert any("SWEEP" in k for k in sweep_kernel_kinds)


def test_runner_raises_when_other_rank_errored_but_local_clean(device, monkeypatch):
    _stub_plan_and_kernels(monkeypatch)
    runner = _make_runner(device=device)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        "sglang.srt.distributed.parallel_state.get_tp_group",
        lambda: SimpleNamespace(device_group=object()),
    )

    def lockstep_all_reduce(tensor, op, group):
        tensor.fill_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", lockstep_all_reduce)

    runner._latest_violation_write_index = 0
    flag = runner._cross_rank_max(local_flag=0)
    assert flag == 1
