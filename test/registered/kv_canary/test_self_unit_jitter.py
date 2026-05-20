"""Host-side unit tests for the kv-canary timing-jitter fuzzer."""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest
import torch

from sglang.jit_kernel.kv_canary.verify import CANARY_SLOT_BYTES, RealKvHashMode
from sglang.srt.kv_canary import endpoint as endpoint_module
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_canary.runner import canary_runner as runner_module
from sglang.srt.kv_canary.runner import jitter as jitter_module
from sglang.srt.kv_canary.runner.canary_runner import CanaryRunner
from sglang.srt.kv_canary.runner.jitter import JitterConfig, JitterRunner, JitterSlot
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-large")


_DEVICE = torch.device("cuda")


def _make_runner(
    *,
    enabled: bool = True,
    per_slot_fire_prob: float = 0.5,
    max_cycles: int = 100_000,
    seed: int = 0,
) -> JitterRunner:
    config = JitterConfig(
        enabled=enabled,
        per_slot_fire_prob=per_slot_fire_prob,
        max_cycles=max_cycles,
        seed=seed,
    )
    return JitterRunner(config=config, device=_DEVICE)


def test_jitter_runner_rejects_disabled_construction() -> None:
    """``JitterRunner`` must only be constructed when enabled; disabled is represented as None at the
    caller layer to keep the hot path branch-free.
    """
    config = JitterConfig(enabled=False)
    with pytest.raises(ValueError, match="enabled=True"):
        JitterRunner(config=config, device=_DEVICE)


def test_jitter_runner_allocates_static_buffer() -> None:
    runner = _make_runner()
    assert runner.cycles.shape == (JitterRunner.NUM_SLOTS,)
    assert runner.cycles.dtype == torch.int64
    assert runner.cycles.device.type == "cuda"

    storage_ptr_before = runner.cycles.data_ptr()
    runner.randomize_for_next_step()
    assert (
        runner.cycles.data_ptr() == storage_ptr_before
    ), "randomize_for_next_step must mutate in place; cuda graph captures the buffer address"


def test_fire_prob_zero_never_fires() -> None:
    """``per_slot_fire_prob=0`` -> all 4 slot cycles stay at 0 across many randomizations."""
    runner = _make_runner(per_slot_fire_prob=0.0, max_cycles=1_000_000)
    for _ in range(200):
        runner.randomize_for_next_step()
    max_val = int(runner.cycles.max().item())
    assert max_val == 0, f"fire_prob=0 must keep cycles at 0, saw max={max_val}"


def test_fire_prob_one_always_fires() -> None:
    runner = _make_runner(per_slot_fire_prob=1.0, max_cycles=10_000)
    seen_zero_slot = False
    for _ in range(50):
        runner.randomize_for_next_step()
        slot_vals = runner.cycles.cpu().tolist()
        if min(slot_vals) <= 0:
            seen_zero_slot = True
    assert not seen_zero_slot, "fire_prob=1 must not produce any 0-cycle slots"


def test_max_cycles_is_respected() -> None:
    cap = 1_000
    runner = _make_runner(per_slot_fire_prob=1.0, max_cycles=cap)
    observed_max = 0
    for _ in range(2000):
        runner.randomize_for_next_step()
        observed_max = max(observed_max, int(runner.cycles.max().item()))
    assert observed_max <= cap
    assert (
        observed_max >= cap // 4
    ), f"expected sampling to approach the cap with fire_prob=1; observed_max={observed_max}"


def test_log_uniform_distribution_shape() -> None:
    """With fire_prob=1 and large max_cycles, sampled values must span multiple decades."""
    cap = 1_000_000
    runner = _make_runner(per_slot_fire_prob=1.0, max_cycles=cap, seed=42)
    bins = [0, 0, 0]  # <100, [100, 10_000), >=10_000
    iters = 4000
    for _ in range(iters):
        runner.randomize_for_next_step()
        for v in runner.cycles.cpu().tolist():
            if v < 100:
                bins[0] += 1
            elif v < 10_000:
                bins[1] += 1
            else:
                bins[2] += 1
    total = sum(bins)
    for i, count in enumerate(bins):
        share = count / total
        assert (
            share > 0.05
        ), f"log-uniform sampler bin {i} too sparse: {count}/{total} = {share:.3f}"


def test_seed_reproducibility() -> None:
    a = _make_runner(per_slot_fire_prob=0.7, max_cycles=10_000, seed=1234)
    b = _make_runner(per_slot_fire_prob=0.7, max_cycles=10_000, seed=1234)
    for _ in range(50):
        a.randomize_for_next_step()
        b.randomize_for_next_step()
    assert torch.equal(a.cycles, b.cycles)


def test_different_seed_produces_different_sequence() -> None:
    a = _make_runner(per_slot_fire_prob=0.7, max_cycles=10_000, seed=1)
    b = _make_runner(per_slot_fire_prob=0.7, max_cycles=10_000, seed=2)
    diverged = False
    for _ in range(20):
        a.randomize_for_next_step()
        b.randomize_for_next_step()
        if not torch.equal(a.cycles, b.cycles):
            diverged = True
            break
    assert diverged, "different seeds must produce different cycles sequences"


def test_launch_slot_executes_on_device(monkeypatch) -> None:
    """``launch_slot`` must invoke ``spin_wait_step`` once per slot and synchronize cleanly.

    Counts kernel launches via monkeypatch so a refactor that silently drops a slot launch fails
    here; wall-time coverage lives in the kernel unit tests under ``jit_kernel/tests``.
    """
    runner = _make_runner(per_slot_fire_prob=1.0, max_cycles=1_000)
    runner.randomize_for_next_step()

    launch_calls: List[int] = []
    real_spin_wait = jitter_module.spin_wait_step

    def _counting_spin_wait(*, cycles: torch.Tensor) -> None:
        launch_calls.append(1)
        real_spin_wait(cycles=cycles)

    monkeypatch.setattr(jitter_module, "spin_wait_step", _counting_spin_wait)

    for slot in JitterSlot:
        runner.launch_slot(slot=slot)
    torch.cuda.synchronize()

    assert len(launch_calls) == JitterRunner.NUM_SLOTS, (
        f"launch_slot must fire exactly once per slot; "
        f"observed {len(launch_calls)} for {JitterRunner.NUM_SLOTS} slots"
    )


def test_step_counter_increments() -> None:
    runner = _make_runner()
    assert runner.step_counter == 0
    runner.randomize_for_next_step()
    runner.randomize_for_next_step()
    assert runner.step_counter == 2


def test_jitter_config_rejects_invalid_fire_prob() -> None:
    with pytest.raises(ValueError, match="per_slot_fire_prob"):
        JitterConfig(enabled=True, per_slot_fire_prob=1.5)


def test_jitter_config_rejects_invalid_max_cycles() -> None:
    with pytest.raises(ValueError, match="max_cycles"):
        JitterConfig(enabled=True, max_cycles=0)


def test_num_slots_matches_jitter_slot_enum() -> None:
    assert JitterRunner.NUM_SLOTS == len(JitterSlot)
    assert {slot.value for slot in JitterSlot} == set(range(JitterRunner.NUM_SLOTS))


def _make_canary_runner_disabled_jitter(*, device: torch.device) -> CanaryRunner:
    config = CanaryConfig(
        mode=CanaryMode.RAISE,
        real_kv_hash_mode=RealKvHashMode.OFF,
        jitter_config=JitterConfig(enabled=False),
    )
    group = CanaryBufferGroup(
        kind=PoolKind.FULL,
        k_head=torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device),
        k_tail=torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device),
        v_head=torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device),
        v_tail=torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device),
        real_kv_sources_k=(),
        real_kv_sources_v=(),
        swa_index_lut=None,
    )
    req_pool = SimpleNamespace(
        req_to_token=torch.zeros(4, 8, dtype=torch.int32, device=device),
        size=4,
    )
    return CanaryRunner(
        config=config,
        buffer_groups_per_pool=[(group,)],
        device=device,
        req_to_token_pool=req_pool,
        per_forward_verify_capacity=4,
        per_forward_write_req_capacity=2,
        per_forward_write_entry_capacity=8,
        sweep_verify_capacity=8,
    )


def _make_forward_batch_for_disabled_jitter(device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: False, is_mixed=lambda: False),
        req_pool_indices=torch.tensor([1, 2], dtype=torch.int64, device=device),
        seq_lens=torch.tensor([3, 4], dtype=torch.int32, device=device),
        extend_prefix_lens=None,
        extend_seq_lens=None,
        input_ids=torch.zeros(2, dtype=torch.int32, device=device),
        positions=torch.zeros(2, dtype=torch.int32, device=device),
        out_cache_loc=torch.zeros(2, dtype=torch.int32, device=device),
    )


def test_jitter_disabled_zero_overhead(monkeypatch) -> None:
    """With ``jitter_config.enabled=False`` the CanaryRunner must allocate no jitter buffer and
    must not launch any ``spin_wait_step`` from ``before_forward`` / ``launch_head_kernels`` /
    ``launch_tail_kernels``.
    """
    device = torch.device("cpu")
    spin_wait_calls: List[int] = []
    monkeypatch.setattr(
        jitter_module,
        "spin_wait_step",
        lambda **kwargs: spin_wait_calls.append(1),
    )
    monkeypatch.setattr(runner_module, "canary_plan_step", lambda **kwargs: None)
    monkeypatch.setattr(endpoint_module, "canary_verify_step", lambda **kwargs: None)
    monkeypatch.setattr(endpoint_module, "canary_write_step", lambda **kwargs: None)

    runner = _make_canary_runner_disabled_jitter(device=device)
    assert (
        runner._jitter is None
    ), "disabled jitter must yield CanaryRunner._jitter is None (no buffer allocation)"

    fb = _make_forward_batch_for_disabled_jitter(device)
    with runner.with_forward_pass(fb):
        runner.launch_head_kernels(fb)
        runner.launch_tail_kernels(fb)

    assert spin_wait_calls == [], (
        f"disabled jitter must not launch any spin_wait_step kernel; "
        f"observed {len(spin_wait_calls)} launches"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
