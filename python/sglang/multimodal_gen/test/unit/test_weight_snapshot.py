"""Snapshot pin allowances follow storage ownership, not request lifetimes."""

import gc
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.runtime.managers.memory_managers import host_memory_budget
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentResidencyManager,
    ComponentUse,
    ResidencyState,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_budget import (
    MIN_HOST_RESERVE_BYTES,
    HostPinBudget,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.memory_occupation_controller import (
    MemoryOccupationController,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.weight_snapshot import (
    capture_weight_snapshot,
    restore_weight_snapshot,
    weight_snapshot,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("pin", [False, True])
@pytest.mark.parametrize("mapped", [False, True])
def test_manager_pins_once_across_requests_and_strategy_rebuilds(tmp_path, pin, mapped):
    module = torch.nn.Linear(4, 4, bias=False)
    if mapped:
        checkpoint = str(tmp_path / "model.safetensors")
        save_file(module.state_dict(), checkpoint)
        module.load_state_dict(load_file(checkpoint), assign=True)
    expected = module.weight.detach().clone()
    pipeline = SimpleNamespace(
        modules={"vae": module},
        _stage_name_mapping={},
        component_residency_strategies={},
    )
    args = SimpleNamespace(
        residency_mode=lambda _: "snapshot-offload", pin_cpu_memory=pin
    )
    manager = ComponentResidencyManager(pipeline, args)
    budget = manager.host_pin_budget
    use = ComponentUse("decode", "vae")
    state = ResidencyState()
    pointer = None
    for _ in range(3):
        strategy = manager.strategy_for("vae", module)
        strategy.prefetch_for_use(module, use, state)
        strategy.wait_for_use(module, use, state)
        snapshot = weight_snapshot(module)
        assert snapshot["weight"].is_pinned() == pin
        assert budget.committed_bytes == (64 if pin else 0)
        torch.testing.assert_close(module.weight, expected.cuda(), rtol=0, atol=0)
        strategy.finish_use(module, use, state)
        if pointer is not None:
            assert module.weight.data_ptr() == pointer
        pointer = module.weight.data_ptr()
        # real server-args replacement invalidates the cached strategy
        manager.refresh_server_args(SimpleNamespace(**vars(args)))
        assert manager.host_pin_budget is budget
    del snapshot
    module.weight = torch.nn.Parameter(expected)
    gc.collect()
    assert budget.committed_bytes == 0


def test_partial_budget_preserves_shared_storage_and_strided_views():
    storage = torch.arange(32, dtype=torch.float32)
    module = torch.nn.Module()
    module.a = torch.nn.Parameter(storage[:16].reshape(4, 4).T)
    module.b = torch.nn.Parameter(storage[16:])
    module.tied = module.a
    module.other = torch.nn.Parameter(torch.ones(32))
    budget = HostPinBudget(available_bytes=MIN_HOST_RESERVE_BYTES + 128)
    capture_weight_snapshot(module, pin_budget=budget, component_name="vae")
    assert budget.committed_bytes == 128
    assert module.a.is_pinned() and module.b.is_pinned()
    assert not module.other.is_pinned()
    assert module.a is module.tied
    assert module.a.stride() == (1, 4)
    assert module.b.storage_offset() == 16
    assert (
        module.a.untyped_storage().data_ptr() == module.b.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(module.a, storage[:16].reshape(4, 4).T, rtol=0, atol=0)
    torch.testing.assert_close(module.b, storage[16:], rtol=0, atol=0)
    restore_weight_snapshot(module)
    backup = module.b.detach()
    del module.a, module.b, module.tied
    gc.collect()
    assert budget.committed_bytes == 128
    del backup
    gc.collect()
    assert budget.committed_bytes == 0


def test_lora_replacement_and_dtype_change_return_pin_allowance():
    module = torch.nn.Linear(4, 4, bias=False)
    pipeline = SimpleNamespace(modules={"transformer": module})
    budget = HostPinBudget(available_bytes=MIN_HOST_RESERVE_BYTES + 128)
    for value in (2.0, 3.0):
        capture_weight_snapshot(module, pin_budget=budget)
        module.cuda()
        assert budget.committed_bytes == 64
        with LoRAPipeline._temporarily_disable_offload(
            pipeline, target="transformer", use_module_names_only=True
        ):
            module.weight = torch.nn.Parameter(torch.full((4, 4), value))
        gc.collect()
        assert budget.committed_bytes == 0
        capture_weight_snapshot(module, pin_budget=budget)
        module.cuda()
        torch.testing.assert_close(
            module.weight, torch.full((4, 4), value, device="cuda")
        )
        restore_weight_snapshot(module)
    module.to(dtype=torch.bfloat16)
    gc.collect()
    assert budget.committed_bytes == 0
    capture_weight_snapshot(module, pin_budget=budget)
    assert budget.committed_bytes == 32
    restore_weight_snapshot(module)


def test_live_headroom_prevents_pin_copy(monkeypatch):
    module = torch.nn.Linear(4, 4, bias=False)
    pointer = module.weight.data_ptr()
    budget = HostPinBudget(available_bytes=16 * 1024**3)
    monkeypatch.setattr(host_memory_budget, "host_memory_available_bytes", lambda: 0)
    capture_weight_snapshot(module, pin_budget=budget)
    assert module.weight.data_ptr() == pointer
    assert not module.weight.is_pinned()
    assert budget.committed_bytes == 0
    restore_weight_snapshot(module)


def test_sleep_keeps_the_pin_lease_and_existing_pins_are_reused():
    module = torch.nn.Linear(4, 4, bias=False)
    budget = HostPinBudget(available_bytes=MIN_HOST_RESERVE_BYTES + 64)
    capture_weight_snapshot(module, pin_budget=budget)
    pointer = module.weight.data_ptr()
    module.cuda()
    pipeline = SimpleNamespace(modules={"transformer": module})
    controller = MemoryOccupationController(pipeline, rank=0, use_fsdp_inference=False)
    controller._move_modules(["transformer"], "cpu")
    assert module.weight.data_ptr() == pointer
    assert budget.committed_bytes == 64
    capture_weight_snapshot(module, pin_budget=budget)
    assert budget.committed_bytes == 64
    restore_weight_snapshot(module)
    del module, pipeline, controller
    gc.collect()
    assert budget.committed_bytes == 0

    module = torch.nn.Linear(4, 4, bias=False)
    module.weight.data = module.weight.detach().pin_memory()
    pointer = module.weight.data_ptr()
    capture_weight_snapshot(module, pin_budget=budget)
    assert module.weight.data_ptr() == pointer
    assert budget.committed_bytes == 0
    restore_weight_snapshot(module)


def test_failed_pin_allocation_returns_allowance(monkeypatch):
    module = torch.nn.Linear(4, 4, bias=False)
    pointer = module.weight.data_ptr()
    budget = HostPinBudget(available_bytes=MIN_HOST_RESERVE_BYTES + 64)

    def fail_pin(storage, device="cuda"):
        raise RuntimeError("pin allocation failed")

    monkeypatch.setattr(torch.UntypedStorage, "pin_memory", fail_pin)
    with pytest.raises(RuntimeError, match="pin allocation failed"):
        capture_weight_snapshot(module, pin_budget=budget)
    assert budget.committed_bytes == 0
    assert module.weight.data_ptr() == pointer
    assert weight_snapshot(module) is None
