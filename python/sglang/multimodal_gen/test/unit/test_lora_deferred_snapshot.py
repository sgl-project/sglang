"""A device-resident LoRA layer takes its unmerge snapshot only when first merged,
pinned while the worker's pin budget covers it."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.lora import linear as lora_linear
from sglang.multimodal_gen.runtime.layers.lora.linear import wrap_with_lora_layer
from sglang.multimodal_gen.runtime.managers.memory_managers import component_manager
from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_budget import (
    HostPinBudget,
)

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _DEVICES)
@torch.no_grad()
def test_dynamic_lora_takes_no_snapshot_and_merge_restores_base(device, monkeypatch):
    torch.manual_seed(0)
    budget = HostPinBudget(available_bytes=1 << 20, reserve_bytes=0)
    monkeypatch.setattr(lora_linear, "_snapshot_pin_budget", lambda: budget)
    base = torch.nn.Linear(8, 16, bias=False, dtype=torch.bfloat16).to(device)
    original = base.weight.detach().clone()
    layer = wrap_with_lora_layer(base, lora_rank=2, lora_alpha=2)
    assert layer.cpu_weight is None

    A = torch.randn(2, 8, dtype=torch.bfloat16, device=device)
    B = torch.randn(16, 2, dtype=torch.bfloat16, device=device)
    layer.set_lora_weights(A, B, clear_existing=True, merge_weights=False)
    assert layer.cpu_weight is None

    layer.set_lora_weights(A, B, clear_existing=True, merge_weights=True)
    assert layer.merged
    assert layer.cpu_weight is not None
    assert layer.cpu_weight.device.type == "cpu"
    if device == "cuda":
        # Pinned, so the unmerge restore is a fast host-to-device copy.
        assert layer.cpu_weight.is_pinned()
        assert budget.committed_bytes == original.numel() * original.element_size()
    torch.testing.assert_close(layer.cpu_weight, original.cpu(), rtol=0, atol=0)
    expected = original.float() + B.float() @ A.float()
    torch.testing.assert_close(base.weight.float(), expected, rtol=2e-2, atol=2e-2)

    layer.unmerge_lora_weights()
    assert layer.cpu_weight is None
    torch.testing.assert_close(base.weight, original, rtol=0, atol=0)


def test_cpu_layer_defers_snapshot_until_merge():
    base = torch.nn.Linear(4, 4, bias=False)
    layer = wrap_with_lora_layer(base, lora_rank=2, lora_alpha=2)
    assert layer.cpu_weight is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("budget", [None, HostPinBudget(available_bytes=0)])
@torch.no_grad()
def test_snapshot_stays_pageable_without_pin_allowance(budget, monkeypatch):
    monkeypatch.setattr(lora_linear, "_snapshot_pin_budget", lambda: budget)
    base = torch.nn.Linear(8, 16, bias=False, dtype=torch.bfloat16).to("cuda")
    original = base.weight.detach().clone()
    layer = wrap_with_lora_layer(base, lora_rank=2, lora_alpha=2)
    A = torch.randn(2, 8, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(16, 2, dtype=torch.bfloat16, device="cuda")
    layer.set_lora_weights(A, B, clear_existing=True, merge_weights=True)

    assert not layer.cpu_weight.is_pinned()
    assert budget is None or budget.committed_bytes == 0
    layer.unmerge_lora_weights()
    assert layer.cpu_weight is None
    torch.testing.assert_close(base.weight, original, rtol=0, atol=0)


@pytest.mark.parametrize("pin_cpu_memory", [True, False])
def test_snapshot_pin_budget_follows_the_pin_setting(pin_cpu_memory, monkeypatch):
    budget = HostPinBudget(available_bytes=1 << 20, reserve_bytes=0)
    manager = SimpleNamespace(
        server_args=SimpleNamespace(pin_cpu_memory=pin_cpu_memory),
        host_pin_budget=budget,
    )
    monkeypatch.setattr(
        component_manager, "peek_global_component_residency_manager", lambda: manager
    )
    assert lora_linear._snapshot_pin_budget() is (budget if pin_cpu_memory else None)


def test_snapshot_pin_budget_is_none_without_a_worker(monkeypatch):
    monkeypatch.setattr(
        component_manager, "peek_global_component_residency_manager", lambda: None
    )
    assert lora_linear._snapshot_pin_budget() is None
