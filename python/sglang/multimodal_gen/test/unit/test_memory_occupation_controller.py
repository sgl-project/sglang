# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang.multimodal_gen.runtime.managers.memory_managers import (
    memory_occupation_controller as memory_controller,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA pinned-memory transfers",
)


class _StridedFp16Module(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(6, device=device, dtype=torch.float16).reshape(2, 3).t()
        )
        self.bias = torch.nn.Parameter(
            (torch.arange(6, device=device, dtype=torch.float16) + 10).reshape(2, 3).t()
        )
        self.register_buffer(
            "offset",
            (torch.arange(6, device=device, dtype=torch.float16) + 20)
            .reshape(2, 3)
            .t(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias + self.offset


@pytest.mark.parametrize(
    "failure_at",
    [1, 2],
    ids=["first_allocation", "after_partial_move"],
)
def test_release_rolls_back_partial_pinned_offload(
    monkeypatch: pytest.MonkeyPatch, failure_at: int
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    module = _StridedFp16Module(device)
    pipeline = object()
    controller = memory_controller.MemoryOccupationController(
        pipeline=pipeline,
        rank=0,
        use_fsdp_inference=False,
    )

    monkeypatch.setattr(
        memory_controller,
        "get_updatable_modules",
        lambda current_pipeline: (
            {"transformer": module}
            if current_pipeline is pipeline
            else pytest.fail("unexpected pipeline")
        ),
    )
    monkeypatch.setattr(
        memory_controller,
        "_is_layerwise_offload_managed",
        lambda _module: False,
    )

    original_tensors = {
        name: (
            tensor.device,
            tensor.detach().clone(),
            tensor.stride(),
        )
        for name, tensor in (
            list(module.named_parameters()) + list(module.named_buffers())
        )
    }
    forward_input = torch.full(
        module.weight.shape,
        0.5,
        device=device,
        dtype=torch.float16,
    )
    expected_output = module(forward_input).detach().clone()

    real_empty_strided = torch.empty_strided
    allocation_count = 0
    successful_allocations = []

    def faulting_empty_strided(*args, **kwargs):
        nonlocal allocation_count
        allocation_count += 1
        if allocation_count == failure_at:
            raise RuntimeError("injected pinned allocation failure")
        pin = real_empty_strided(*args, **kwargs)
        successful_allocations.append(pin)
        return pin

    monkeypatch.setattr(torch, "empty_strided", faulting_empty_strided)

    with pytest.raises(RuntimeError, match="injected pinned allocation failure"):
        controller.release_memory_occupation()

    assert allocation_count == failure_at
    assert len(successful_allocations) == failure_at - 1
    assert all(pin.is_pinned() for pin in successful_allocations)
    assert not controller.is_sleeping()
    assert controller.resume_memory_occupation() == {
        "success": True,
        "sleeping": False,
        "message": "already awake",
    }

    current_tensors = dict(module.named_parameters()) | dict(module.named_buffers())
    assert current_tensors.keys() == original_tensors.keys()
    for name, tensor in current_tensors.items():
        original_device, original_value, original_stride = original_tensors[name]
        assert tensor.device == original_device
        assert tensor.stride() == original_stride
        torch.testing.assert_close(tensor, original_value, rtol=0, atol=0)

    torch.testing.assert_close(
        module(forward_input),
        expected_output,
        rtol=0,
        atol=0,
    )
