import pytest
import torch
from torch import nn
from torch.overrides import TorchFunctionMode

import sglang.multimodal_gen.runtime.layers.lora.linear as lora_linear
from sglang.multimodal_gen.runtime.layers.lora.linear import (
    LinearWithLoRA,
    _compute_lora_delta,
)


class _AddmmInplaceRecorder(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.Tensor.addmm_:
            self.count += 1
        return func(*args, **(kwargs or {}))


def test_stacked_lora_delta_preserves_projection_order():
    x = torch.tensor([[2.0, 3.0]])
    lora_a = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    lora_b = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])

    actual = _compute_lora_delta(x, lora_a, lora_b)

    torch.testing.assert_close(actual, torch.tensor([[2.0, 4.0, 9.0, 12.0]]))


def test_lora_merge_unmerge_handles_inference_base_weight():
    with torch.inference_mode():
        base_layer = nn.Linear(4, 3, bias=False)

    layer = LinearWithLoRA(base_layer, lora_rank=2, lora_alpha=2)
    base_weight = layer.cpu_weight.clone()

    assert layer.base_layer.weight.is_inference()
    assert not base_weight.is_inference()

    lora_a = torch.ones(2, 4)
    lora_b = torch.full((3, 2), 0.5)
    expected_merged = base_weight + lora_b @ lora_a

    with torch.inference_mode(False):
        layer.set_lora_weights(
            lora_a,
            lora_b,
            clear_existing=True,
            merge_weights=True,
        )

    assert layer.merged
    assert not layer.base_layer.weight.is_inference()
    assert torch.allclose(layer.base_layer.weight, expected_merged)

    with torch.inference_mode(False):
        layer.unmerge_lora_weights()

    assert not layer.merged
    assert not layer.base_layer.weight.is_inference()
    assert torch.allclose(layer.base_layer.weight, base_weight)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_chunked_2d_merge_accumulates_directly_with_addmm(
    monkeypatch, dtype, noncontiguous
):
    torch.manual_seed(2)
    out_features, in_features, rank = 5, 7, 3
    base = nn.Linear(in_features, out_features, bias=False)
    layer = LinearWithLoRA(base, lora_rank=rank, lora_alpha=rank)
    if noncontiguous:
        data = torch.randn(in_features, out_features, dtype=dtype).T
        lora_a = torch.randn(in_features, rank, dtype=dtype).T
        lora_b = torch.randn(rank, out_features, dtype=dtype).T
    else:
        data = torch.randn(out_features, in_features, dtype=dtype)
        lora_a = torch.randn(rank, in_features, dtype=dtype)
        lora_b = torch.randn(out_features, rank, dtype=dtype)
    lora_a_2 = torch.randn_like(lora_a)
    lora_b_2 = torch.randn_like(lora_b)
    entries = [
        (lora_a, lora_b, None, -0.375, rank, rank, None),
        (lora_a_2, lora_b_2, None, 0.25, rank, 2 * rank, None),
    ]
    expected = data.clone(memory_format=torch.preserve_format)
    for entry in entries:
        a, b, _, strength, lora_rank, alpha, _ = entry
        scale = strength * alpha / lora_rank
        for start in range(0, out_features, 2):
            expected[start : start + 2].add_(b[start : start + 2] @ a, alpha=scale)
    storage_ptr = data.data_ptr()
    monkeypatch.setattr(
        lora_linear,
        "LORA_MERGE_CHUNK_BYTES",
        2 * in_features * data.element_size(),
    )
    recorder = _AddmmInplaceRecorder()

    with recorder:
        layer._merge_lora_into_data(data, entries)

    assert recorder.count == 6
    assert data.data_ptr() == storage_ptr
    atol = {torch.float32: 1e-5, torch.float16: 5e-3, torch.bfloat16: 3e-2}[dtype]
    rtol = atol
    torch.testing.assert_close(data, expected, atol=atol, rtol=rtol)
