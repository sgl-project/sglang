"""Regression tests for destination-backed CUDA-graph LoRA-B expansion."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/lora/torch_ops/graph_lora_ops.py"
)
_SPEC = importlib.util.spec_from_file_location("_graph_lora_ops", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_GRAPH_LORA_OPS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GRAPH_LORA_OPS)
sgemm_lora_b_graph_fwd = _GRAPH_LORA_OPS.sgemm_lora_b_graph_fwd

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _case(dtype: torch.dtype = torch.float32, num_loras: int = 2):
    torch.manual_seed(7)
    num_tokens = 4
    rank = 2
    slice_widths = (3, 4)
    slice_offsets = torch.tensor((0, 3, 7), dtype=torch.int32)
    inputs = torch.randn(num_tokens, len(slice_widths) * rank, dtype=dtype)
    weights = torch.randn(num_loras, sum(slice_widths), rank, dtype=dtype)
    weight_indices = torch.tensor((0, 1, -1, 0), dtype=torch.int32)
    seg_lens = torch.ones(num_tokens, dtype=torch.int32)

    backing = torch.randn(num_tokens, sum(slice_widths) + 2, dtype=dtype)
    base_output = backing[:, 1:-1]
    return (
        inputs,
        weights,
        weight_indices,
        seg_lens,
        slice_offsets,
        backing,
        base_output,
    )


def _reference(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    slice_offsets: torch.Tensor,
    base_output: torch.Tensor,
) -> torch.Tensor:
    expected = base_output.clone()
    num_slices = len(slice_offsets) - 1
    rank = inputs.shape[-1] // num_slices
    for lora_idx in range(weights.shape[0]):
        rows = (weight_indices == lora_idx).unsqueeze(1)
        masked_inputs = torch.where(rows, inputs, 0)
        for slice_idx in range(num_slices):
            input_start = slice_idx * rank
            input_end = input_start + rank
            output_start = int(slice_offsets[slice_idx])
            output_end = int(slice_offsets[slice_idx + 1])
            expected[:, output_start:output_end].add_(
                masked_inputs[:, input_start:input_end]
                @ weights[lora_idx, output_start:output_end].t()
            )
    return expected


def test_inference_accumulates_directly_into_packed_output(monkeypatch):
    (
        inputs,
        weights,
        weight_indices,
        seg_lens,
        slice_offsets,
        backing,
        base_output,
    ) = _case(num_loras=4)
    expected = _reference(inputs, weights, weight_indices, slice_offsets, base_output)
    padding_before = backing[:, (0, -1)].clone()

    real_addmm_ = torch.Tensor.addmm_
    calls: list[torch.Tensor] = []

    def recording_addmm_(self, mat1, mat2, *, beta=1, alpha=1):
        calls.append(self)
        return real_addmm_(self, mat1, mat2, beta=beta, alpha=alpha)

    monkeypatch.setattr(torch.Tensor, "addmm_", recording_addmm_)
    with torch.inference_mode():
        actual = sgemm_lora_b_graph_fwd(
            inputs,
            weights,
            weight_indices,
            seg_lens,
            slice_offsets,
            base_output,
        )

    assert len(calls) == weights.shape[0] * (len(slice_offsets) - 1)
    assert all(
        destination.untyped_storage().data_ptr()
        == base_output.untyped_storage().data_ptr()
        for destination in calls
    )
    assert actual.data_ptr() == base_output.data_ptr()
    assert not actual.is_contiguous()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(backing[:, (0, -1)], padding_before)


def _forbid_addmm_(*args, **kwargs):
    raise AssertionError("the direct-accumulation fast path must not run")


@pytest.mark.parametrize("num_loras", (1, 2, 3))
def test_small_lora_pool_uses_existing_fallback(monkeypatch, num_loras):
    (
        inputs,
        weights,
        weight_indices,
        seg_lens,
        slice_offsets,
        _,
        base_output,
    ) = _case(num_loras=num_loras)
    expected = _reference(inputs, weights, weight_indices, slice_offsets, base_output)

    monkeypatch.setattr(torch.Tensor, "addmm_", _forbid_addmm_)
    with torch.inference_mode():
        actual = sgemm_lora_b_graph_fwd(
            inputs,
            weights,
            weight_indices,
            seg_lens,
            slice_offsets,
            base_output,
        )

    torch.testing.assert_close(actual, expected)


def test_autograd_uses_existing_fallback_and_preserves_gradients(monkeypatch):
    inputs, weights, weight_indices, seg_lens, slice_offsets, _, _ = _case(
        torch.float64, num_loras=4
    )
    actual_inputs = inputs.clone().requires_grad_()
    actual_weights = weights.clone().requires_grad_()
    expected_inputs = inputs.clone().requires_grad_()
    expected_weights = weights.clone().requires_grad_()

    monkeypatch.setattr(torch.Tensor, "addmm_", _forbid_addmm_)
    actual = sgemm_lora_b_graph_fwd(
        actual_inputs,
        actual_weights,
        weight_indices,
        seg_lens,
        slice_offsets,
    )
    expected = _reference(
        expected_inputs,
        expected_weights,
        weight_indices,
        slice_offsets,
        torch.zeros_like(actual),
    )

    actual.square().sum().backward()
    expected.square().sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_inputs.grad, expected_inputs.grad)
    torch.testing.assert_close(actual_weights.grad, expected_weights.grad)


def test_mixed_output_dtype_uses_existing_fallback(monkeypatch):
    (
        inputs,
        weights,
        weight_indices,
        seg_lens,
        slice_offsets,
        _,
        low_precision_base,
    ) = _case(torch.float16, num_loras=4)
    backing = torch.randn(low_precision_base.shape[0], low_precision_base.shape[1] + 2)
    base_output = backing[:, 1:-1]
    expected = _reference(inputs, weights, weight_indices, slice_offsets, base_output)

    monkeypatch.setattr(torch.Tensor, "addmm_", _forbid_addmm_)
    with torch.inference_mode():
        actual = sgemm_lora_b_graph_fwd(
            inputs,
            weights,
            weight_indices,
            seg_lens,
            slice_offsets,
            base_output,
        )

    assert actual.dtype == torch.float32
    assert actual.data_ptr() == base_output.data_ptr()
    torch.testing.assert_close(actual, expected)


def test_torch_compile_fullgraph_uses_direct_accumulation():
    (
        inputs,
        weights,
        weight_indices,
        seg_lens,
        slice_offsets,
        _,
        base_output,
    ) = _case(num_loras=4)
    expected = _reference(inputs, weights, weight_indices, slice_offsets, base_output)

    def run_compiled(inputs, weights, weight_indices, base_output):
        return sgemm_lora_b_graph_fwd(
            inputs,
            weights,
            weight_indices,
            seg_lens,
            slice_offsets,
            base_output,
        )

    captured_graphs = []

    def recording_backend(graph_module, _example_inputs):
        captured_graphs.append(graph_module)
        return graph_module.forward

    compiled = torch.compile(run_compiled, backend=recording_backend, fullgraph=True)
    with (
        patch.dict(sys.modules, {sgemm_lora_b_graph_fwd.__module__: _GRAPH_LORA_OPS}),
        torch.inference_mode(),
    ):
        actual = compiled(inputs, weights, weight_indices, base_output)

    torch.testing.assert_close(actual, expected)
    assert len(captured_graphs) == 1
    assert any(
        node.op == "call_method" and node.target == "addmm_"
        for node in captured_graphs[0].graph.nodes
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
