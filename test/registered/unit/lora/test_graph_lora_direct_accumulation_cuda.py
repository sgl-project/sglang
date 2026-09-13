"""CUDA Graph replay coverage for destination-backed LoRA-B expansion."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/lora/torch_ops/graph_lora_ops.py"
)
_SPEC = importlib.util.spec_from_file_location("_graph_lora_cuda_ops", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_GRAPH_LORA_OPS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GRAPH_LORA_OPS)
sgemm_lora_b_graph_fwd = _GRAPH_LORA_OPS.sgemm_lora_b_graph_fwd

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


_CUDA_GRAPH_AVAILABLE = bool(
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability()[0] >= 8
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
        masked_inputs = torch.where(
            (weight_indices == lora_idx).unsqueeze(1), inputs, 0
        )
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


@pytest.mark.skipif(
    not _CUDA_GRAPH_AVAILABLE,
    reason="requires an NVIDIA GPU with BF16 and CUDA Graph support",
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_changed_inputs_replay_into_packed_output(dtype):
    device = torch.device("cuda")
    num_tokens = 8
    num_loras = 4
    rank = 4
    total_output_dim = 12
    slice_offsets = torch.tensor((0, 5, 12), dtype=torch.int32)
    seg_lens = torch.ones(num_tokens, dtype=torch.int32, device=device)
    weight_indices = torch.tensor(
        (0, 1, 2, 3, -1, 0, 2, 3), dtype=torch.int32, device=device
    )

    torch.manual_seed(17)
    static_inputs = torch.randn(num_tokens, 2 * rank, dtype=dtype, device=device)
    static_weights = torch.randn(
        num_loras, total_output_dim, rank, dtype=dtype, device=device
    )
    base_template = torch.randn(
        num_tokens, total_output_dim + 2, dtype=dtype, device=device
    )
    static_backing = base_template.clone()
    static_output = static_backing[:, 1:-1]

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream), torch.inference_mode():
        sgemm_lora_b_graph_fwd(
            static_inputs,
            static_weights,
            weight_indices,
            seg_lens,
            slice_offsets,
            static_output,
        )
    torch.cuda.current_stream().wait_stream(side_stream)
    static_backing.copy_(base_template)

    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        captured_output = sgemm_lora_b_graph_fwd(
            static_inputs,
            static_weights,
            weight_indices,
            seg_lens,
            slice_offsets,
            static_output,
        )

    for replay_seed in (23, 29):
        torch.manual_seed(replay_seed)
        new_inputs = torch.randn_like(static_inputs)
        new_weights = torch.randn_like(static_weights)
        static_inputs.copy_(new_inputs)
        static_weights.copy_(new_weights)
        static_backing.copy_(base_template)

        graph.replay()
        torch.cuda.synchronize()

        expected = _reference(
            new_inputs,
            new_weights,
            weight_indices,
            slice_offsets,
            base_template[:, 1:-1],
        )
        assert captured_output.data_ptr() == static_output.data_ptr()
        assert not captured_output.is_contiguous()
        torch.testing.assert_close(captured_output, expected, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(
            static_backing[:, (0, -1)], base_template[:, (0, -1)]
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
