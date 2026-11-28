"""The fused NVFP4 embedding kernel must be bit-exact with the torch fallback.

The fallback itself is checked against an independent per-element reference in
test/registered/unit/layers/quantization/test_nvfp4_embedding.py.
"""

import sys
from unittest import mock

import pytest
import torch
from torch._dynamo.testing import EagerAndRecordGraphs

from sglang.kernels.ops.embeddings import nvfp4_embedding as kernel_module
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptNvFp4EmbeddingMethod,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _fused_kernel_supported() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    return torch.cuda.get_device_capability() >= (8, 9)


pytestmark = pytest.mark.skipif(
    not _fused_kernel_supported(), reason="The fused kernel needs NVIDIA SM 8.9+."
)

GROUP_SIZE = 16
VOCAB_SIZE = 257

# Include the E4M3 extremes: the smallest subnormal and the largest finite value.
_SCALE_VALUES = torch.tensor([2.0**-9, 0.25, 0.5, 0.75, 1.0, 1.5, 3.0, 448.0])


def _make_layer(hidden_size, params_dtype=torch.bfloat16):
    method = ModelOptNvFp4EmbeddingMethod(
        ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=GROUP_SIZE)
    )
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=hidden_size,
        output_partition_sizes=[VOCAB_SIZE],
        input_size=hidden_size,
        output_size=VOCAB_SIZE,
        params_dtype=params_dtype,
    )

    generator = torch.Generator().manual_seed(0)
    # Random bytes put every 4-bit code in both nibble positions.
    layer.weight.data.copy_(
        torch.randint(
            0, 256, layer.weight.shape, dtype=torch.uint8, generator=generator
        )
    )
    picks = torch.randint(
        0, len(_SCALE_VALUES), layer.weight_scale.shape, generator=generator
    )
    layer.weight_scale.data.copy_(_SCALE_VALUES[picks].to(torch.float8_e4m3fn))
    layer.weight_scale_2.data.fill_(0.375)
    layer.cuda()
    method.process_weights_after_loading(layer)
    assert layer.use_fused_nvfp4_embedding
    return method, layer


def _fallback(method, layer, ids):
    with mock.patch.object(
        modelopt_quant, "_can_use_fused_nvfp4_embedding", return_value=False
    ):
        return method.embedding(layer, ids)


def _fused(method, layer, ids):
    with mock.patch.object(
        kernel_module, "nvfp4_embedding", wraps=kernel_module.nvfp4_embedding
    ) as kernel:
        output = method.embedding(layer, ids)
    assert kernel.call_count == 1, "An eligible lookup did not use the fused kernel."
    return output


# 2560 is not a power of two and spans several column blocks.
@pytest.mark.parametrize("hidden_size", [32, 160, 2560])
@pytest.mark.parametrize("params_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_matches_fallback_bit_exactly(hidden_size, params_dtype, id_dtype):
    method, layer = _make_layer(hidden_size, params_dtype)
    ids = torch.tensor(
        [0, VOCAB_SIZE - 1, 3, 91, 17, 128, 17], dtype=id_dtype, device="cuda"
    )

    actual = _fused(method, layer, ids)
    expected = _fallback(method, layer, ids)

    assert actual.dtype == params_dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "ids",
    [
        pytest.param(lambda: torch.tensor(5), id="scalar"),
        pytest.param(lambda: torch.tensor([[0, 7], [3, 3]]), id="2d"),
        pytest.param(lambda: torch.arange(16)[::3], id="non-contiguous"),
        pytest.param(lambda: torch.tensor([-1, -VOCAB_SIZE, 0]), id="negative"),
        pytest.param(lambda: torch.empty((0, 4), dtype=torch.long), id="empty"),
    ],
)
def test_id_layouts_match_fallback(ids):
    method, layer = _make_layer(hidden_size=160)
    ids = ids().cuda()

    actual = _fused(method, layer, ids)
    expected = _fallback(method, layer, ids)

    assert actual.shape == (*ids.shape, 160)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_torch_compile_matches_fallback():
    """--enable-torch-compile traces the lookup; it must not graph-break."""
    torch._dynamo.reset()
    method, layer = _make_layer(hidden_size=160)
    ids = torch.tensor([0, 5, VOCAB_SIZE - 1, 17], device="cuda")

    actual = torch.compile(method.embedding, fullgraph=True)(layer, ids)
    expected = _fallback(method, layer, ids)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_torch_compile_traces_fused_kernel():
    """The fallback is bit-identical, so only the graph shows which path compiled."""
    torch._dynamo.reset()
    method, layer = _make_layer(hidden_size=160)
    ids = torch.tensor([0, 5, VOCAB_SIZE - 1, 17], device="cuda")
    backend = EagerAndRecordGraphs()

    torch.compile(method.embedding, backend=backend, fullgraph=True)(layer, ids)

    assert len(backend.graphs) == 1
    targets = [str(node.target) for node in backend.graphs[0].graph.nodes]
    assert any("triton_kernel_wrapper" in target for target in targets), targets


def test_cuda_graph_replay_reads_updated_ids():
    method, layer = _make_layer(hidden_size=160)
    ids = torch.zeros(5, dtype=torch.long, device="cuda")
    # Compile outside capture.
    method.embedding(layer, ids)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = method.embedding(layer, ids)

    for new_ids in ([7, 6, 5, 4, 3], [VOCAB_SIZE - 1, VOCAB_SIZE - 1, 0, 1, -1]):
        ids.copy_(torch.tensor(new_ids, device="cuda"))
        graph.replay()
        expected = _fallback(method, layer, ids)
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
