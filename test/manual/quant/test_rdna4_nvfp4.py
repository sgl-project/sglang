"""Fail-closed numerical test for the exact gfx1201 NVFP4 JIT backend."""

import pytest
import torch

from sglang.kernels.ops.gemm.rdna4_nvfp4 import (
    is_rdna4_nvfp4_device,
    rdna4_nvfp4_linear,
)
from sglang.srt.layers.quantization.petit import PetitNvFp4Config
from sglang.srt.layers.quantization.petit_utils import RDNA4_NVFP4_BACKEND
from sglang.test.layer_ut_utils import (
    init_single_process_dist,
    load_linear_weights,
    make_tp1_column_parallel_linear,
)

E2M1_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _dequantize_weight(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    packed_cpu = packed.cpu().long()
    n, packed_k = packed_cpu.shape
    weight = torch.empty((n, packed_k * 2), dtype=torch.float32)
    weight[:, 0::2] = E2M1_VALUES[packed_cpu & 0xF]
    weight[:, 1::2] = E2M1_VALUES[packed_cpu >> 4]
    weight *= block_scale.float().cpu().repeat_interleave(16, dim=1)
    return weight * global_scale.float().cpu()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 1, 16),
        (1, 17, 528),
        (2, 3, 32),
        (7, 17, 272),
        (9, 33, 528),
        (127, 31, 272),
        (128, 64, 528),
        (129, 65, 272),
    ],
)
def test_rdna4_nvfp4_matches_independent_oracle(dtype, m, n, k):
    if not is_rdna4_nvfp4_device():
        raise RuntimeError(
            "This fail-closed test requires an exact gfx1201 ROCm device."
        )

    torch.manual_seed(20260828 + m + n + k)
    input_tensor = torch.randn((m, k), dtype=dtype, device="cuda")
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    block_scale = (
        (torch.rand((n, k // 16), dtype=torch.float32) * 3.75 + 0.0625)
        .to(torch.float8_e4m3fn)
        .cuda()
    )
    global_scale = torch.tensor([0.125], dtype=torch.float32, device="cuda")

    output = rdna4_nvfp4_linear(
        input_tensor,
        packed,
        block_scale,
        global_scale,
    )
    dequantized_weight = _dequantize_weight(
        packed,
        block_scale,
        global_scale,
    )
    reference = input_tensor.float().cpu() @ dequantized_weight.T
    rounded_reference = reference.to(dtype).float()

    torch.testing.assert_close(
        output.float().cpu(),
        rounded_reference,
        rtol=2e-2,
        atol=1.0 if dtype == torch.float16 else 0.5,
    )


def test_rdna4_nvfp4_real_linear_layer_path():
    if not is_rdna4_nvfp4_device():
        raise RuntimeError(
            "This fail-closed test requires an exact gfx1201 ROCm device."
        )

    init_single_process_dist()
    m, n, k = 3, 17, 32
    config = PetitNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo="FP8",
        group_size=16,
        exclude_modules=[],
    )
    layer = make_tp1_column_parallel_linear(config, n, k)

    torch.manual_seed(20260828)
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    block_scale = (
        (torch.rand((n, k // 16), dtype=torch.float32) * 3.75 + 0.0625)
        .to(torch.float8_e4m3fn)
        .cuda()
    )
    global_scale = torch.tensor([0.125], dtype=torch.float32, device="cuda")
    load_linear_weights(
        layer,
        weight=packed,
        weight_scale=block_scale,
        weight_scale_2=global_scale,
        input_scale=torch.ones(1, dtype=torch.float32, device="cuda"),
    )
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.nvfp4_backend == RDNA4_NVFP4_BACKEND

    input_tensor = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    output, _ = layer(input_tensor)
    dequantized_weight = _dequantize_weight(
        packed,
        block_scale,
        global_scale,
    )
    reference = input_tensor.float().cpu() @ dequantized_weight.T
    torch.testing.assert_close(
        output.float().cpu(),
        reference.to(torch.bfloat16).float(),
        rtol=2e-2,
        atol=0.5,
    )


@pytest.mark.parametrize("m", [1, 7])
def test_rdna4_nvfp4_cuda_graph_replay(m):
    if not is_rdna4_nvfp4_device():
        raise RuntimeError(
            "This fail-closed test requires an exact gfx1201 ROCm device."
        )

    n, k = 17, 272
    torch.manual_seed(20260828 + m)
    input_tensor = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    block_scale = (
        (torch.rand((n, k // 16), dtype=torch.float32) * 3.75 + 0.0625)
        .to(torch.float8_e4m3fn)
        .cuda()
    )
    global_scale = torch.tensor([0.125], dtype=torch.float32, device="cuda")

    rdna4_nvfp4_linear(input_tensor, packed, block_scale, global_scale)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = rdna4_nvfp4_linear(
            input_tensor,
            packed,
            block_scale,
            global_scale,
        )
    graph.replay()
    torch.cuda.synchronize()

    dequantized_weight = _dequantize_weight(
        packed,
        block_scale,
        global_scale,
    )
    reference = input_tensor.float().cpu() @ dequantized_weight.T
    torch.testing.assert_close(
        output.float().cpu(),
        reference.to(torch.bfloat16).float(),
        rtol=2e-2,
        atol=0.5,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
