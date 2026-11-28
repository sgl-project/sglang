import sys
from types import SimpleNamespace

import pytest
import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.srt.layers.quantization.marlin_utils import check_marlin_supported
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    nvfp4_marlin_process_global_scale,
    prepare_nvfp4_layer_for_marlin,
)
from sglang.srt.utils.device import (
    is_sm80_supported,
    is_sm90_supported,
    is_sm120_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import make_nvfp4_weight_and_ref

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")
register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.skipif(
    not (is_sm80_supported() or is_sm90_supported() or is_sm120_supported()),
    reason="NVFP4 Marlin fallback tests require CUDA SM8X/SM9X/SM120",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_marlin_support_and_scale_transforms_sm80_sm90_sm120(dtype):
    major, minor = torch.cuda.get_device_capability()
    capability = major * 10 + minor
    assert check_marlin_supported(
        scalar_types.float4_e2m1f,
        group_size=16,
        has_zp=False,
        device_capability=capability,
    )

    global_scale = torch.tensor(1.0, dtype=dtype, device="cuda")
    actual_global_scale = nvfp4_marlin_process_global_scale(global_scale)
    assert actual_global_scale.is_cuda
    assert actual_global_scale.ndim == 1
    assert actual_global_scale.numel() == 1
    if dtype == torch.float16:
        assert actual_global_scale.item() == 128.0
    else:
        assert actual_global_scale.item() == 2.0**119


@pytest.mark.skipif(
    not (is_sm80_supported() or is_sm90_supported() or is_sm120_supported()),
    reason="NVFP4 Marlin dense numeric test requires CUDA SM80, SM86, SM90 or SM120",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_marlin_dense_matches_dequant_reference(dtype):
    torch.manual_seed(0)

    size_m = 17
    size_k = 256
    size_n = 192
    group_size = 16

    a_input = torch.randn((size_m, size_k), dtype=dtype, device="cuda") / 10
    fp4_weight, scales, global_scale, weight_ref = make_nvfp4_weight_and_ref(
        size_n, size_k, dtype, group_size=group_size
    )

    layer = torch.nn.Module()
    layer.quant_config = SimpleNamespace(group_size=group_size)
    layer.output_size_per_partition = size_n
    layer.input_size_per_partition = size_k
    layer.params_dtype = dtype
    layer.weight = torch.nn.Parameter(fp4_weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(
        global_scale.reshape(1), requires_grad=False
    )
    prepare_nvfp4_layer_for_marlin(layer)

    output = apply_fp4_marlin_linear(
        a_input,
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        layer.workspace,
        size_n,
        size_k,
        use_fp32_reduce=True,
    )

    output_ref = torch.matmul(a_input, weight_ref.T)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, output_ref, rtol=0.04, atol=0.04)


@pytest.mark.skipif(
    not (is_sm80_supported() or is_sm90_supported() or is_sm120_supported()),
    reason="NVFP4 Marlin dense numeric test requires CUDA SM80, SM86, SM90 or SM120",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_marlin_dense_subnormal_block_scales(dtype):
    # E4M3 subnormal block scales (below 2^-6) occur in tensors whose largest
    # block scale is near 448, and in checkpoints exported with a global scale
    # of 1. Both activation dtypes must decode them exactly, and zero scales
    # must still give zero weights.
    torch.manual_seed(0)
    size_n, size_k, group_size = 256, 1024, 16
    e2m1 = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    codes = torch.randint(0, 16, (size_n, size_k), dtype=torch.uint8)

    # Half of the output channels use scale 448, the other half zero, the seven
    # E4M3 subnormals and the smallest normal value, 2^-6.
    small = torch.tensor([0.0] + [i * 2**-9 for i in range(1, 8)] + [2**-6])
    scales = torch.full((size_n, size_k // group_size), 448.0)
    scales[size_n // 2 :] = small.repeat(size_k // group_size // len(small) + 1)[
        : size_k // group_size
    ]
    scales = scales.to(torch.float8_e4m3fn)
    global_scale = 2.0**-6

    layer = torch.nn.Module()
    layer.quant_config = SimpleNamespace(group_size=group_size)
    layer.output_size_per_partition = size_n
    layer.input_size_per_partition = size_k
    layer.params_dtype = dtype
    layer.weight = torch.nn.Parameter(
        (codes[:, 0::2] | (codes[:, 1::2] << 4)).cuda(), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(scales.cuda(), requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(
        torch.tensor([global_scale], dtype=dtype, device="cuda"), requires_grad=False
    )
    prepare_nvfp4_layer_for_marlin(layer)

    # Rows of the identity: the output is the dequantized weight, exactly.
    output = apply_fp4_marlin_linear(
        torch.eye(size_k, dtype=dtype, device="cuda"),
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        layer.workspace,
        size_n,
        size_k,
        use_fp32_reduce=True,
    )
    weight_ref = e2m1[codes.long()] * scales.float().repeat_interleave(group_size, 1)
    torch.testing.assert_close(
        output.float().cpu(), (weight_ref * global_scale).T, rtol=0, atol=0
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
