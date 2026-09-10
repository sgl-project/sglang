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
from sglang.srt.utils.common import (
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
    reason="NVFP4 Marlin requires CUDA SM8X/SM9X/SM120",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 33])
@pytest.mark.parametrize("prescale", ["awq", "identity", "plain"])
def test_modelopt_nvfp4_marlin_prescale(dtype, num_tokens, prescale, monkeypatch):
    from sglang.srt.layers.linear import RowParallelLinear
    from sglang.srt.layers.quantization import fp4_utils
    from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
    from sglang.test.layer_ut_utils import init_single_process_dist, load_linear_weights

    init_single_process_dist()
    monkeypatch.setattr(
        fp4_utils, "FP4_GEMM_RUNNER_BACKEND", Fp4GemmRunnerBackend.MARLIN
    )
    torch.manual_seed(7209)
    config = ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
        exclude_modules=[],
        packed_modules_mapping={},
        is_awq=prescale != "plain",
    )
    layer = RowParallelLinear(
        256,
        128,
        bias=False,
        params_dtype=dtype,
        quant_config=config,
        tp_rank=0,
        tp_size=1,
        prefix="model.layers.0.mlp.down_proj",
    ).cuda()
    packed, scales, global_scale, weight_ref = make_nvfp4_weight_and_ref(
        128, 256, dtype
    )
    load_linear_weights(
        layer,
        weight=packed,
        weight_scale=scales,
        weight_scale_2=global_scale.float(),
        input_scale=torch.ones((), device="cuda"),
    )
    channel_scale = torch.ones(256, dtype=dtype, device="cuda")
    if prescale == "awq":
        channel_scale = torch.linspace(0.25, 4, 256, dtype=dtype, device="cuda")
    if config.is_awq:
        layer.pre_quant_scale.weight_loader(layer.pre_quant_scale, channel_scale)
    layer.quant_method.process_weights_after_loading(layer)
    x = torch.randn(num_tokens, 256, dtype=dtype, device="cuda") / 10
    actual = layer(x)[0].float()
    expected = (x * channel_scale).float() @ weight_ref.float().T
    relative_l2 = (actual - expected).norm() / expected.norm()
    assert relative_l2.item() < 0.01


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
