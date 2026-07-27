from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_mxfp4_marlin_quant_info_uses_processed_layer_parameters():
    layer = SimpleNamespace(
        w13_weight=torch.zeros(2, 256, 16, dtype=torch.uint8),
        w2_weight=torch.zeros(2, 32, 64, dtype=torch.uint8),
        w13_weight_scale=torch.zeros(2, 256, 1, dtype=torch.float8_e8m0fnu),
        w2_weight_scale=torch.zeros(2, 32, 4, dtype=torch.float8_e8m0fnu),
        w13_weight_bias=torch.zeros(2, 256, dtype=torch.bfloat16),
        w2_weight_bias=torch.zeros(2, 32, dtype=torch.bfloat16),
    )
    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    method.use_marlin = True

    quant_info = method.get_marlin_quant_info(layer)

    assert quant_info.w13_qweight is layer.w13_weight
    assert quant_info.w2_qweight is layer.w2_weight
    assert quant_info.w13_scales is layer.w13_weight_scale
    assert quant_info.w2_scales is layer.w2_weight_scale
    assert quant_info.w13_bias is layer.w13_weight_bias
    assert quant_info.w2_bias is layer.w2_weight_bias
    assert quant_info.weight_bits == 4
    assert quant_info.is_k_full
