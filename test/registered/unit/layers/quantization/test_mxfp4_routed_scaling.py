import sys

import pytest

from sglang.srt.layers.moe.fused_moe_triton.layer import (
    _fuses_routed_scaling_factor_in_topk,
)
from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
    Mxfp4FlashinferCutlassMoEMethod,
)
from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    Mxfp4FlashinferTrtllmMoEMethod,
)
from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_native_mxfp4_methods_fold_routed_scaling_into_topk_weights():
    methods = (
        Mxfp4MarlinMoEMethod.__new__(Mxfp4MarlinMoEMethod),
        Mxfp4FlashinferCutlassMoEMethod.__new__(Mxfp4FlashinferCutlassMoEMethod),
        Mxfp4FlashinferTrtllmMoEMethod.__new__(Mxfp4FlashinferTrtllmMoEMethod),
    )

    assert all(_fuses_routed_scaling_factor_in_topk(method) for method in methods)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
