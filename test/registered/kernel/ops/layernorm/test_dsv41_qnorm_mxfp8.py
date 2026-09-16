"""Model Q normalization preserves deterministic execution modes."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import flashinfer
import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="The production fast paths target SM10x",
)


@pytest.mark.parametrize(
    "batch_invariant,deterministic", [(True, False), (False, True)]
)
def test_deterministic_modes_keep_original_norm(batch_invariant, deterministic):
    from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend
    from sglang.srt.models.deepseek_v4 import MQALayer

    class Norm:
        weight = torch.ones(1280, device="cuda", dtype=torch.bfloat16)
        variance_epsilon = 1e-6

        def __call__(self, x):
            return flashinfer.norm.rmsnorm(x, self.weight, self.variance_epsilon)

    layer = SimpleNamespace(
        is_dsv41=True,
        q_norm=Norm(),
        wq_b=SimpleNamespace(
            quant_method=SimpleNamespace(
                mxfp8_dense_backend=Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL,
                use_mxfp8=True,
            )
        ),
    )
    x = torch.randn(6, 1280, device="cuda", dtype=torch.bfloat16)
    runtime = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=deterministic)
    )
    with (
        patch(
            "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
            return_value=batch_invariant,
        ),
        patch("sglang.srt.runtime_context.get_exec", return_value=runtime),
        patch(
            "sglang.kernels.ops.layernorm.mxfp8_epilogue.rmsnorm_mxfp8",
            side_effect=AssertionError("deterministic norm changed"),
        ),
    ):
        y, q = MQALayer._normalize_q_lora(layer, x)
    assert q is y
    assert torch.equal(y, layer.q_norm(x))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
