"""A compressed-tensors per-channel W8A8-FP8 linear on aiter must accept the
(fp8, per-token scale) tuple that GLM-4.x's fused RMSNorm + FP8 quant emits
(ROCM-31378: AssertionError in apply_fp8_linear during CUDA graph capture)."""

import os
import unittest

os.environ.setdefault("SGLANG_USE_AITER", "1")

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _build_channel_fp8_layer(n: int, k: int):
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW8A8Fp8,
    )

    scheme = CompressedTensorsW8A8Fp8(
        weight_quant=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=True,
            dynamic=False,
        ),
        is_static_input_scheme=False,
    )
    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        input_size_per_partition=k,
        output_partition_sizes=[n],
        input_size=k,
        output_size=n,
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )
    w = torch.randn(n, k, device="cuda") / k**0.5
    w_scale = w.abs().amax(dim=1, keepdim=True) / torch.finfo(torch.float8_e4m3fn).max
    w_fp8 = (w / w_scale).to(torch.float8_e4m3fn)
    layer.weight.data = w_fp8
    layer.weight_scale.data = w_scale.float()
    # The reference uses the checkpoint values, before any fnuz/shuffle rewrite.
    w_ref = w_fp8.float() * w_scale.float()
    scheme.process_weights_after_loading(layer)
    return scheme, layer, w_ref


@unittest.skipUnless(is_hip() and torch.cuda.is_available(), "ROCm GPU required")
class TestW8A8Fp8PerTokenTuple(CustomTestCase):
    def test_fused_norm_per_token_tuple_matches_reference(self):
        """M > 1 tokens: the tuple path must not assert and must match the math."""
        from sglang.srt.layers.layer_boundary.residual.add_norm import (
            _fused_rmsnorm_fp8_per_token_quant,
        )

        torch.manual_seed(0)
        n, k, m = 512, 1024, 24
        scheme, layer, w_ref = _build_channel_fp8_layer(n, k)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        norm_weight = torch.ones(k, device="cuda", dtype=torch.bfloat16)

        q, scale = _fused_rmsnorm_fp8_per_token_quant(x, norm_weight, 1e-6)
        self.assertEqual(tuple(scale.reshape(m, -1).shape), (m, 1))

        out = scheme.apply_weights(layer, (q, scale))

        x_ref = q.float() * scale.float().reshape(m, 1)
        expected = x_ref @ w_ref.T
        torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
