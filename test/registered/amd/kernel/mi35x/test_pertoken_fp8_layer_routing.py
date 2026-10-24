"""Genuine layer-path routing test for the per-token fp8 fold (MI35x).

Unlike the kernel-level parity test (which calls ``apply_fp8_linear`` directly),
this drives the fold's ``(fp8, scale[M,1], dtype)`` tuple through a REAL
``ColumnParallelLinear`` whose weight is quantized AND preshuffled by the actual
Quark ``W8A8Fp8`` scheme -- ``create_weights`` -> load -> ``process_weights_after_
loading`` (which runs ``shuffle_weight`` and sets ``_fp8_weight_preshuffled``) --
then calls ``layer.forward()``. That exercises the production routing:

    ColumnParallelLinear.forward
        -> quant_method.apply (QuarkLinearMethod)
            -> scheme.apply_weights (QuarkW8A8Fp8)
                -> apply_fp8_linear -> gemm_a8w8_bpreshuffle

For q_b (12288) and kv_b (16384) widths, at M=1 (decode) and M>1 (prefill), in
BF16 and FP16, it asserts:
  * the preshuffle marker landed (``_fp8_weight_preshuffled is True``),
  * the folded-tuple forward matches the plain-activation forward, and
  * dtype is preserved.

Requires ROCm/aiter on gfx95 (MI35x); skipped elsewhere.

NOTE: the fused ENTRY proj is a ``ReplicatedLinear`` whose folding also depends on
the ``__init__``-time ``fp8_pending`` re-resolve in ``deepseek_v2.py`` -- that end
of the path is exercised by the e2e model run (eager + graph, TP for fused-AR),
which is the companion evidence for this review point.
"""

import types
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")

# Detect the target hardware defensively: a non-ROCm / non-gfx95 (or otherwise
# broken) environment simply skips this test.
try:
    from sglang.srt.models.deepseek_common.utils import (
        _use_aiter,
        _use_aiter_bpreshuffle_gfx95,
    )

    _HAS_PATH = (
        _use_aiter and _use_aiter_bpreshuffle_gfx95 and torch.cuda.is_available()
    )
except Exception:
    _HAS_PATH = False

# On the gfx95 runner (where this coverage is required) the Quark/linear imports
# are NOT guarded: an API move or an incompatible pinned Quark/AITER build must
# fail loudly here rather than silently degrade into a green skip and lose MI35x
# layer-routing coverage.
if _HAS_PATH:
    from sglang.srt.layers.linear import ColumnParallelLinear
    from sglang.srt.layers.quantization.quark.quark import QuarkLinearMethod
    from sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
        QuarkW8A8Fp8,
    )
    from sglang.srt.models.deepseek_common.utils import _is_per_channel_dynamic_fp8

# (input, output) widths for the two ColumnParallelLinear attn projections.
_PROJ_WIDTHS = {"q_b": (1536, 12288), "kv_b": (512, 16384)}

# Quark *self_attn* spec: per-channel fp8 weight, dynamic per-channel fp8 input.
_WEIGHT_CFG = {"dtype": "fp8_e4m3", "is_dynamic": False, "qscheme": "per_channel"}
_INPUT_CFG = {"dtype": "fp8_e4m3", "is_dynamic": True, "qscheme": "per_channel"}


@unittest.skipUnless(_HAS_PATH, "requires ROCm/aiter gfx95 (MI35x)")
class TestPerTokenFp8LayerRouting(CustomTestCase):
    def _build_quark_linear(self, K, N, device):
        # A real ColumnParallelLinear, wired to the real Quark W8A8Fp8 scheme via
        # QuarkLinearMethod (which just delegates create/process/apply to the
        # scheme). tp_size=1 keeps it single-rank (no all-gather / process group).
        layer = ColumnParallelLinear(K, N, bias=False, tp_size=1, tp_rank=0)
        layer.scheme = QuarkW8A8Fp8(_WEIGHT_CFG, _INPUT_CFG)
        layer.quant_method = QuarkLinearMethod(
            types.SimpleNamespace(online_scheme=None)
        )
        # Real Quark create_weights: registers fp8 weight + per-channel scale +
        # the _fp8_weight_preshuffled default.
        layer.quant_method.create_weights(
            layer,
            input_size_per_partition=K,
            output_partition_sizes=[N],
            input_size=K,
            output_size=N,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *a, **k: None,
        )
        # Load quantized weights (fp8 weight, per-channel fp32 scale).
        layer.weight.data = (torch.randn(N, K, device=device) * 0.1).to(
            torch.float8_e4m3fn
        )
        layer.weight_scale.data = (
            torch.rand(N, dtype=torch.float32, device=device) * 0.05 + 0.01
        )
        # Real process_weights_after_loading: shuffle_weight + marker + scale view.
        layer.quant_method.process_weights_after_loading(layer)
        return layer

    def test_layer_forward_folded_tuple_matches_plain(self):
        device = "cuda"
        for name, (K, N) in _PROJ_WIDTHS.items():
            layer = self._build_quark_linear(K, N, device)
            # The marker and eligibility must hold on the real, processed layer.
            self.assertTrue(getattr(layer, "_fp8_weight_preshuffled", False))
            self.assertTrue(_is_per_channel_dynamic_fp8(layer))
            for M in (1, 4):
                for dtype in (torch.bfloat16, torch.float16):
                    with self.subTest(proj=name, tokens=M, dtype=dtype):
                        x = torch.randn(M, K, dtype=dtype, device=device) * 0.1

                        # Plain activation through the real layer (reference).
                        ref, _ = layer(x)

                        # Folded (fp8, scale[M,1], dtype) tuple through the SAME
                        # layer.forward -> quant_method.apply -> scheme -> GEMM.
                        import sglang.srt.layers.quantization.fp8_utils as fp8u

                        qx, x_scale = fp8u.per_token_group_quant_fp8(
                            x, group_size=x.shape[1]
                        )
                        fused, _ = layer((qx, x_scale, dtype))

                        self.assertEqual(fused.dtype, dtype)
                        self.assertEqual(ref.dtype, dtype)
                        self.assertEqual(list(fused.shape), [M, N])
                        torch.testing.assert_close(
                            fused.float(), ref.float(), rtol=2e-2, atol=2e-2
                        )


if __name__ == "__main__":
    unittest.main()
