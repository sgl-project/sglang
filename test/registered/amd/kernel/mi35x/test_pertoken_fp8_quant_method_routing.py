"""MI35x routing test: per-token fp8 tuple through the real quant-method apply.

The GEMM-level parity test (test_pertoken_fp8_fold_parity.py) calls
``apply_fp8_linear`` directly and therefore bypasses the projection's quant
method -- exactly the layer where the fold's ``(fp8, scale[M,1], dtype)`` tuple
is actually routed. This test drives the routing:

  * Quark ``W8A8Fp8`` (this checkpoint's attn scheme) passes the tuple straight
    into ``apply_fp8_linear`` -> per-channel bpreshuffle GEMM.
  * The native ``Fp8LinearMethod`` / compressed-tensors schemes instead unwrap
    the tuple and call ``apply_fp8_linear(input=qx, input_scale=scale[M,1])``,
    which previously asserted ``input_scale.numel() == 1`` and crashed for M > 1.

It asserts no crash, numerical parity vs the non-folded path, and dtype
preservation, across the entry (2112), q_b (12288), and kv_b (16384) projection
widths at M = 1 (decode) and M > 1 (prefill).

Requires ROCm/aiter on gfx95 (MI35x); skipped elsewhere.

DRAFT -- validate on MI35x. If a scheme constructor/layer-attr shape differs
from what is assumed here, adjust ``_make_layer`` / ``_make_scheme``; the
assertions (parity + dtype + no-crash for M>1) are the contract to keep.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")

try:
    import sglang.srt.layers.quantization.fp8_utils as fp8u
    from sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
        QuarkW8A8Fp8,
    )
    from sglang.srt.models.deepseek_common.utils import (
        _use_aiter,
        _use_aiter_bpreshuffle_gfx95,
    )

    _HAS_PATH = (
        _use_aiter and _use_aiter_bpreshuffle_gfx95 and torch.cuda.is_available()
    )
except Exception:
    _HAS_PATH = False

# Attn projection output widths for this checkpoint (per-channel dynamic fp8).
#   entry fused_qkv_a = q_lora(1536) + kv_lora(512) + qk_rope(64) = 2112
#   q_b  = num_heads(64) * (qk_nope(128) + qk_rope(64)) = 12288
#   kv_b = num_heads(64) * (qk_nope(128) + v_head(128)) = 16384
_PROJ_WIDTHS = {"entry": (7168, 2112), "q_b": (1536, 12288), "kv_b": (512, 16384)}


def _make_scheme():
    # Mirrors the Quark config layer_quant_config["*self_attn*"]: per-channel
    # fp8 weight, dynamic per-channel fp8 input.
    weight_cfg = {"dtype": "fp8_e4m3", "is_dynamic": False, "qscheme": "per_channel"}
    input_cfg = {"dtype": "fp8_e4m3", "is_dynamic": True, "qscheme": "per_channel"}
    return QuarkW8A8Fp8(weight_cfg, input_cfg)


def _make_layer(K, N, device):
    # apply_fp8_linear consumes weight as (K, N) (it feeds weight.T -> (N, K));
    # per-channel weight scale is 1-D [N]; dynamic activation => input_scale None.
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        (torch.randn(K, N, device=device) * 0.1).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.rand(N, dtype=torch.float32, device=device) * 0.05 + 0.01,
        requires_grad=False,
    )
    layer.input_scale = None
    return layer


@unittest.skipUnless(
    _HAS_PATH, "requires ROCm/aiter gfx95 (MI35x) with bpreshuffle GEMM"
)
class TestPerTokenFp8QuantMethodRouting(CustomTestCase):
    def _prequant(self, x):
        # Same per-token quant apply_fp8_linear uses internally on the non-folded
        # path, so folded and reference share identical (qx, scale).
        return fp8u.per_token_group_quant_fp8(x, group_size=x.shape[1])

    def test_quark_scheme_routes_tuple(self):
        scheme = _make_scheme()
        device = "cuda"
        for name, (K, N) in _PROJ_WIDTHS.items():
            layer = _make_layer(K, N, device)
            for M in (1, 4):
                for dtype in (torch.bfloat16, torch.float16):
                    with self.subTest(proj=name, tokens=M, dtype=dtype):
                        x = torch.randn(M, K, dtype=dtype, device=device) * 0.1

                        # Non-folded: plain activation through the real scheme.
                        ref = scheme.apply_weights(layer, x)

                        # Folded: (fp8, scale[M,1], dtype) tuple routed through
                        # the real scheme (Quark passes it straight through).
                        qx, x_scale = self._prequant(x)
                        fused = scheme.apply_weights(layer, (qx, x_scale, dtype))

                        self.assertEqual(fused.dtype, dtype)
                        self.assertEqual(ref.dtype, dtype)
                        self.assertEqual(list(fused.shape), [M, N])
                        torch.testing.assert_close(
                            fused.float(), ref.float(), rtol=2e-2, atol=2e-2
                        )

    def test_unwrapped_per_token_scale_does_not_assert(self):
        # Guards the native Fp8LinearMethod / compressed-tensors path, which
        # unwraps the tuple and calls apply_fp8_linear with a per-token [M, 1]
        # scale. This is the exact call that used to hit
        # `assert input_scale.numel() == 1` for M > 1.
        device = "cuda"
        K, N = _PROJ_WIDTHS["entry"]
        layer = _make_layer(K, N, device)
        for M in (1, 4):
            for dtype in (torch.bfloat16, torch.float16):
                with self.subTest(tokens=M, dtype=dtype):
                    x = torch.randn(M, K, dtype=dtype, device=device) * 0.1
                    ref = fp8u.apply_fp8_linear(
                        input=x,
                        weight=layer.weight,
                        weight_scale=layer.weight_scale,
                        use_per_token_if_dynamic=True,
                    )
                    qx, x_scale = self._prequant(x)
                    # Exactly what the unwrapping methods pass post-unwrap.
                    out = fp8u.apply_fp8_linear(
                        input=qx,
                        weight=layer.weight,
                        weight_scale=layer.weight_scale,
                        input_scale=x_scale,
                        pre_quant_output_dtype=dtype,
                        use_per_token_if_dynamic=True,
                    )
                    self.assertEqual(out.dtype, dtype)
                    torch.testing.assert_close(
                        out.float(), ref.float(), rtol=2e-2, atol=2e-2
                    )


if __name__ == "__main__":
    unittest.main()
