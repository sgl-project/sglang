"""gfx1250 GPT-OSS W4A8 MoE with SGLANG_GFX1250_W4A8_FLYDSL: FlyDSL a8w4 from
SGLANG_GFX1250_W4A8_FLYDSL_MIN_TOKENS tokens up, gluon below it over zero-copy
views of the same (FlyDSL-layout) weights.

Synthetic MXFP4 experts with gpt-oss's hidden size 2880, which is not a
multiple of 128: FlyDSL's weights are padded to 3072 and the gluon views cut at
2944. The intermediate size is the TP1 one (2880) and the TP2 one (1440).
Guards:
- dispatch: MIN_TOKENS - 1 tokens run the gluon path, MIN_TOKENS and
  MIN_TOKENS + 1 run FlyDSL;
- the gluon views give exactly the stock Triton result. At K = 2880 the
  GFX1250_SCALE scale descriptor ends at K // 32 groups while the layout keeps
  K in 128-wide chunks, so views cut at the real dims read the last chunk's
  scales as 0;
- FlyDSL stays close to an fp32 dequantized reference with nonzero biases. An
  AITER without ROCm/aiter#6663 adds the first N tile's bias to every N tile;
- outputs are finite, and the calls leave the weights unchanged;
- with the switch off, the weights and outputs are the stock path's.
"""

import functools
import unittest
from unittest import mock

import torch

from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.utils.common import is_gfx1250_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

_HIDDEN = 2880
_NUM_EXPERTS = 8
_TOP_K = 4
_ALPHA = 1.702
_LIMIT = 7.0
_MIN_TOKENS = 384
# Static per-tensor FP8 activation scales. The Triton path clips activations at
# 448 * scale and FlyDSL (per-1x32 MXFP8) does not, so inputs beyond the range
# would make two correct paths disagree: x stays within +-6 < 7, and the SwiGLU
# output is bounded by limit * (limit + 1) = 56.
_A13_SCALE = 7.0 / 448
_A2_SCALE = 56.0 / 448
# FlyDSL quantizes activations to MXFP8 (per-1x32 power-of-two scales) and
# rounds GEMM 1's output to bf16 first: rel. L2 0.012-0.018 from the reference
# here, the Triton path 0.003. Without ROCm/aiter#6663's bias fix: 0.55.
_FLYDSL_REL_L2_TOL = 0.03

_FP4_VALUES = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)


def _flydsl_available() -> bool:
    if not is_gfx1250_supported():
        return False
    try:
        import aiter.ops.flydsl.grouped_moe_gfx1250  # noqa: F401
    except ImportError:
        return False
    return True


def _dequant_mxfp4(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """[..., N, K // 2] uint8 (low nibble first), [..., N, K // 32] e8m0 -> fp32."""
    lut = torch.tensor(_FP4_VALUES, device=w.device)
    v = torch.stack([lut[(w & 15).long()], lut[(w >> 4).long()]], dim=-1)
    return v.flatten(-2) * torch.exp2(scale.float() - 127).repeat_interleave(32, -1)


def _fp8(x: torch.Tensor, scale: float) -> torch.Tensor:
    return (x / scale).to(torch.float8_e4m3fn).float() * scale


def _reference(x, router_logits, w13, w13_scale, w13_bias, w2, w2_scale, w2_bias):
    """gpt-oss MoE in fp32 over the loaded (gate/up separated) weights, with
    both GEMMs' activations rounded to FP8 at the static scales."""
    inter = w13.shape[1] // 2
    vals, ids = torch.topk(router_logits.float(), _TOP_K, dim=-1)
    weights = torch.softmax(vals, dim=-1)
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in ids.unique().tolist():
        rows, slot = (ids == e).nonzero(as_tuple=True)
        h = _fp8(x[rows].float(), _A13_SCALE) @ _dequant_mxfp4(w13[e], w13_scale[e]).T
        h = h + w13_bias[e]
        gate = h[:, :inter].clamp(max=_LIMIT)
        up = h[:, inter:].clamp(-_LIMIT, _LIMIT)
        act = _fp8(gate * torch.sigmoid(_ALPHA * gate) * (up + 1), _A2_SCALE)
        y = act @ _dequant_mxfp4(w2[e], w2_scale[e]).T + w2_bias[e]
        out.index_add_(0, rows, y * weights[rows, slot, None])
    return out


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@unittest.skipUnless(_flydsl_available(), "requires an AMD gfx1250 GPU with FlyDSL")
class TestGptOssW4A8FlydslMoEGfx1250(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.moe.fused_moe_triton import aiter_mxfp4_w4a8_moe
        from sglang.srt.layers.quantization.quark.schemes import (
            quark_w4a8_mxfp4_moe,
        )

        cls.moe = aiter_mxfp4_w4a8_moe
        cls.scheme_module = quark_w4a8_mxfp4_moe
        cls.gen = torch.Generator(device="cuda").manual_seed(0)

    def _random_loaded_weights(self, inter: int):
        """What the Quark loader leaves in the layer: gate/up separated."""

        def fp4(*shape):
            return torch.randint(
                0, 256, shape, generator=self.gen, device="cuda", dtype=torch.uint8
            )

        def e8m0(*shape):
            # 2^-7 .. 2^-5: gate/up pre-activations of a few units.
            return torch.randint(
                120, 123, shape, generator=self.gen, device="cuda", dtype=torch.uint8
            )

        def randn(*shape):
            return torch.randn(shape, generator=self.gen, device="cuda")

        return dict(
            w13_weight=fp4(_NUM_EXPERTS, 2 * inter, _HIDDEN // 2),
            w13_weight_scale=e8m0(_NUM_EXPERTS, 2 * inter, _HIDDEN // 32),
            w13_weight_bias=randn(_NUM_EXPERTS, 2 * inter),
            w2_weight=fp4(_NUM_EXPERTS, _HIDDEN, inter // 2),
            w2_weight_scale=e8m0(_NUM_EXPERTS, _HIDDEN, inter // 32),
            w2_weight_bias=2 * randn(_NUM_EXPERTS, _HIDDEN),
        )

    def _make_layer(self, loaded: dict, flydsl: bool):
        scheme = self.scheme_module.QuarkW4A8MXFp4MoE(
            weight_config=dict(
                dtype="fp4",
                qscheme="per_group",
                group_size=32,
                is_dynamic=False,
                scale_format="e8m0",
            ),
            input_config=dict(dtype="fp8_e4m3", qscheme="per_tensor", is_dynamic=False),
        )
        layer = torch.nn.Module()
        inter = loaded["w2_weight"].shape[-1] * 2
        with torch.device("cuda"):
            scheme.create_weights(
                layer,
                num_experts=_NUM_EXPERTS,
                hidden_size=_HIDDEN,
                intermediate_size_per_partition=inter,
                params_dtype=torch.bfloat16,
                with_bias=True,
            )
        for name, value in loaded.items():
            getattr(layer, name).data.copy_(value)
        layer.w13_input_scale.data.fill_(_A13_SCALE)
        layer.w2_input_scale.data.fill_(_A2_SCALE)
        with mock.patch.object(self.moe, "FLYDSL_ENABLED", flydsl):
            scheme.process_weights_after_loading(layer)
        scheme.create_moe_runner(
            layer,
            MoeRunnerConfig(
                activation="swiglu", gemm1_alpha=_ALPHA, gemm1_clamp_limit=_LIMIT
            ),
        )
        return scheme, layer

    def _inputs(self, num_tokens: int):
        x = torch.randn(num_tokens, _HIDDEN, generator=self.gen, device="cuda")
        x = x.clamp(-6, 6).bfloat16()
        # Distinct logits: bf16 ties at the 4th/5th expert would let the Triton
        # routing and torch.topk pick different, equally valid experts.
        logits = torch.stack(
            [
                torch.randperm(_NUM_EXPERTS, generator=self.gen, device="cuda")
                for _ in range(num_tokens)
            ]
        )
        return x, (logits * 0.5).bfloat16()

    def _apply(self, scheme, layer, x, router_logits):
        topk_weights, topk_ids = torch.topk(router_logits.float(), _TOP_K, dim=-1)
        dispatch_output = StandardDispatchOutput(
            hidden_states=x,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights.softmax(dim=-1), topk_ids.int(), router_logits
            ),
        )
        with mock.patch.object(self.moe, "FLYDSL_MIN_TOKENS", _MIN_TOKENS):
            return scheme.apply_weights(layer, dispatch_output).hidden_states

    def _triton(self, stock, x, router_logits):
        """The stock path's layout on the Triton backend."""
        return self.moe.aiter_w4a8_gfx1250_forward(
            hidden_states=x,
            router_logits=router_logits,
            topk=_TOP_K,
            a13_scale=torch.tensor(_A13_SCALE, device="cuda"),
            a2_scale=torch.tensor(_A2_SCALE, device="cuda"),
            gemm1_alpha=_ALPHA,
            gemm1_limit=_LIMIT,
            backend="triton",
            **stock,
        )

    def _check_switch(self, inter: int):
        loaded = self._random_loaded_weights(inter)
        stock = dict(
            zip(
                loaded.keys(),
                self.moe.prepare_w4a8_gfx1250_weights(
                    *[t.clone() for t in loaded.values()]
                ),
            )
        )
        scheme, layer = self._make_layer(loaded, flydsl=True)
        fw = scheme.flydsl_weights
        self.assertIsNotNone(fw)
        # One weight copy: the parameters are FlyDSL's buffers.
        self.assertEqual(layer.w13_weight.data_ptr(), fw.w13.data_ptr())
        self.assertEqual(layer.w2_weight.data_ptr(), fw.w2.data_ptr())
        params = {n: p.detach().clone() for n, p in layer.named_parameters()}

        for num_tokens in (1, _MIN_TOKENS - 1, _MIN_TOKENS, _MIN_TOKENS + 1, 1024):
            with self.subTest(inter=inter, num_tokens=num_tokens):
                x, router_logits = self._inputs(num_tokens)
                out = self._apply(scheme, layer, x, router_logits)
                self.assertEqual(tuple(out.shape), (num_tokens, _HIDDEN))
                self.assertTrue(torch.isfinite(out).all())
                triton = self._triton(stock, x, router_logits)
                if num_tokens < _MIN_TOKENS:
                    # Gluon over FlyDSL's weights: the Triton result, bit for bit.
                    self.assertTrue(
                        torch.equal(out, triton),
                        f"gluon vs triton max abs diff "
                        f"{(out.float() - triton.float()).abs().max().item()}",
                    )
                else:
                    # FlyDSL: not the gluon/Triton result, but within tolerance
                    # of the fp32 reference.
                    self.assertFalse(torch.equal(out, triton))
                    ref = _reference(x, router_logits, *loaded.values())
                    rel = _rel_l2(out, ref)
                    self.assertLess(rel, _FLYDSL_REL_L2_TOL, f"FlyDSL rel L2 {rel}")

        for name, p in layer.named_parameters():
            self.assertTrue(torch.equal(p, params[name]), f"{name} changed")

    def test_switch_tp1_intermediate(self):
        self._check_switch(inter=2880)

    def test_switch_tp2_intermediate(self):
        self._check_switch(inter=1440)

    def test_switch_off_keeps_the_stock_path(self):
        loaded = self._random_loaded_weights(1440)
        stock = dict(
            zip(
                loaded.keys(),
                self.moe.prepare_w4a8_gfx1250_weights(
                    *[t.clone() for t in loaded.values()]
                ),
            )
        )
        scheme, layer = self._make_layer(loaded, flydsl=False)
        self.assertIsNone(scheme.flydsl_weights)
        for name, expected in stock.items():
            p = getattr(layer, name)
            self.assertEqual(p.shape, expected.shape, name)
            self.assertEqual(p.stride(), expected.stride(), name)
            self.assertTrue(torch.equal(p, expected), name)
        # Pin the stock path's backend to Triton, as the gfx1250 recipes do: its
        # default (gluon over the unshuffled layout) is not reproducible run to
        # run from a few hundred tokens on this AITER.
        triton_forward = functools.partial(
            self.moe.aiter_w4a8_gfx1250_forward, backend="triton"
        )
        for num_tokens in (1, _MIN_TOKENS, 1024):
            with self.subTest(num_tokens=num_tokens):
                x, router_logits = self._inputs(num_tokens)
                with mock.patch.object(
                    self.moe, "aiter_w4a8_gfx1250_forward", triton_forward
                ):
                    out = self._apply(scheme, layer, x, router_logits)
                self.assertTrue(torch.equal(out, self._triton(stock, x, router_logits)))


if __name__ == "__main__":
    unittest.main()
