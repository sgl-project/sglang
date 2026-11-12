"""Kimi-K3 prefill FP8 copies (k3_rocm_prefill_fp8): parsing, thresholds,
row chunks, quantization, layer selection and the BF16 fallback."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.quantization import k3_rocm_prefill_fp8 as p
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _linear(shape, dtype=torch.bfloat16, quant_method=None, device="meta"):
    return SimpleNamespace(
        weight=torch.nn.Parameter(
            torch.empty(shape, dtype=dtype, device=device), requires_grad=False
        ),
        quant_method=quant_method or UnquantizedLinearMethod(),
    )


def _fake_copy(group, weight, min_tokens):
    """A PrefillFP8Weight without aiter: only the fields the checks read."""
    fp8 = object.__new__(p.PrefillFP8Weight)
    fp8.group = group
    fp8.n, fp8.k = weight.shape
    fp8.min_tokens = min_tokens
    fp8.source_ptr = weight.data_ptr()
    return fp8


class TestParseGroups(CustomTestCase):
    def test_keeps_order_and_drops_duplicates(self):
        self.assertEqual(
            p.parse_groups(" o_proj,in_proj,o_proj ,"), ("o_proj", "in_proj")
        )

    def test_empty_is_off(self):
        self.assertEqual(p.parse_groups(""), ())

    def test_front_points_to_the_variable_that_serves_it(self):
        with self.assertRaisesRegex(ValueError, "SGLANG_K3_TARGET_DENSE_FP8=front"):
            p.parse_groups("in_proj,front")

    def test_unknown_group_is_refused(self):
        with self.assertRaisesRegex(ValueError, "'qkv'"):
            p.parse_groups("qkv")


class TestThresholdsAndChunks(CustomTestCase):
    def test_group_threshold_raises_the_base(self):
        self.assertEqual(p.group_min_tokens("in_proj", 1024), 1024)
        self.assertEqual(p.group_min_tokens("kv_b", 1024), 14336)
        self.assertEqual(p.group_min_tokens("kv_b", 20000), 20000)

    def test_every_group_threshold_names_a_known_group(self):
        self.assertLessEqual(set(p.GROUP_MIN_TOKENS), set(p.GROUP_SHAPES))

    def test_row_chunks_cover_the_batch_within_the_bound(self):
        self.assertEqual(p.row_chunks(0), [])
        self.assertEqual(p.row_chunks(10, rows=4), [(0, 4), (4, 8), (8, 10)])
        m = 3 * p.MAX_ROWS_PER_CALL + 5
        chunks = p.row_chunks(m)
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], m)
        self.assertTrue(all(b - a <= p.MAX_ROWS_PER_CALL for a, b in chunks))

    def test_calls_stay_within_the_run_to_run_stable_rows(self):
        self.assertLessEqual(p.MAX_ROWS_PER_CALL, 16384)
        self.assertEqual(p.row_chunks(32768), [(0, 16384), (16384, 32768)])


class TestQuantizePerChannel(CustomTestCase):
    def test_scale_per_row_and_bounded_error(self):
        torch.manual_seed(0)
        w = torch.randn(64, 128, dtype=torch.bfloat16)
        w[3] *= 100  # a row with a large range must not change the others' scale
        q, scale = p.quantize_per_channel(w, torch.float8_e4m3fn)
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.shape, (64, 1))
        self.assertEqual(scale.dtype, torch.float32)
        rel = (q.float() * scale - w.float()).abs().amax(dim=1) / w.float().abs().amax(
            dim=1
        )
        self.assertLess(rel.max().item(), 0.07)

    def test_zero_row_stays_finite(self):
        q, scale = p.quantize_per_channel(
            torch.zeros(2, 8, dtype=torch.bfloat16), torch.float8_e4m3fn
        )
        self.assertTrue(torch.isfinite(scale).all())
        self.assertEqual(q.float().abs().sum().item(), 0.0)

    def test_fnuz_uses_its_own_range(self):
        w = torch.full((1, 4), 1000.0)
        q, scale = p.quantize_per_channel(w, torch.float8_e4m3fnuz)
        self.assertAlmostEqual(scale.item(), 1000.0 / 240.0, places=3)


class TestLayerSelection(CustomTestCase):
    def test_plain_bf16_linear_of_the_group_shape(self):
        self.assertTrue(p.is_plain_bf16_linear(_linear((7168, 1536)), "o_proj"))

    def test_wrong_shape_dtype_or_method_is_left_bf16(self):
        self.assertFalse(p.is_plain_bf16_linear(_linear((7168, 1024)), "o_proj"))
        self.assertFalse(
            p.is_plain_bf16_linear(_linear((7168, 1536), torch.float16), "o_proj")
        )

        class OtherMethod(UnquantizedLinearMethod):
            pass

        self.assertFalse(
            p.is_plain_bf16_linear(
                _linear((7168, 1536), quant_method=OtherMethod()), "o_proj"
            )
        )

    def test_layer_converted_by_target_dense_fp8_is_skipped(self):
        converted = _linear((6144, 7168), dtype=torch.float8_e4m3fn)
        converted.quant_method = object()  # StaticFP8LinearMethod
        self.assertFalse(p.is_plain_bf16_linear(converted, "in_proj"))

    def _model(self):
        kda = SimpleNamespace(
            self_attn=SimpleNamespace(fused_qkvg_proj="qkvg", o_proj="o"),
            mlp=SimpleNamespace(
                routed_expert_up_proj="up",
                shared_experts=SimpleNamespace(down_proj="sdown"),
            ),
        )
        mla = SimpleNamespace(
            self_attn=SimpleNamespace(
                o_proj="o2",
                fused_qkv_a_proj_with_mqa="qkv_a",
                g_proj="g",
                q_b_proj="q_b",
                kv_b_proj="kv_b",
            ),
            mlp=SimpleNamespace(),
        )
        missing = SimpleNamespace()  # PPMissingLayer
        return SimpleNamespace(model=SimpleNamespace(layers=[kda, mla, missing]))

    def test_targets_are_the_selected_groups_only(self):
        targets = list(p.iter_targets(self._model(), ("o_proj", "kv_b", "sdown")))
        self.assertEqual(
            [(g, name) for g, name, _, _ in targets],
            [
                ("o_proj", "layers.0.self_attn.o_proj"),
                ("sdown", "layers.0.mlp.shared_experts.down_proj"),
                ("o_proj", "layers.1.self_attn.o_proj"),
                ("kv_b", "layers.1.self_attn.kv_b_proj"),
            ],
        )

    def test_shared_down_target_carries_its_moe(self):
        model = self._model()
        (target,) = p.iter_targets(model, ("sdown",))
        self.assertIs(target[3], model.model.layers[0].mlp)


class TestBuild(CustomTestCase):
    def test_unset_is_a_no_op(self):
        with mock.patch.object(
            p.envs.SGLANG_K3_PREFILL_DENSE_FP8, "get", return_value=""
        ):
            p.build_prefill_fp8_copies(model=None)

    def test_set_without_rocm_fails_loudly(self):
        with (
            mock.patch.object(
                p.envs.SGLANG_K3_PREFILL_DENSE_FP8, "get", return_value="o_proj"
            ),
            mock.patch("sglang.srt.utils.is_hip", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "ROCm"):
                p.build_prefill_fp8_copies(model=None)


class TestFallback(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.layer = _linear((16, 32), device="cpu")
        self.layer.weight.data.normal_()
        self.method = p.PrefillFP8LinearMethod(
            _fake_copy("o_proj", self.layer.weight.data, min_tokens=4)
        )

    def test_declined_call_goes_to_the_stock_method(self):
        x = torch.randn(2, 32, dtype=torch.bfloat16)
        with mock.patch.object(
            UnquantizedLinearMethod, "apply", return_value="bf16"
        ) as stock:
            self.assertEqual(self.method.apply(self.layer, x), "bf16")
        stock.assert_called_once_with(self.layer, x, None)

    def test_rows_below_the_threshold_are_declined(self):
        self.assertEqual(self.method.fp8.min_tokens, 4)
        x = torch.randn(2, 32, dtype=torch.bfloat16)
        self.assertFalse(self.method.fp8.admits(self.layer, x, None))

    def test_bias_takes_bf16(self):
        x = torch.randn(8, 32, dtype=torch.bfloat16)
        bias = torch.randn(16, dtype=torch.bfloat16)
        self.assertFalse(self.method.fp8.admits(self.layer, x, bias))

    def test_replaced_weight_takes_bf16(self):
        x = torch.randn(8, 32, dtype=torch.bfloat16)
        self.layer.weight = torch.nn.Parameter(
            torch.randn(16, 32, dtype=torch.bfloat16), requires_grad=False
        )
        self.assertFalse(self.method.fp8.admits(self.layer, x, None))

    def test_non_gpu_input_takes_bf16_and_raw_call_declines(self):
        x = torch.randn(8, 32, dtype=torch.bfloat16)
        out = torch.empty(8, 16, dtype=torch.bfloat16)
        self.assertFalse(self.method.fp8.admits(self.layer, x, None))
        self.assertFalse(self.method.apply_raw_into(self.layer, x, out))


if __name__ == "__main__":
    unittest.main()
