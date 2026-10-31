"""--nvfp4-ckpt-fp8-dense-modules: which BF16 linears of an NVFP4 config go block-FP8, and the load-time quantization."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

# isort: off
import sglang.srt.layers.quantization.modelopt_quant as mq
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    quant_weight_block_fp8_online,
)
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    per_block_cast_to_fp8,
)

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

# isort: on
from sglang.test.test_utils import CustomTestCase

BLOCK = [128, 128]
# GLM-5.x per-rank dense shapes (out, in): fused q_a+kv_a (ragged rows), q_b, o_proj, shared experts.
SHAPES = [(2624, 6144), (16384, 2048), (6144, 16384), (4096, 6144), (6144, 2048)]
# The module list documented for DeepSeek-V3 / GLM-5 NVFP4 checkpoints.
DSV3_MODULES = ["q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj", "wq_b"]
DSV3_MODULES += ["gate_proj", "up_proj", "down_proj"]


def _weight(shape, seed):
    # Explicit CPU device: other tests may leave a CUDA default device set.
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(shape, generator=g, device="cpu") * 0.02
    w[:7, :9] *= 40  # one outlier block
    return w.to(torch.bfloat16)


def _linear():
    layer = ReplicatedLinear.__new__(ReplicatedLinear)  # isinstance checks only
    torch.nn.Module.__init__(layer)
    return layer


def _lm_head():
    layer = ParallelLMHead.__new__(ParallelLMHead)
    torch.nn.Module.__init__(layer)
    return layer


class _StubOnlineMethod:
    def __init__(self, quant_config, *, prefix):
        self.quant_config = quant_config
        self.prefix = prefix


class TestFp8DenseDispatch(CustomTestCase):
    def _config(self):
        qc = ModelOptFp4Config.from_config(
            {
                "quantization": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                    "kv_cache_quant_algo": "FP8",
                    "exclude_modules": ["lm_head", "model.layers.*.self_attn*"],
                }
            }
        )
        # As DeepseekV2ForCausalLM registers it; gate_up_proj uses the shared fallback.
        qc.update_packed_modules_mapping(
            {"fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"]}
        )
        return qc

    def _method(self, qc, layer, prefix, modules):
        with (
            patch.object(mq, "_online_fp8_dense_modules", return_value=modules),
            patch.object(mq, "ModelOptOnlineBlockFp8LinearMethod", _StubOnlineMethod),
        ):
            return qc._unquantized_linear_method(layer, prefix)

    def assertFp8(self, qc, prefix, modules):
        m = self._method(qc, _linear(), prefix, modules)
        self.assertIsInstance(m, _StubOnlineMethod, prefix)
        self.assertEqual(m.quant_config.weight_block_size, [128, 128])
        self.assertTrue(m.quant_config.is_checkpoint_fp8_serialized)

    def assertBf16(self, qc, prefix, modules):
        m = self._method(qc, _linear(), prefix, modules)
        self.assertIsInstance(m, UnquantizedLinearMethod, prefix)

    def test_off_when_unset(self):
        self.assertBf16(self._config(), "model.layers.3.self_attn.o_proj", None)

    def test_listed_modules_only(self):
        qc = self._config()
        for prefix in (
            "model.layers.3.self_attn.fused_qkv_a_proj_with_mqa",
            "model.layers.3.self_attn.q_b_proj",
            "model.layers.3.self_attn.o_proj",
            "model.layers.3.self_attn.indexer.wq_b",
            "model.layers.3.mlp.shared_experts.gate_up_proj",
            "model.layers.0.mlp.down_proj",
        ):
            self.assertFp8(qc, prefix, DSV3_MODULES)
        for prefix in (
            "model.layers.10.self_attn.kv_b_proj",
            "model.layers.3.self_attn.indexer.wk",
            "model.layers.3.self_attn.indexer.weights_proj",
            "model.layers.3.mlp.gate",
        ):
            self.assertBf16(qc, prefix, DSV3_MODULES)

    def test_dotted_paths_match_on_module_boundaries(self):
        qc = self._config()
        self.assertFp8(qc, "model.layers.0.mlp.down_proj", ["mlp.down_proj"])
        self.assertBf16(
            qc, "model.layers.3.mlp.shared_experts.down_proj", ["mlp.down_proj"]
        )
        self.assertFp8(
            qc, "model.layers.3.mlp.shared_experts.down_proj", ["shared_experts"]
        )
        # mlp.gate is the MoE router's name; it must not select the fused dense gate_up_proj.
        self.assertBf16(qc, "model.layers.0.mlp.gate_up_proj", ["mlp.gate"])
        self.assertBf16(qc, "model.layers.3.self_attn.o_proj", ["proj"])

    def test_glob_patterns_match_checkpoint_paths(self):
        """Checkpoint wildcard syntax must not silently select zero BF16 layers."""
        qc = self._config()
        for pattern, prefix, selected in (
            (
                "model.layers.*.self_attn.o_proj",
                "model.layers.3.self_attn.o_proj",
                True,
            ),
            (
                "model.layers.*.self_attn.o_proj",
                "model.layers.3.self_attn.q_b_proj",
                False,
            ),
            ("model.layers.1.*", "model.layers.1.self_attn.o_proj", True),
            ("model.layers.1.*", "model.layers.10.self_attn.o_proj", False),
            ("model.layers.1.*", "modelXlayers.1.self_attn.o_proj", False),
            ("o_*", "model.layers.3.self_attn.o_proj", True),
            (
                "model.layers.*.self_attn.o_proj",
                "language_model.model.layers.3.self_attn.o_proj",
                True,
            ),
            ("*.mlp.gate", "model.layers.0.mlp.gate_up_proj", False),
            (
                "*.mlp.shared_experts.*",
                "model.layers.3.mlp.shared_experts.down_proj",
                True,
            ),
            ("*.mlp.shared_experts.*", "model.layers.3.mlp.gate", False),
        ):
            with self.subTest(pattern=pattern, prefix=prefix):
                if selected:
                    self.assertFp8(qc, prefix, [pattern])
                else:
                    self.assertBf16(qc, prefix, [pattern])

    def test_glob_fused_layers_require_all_checkpoint_parts(self):
        """Wildcard matching must validate each checkpoint shard before fusing."""
        qc = self._config()
        for prefix, complete, partial in (
            (
                "model.layers.3.mlp.gate_up_proj",
                ["model.layers.*.mlp.gate_proj", "*.mlp.up_proj"],
                ["*.mlp.gate_proj"],
            ),
            (
                "model.layers.3.self_attn.fused_qkv_a_proj_with_mqa",
                ["*.self_attn.q_a_proj", "*.self_attn.kv_a_proj_with_mqa"],
                ["*.self_attn.q_a_proj"],
            ),
            (
                "model.layers.3.self_attn.qkv_proj",
                ["*.self_attn.q_proj", "*.self_attn.k_proj", "v_proj"],
                ["*.self_attn.q_proj", "*.self_attn.k_proj"],
            ),
        ):
            with self.subTest(prefix=prefix):
                self.assertFp8(qc, prefix, complete)
                self.assertFp8(qc, prefix, ["*"])
                with self.assertRaisesRegex(ValueError, "all parts of the fused"):
                    self._method(qc, _linear(), prefix, partial)
                self.assertBf16(qc, prefix, [prefix.rsplit(".", 1)[-1]])

    def test_glob_selection_does_not_requantize_checkpoint_fp4(self):
        qc = self._config()
        checkpoint_method = object()
        with (
            patch.object(mq, "_online_fp8_dense_modules", return_value=["*"]),
            patch.object(mq, "ModelOptOnlineBlockFp8LinearMethod", _StubOnlineMethod),
            patch.object(mq, "ModelOptFp4LinearMethod", return_value=checkpoint_method),
        ):
            self.assertIsInstance(
                qc.get_quant_method(_linear(), "model.layers.3.self_attn.o_proj"),
                _StubOnlineMethod,
            )
            self.assertIs(
                qc.get_quant_method(_linear(), "model.layers.3.mlp.down_proj"),
                checkpoint_method,
            )

    def test_glob_selection_summary_uses_the_dispatch_matcher(self):
        qc = self._config()
        modules = ["*.mlp.gate_proj", "*.mlp.up_proj", "*.self_attn.o_prjo"]
        layer = _linear()
        with (
            patch.object(mq, "_online_fp8_dense_modules", return_value=modules),
            patch.object(mq, "ModelOptOnlineBlockFp8LinearMethod", _StubOnlineMethod),
        ):
            layer.quant_method = qc._unquantized_linear_method(
                layer, "model.layers.0.mlp.gate_up_proj"
            )
            with self.assertLogs(mq.logger, level="INFO") as logs:
                qc.validate_online_fp8_selection(torch.nn.Sequential(layer))
        self.assertIn("selected 1 linear modules", logs.output[0])
        self.assertIn("for *.self_attn.o_prjo.", logs.output[1])
        self.assertNotIn("*.mlp.", logs.output[1])

    def test_exclusion_matcher_keeps_existing_modelopt_semantics(self):
        qc = self._config()
        for pattern, prefix, excluded in (
            ("model.layers.1.*", "model.layers.1.self_attn.o_proj", True),
            ("model.layers.1.*", "model.layers.10.self_attn.o_proj", False),
            ("model.layers.*.self_attn*", "model.layers.3.mlp.down_proj", False),
            (
                "model.layers.*.self_attn*",
                "language_model.model.layers.3.self_attn.o_proj",
                True,
            ),
            ("mtp*", "model.mtp_layers.o_proj", True),
            ("q_a_proj", "model.layers.3.self_attn.fused_qkv_a_proj_with_mqa", False),
            (
                "kv_a_proj_with_mqa",
                "model.layers.3.self_attn.fused_qkv_a_proj_with_mqa",
                True,
            ),
            ("q_a_proj", "model.layers.3.self_attn.o_proj", False),
        ):
            with self.subTest(pattern=pattern, prefix=prefix):
                qc.exclude_modules = [pattern]
                self.assertEqual(qc.is_layer_excluded(prefix), excluded)

    def test_fused_layer_needs_all_parts(self):
        qc = self._config()
        self.assertFp8(qc, "model.layers.3.mlp.gate_up_proj", ["gate_proj", "up_proj"])
        # Half of a fused weight cannot be quantized on its own.
        for prefix, modules in (
            ("model.layers.3.mlp.gate_up_proj", ["gate_proj"]),
            ("model.layers.3.self_attn.fused_qkv_a_proj_with_mqa", ["q_a_proj"]),
        ):
            with self.assertRaisesRegex(ValueError, "all parts of the fused"):
                self._method(qc, _linear(), prefix, modules)

    def test_draft_model_keeps_checkpoint_precision(self):
        from sglang.srt.layers.moe.utils import draft_model_build_scope
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        configs = [
            self._config(),
            mq.ModelOptMixedPrecisionConfig.from_config(
                {
                    "quant_algo": "MIXED_PRECISION",
                    "packed_modules_mapping": {},
                    "quantized_layers": {
                        "model.layers.0.mlp.experts.0.gate_proj": {
                            "quant_algo": "NVFP4",
                            "group_size": 16,
                        }
                    },
                }
            ),
        ]
        with (
            patch.object(mq.get_flags().moe, "in_speculative_scope", False),
            patch.object(mq, "_online_fp8_dense_modules", return_value=["*"]),
            patch.object(mq, "ModelOptOnlineBlockFp8LinearMethod", _StubOnlineMethod),
        ):
            for qc in configs:
                with self.subTest(config=qc.get_name()):

                    def method():
                        return qc.get_quant_method(
                            _linear(), "model.layers.78.self_attn.o_proj"
                        )

                    self.assertIsInstance(method(), _StubOnlineMethod)
                    with self.assertRaisesRegex(
                        RuntimeError, "draft construction failed"
                    ):
                        with draft_model_build_scope():
                            self.assertIsInstance(method(), UnquantizedLinearMethod)
                            with draft_model_build_scope():
                                self.assertIsInstance(method(), UnquantizedLinearMethod)
                            self.assertIsInstance(method(), UnquantizedLinearMethod)
                            raise RuntimeError("draft construction failed")
                    self.assertFalse(mq.get_flags().moe.in_speculative_scope)
                    self.assertIsInstance(method(), _StubOnlineMethod)

    def test_selection_summary_reports_misses_not_requested_conversions(self):
        qc = self._config()
        modules = ["gate_proj", "up_proj", "gate_up_proj", "o_prjo"]
        layer = _linear()
        prefix = "model.layers.0.mlp.gate_up_proj"
        model = torch.nn.Sequential(layer)
        with (
            patch.object(mq, "_online_fp8_dense_modules", return_value=modules),
            patch.object(mq, "ModelOptOnlineBlockFp8LinearMethod", _StubOnlineMethod),
        ):
            layer.quant_method = qc._unquantized_linear_method(layer, prefix)
            with self.assertLogs(mq.logger, level="INFO") as logs:
                qc.validate_online_fp8_selection(model)
        self.assertIn("selected 1 linear modules", logs.output[0])
        self.assertIn("for gate_up_proj, o_prjo.", logs.output[1])
        self.assertIn("Use gate_proj up_proj", logs.output[1])

    def test_selection_summary_warns_when_nothing_matches(self):
        qc = self._config()
        with (
            patch.object(mq, "_online_fp8_dense_modules", return_value=["o_prjo"]),
            self.assertLogs(mq.logger, level="INFO") as logs,
        ):
            qc.validate_online_fp8_selection(torch.nn.Module())
        self.assertIn("selected 0 linear modules", logs.output[0])
        self.assertIn("for o_prjo.", logs.output[1])

    def test_lm_head_never(self):
        for modules in (["lm_head"], ["*"], ["*head"]):
            with self.subTest(modules=modules):
                m = self._method(self._config(), _lm_head(), "lm_head", modules)
                self.assertIsInstance(m, UnquantizedLinearMethod)

    def test_non_nvfp4_config_does_not_opt_in(self):
        qc = mq.ModelOptFp8Config(is_checkpoint_fp8_serialized=True)
        self.assertFalse(qc.supports_online_fp8_dense)
        m = self._method(qc, _linear(), "model.layers.3.self_attn.o_proj", ["o_proj"])
        self.assertIsInstance(m, UnquantizedLinearMethod)
        self.assertTrue(mq.ModelOptMixedPrecisionConfig.supports_online_fp8_dense)


class TestFp8DenseOnlineQuant(CustomTestCase):
    def test_layout_and_error(self):
        for shape in SHAPES:
            w = _weight(shape, seed=shape[0])
            for pow2 in (False, True):
                q, s = quant_weight_block_fp8_online(w, BLOCK, pow2)
                self.assertEqual(q.dtype, torch.float8_e4m3fn)
                self.assertEqual(tuple(q.shape), shape)
                self.assertEqual(s.dtype, torch.float32)
                self.assertEqual(
                    tuple(s.shape), (-(-shape[0] // 128), -(-shape[1] // 128))
                )
                self.assertTrue(bool(torch.isfinite(s).all() and (s > 0).all()))
                d = block_quant_dequant(q, s, BLOCK, torch.float32)
                rel = ((d - w.float()).norm() / w.float().norm()).item()
                self.assertLess(rel, 0.035, f"{shape} pow2={pow2}: rel err {rel}")

    def test_fp32_scale_is_amax_over_448(self):
        w = _weight((512, 1024), seed=1)
        _, s = quant_weight_block_fp8_online(w, BLOCK, pow2_scale=False)
        amax = w.float().view(4, 128, 8, 128).abs().amax(dim=(1, 3))
        torch.testing.assert_close(s, amax / 448.0)

    def test_pow2_scales_requant_exactly(self):
        # DeepGEMM UE8M0 requant (dequant -> BF16 -> per_block_cast_to_fp8) must be a no-op.
        for shape in ((4096, 6144), (16384, 2048), (6144, 16384)):
            w = _weight(shape, seed=7)
            q, s = quant_weight_block_fp8_online(w, BLOCK, pow2_scale=True)
            torch.testing.assert_close(s, torch.exp2(torch.round(torch.log2(s))))
            d = block_quant_dequant(q, s, BLOCK, torch.float32)
            q2, s2 = per_block_cast_to_fp8(
                block_quant_dequant(q, s, BLOCK, torch.bfloat16)
            )
            torch.testing.assert_close(
                block_quant_dequant(q2, s2, BLOCK, torch.float32), d, rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
