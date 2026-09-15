"""Persimmon per-module quantization selection with real CPU model layers."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import PersimmonConfig

from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp8Config,
    ModelOptFp8LinearMethod,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.persimmon import PersimmonForCausalLM
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPersimmonQuantizationPrefix(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(get_context().override_server_args())
        self.enterContext(
            get_parallel().override(
                tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0
            )
        )
        pp_group = SimpleNamespace(
            is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1
        )
        self.enterContext(
            patch("sglang.srt.models.persimmon.get_pp_group", return_value=pp_group)
        )
        self.enterContext(patch("sglang.srt.distributed.parallel_state._TP", pp_group))
        # No attention forward is needed. Construct the real RoPE cache without
        # importing an optional device kernel on CPU-only test installations.
        self.enterContext(
            patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
        )
        self.enterContext(patch.dict(os.environ, SGLANG_FP8_IGNORED_LAYERS=""))
        self.enterContext(torch.device("cpu"))

    def _model(self, quant_config=None, prefix=""):
        config = PersimmonConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=64,
            max_position_embeddings=16,
        )
        return PersimmonForCausalLM(
            config=config, quant_config=quant_config, prefix=prefix
        )

    def _projections(self, model):
        return {
            name: module
            for name, module in model.named_modules()
            if isinstance(
                getattr(module, "quant_method", None),
                (Fp8LinearMethod, ModelOptFp8LinearMethod, UnquantizedLinearMethod),
            )
            and name != "lm_head"
        }

    def test_selective_fp8_exclusions(self):
        projections = (
            "self_attn.query_key_value",
            "self_attn.dense",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        )
        for prefix in ("", "language_model"):
            for projection in projections:
                target = f"model.layers.1.{projection}"
                qualified = f"{prefix}.{target}" if prefix else target
                with self.subTest(prefix=prefix, projection=projection):
                    model = self._model(
                        Fp8Config(ignored_layers=[qualified]), prefix=prefix
                    )
                    modules = self._projections(model)
                    self.assertEqual(len(modules), 8)
                    for name, layer in modules.items():
                        expected = (
                            UnquantizedLinearMethod
                            if name == target
                            else Fp8LinearMethod
                        )
                        self.assertIsInstance(layer.quant_method, expected, name)

    def test_checkpoint_modules_to_not_convert(self):
        target = "model.layers.0.mlp.dense_h_to_4h"
        quant_config = Fp8Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "modules_to_not_convert": [target],
            }
        )
        model = self._model(quant_config)
        self.assertIsInstance(
            model.model.layers[0].mlp.dense_h_to_4h.quant_method,
            UnquantizedLinearMethod,
        )
        self.assertIsInstance(
            model.model.layers[1].mlp.dense_h_to_4h.quant_method, Fp8LinearMethod
        )

    def test_modelopt_head_exclusion(self):
        for prefix in ("", "language_model"):
            head_name = f"{prefix}.lm_head" if prefix else "lm_head"
            with self.subTest(prefix=prefix):
                model = self._model(
                    ModelOptFp8Config(
                        exclude_modules=[head_name], packed_modules_mapping={}
                    ),
                    prefix=prefix,
                )
                self.assertIsInstance(
                    model.lm_head.quant_method, UnquantizedLinearMethod
                )
                for layer in self._projections(model).values():
                    self.assertIsInstance(layer.quant_method, ModelOptFp8LinearMethod)

    def test_no_exclusions_preserve_quantization(self):
        for config, expected in (
            (Fp8Config(), Fp8LinearMethod),
            (ModelOptFp8Config(packed_modules_mapping={}), ModelOptFp8LinearMethod),
        ):
            with self.subTest(method=config.get_name()):
                model = self._model(config)
                for layer in self._projections(model).values():
                    self.assertIsInstance(layer.quant_method, expected)
                if isinstance(config, ModelOptFp8Config):
                    self.assertIsInstance(model.lm_head.quant_method, expected)

    def test_unquantized_model_is_unchanged(self):
        model = self._model()
        for layer in self._projections(model).values():
            self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)

    def test_excluded_mlp_matches_dense_reference(self):
        ignored = [
            f"model.layers.0.mlp.{name}" for name in ("dense_h_to_4h", "dense_4h_to_h")
        ]
        model = self._model(Fp8Config(ignored_layers=ignored))
        mlp = model.model.layers[0].mlp
        for layer in (mlp.dense_h_to_4h, mlp.dense_4h_to_h):
            self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)
        generator = torch.Generator().manual_seed(42)
        with torch.no_grad():
            for parameter in mlp.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
            hidden = torch.randn(3, 32, generator=generator)
            intermediate = F.linear(
                hidden, mlp.dense_h_to_4h.weight, mlp.dense_h_to_4h.bias
            )
            expected = F.linear(
                F.relu(intermediate).square(),
                mlp.dense_4h_to_h.weight,
                mlp.dense_4h_to_h.bias,
            )
            torch.testing.assert_close(mlp(hidden), expected)


if __name__ == "__main__":
    unittest.main()
