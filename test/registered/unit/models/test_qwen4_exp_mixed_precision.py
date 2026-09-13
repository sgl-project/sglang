"""Regression tests for Qwen4-Exp on three-precision ModelOpt MIXED_PRECISION
checkpoints (`nvidia/Qwen3.8-Flash-Next-NVFP4`: NVFP4 experts + FP8 PLE table
+ block-FP8 MTP experts).

Each case guards a decision that used to be made before consulting the
per-layer `quantized_layers` map:
  1. The PLE n-gram table's storage dtype is fixed at construction. Building it
     bf16 and switching to fp8 in load_weights is impossible once the table
     sits in pinned host memory (`--ple-offload-embedding`, default on), so
     the FP8 entry in `quantized_layers` must be honoured at construction.
  2. The draft (MTP) module dropped its quant config for every modelopt_mixed
     checkpoint. With `mtp.*.experts` listed as block-FP8 that cast the fp8
     expert values to bf16 without their scales and silently discarded
     `weight_scale_inv` (accept length fell from ~3.0 to ~1.6 with no error).
"""

import unittest

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.models.qwen3_5_mtp import _mtp_quant_config
from sglang.srt.models.qwen4_exp import _ple_table_is_fp8
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
MTP_EXPERTS = "mtp.layers.0.mlp.experts"


def _mixed_config(quantized_layers):
    return ModelOptMixedPrecisionConfig.from_config(
        {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "exclude_modules": [],
                "group_size": 16,
                "quantized_layers": quantized_layers,
            }
        }
    )


class _Cfg:
    ple_embedding_dtype = None


class TestPLETableDtype(CustomTestCase):
    def test_mixed_precision_fp8_ple_entry_selects_fp8_storage(self):
        quant_config = _mixed_config(
            {
                PLE_PREFIX: {"quant_algo": "FP8"},
                "model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"},
            }
        )
        self.assertTrue(_ple_table_is_fp8(_Cfg(), quant_config, PLE_PREFIX))
        # A checkpoint that quantizes only the experts keeps the table bf16.
        self.assertFalse(
            _ple_table_is_fp8(
                _Cfg(),
                _mixed_config(
                    {
                        "model.language_model.layers.0.mlp.experts": {
                            "quant_algo": "NVFP4"
                        }
                    }
                ),
                PLE_PREFIX,
            )
        )

    def test_hf_style_prefix_resolves_the_same_entry(self):
        # Multimodal prefixes drift between `model.language_model.` and
        # `language_model.model.`; both must find the FP8 entry.
        quant_config = _mixed_config({PLE_PREFIX: {"quant_algo": "FP8"}})
        drifted = PLE_PREFIX.replace("model.language_model.", "language_model.model.")
        self.assertTrue(_ple_table_is_fp8(_Cfg(), quant_config, drifted))


class TestMTPQuantConfig(CustomTestCase):
    def test_mixed_precision_keeps_config_when_mtp_layers_are_quantized(self):
        quant_config = _mixed_config(
            {
                "model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"},
                MTP_EXPERTS: {"quant_algo": "FP8_BLOCK_SCALES", "group_size": 128},
            }
        )
        self.assertIs(_mtp_quant_config(quant_config), quant_config)

    def test_mixed_precision_drops_config_when_mtp_ships_bf16(self):
        quant_config = _mixed_config(
            {"model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"}}
        )
        self.assertIsNone(_mtp_quant_config(quant_config))


if __name__ == "__main__":
    unittest.main()
