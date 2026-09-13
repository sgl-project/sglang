"""Unit tests for compressed-tensors lm_head scheme resolution — CPU-only."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from sglang.test.test_utils import CustomTestCase

_FP8_WEIGHTS = {
    "num_bits": 8,
    "type": "float",
    "strategy": "channel",
    "symmetric": True,
    "dynamic": False,
}
_FP8_DYNAMIC_ACTS = {
    "num_bits": 8,
    "type": "float",
    "strategy": "token",
    "symmetric": True,
    "dynamic": True,
}


def _config(targets, ignore=()):
    return CompressedTensorsConfig.from_config(
        {
            "format": "float-quantized",
            "quant_method": "compressed-tensors",
            "ignore": list(ignore),
            "config_groups": {
                "group_0": {
                    "targets": list(targets),
                    "weights": _FP8_WEIGHTS,
                    "input_activations": _FP8_DYNAMIC_ACTS,
                }
            },
        }
    )


class _Head(torch.nn.Module):
    pass


_GET_LINEAR_SCHEME = (
    "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors."
    "CompressedTensorsConfig.get_linear_scheme"
)


class TestGetLmHeadScheme(CustomTestCase):
    """The head resolves a scheme only when a config target names it by
    layer name; module-type targets and ignored heads stay unquantized."""

    def test_regex_target_resolves(self):
        config = _config(["re:.*lm_head", "re:.*mlp\\.down_proj$"])
        head = _Head()
        with patch(_GET_LINEAR_SCHEME, return_value="scheme") as mock_resolve:
            scheme = config.get_lm_head_scheme(head, "lm_head")
        self.assertEqual(scheme, "scheme")
        mock_resolve.assert_called_once_with(
            layer=head, layer_name="lm_head", matched_target="re:.*lm_head"
        )

    def test_exact_target_resolves(self):
        config = _config(["lm_head"])
        with patch(_GET_LINEAR_SCHEME, return_value="scheme"):
            self.assertEqual(config.get_lm_head_scheme(_Head(), "lm_head"), "scheme")

    def test_ignored_head_is_none(self):
        config = _config(["re:.*lm_head"], ignore=["lm_head"])
        with patch(_GET_LINEAR_SCHEME) as mock_resolve:
            self.assertIsNone(config.get_lm_head_scheme(_Head(), "lm_head"))
        mock_resolve.assert_not_called()

    def test_module_type_target_is_none(self):
        # llm-compressor emits "Linear" for decoder linears; an unmentioned
        # head must stay on the unquantized path instead of tripping
        # find_matched_target's unmatched-layer error.
        config = _config(["Linear"])
        with patch(_GET_LINEAR_SCHEME) as mock_resolve:
            self.assertIsNone(config.get_lm_head_scheme(_Head(), "lm_head"))
        mock_resolve.assert_not_called()

    def test_no_layer_name_is_none(self):
        config = _config(["re:.*lm_head"])
        self.assertIsNone(config.get_lm_head_scheme(_Head(), None))

    def test_prefixed_head_with_plain_target_resolves(self):
        """Bug regression: `check_equal_or_regex_match` accepts the dotted
        suffix ("lm_head" target for a "language_model.lm_head" prefix) but
        `find_matched_target`'s name pass is exact/regex only, so re-deriving
        the match downstream raised ValueError at load instead of resolving
        the scheme. The suffix match must be carried through."""
        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            CompressedTensorsW8A8Fp8,
        )

        config = _config(["lm_head"])
        with patch(
            "sglang.srt.layers.quantization.compressed_tensors."
            "compressed_tensors.CompressedTensorsConfig._check_scheme_supported",
            return_value=True,
        ):
            scheme = config.get_lm_head_scheme(_Head(), "language_model.lm_head")
        self.assertIsInstance(scheme, CompressedTensorsW8A8Fp8)

    def test_block_quantized_head_is_rejected(self):
        """Bug regression: a block-FP8 head resolves to a weight_scale whose
        first dim is vocab/block_n, which the vocab-parallel weight loader
        (asserting dim0 == vocab size on output_dim=0 params) cannot load
        even at TP=1. Reject loudly instead of asserting mid-load."""
        block_weights = dict(_FP8_WEIGHTS, strategy="block", block_structure=[128, 128])
        config = CompressedTensorsConfig.from_config(
            {
                "format": "float-quantized",
                "quant_method": "compressed-tensors",
                "ignore": [],
                "config_groups": {
                    "group_0": {
                        "targets": ["re:.*lm_head"],
                        "weights": block_weights,
                        "input_activations": _FP8_DYNAMIC_ACTS,
                    }
                },
            }
        )
        with self.assertRaises(NotImplementedError):
            config.get_lm_head_scheme(_Head(), "lm_head")


class TestGetQuantMethodLmHead(CustomTestCase):
    def _head(self):
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        # __new__ is enough: get_quant_method only isinstance-checks the
        # layer and attaches `scheme` to it.
        return ParallelLMHead.__new__(ParallelLMHead)

    def test_quantized_head_gets_linear_method(self):
        config = _config(["re:.*lm_head"])
        head = self._head()
        with patch.object(config, "get_lm_head_scheme", return_value="scheme"):
            method = config.get_quant_method(head, "lm_head")
        self.assertIsInstance(method, CompressedTensorsLinearMethod)
        self.assertEqual(head.scheme, "scheme")

    def test_unquantized_head_falls_back(self):
        config = _config(["Linear"])
        head = self._head()
        self.assertIsNone(config.get_quant_method(head, "lm_head"))


if __name__ == "__main__":
    unittest.main()
