"""Routing contract for --quantization-overrides."""

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.quantization import (
    ONLINE_QUANTIZATION_METHODS,
)
from sglang.multimodal_gen.runtime.layers.quantization.mixed_precision import (
    MixedPrecisionConfig,
    PrecisionRule,
    parse_precision_rules,
)

_FORMATS = ONLINE_QUANTIZATION_METHODS


class _StubMethod:
    def __init__(self, format_name: str) -> None:
        self.format_name = format_name


class _StubConfig:
    """Stands in for a real sub-config so routing needs no quantization kernels."""

    def __init__(self, format_name: str, ignored_layers=None, **_kwargs) -> None:
        self.format_name = format_name
        self.ignored_layers = list(ignored_layers or [])

    def get_quant_method(self, layer, prefix: str):
        del layer
        if any(prefix.startswith(ignored) for ignored in self.ignored_layers):
            return None
        return _StubMethod(self.format_name)


def _stub_quantization_config(format_name: str):
    return lambda **kwargs: _StubConfig(format_name, **kwargs)


def _build_config(base_format: str, rules, ignored_layers=None):
    with patch(
        "sglang.multimodal_gen.runtime.layers.quantization.get_quantization_config",
        side_effect=_stub_quantization_config,
    ):
        return MixedPrecisionConfig(
            base_format=base_format,
            rules=tuple(rules),
            ignored_layers=ignored_layers,
        )


class TestPrecisionRuleParsing(unittest.TestCase):
    def test_entry_without_format_is_rejected(self):
        for entry in ("blocks.0", "blocks.0=", "=fp8", "blocks.0=="):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                parse_precision_rules([entry], supported_formats=_FORMATS)

    def test_checkpoint_format_is_rejected(self):
        """A format fixed on disk cannot be chosen per layer at load time."""
        with self.assertRaisesRegex(ValueError, "not quantized after load"):
            parse_precision_rules(
                ["blocks.0=modelopt_fp4"],
                supported_formats=_FORMATS,
            )

    def test_duplicate_rules_collapse(self):
        rules = parse_precision_rules(
            ["blocks.0=fp8", "blocks.0=fp8"],
            supported_formats=_FORMATS,
        )

        self.assertEqual((PrecisionRule(pattern="blocks.0", format="fp8"),), rules)


class TestMixedPrecisionRouting(unittest.TestCase):
    def test_unclaimed_layer_takes_the_base_format(self):
        config = _build_config(
            "mxfp4", [PrecisionRule(pattern="blocks.0", format="fp8")]
        )

        self.assertEqual("mxfp4", config.format_for("blocks.7.ffn.fc_in"))

    def test_block_index_does_not_match_a_longer_index(self):
        """`blocks.1` must not claim `blocks.10`, or a rule silently spreads."""
        config = _build_config(
            "mxfp4", [PrecisionRule(pattern="blocks.1", format="fp8")]
        )

        self.assertEqual("fp8", config.format_for("blocks.1.to_q"))
        self.assertEqual("mxfp4", config.format_for("blocks.10.to_q"))
        self.assertEqual("mxfp4", config.format_for("blocks.19.ffn.fc_out"))

    def test_glob_pattern_matches_the_whole_prefix(self):
        config = _build_config(
            "mxfp4", [PrecisionRule(pattern="*.ffn.fc_out", format="fp8")]
        )

        self.assertEqual("fp8", config.format_for("blocks.12.ffn.fc_out"))
        self.assertEqual("mxfp4", config.format_for("blocks.12.ffn.fc_in"))

    def test_first_matching_rule_wins(self):
        config = _build_config(
            "mxfp4",
            [
                PrecisionRule(pattern="blocks.3", format="convrot_int8"),
                PrecisionRule(pattern="blocks.3", format="fp8"),
            ],
        )

        self.assertEqual("convrot_int8", config.format_for("blocks.3.to_q"))

    def test_quant_method_comes_from_the_routed_sub_config(self):
        config = _build_config(
            "mxfp4", [PrecisionRule(pattern="blocks.0", format="fp8")]
        )
        layer = torch.nn.Linear(8, 8)

        self.assertEqual(
            "fp8", config.get_quant_method(layer, "blocks.0.to_q").format_name
        )
        self.assertEqual(
            "mxfp4", config.get_quant_method(layer, "blocks.5.to_q").format_name
        )

    def test_ignored_layers_reach_every_sub_config(self):
        """An ignored layer stays unquantized whichever rule claims it."""
        config = _build_config(
            "mxfp4",
            [PrecisionRule(pattern="blocks.0", format="fp8")],
            ignored_layers=["blocks.0.to_q", "blocks.5.to_q"],
        )
        layer = torch.nn.Linear(8, 8)

        self.assertIsNone(config.get_quant_method(layer, "blocks.0.to_q"))
        self.assertIsNone(config.get_quant_method(layer, "blocks.5.to_q"))


class TestServerArgsValidation(unittest.TestCase):
    def _server_args(self, **fields):
        from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs

        with patch.object(ServerArgs, "__post_init__", lambda self: None):
            server_args = ServerArgs(model_path="test")
        for name, value in fields.items():
            setattr(server_args, name, value)
        return server_args

    def test_overrides_require_an_online_quantization(self):
        server_args = self._server_args(
            quantization="modelopt_fp4", quantization_overrides=["blocks.0=fp8"]
        )

        with self.assertRaisesRegex(ValueError, "fixed in the checkpoint"):
            server_args._normalize_quantization_overrides()

    def test_overrides_require_a_base_quantization(self):
        server_args = self._server_args(
            quantization=None, quantization_overrides=["blocks.0=fp8"]
        )

        with self.assertRaisesRegex(ValueError, "must name an online"):
            server_args._normalize_quantization_overrides()

    def test_absent_overrides_normalize_to_none(self):
        server_args = self._server_args(quantization=None, quantization_overrides=[])
        server_args._normalize_quantization_overrides()

        self.assertIsNone(server_args.quantization_overrides)


if __name__ == "__main__":
    unittest.main()
