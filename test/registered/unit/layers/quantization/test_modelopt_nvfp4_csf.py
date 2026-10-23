"""Unit tests for srt/layers/quantization/nvfp4_csf.py and its modelopt_mixed wiring."""

import unittest
from unittest.mock import Mock, patch

import torch

import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig
from sglang.srt.layers.quantization.nvfp4_csf import (
    decode_csf_scales,
    decode_csf_weight_scales,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The routed-expert prefix of FP4-Local's first layer, as quantized_layers keys it
EXPERTS = "model.language_model.layers.0.mlp.experts"


def _records(words: list[int]) -> torch.Tensor:
    """A uint32 exception stream holding WORDS, each below 2**31."""
    return torch.tensor(words, dtype=torch.int32).view(torch.uint32)


def _encode(scale_bytes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The CSF fixed and exception streams of a uint8 `[rows, columns]` scale matrix."""
    rows, columns = scale_bytes.shape
    raw = scale_bytes.to(torch.int64)
    bases = raw.min(dim=1, keepdim=True).values.clamp(max=240)
    offsets = raw - bases
    is_exception = offsets > 15
    offsets = torch.where(is_exception, 0, offsets)
    packed = offsets[:, 0::2] | (offsets[:, 1::2] << 4)
    fixed = torch.cat(
        (bases.reshape(rows // 16, 16), packed.reshape(rows // 16, -1)), dim=1
    ).to(torch.uint8)
    positions = is_exception.reshape(-1).nonzero().reshape(-1)
    words = (raw.reshape(-1)[positions] << 24) | positions
    return fixed, words.to(torch.int32).view(torch.uint32)


def _random_scales(rows: int, columns: int, exception_count: int) -> torch.Tensor:
    """Positive E4M3 scale bytes inside one window, with EXCEPTION_COUNT outliers."""
    generator = torch.Generator().manual_seed(rows * columns)
    scale_bytes = torch.randint(
        0x30, 0x40, (rows, columns), dtype=torch.uint8, generator=generator
    )
    outliers = torch.randperm(rows * columns, generator=generator)[:exception_count]
    scale_bytes.view(-1)[outliers] = 0x7E
    return scale_bytes


def _fp4_local_like_config(**csf_overrides) -> ModelOptMixedPrecisionConfig:
    """A config with one layer of each recipe FP4-Local's quantized_layers uses."""
    csf_layer = {"quant_algo": "NVFP4", "group_size": 16}
    csf_layer["weight_scale_encoding"] = "csf"
    csf_layer.update(csf_overrides)
    return ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                EXPERTS: csf_layer,
                "model.language_model.layers.0.self_attn.q_proj": {
                    "quant_algo": "MXFP8",
                    "group_size": 32,
                },
                "model.language_model.embed_tokens_per_layer": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
                "mtp.layers.0.mlp.experts": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                },
            },
        }
    )


class TestDecodeCsfScales(CustomTestCase):
    def test_decodes_the_flashinfer_layout(self):
        # One 16 x 4 slab: row r has base 0x30 + r, and offset byte 0x21 puts
        # 1 in the even column (low nibble) and 2 in the odd one (high nibble).
        bases = torch.arange(0x30, 0x40, dtype=torch.uint8)
        nibbles = torch.tensor([0x21, 0x43], dtype=torch.uint8).repeat(16)
        fixed = torch.cat((bases, nibbles)).reshape(1, 48)
        # Position 1 * 4 + 2 (row 1, column 2) holds byte 0x7E.
        exceptions = _records([(0x7E << 24) | 6])

        scales = decode_csf_scales("layer", fixed, exceptions)

        offsets = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
        expected = bases.reshape(16, 1) + offsets
        expected[1, 2] = 0x7E
        self.assertEqual(scales.dtype, torch.float8_e4m3fn)
        self.assertTrue(torch.equal(scales.view(torch.uint8), expected))

    def test_round_trip_is_lossless(self):
        # Flash-Next's gate/up and down scale shapes, plus a single slab.
        shapes = ((640, 160, 64), (2560, 40, 64), (16, 4, 3))
        for rows, columns, exception_count in shapes:
            with self.subTest(rows=rows, columns=columns):
                scale_bytes = _random_scales(rows, columns, exception_count)
                fixed, exceptions = _encode(scale_bytes)
                self.assertEqual(exceptions.numel(), exception_count)

                scales = decode_csf_scales("layer", fixed, exceptions)

                self.assertTrue(torch.equal(scales.view(torch.uint8), scale_bytes))

    def test_malformed_streams_raise(self):
        fixed, exceptions = _encode(_random_scales(32, 4, 0))
        records = [(0x7E << 24) | position for position in (3, 9)]
        bad_base = fixed.clone()
        bad_base[0, 5] = 241
        cases = {
            "fixed dtype": (fixed.to(torch.int16), exceptions),
            "fixed rank": (fixed.reshape(-1), exceptions),
            "empty fixed": (fixed[:0], exceptions),
            "slab width": (fixed[:, :47], exceptions),
            "base above 240": (bad_base, exceptions),
            "exceptions dtype": (fixed, exceptions.view(torch.int32)),
            "unsorted": (fixed, _records(records[::-1])),
            "duplicate": (fixed, _records(records[:1] * 2)),
            "out of range": (fixed, _records([32 * 4])),
        }
        for case, (case_fixed, case_exceptions) in cases.items():
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, "layer"):
                decode_csf_scales("layer", case_fixed, case_exceptions)


class TestDecodeCsfWeightScales(CustomTestCase):
    def test_stream_pairs_become_weight_scales(self):
        scale_bytes = _random_scales(32, 4, 2)
        fixed, exceptions = _encode(scale_bytes)
        prefix = f"{EXPERTS}.7.down_proj"
        weights = [
            (f"{prefix}.weight", torch.zeros(1)),
            (f"{prefix}.weight_scale.nvfp4_csf_exceptions", exceptions),
            (f"{prefix}.weight_scale_2", torch.ones(1)),
            (f"{prefix}.weight_scale.nvfp4_csf_fixed", fixed),
        ]

        decoded = list(
            decode_csf_weight_scales(weights, scale_encoding_of=lambda _: "csf")
        )

        names = [name for name, _ in decoded]
        self.assertEqual(
            names,
            [f"{prefix}.weight", f"{prefix}.weight_scale_2", f"{prefix}.weight_scale"],
        )
        self.assertTrue(torch.equal(decoded[2][1].view(torch.uint8), scale_bytes))

    def test_unpaired_stream_raises(self):
        fixed, _ = _encode(_random_scales(16, 4, 0))
        weights = [("layer.weight_scale.nvfp4_csf_fixed", fixed)]

        with self.assertRaisesRegex(ValueError, "no partner.*layer"):
            list(decode_csf_weight_scales(weights, scale_encoding_of=lambda _: "csf"))

    def test_pair_on_a_layer_not_marked_csf_raises(self):
        fixed, exceptions = _encode(_random_scales(16, 4, 0))
        weights = [
            ("layer.weight_scale.nvfp4_csf_fixed", fixed),
            ("layer.weight_scale.nvfp4_csf_exceptions", exceptions),
        ]

        with self.assertRaisesRegex(ValueError, "layer.*None"):
            list(decode_csf_weight_scales(weights, scale_encoding_of=lambda _: None))


class TestModelOptMixedPrecisionCsf(CustomTestCase):
    def test_scale_encoding_resolves_per_expert_names(self):
        config = _fp4_local_like_config()

        self.assertTrue(config.has_csf_scales)
        expected = {
            f"{EXPERTS}.7.down_proj": "csf",
            "model.layers.0.mlp.experts.7.gate_proj": "csf",
            "model.language_model.layers.0.self_attn.q_proj": None,
            "model.language_model.embed_tokens_per_layer": None,
            "mtp.layers.0.mlp.experts.3.gate_proj": None,
            "model.language_model.layers.1.mlp.experts.0.up_proj": None,
        }
        for prefix, encoding in expected.items():
            with self.subTest(prefix=prefix):
                self.assertEqual(config.scale_encoding_of(prefix), encoding)

    def test_unsupported_recipes_raise(self):
        cases = {
            "unknown encoding": {"weight_scale_encoding": "zstd"},
            "MXFP8 with csf": {"quant_algo": "MXFP8", "group_size": 32},
            "group size 32": {"group_size": 32},
        }
        for case, overrides in cases.items():
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, EXPERTS):
                _fp4_local_like_config(**overrides)

    def test_loader_decodes_before_load_weights(self):
        scale_bytes = _random_scales(16, 4, 1)
        fixed, exceptions = _encode(scale_bytes)
        prefix = f"{EXPERTS}.0.gate_proj"
        loaded = {}
        model = Mock(quant_config=_fp4_local_like_config())
        model.load_weights.side_effect = lambda weights: loaded.update(weights)

        with patch.object(loader_mod, "is_cuda_alike", return_value=False):
            DefaultModelLoader.load_weights_only(
                model,
                iter(
                    [
                        (f"{prefix}.weight_scale.nvfp4_csf_fixed", fixed),
                        (f"{prefix}.weight_scale.nvfp4_csf_exceptions", exceptions),
                    ]
                ),
                torch.device("cpu"),
            )

        self.assertEqual(list(loaded), [f"{prefix}.weight_scale"])
        decoded = loaded[f"{prefix}.weight_scale"]
        self.assertTrue(torch.equal(decoded.view(torch.uint8), scale_bytes))


if __name__ == "__main__":
    unittest.main()
