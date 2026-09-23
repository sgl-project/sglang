"""`Mxfp4MarlinComparable` must see through the Marlin repack and through equivalent MXFP4 encodings."""

import unittest

import torch

from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.srt.utils.weight_checker import _build_quantized_set
from sglang.srt.utils.weight_checker_comparator import (
    Mxfp4MarlinComparable,
    compare_weights,
    select_comparable_weight,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

_E2M1_VALUES = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))
# e2m1 code of twice the value, for codes whose value doubles within range
_DOUBLED_CODE = {0: 0, 1: 2, 2: 4, 3: 5, 4: 6, 5: 7}

_NUM_EXPERTS, _HIDDEN, _INTERMEDIATE = 2, 256, 128


def _random_checkpoint(device):
    """Checkpoint-layout MXFP4 experts: packed nibbles `(E, N, K/2)` and e8m0 bytes `(E, N, K/32)`.

    Every other block keeps its values at or below 3 so it has a second, equivalent encoding.
    """
    generator = torch.Generator(device="cpu").manual_seed(0)

    def packed(n, k):
        magnitude = torch.randint(
            0, 8, (_NUM_EXPERTS, n, k // 32, 32), generator=generator
        )
        magnitude[:, :, 0::2] %= 6
        sign = torch.randint(0, 2, magnitude.shape, generator=generator) << 3
        codes = (magnitude | sign).reshape(_NUM_EXPERTS, n, k).to(torch.uint8)
        return codes[..., 0::2] | (codes[..., 1::2] << 4)

    def scales(n, k):
        return torch.randint(
            118, 136, (_NUM_EXPERTS, n, k // 32), dtype=torch.uint8, generator=generator
        )

    return {
        "w13_weight": packed(2 * _INTERMEDIATE, _HIDDEN).to(device),
        "w13_weight_scale": scales(2 * _INTERMEDIATE, _HIDDEN).to(device),
        "w2_weight": packed(_HIDDEN, _INTERMEDIATE).to(device),
        "w2_weight_scale": scales(_HIDDEN, _INTERMEDIATE).to(device),
    }


def _dequantize_checkpoint(packed, scales):
    nibbles = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(
        *packed.shape[:-1], -1
    )
    values = _E2M1_VALUES.to(packed.device)[(nibbles & 0x7).long()] * (
        1.0 - 2.0 * (nibbles >> 3).float()
    )
    scale = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=-1)
    return (values * scale).to(torch.bfloat16)


def _repacked_layer(checkpoint):
    layer = torch.nn.Module()
    layer.orig_dtype = torch.bfloat16
    for name, tensor in checkpoint.items():
        layer.register_parameter(
            name, torch.nn.Parameter(tensor.clone(), requires_grad=False)
        )
    prepare_moe_mxfp4_layer_for_marlin(layer)
    return layer


def _re_encode_eligible_blocks(packed, scales):
    """Halve the scale and double the nibbles of the blocks that can take it: same values, new bytes."""
    nibbles = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(
        *packed.shape[:-1], -1, 32
    )
    magnitude = nibbles & 0x7
    eligible = (magnitude <= 5).all(dim=-1)
    doubled = magnitude.clone()
    for code, twice in _DOUBLED_CODE.items():
        doubled[magnitude == code] = twice
    new_nibbles = torch.where(
        eligible.unsqueeze(-1), doubled | (nibbles & 0x8), nibbles
    )
    new_nibbles = new_nibbles.reshape(*packed.shape[:-1], -1)
    new_packed = new_nibbles[..., 0::2] | (new_nibbles[..., 1::2] << 4)
    new_scales = torch.where(eligible, scales - 1, scales)
    assert eligible.any(), "the fixture must exercise at least one re-encoded block"
    return new_packed.contiguous(), new_scales.contiguous()


class TestMxfp4MarlinComparable(CustomTestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.checkpoint = _random_checkpoint(self.device)
        self.layer = _repacked_layer(self.checkpoint)

    def _comparable(self, layer, prefix):
        return Mxfp4MarlinComparable(
            getattr(layer, f"{prefix}_weight"), getattr(layer, f"{prefix}_weight_scale")
        )

    def test_dequantize_undoes_the_marlin_repack(self):
        for prefix in ("w13", "w2"):
            expected = _dequantize_checkpoint(
                self.checkpoint[f"{prefix}_weight"],
                self.checkpoint[f"{prefix}_weight_scale"],
            )
            actual = self._comparable(self.layer, prefix).dequantize()
            self.assertEqual(tuple(actual.shape), tuple(expected.shape), prefix)
            self.assertTrue(torch.equal(actual, expected), prefix)

    def test_equivalent_encodings_compare_equal(self):
        re_encoded = dict(self.checkpoint)
        for prefix in ("w13", "w2"):
            re_encoded[f"{prefix}_weight"], re_encoded[f"{prefix}_weight_scale"] = (
                _re_encode_eligible_blocks(
                    self.checkpoint[f"{prefix}_weight"],
                    self.checkpoint[f"{prefix}_weight_scale"],
                )
            )
            self.assertFalse(
                torch.equal(
                    re_encoded[f"{prefix}_weight"], self.checkpoint[f"{prefix}_weight"]
                )
            )
        other = _repacked_layer(re_encoded)
        for prefix in ("w13", "w2"):
            self.assertFalse(
                torch.equal(
                    getattr(other, f"{prefix}_weight"),
                    getattr(self.layer, f"{prefix}_weight"),
                )
            )
            result = compare_weights(
                self._comparable(self.layer, prefix), self._comparable(other, prefix)
            )
            self.assertTrue(result.equal, (prefix, result))

    def _compare_with_nibble_flipped(self, mask):
        changed = dict(self.checkpoint)
        changed["w2_weight"] = self.checkpoint["w2_weight"].clone()
        changed["w2_weight"][0, 3, 5] ^= mask
        other = _repacked_layer(changed)
        return compare_weights(
            self._comparable(self.layer, "w2"), self._comparable(other, "w2")
        )

    def test_a_changed_value_is_reported(self):
        # flipping bit 2 of the e2m1 code moves the value by at least four ulps
        result = self._compare_with_nibble_flipped(0x4)
        self.assertFalse(result.equal)
        self.assertGreater(result.num_exceed, 0)
        self.assertGreater(result.max_abs_err, 0.0)

    def test_a_one_ulp_change_stays_within_the_quantization_tolerance(self):
        result = self._compare_with_nibble_flipped(0x2)
        self.assertFalse(result.equal)
        self.assertEqual(result.num_exceed, 0)

    def test_accepts_the_checkers_shuffle_flag(self):
        # _build_check_entries passes is_shuffled to every quantized comparable
        Mxfp4MarlinComparable(
            self.layer.w2_weight, self.layer.w2_weight_scale, is_shuffled=False
        )
        with self.assertRaises(AssertionError):
            Mxfp4MarlinComparable(
                self.layer.w2_weight, self.layer.w2_weight_scale, is_shuffled=True
            )

    def test_chunking_matches_the_whole(self):
        comparable = self._comparable(self.layer, "w13")
        whole = comparable.dequantize()
        chunks = torch.cat([dq for dq, _ in comparable.iter_chunks()])
        self.assertTrue(torch.equal(chunks, whole))


class TestRouting(CustomTestCase):
    def _method(self, use_marlin):
        method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
        method.use_marlin = use_marlin
        return method

    def test_marlin_experts_get_the_comparable(self):
        self.assertIs(
            select_comparable_weight(self._method(True)), Mxfp4MarlinComparable
        )

    def test_other_mxfp4_backends_are_rejected(self):
        with self.assertRaises(NotImplementedError):
            select_comparable_weight(self._method(False))

    def test_quantized_set_pairs_weight_with_weight_scale(self):
        experts = torch.nn.Module()
        experts.quant_method = self._method(True)
        for name in (
            "w13_weight",
            "w13_weight_scale",
            "w2_weight",
            "w2_weight_scale",
            "w2_weight_bias",
        ):
            experts.register_parameter(
                name, torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            )
        model = torch.nn.Module()
        model.experts = experts

        quantized = _build_quantized_set(model)

        self.assertEqual(
            {name: qw.scale_name for name, qw in quantized.items()},
            {
                "experts.w13_weight": "experts.w13_weight_scale",
                "experts.w2_weight": "experts.w2_weight_scale",
            },
        )
        self.assertTrue(
            all(qw.comparable_cls is Mxfp4MarlinComparable for qw in quantized.values())
        )


if __name__ == "__main__":
    unittest.main()
