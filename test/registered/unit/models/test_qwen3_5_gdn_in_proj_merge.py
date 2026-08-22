"""Unit tests for the Qwen3.5 GDN in_proj_qkvzba merge.

Folding in_proj_ba into in_proj_qkvz gives one padded GEMM, so the merged layer
must produce exactly what the separate projections produce and the weight loader
must reach the same rows -- six shards into one parameter instead of four into one
and two into another. That logic is device-independent; the Triton split kernel it
feeds is covered by registered/ops/test_gdn_fused_proj_amd.py.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.srt.models import qwen3_5
from sglang.srt.models.qwen3_5 import (
    _GEMM_N_ALIGN,
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    _gdn_input_proj_stacked_mapping,
)
from sglang.test.test_utils import CustomTestCase

_HIDDEN = 256
_PREFIX = "model.layers.0.linear_attn"
_NO_LORA = SimpleNamespace(lora_paths=None, enable_lora=False)
# 64+64+128+128+8+8 = 400 -> padded to 512; 128+128+128+128+64+64 = 640 -> no pad.
_PADDED = (64, 128, 8)
_ALIGNED = (128, 128, 64)

_FP8 = {
    "bias": None,
    "output_tensors": None,
    "weight": {"dtype": "fp8_e4m3", "qscheme": "per_channel", "is_dynamic": False},
    "input_tensors": {
        "dtype": "fp8_e4m3",
        "qscheme": "per_channel",
        "is_dynamic": True,
    },
}


def _quark_fp8_config():
    """One fp8 scheme over all four shards, which is what makes them mergeable."""
    return QuarkConfig.from_config(
        {
            "quant_method": "quark",
            "export": {"kv_cache_group": [], "pack_method": "reorder"},
            "global_quant_config": _FP8,
            "layer_quant_config": {"*linear_attn*": _FP8},
            "layer_type_quant_config": {},
            "exclude": [],
            # The merged name resolves to its shards through this mapping, so use
            # the one the model ships rather than a copy that can drift from it.
            "packed_modules_mapping": Qwen3_5ForCausalLM.packed_modules_mapping,
        }
    )


class _StubModel:
    def __init__(self, *names):
        self._names = names

    def named_parameters(self):
        return [(name, None) for name in self._names]


def _make_merged(
    key_dim, value_dim, num_v_heads, tp_size=1, lora=_NO_LORA, quant_config=None
):
    with patch.object(qwen3_5, "_fuse_gdn_qkvzba", True), patch.object(
        qwen3_5, "get_lora", lambda: lora
    ):
        return Qwen3_5GatedDeltaNet.create_qkvzba_proj(
            None,
            hidden_size=_HIDDEN,
            key_dim=key_dim,
            value_dim=value_dim,
            num_v_heads=num_v_heads,
            quant_config=quant_config,
            prefix=f"{_PREFIX}.in_proj_qkvzba",
            tp_rank=0,
            tp_size=tp_size,
        )


def _make_separate(key_dim, value_dim, num_v_heads, tp_size=1, quant_config=None):
    qkvz = Qwen3_5GatedDeltaNet.create_qkvz_proj(
        None,
        hidden_size=_HIDDEN,
        key_dim=key_dim,
        value_dim=value_dim,
        quant_config=quant_config,
        prefix=f"{_PREFIX}.in_proj_qkvz",
        tp_rank=0,
        tp_size=tp_size,
    )
    ba = Qwen3_5GatedDeltaNet.create_ba_proj(
        None,
        hidden_size=_HIDDEN,
        num_v_heads=num_v_heads,
        quant_config=quant_config,
        prefix=f"{_PREFIX}.in_proj_ba",
        tp_rank=0,
        tp_size=tp_size,
    )
    return qkvz, ba


def _checkpoint_shards(key_dim, value_dim, num_v_heads):
    """q, k, v, z, b, a as the checkpoint stores them."""
    torch.manual_seed(0)
    sizes = [key_dim, key_dim, value_dim, value_dim, num_v_heads, num_v_heads]
    return [torch.randn(n, _HIDDEN) for n in sizes]


def _load(layer, shards):
    for shard_id, shard in enumerate(shards):
        layer.weight_loader(layer.weight, shard, shard_id)


class TestGdnInProjMerge(CustomTestCase):
    def _assert_merged_matches_separate(self, key_dim, value_dim, num_v_heads):
        shards = _checkpoint_shards(key_dim, value_dim, num_v_heads)
        merged = _make_merged(key_dim, value_dim, num_v_heads)
        self.assertIsNotNone(merged)
        qkvz, ba = _make_separate(key_dim, value_dim, num_v_heads)

        _load(merged, shards)
        _load(qkvz, shards[:4])
        _load(ba, shards[4:])

        x = torch.randn(7, _HIDDEN)
        merged_out, _ = merged(x)
        split = SimpleNamespace(
            qkvz_width=2 * key_dim + 2 * value_dim, ba_width=2 * num_v_heads
        )
        got_qkvz, got_ba = Qwen3_5GatedDeltaNet._split_qkvzba(split, merged_out)

        want_qkvz, _ = qkvz(x)
        want_ba, _ = ba(x)
        torch.testing.assert_close(got_qkvz, want_qkvz)
        torch.testing.assert_close(got_ba, want_ba)

    def test_merged_matches_separate_when_padded(self):
        self._assert_merged_matches_separate(*_PADDED)

    def test_merged_matches_separate_when_already_aligned(self):
        self._assert_merged_matches_separate(*_ALIGNED)

    def test_padding_rows_stay_zero_after_loading(self):
        key_dim, value_dim, num_v_heads = _PADDED
        used = 2 * key_dim + 2 * value_dim + 2 * num_v_heads
        merged = _make_merged(key_dim, value_dim, num_v_heads)

        self.assertEqual(merged.weight.shape[0] % _GEMM_N_ALIGN, 0)
        self.assertGreater(merged.weight.shape[0], used)
        _load(merged, _checkpoint_shards(key_dim, value_dim, num_v_heads))
        self.assertEqual(merged.weight[used:].count_nonzero().item(), 0)

    def test_aligned_shapes_are_not_padded(self):
        key_dim, value_dim, num_v_heads = _ALIGNED
        merged = _make_merged(key_dim, value_dim, num_v_heads)
        self.assertEqual(
            merged.weight.shape[0], 2 * key_dim + 2 * value_dim + 2 * num_v_heads
        )

    def test_padding_accounts_for_tensor_parallel_shards(self):
        key_dim, value_dim, num_v_heads = _PADDED
        merged = _make_merged(key_dim, value_dim, num_v_heads, tp_size=2)
        self.assertEqual(merged.weight.shape[0] % _GEMM_N_ALIGN, 0)

    def test_disabled_keeps_projections_separate(self):
        with patch.object(qwen3_5, "_fuse_gdn_qkvzba", False):
            self.assertIsNone(
                Qwen3_5GatedDeltaNet.create_qkvzba_proj(
                    None,
                    hidden_size=_HIDDEN,
                    key_dim=64,
                    value_dim=128,
                    num_v_heads=8,
                    quant_config=None,
                    prefix=f"{_PREFIX}.in_proj_qkvzba",
                    tp_rank=0,
                    tp_size=1,
                )
            )

    def test_lora_keeps_projections_separate(self):
        # supported_lora_modules names in_proj_qkvz, which the merge removes.
        lora = SimpleNamespace(lora_paths=["some-adapter"], enable_lora=True)
        self.assertIsNone(_make_merged(*_PADDED, lora=lora))


class TestGdnInProjMergeQuantized(CustomTestCase):
    """The fp8 checkpoint layout, where every shard also carries a weight_scale."""

    def setUp(self):
        key_dim, value_dim, num_v_heads = _PADDED
        self.qkvz_width = 2 * key_dim + 2 * value_dim
        self.ba_width = 2 * num_v_heads
        self.merged = _make_merged(*_PADDED, quant_config=_quark_fp8_config())

    def _load_scales(self, layer, scales):
        for shard_id, scale in enumerate(scales):
            layer.weight_loader(layer.weight_scale, scale, shard_id)

    def test_uniform_fp8_scheme_builds_the_merged_layer(self):
        """A scheme the merged layer rejects turns into the separate-projection path."""
        self.assertIsNotNone(self.merged)

    def test_quantized_shard_rows_match_the_separate_projections(self):
        """Six weights and six per-channel scales land where qkvz and ba put them."""
        qkvz, ba = _make_separate(*_PADDED, quant_config=_quark_fp8_config())
        shards = [
            shard.to(torch.float8_e4m3fn) for shard in _checkpoint_shards(*_PADDED)
        ]
        torch.manual_seed(1)
        scales = [torch.rand(shard.shape[0]) + 0.5 for shard in shards]

        _load(self.merged, shards)
        _load(qkvz, shards[:4])
        _load(ba, shards[4:])
        self._load_scales(self.merged, scales)
        self._load_scales(qkvz, scales[:4])
        self._load_scales(ba, scales[4:])

        ba_end = self.qkvz_width + self.ba_width
        for got, want in (
            (self.merged.weight[: self.qkvz_width], qkvz.weight),
            (self.merged.weight[self.qkvz_width : ba_end], ba.weight),
            (self.merged.weight_scale[: self.qkvz_width], qkvz.weight_scale),
            (self.merged.weight_scale[self.qkvz_width : ba_end], ba.weight_scale),
        ):
            torch.testing.assert_close(got.float(), want.float(), rtol=0, atol=0)

    def test_quantized_padding_rows_stay_zero(self):
        self.assertEqual(self.merged.weight.shape[0] % _GEMM_N_ALIGN, 0)
        used = self.qkvz_width + self.ba_width
        self.assertGreater(self.merged.weight.shape[0], used)
        self.assertEqual(self.merged.weight[used:].float().count_nonzero().item(), 0)


class TestGdnInProjStackedMapping(CustomTestCase):
    def test_merged_checkpoint_shards_map_onto_the_merged_param(self):
        model = _StubModel(f"{_PREFIX}.in_proj_qkvzba.weight")
        self.assertEqual(
            _gdn_input_proj_stacked_mapping(model),
            [
                ("in_proj_qkvzba.", "in_proj_qkv.", (0, 1, 2)),
                ("in_proj_qkvzba.", "in_proj_z.", 3),
                ("in_proj_qkvzba.", "in_proj_b.", 4),
                ("in_proj_qkvzba.", "in_proj_a.", 5),
            ],
        )

    def test_separate_params_keep_the_two_projection_mapping(self):
        model = _StubModel(
            f"{_PREFIX}.in_proj_qkvz.weight", f"{_PREFIX}.in_proj_ba.weight"
        )
        self.assertEqual(
            _gdn_input_proj_stacked_mapping(model),
            [
                ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
                ("in_proj_qkvz.", "in_proj_z.", 3),
                ("in_proj_ba.", "in_proj_b.", 0),
                ("in_proj_ba.", "in_proj_a.", 1),
            ],
        )


if __name__ == "__main__":
    unittest.main()
