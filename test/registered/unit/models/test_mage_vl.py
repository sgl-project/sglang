"""
Unit tests for Mage-VL pure helper logic: cu_seqlens windowing, HF checkpoint
key remapping, the interleaved RoPE rotate_half, and the norm-type / rotary
split assertions.

No server, no model weight loading, no GPU required.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang.srt.models.mage_vl import (
    MageVLForConditionalGeneration,
    VisionRotaryEmbedding,
    _get_norm,
    _mage_vl_rotate_half_interleaved,
    build_cu_seqlens,
    map_hf_name,
)


class TestBuildCuSeqlens(unittest.TestCase):
    def test_grid_within_fixed_t_is_not_split(self):
        cu_seqlens, max_seqlen = build_cu_seqlens(
            torch.tensor([[2, 3, 3]]), total_patches=18, fixed_t=4
        )
        self.assertEqual(cu_seqlens.tolist(), [0, 18])
        self.assertEqual(max_seqlen, 18)

    def test_grid_exceeding_fixed_t_splits_into_windows(self):
        # t=6, fixed_t=4 -> one full window of 4*2*2=16 plus a remainder of 2*2*2=8.
        cu_seqlens, max_seqlen = build_cu_seqlens(
            torch.tensor([[6, 2, 2]]), total_patches=24, fixed_t=4
        )
        self.assertEqual(cu_seqlens.tolist(), [0, 16, 24])
        self.assertEqual(max_seqlen, 16)

    def test_multiple_grids_offsets_accumulate(self):
        # The second grid's windows must continue from the first grid's running
        # total rather than restarting from 0.
        cu_seqlens, max_seqlen = build_cu_seqlens(
            torch.tensor([[2, 3, 3], [6, 2, 2]]), total_patches=18 + 24, fixed_t=4
        )
        self.assertEqual(cu_seqlens.tolist(), [0, 18, 34, 42])
        self.assertEqual(max_seqlen, 18)

    def test_raises_on_patch_count_mismatch(self):
        # If the caller passes a total_patches that disagrees with grid_thw, the
        # window split must fail loudly instead of silently returning a
        # truncated cu_seqlens (which would corrupt downstream attention).
        with self.assertRaises(ValueError):
            build_cu_seqlens(torch.tensor([[2, 3, 3]]), total_patches=99, fixed_t=4)

    def test_none_grid_thw_returns_single_span(self):
        cu_seqlens, max_seqlen = build_cu_seqlens(None, total_patches=10)
        self.assertEqual(cu_seqlens.tolist(), [0, 10])
        self.assertEqual(max_seqlen, 10)


class TestMapHfName(unittest.TestCase):
    def test_vision_self_attn_qkv_remapped(self):
        self.assertEqual(
            map_hf_name("model.visual.encoder.layers.0.self_attn.qkv.weight"),
            "visual.encoder.layers.0.attn.qkv_proj.weight",
        )

    def test_vision_self_attn_proj_remapped(self):
        self.assertEqual(
            map_hf_name("model.visual.encoder.layers.0.self_attn.proj.bias"),
            "visual.encoder.layers.0.attn.proj.bias",
        )

    def test_language_model_prefix_remapped(self):
        self.assertEqual(
            map_hf_name("model.language_model.layers.0.self_attn.q_proj.weight"),
            "model.layers.0.self_attn.q_proj.weight",
        )

    def test_unrelated_key_unchanged(self):
        self.assertEqual(map_hf_name("lm_head.weight"), "lm_head.weight")


class TestGetNorm(unittest.TestCase):
    def _cfg(self, **overrides):
        return SimpleNamespace(hidden_size=8, layer_norm_eps=1e-5, **overrides)

    def test_layer_norm_and_rms_norm_selection(self):
        cases = [
            ("layer_norm", nn.LayerNorm),
            ("rms_norm", nn.RMSNorm),
        ]
        for norm_type, expected_cls in cases:
            with self.subTest(norm_type=norm_type):
                norm = _get_norm(self._cfg(layer_norm_type=norm_type))
                self.assertIsInstance(norm, expected_cls)

    def test_missing_attr_defaults_to_layer_norm(self):
        norm = _get_norm(self._cfg())
        self.assertIsInstance(norm, nn.LayerNorm)

    def test_unknown_norm_type_raises(self):
        # Regression guard: a typo like "rmsnorm" used to silently fall through
        # to LayerNorm instead of raising, producing a model with the wrong
        # normalization and no error.
        with self.assertRaises(ValueError):
            _get_norm(self._cfg(layer_norm_type="rmsnorm"))


class TestRotateHalfInterleaved(unittest.TestCase):
    def test_interleaves_adjacent_pairs(self):
        # HF Mage-VL pairs adjacent dims (x0,x1), (x2,x3), ... and rotates each
        # pair to (-x1,x0), (-x3,x2). sglang's default rotate_half instead
        # pairs the two halves of the tensor, which is numerically different
        # and silently produces a wrong vision tower output.
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = _mage_vl_rotate_half_interleaved(x)
        self.assertTrue(torch.equal(out, torch.tensor([[-2.0, 1.0, -4.0, 3.0]])))


class TestVisionRotaryEmbeddingSplit(unittest.TestCase):
    def test_computes_4_6_6_split_of_half_head_dim(self):
        config = SimpleNamespace(hidden_size=64, num_attention_heads=1, rope_theta=10000.0)
        rope = VisionRotaryEmbedding(config)
        self.assertEqual(rope.head_dim, 64)
        self.assertEqual(rope.half, 32)
        self.assertEqual((rope.t_size, rope.h_size, rope.w_size), (8, 12, 12))

    def test_invalid_head_dim_raises(self):
        # half = head_dim // 2 = 24 is not divisible by 16, so the 4:6:6 unit
        # split is undefined for this head_dim.
        config = SimpleNamespace(hidden_size=48, num_attention_heads=1, rope_theta=10000.0)
        with self.assertRaises(AssertionError):
            VisionRotaryEmbedding(config)


class _FakeParam:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, param, loaded_weight, *args, **kwargs):
        self.loaded = (loaded_weight, args, kwargs)


class TestLoadWeightsStackedMerge(unittest.TestCase):
    def _make_model(self, named_parameters):
        model = object.__new__(MageVLForConditionalGeneration)
        model.named_parameters = lambda remove_duplicate=True: iter(
            named_parameters.items()
        )
        return model

    def test_text_qkv_and_gate_up_are_merged_by_shard(self):
        params = {
            "model.layers.0.self_attn.qkv_proj.weight": _FakeParam(),
            "model.layers.0.mlp.gate_up_proj.weight": _FakeParam(),
        }
        model = self._make_model(params)
        model.load_weights(
            [
                (
                    "model.language_model.layers.0.self_attn.q_proj.weight",
                    torch.tensor([1.0]),
                ),
                (
                    "model.language_model.layers.0.self_attn.k_proj.weight",
                    torch.tensor([2.0]),
                ),
                (
                    "model.language_model.layers.0.self_attn.v_proj.weight",
                    torch.tensor([3.0]),
                ),
                (
                    "model.language_model.layers.0.mlp.gate_proj.weight",
                    torch.tensor([4.0]),
                ),
                (
                    "model.language_model.layers.0.mlp.up_proj.weight",
                    torch.tensor([5.0]),
                ),
            ]
        )
        # Only the last-loaded shard's call is retained by the fake, but each
        # call must have used the qkv_proj/gate_up_proj target with its shard id.
        qkv_loaded = params["model.layers.0.self_attn.qkv_proj.weight"].loaded
        gate_up_loaded = params["model.layers.0.mlp.gate_up_proj.weight"].loaded
        self.assertEqual(qkv_loaded, (torch.tensor([3.0]), ("v",), {}))
        self.assertEqual(gate_up_loaded, (torch.tensor([5.0]), (1,), {}))

    def test_vision_qkv_bypasses_stacked_merge(self):
        # Vision qkv is already a single fused projection in the checkpoint, so
        # it must load directly (no shard id) rather than being routed through
        # the text q/k/v stacked-merge table.
        param = _FakeParam()
        model = self._make_model(
            {"visual.encoder.layers.0.attn.qkv_proj.weight": param}
        )
        model.load_weights(
            [
                (
                    "model.visual.encoder.layers.0.self_attn.qkv.weight",
                    torch.tensor([6.0]),
                )
            ]
        )
        self.assertEqual(param.loaded, (torch.tensor([6.0]), (), {}))

    def test_unmapped_weight_key_raises_keyerror(self):
        # A checkpoint key that maps to no known parameter must fail loudly
        # instead of being silently dropped.
        model = self._make_model({})
        with self.assertRaises(KeyError):
            model.load_weights(
                [("model.language_model.layers.0.some_new_param.weight", torch.tensor([1.0]))]
            )


if __name__ == "__main__":
    unittest.main()
