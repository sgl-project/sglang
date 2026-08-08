"""GDN fused split/reshape/cat kernel and in_proj_qkvzba merge gating on AMD.

Merging in_proj_ba into in_proj_qkvz feeds the kernel column slices of one wider
projection rather than two dense tensors, and is only valid when all four checkpoint
shards resolve to one quantization scheme. Quark enforces that because the merged name
is listed in packed_modules_mapping; create_qkvzba_proj turns a mismatch into the
separate-projection path. The merge is default-off behind SGLANG_GDN_FUSE_QKVZBA, so a
plain green CI run never executes it.
"""

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

import copy
import unittest

import torch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.test.test_utils import CustomTestCase

_DEVICE = "cuda"
_HEAD_K_DIM = _HEAD_V_DIM = 128

_LAYER = "model.layers.0.linear_attn"
_MERGED_PREFIX = f"{_LAYER}.in_proj_qkvzba"
_SHARDS = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]
_MAPPING = {"in_proj_qkvzba": _SHARDS}

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
_OTHER_DTYPE = copy.deepcopy(_FP8)
_OTHER_DTYPE["weight"]["dtype"] = "fp4"


def _reference_split_reshape_cat(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v):
    k_dim = num_heads_qk * _HEAD_K_DIM
    v_dim = num_heads_v * _HEAD_V_DIM
    query, key, value, z = mixed_qkvz.split([k_dim, k_dim, v_dim, v_dim], dim=-1)
    b, a = mixed_ba.split([num_heads_v, num_heads_v], dim=-1)
    z = z.reshape(z.size(0), -1, _HEAD_V_DIM)
    mixed_qkv = torch.cat(
        [x.reshape(x.shape[0], -1) for x in (query, key, value)], dim=-1
    )
    return mixed_qkv, z, b.contiguous(), a.contiguous()


def _build_inputs(seq_len, num_heads_qk, num_heads_v, merged):
    """The qkvz/ba pair, either as separate tensors or as slices of one projection."""
    qkvz_width = 2 * num_heads_qk * _HEAD_K_DIM + 2 * num_heads_v * _HEAD_V_DIM
    ba_width = 2 * num_heads_v
    torch.manual_seed(0)
    if not merged:
        return (
            torch.randn(seq_len, qkvz_width, dtype=torch.bfloat16, device=_DEVICE),
            torch.randn(seq_len, ba_width, dtype=torch.bfloat16, device=_DEVICE),
        )
    # 64 extra columns stand in for the alignment padding the merged layer carries.
    projected = torch.randn(
        seq_len, qkvz_width + ba_width + 64, dtype=torch.bfloat16, device=_DEVICE
    )
    return projected[:, :qkvz_width], projected[:, qkvz_width : qkvz_width + ba_width]


def _quant_method(layer_quant_config, exclude=(), mapping=_MAPPING):
    quant_config = QuarkConfig.from_config(
        {
            "quant_method": "quark",
            "export": {"kv_cache_group": [], "pack_method": "reorder"},
            "global_quant_config": _FP8,
            "layer_quant_config": layer_quant_config,
            "layer_type_quant_config": {},
            "exclude": list(exclude),
            "packed_modules_mapping": mapping,
        }
    )
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    return quant_config.get_quant_method(layer, _MERGED_PREFIX)


class TestGdnFusedProjAmd(CustomTestCase):
    def test_split_reshape_cat_contiguous(self):
        from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
            fused_qkvzba_split_reshape_cat_contiguous,
        )

        for merged in (False, True):
            for seq_len in (1, 4, 37, 256):
                for num_heads_qk, num_heads_v in ((16, 32), (8, 32), (4, 16)):
                    with self.subTest(
                        merged=merged,
                        seq_len=seq_len,
                        num_heads_qk=num_heads_qk,
                        num_heads_v=num_heads_v,
                    ):
                        mixed_qkvz, mixed_ba = _build_inputs(
                            seq_len, num_heads_qk, num_heads_v, merged
                        )
                        self.assertEqual(
                            mixed_qkvz.stride(0) != mixed_qkvz.shape[1], merged
                        )

                        out = fused_qkvzba_split_reshape_cat_contiguous(
                            mixed_qkvz,
                            mixed_ba,
                            num_heads_qk,
                            num_heads_v,
                            _HEAD_K_DIM,
                            _HEAD_V_DIM,
                        )
                        ref = _reference_split_reshape_cat(
                            mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v
                        )
                        for got, want in zip(out, ref):
                            torch.testing.assert_close(got, want, rtol=0, atol=0)


class TestGdnInProjMergeGate(CustomTestCase):
    def test_uniform_scheme_is_mergeable(self):
        method = _quant_method({"*linear_attn*": _FP8})
        self.assertEqual(type(method).__name__, "QuarkLinearMethod")

    def test_uniformly_excluded_is_mergeable_unquantized(self):
        method = _quant_method(
            {"*linear_attn*": _FP8},
            exclude=[f"{_LAYER}.{shard}" for shard in _SHARDS],
        )
        self.assertIsInstance(method, UnquantizedLinearMethod)

    def test_ba_left_unquantized_is_rejected(self):
        # What amd/Qwen3.5-397B-A17B-MXFP4-AttnFP8 ships: in_proj_a and in_proj_b are
        # excluded while qkvz is quantized, so that checkpoint keeps them separate.
        with self.assertRaises(ValueError):
            _quant_method(
                {"*linear_attn*": _FP8},
                exclude=[f"{_LAYER}.in_proj_b", f"{_LAYER}.in_proj_a"],
            )

    def test_ba_with_another_dtype_is_rejected(self):
        with self.assertRaises(ValueError):
            _quant_method(
                {
                    "*in_proj_b": _OTHER_DTYPE,
                    "*in_proj_a": _OTHER_DTYPE,
                    "*linear_attn*": _FP8,
                }
            )

    def test_detection_needs_the_merged_name_in_packed_modules_mapping(self):
        self.assertEqual(
            Qwen3_5ForCausalLM.packed_modules_mapping.get("in_proj_qkvzba"), _SHARDS
        )
        # Without the entry the same mixed checkpoint resolves to a quantized merged
        # layer over unquantized b/a weights instead of raising.
        method = _quant_method(
            {"*linear_attn*": _FP8},
            exclude=[f"{_LAYER}.in_proj_b", f"{_LAYER}.in_proj_a"],
            mapping={},
        )
        self.assertEqual(type(method).__name__, "QuarkLinearMethod")


if __name__ == "__main__":
    unittest.main()
