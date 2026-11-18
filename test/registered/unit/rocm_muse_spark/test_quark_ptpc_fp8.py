"""Quark PTPC-FP8 Muse Spark on ROCm: checkpoint contract and attention projection loaders (CPU).

The serialized export (amd/Muse-Spark-Open-816B-A42B-Quark-PTPC-FP8) stores FP8
e4m3 attention and expert weights with one fp32 scale per output row. These cases
check which Quark configs the model accepts, and how the fused q/k/v/o-gate and o
projections built with the real Quark W8A8 FP8 scheme place FP8 rows and their
scales on every attention-TP rank, KV-head replicas included. Loading the whole
ROCm model is covered by ``test_quark_ptpc_fp8_model_load.py`` (it needs the
ROCm image's kernel libraries).
"""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.srt.rocm_muse_spark.layers import QKVOGateParallelLinear
from sglang.srt.rocm_muse_spark.quark_ptpc_fp8 import (
    check_rope_rows,
    excluded_expert_layers,
    ptpc_fp8_rejection,
)
from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels.muse_spark_fp8_moe import FP8, quantize_rows
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PACKED_MODULES = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}

# Quantizers as the release config.json spells them (Quark 0.12.post1).
_WEIGHT = {
    "dtype": "fp8_e4m3",
    "is_dynamic": False,
    "qscheme": "per_channel",
    "ch_axis": 0,
    "group_size": None,
    "block_size": None,
    "symmetric": True,
    "round_method": "half_even",
    "scale_type": "float",
    "zero_point_type": "int32",
    "scale_format": None,
    "scale_calculation_mode": None,
    "mx_element_dtype": None,
    "observer_cls": "PerChannelMinMaxObserver",
    "is_scale_quant": False,
    "enable_buffer_reuse": False,
    "max_input_numel": 4194304,
}
_INPUT = {**_WEIGHT, "is_dynamic": True, "ch_axis": 1}
NUM_LAYERS = 2
NUM_EXPERTS = 8
HEADS, KV_HEADS, HEAD_DIM, HIDDEN = 8, 2, 16, 64
MOE_HIDDEN, MOE_INTER = 32, 32
ATTN = "model.language_model.layers.{}.self_attn"
EXPERT = "model.language_model.layers.{}.mlp.experts.{}.{}"


def release_quant_config(num_layers: int = 62) -> dict:
    """The release's quantization_config: global PTPC, BF16 router / pooler / lm_head."""
    exclude = ["lm_head"]
    for layer in range(num_layers):
        exclude += [
            f"model.language_model.layers.{layer}.mlp.{m}"
            for m in ("gate", "post_expert_proj", "pre_expert_proj")
        ]
    exclude += ["model.audio_tower.proj", "model.vision_tower.encoder.layers.0.mlp.fc1"]
    return {
        "global_quant_config": {
            "input_tensors": dict(_INPUT),
            "output_tensors": None,
            "weight": dict(_WEIGHT),
            "bias": None,
            "target_device": None,
        },
        "exclude": exclude,
        "algo_config": None,
        "softmax_quant_spec": None,
        "quant_method": "quark",
        "layer_type_quant_config": {},
        "layer_quant_config": {},
        "kv_cache_quant_config": {},
        "kv_cache_post_rope": False,
        "quant_mode": "eager_mode",
        "version": "0.12.post1+f7d8cef",
        "export": {
            "kv_cache_group": [],
            "min_kv_scale": 0.0,
            "pack_method": "reorder",
            "weight_format": "real_quantized",
            "weight_merge_groups": None,
        },
    }


def quark(config: dict) -> QuarkConfig:
    config = copy.deepcopy(config)
    config["packed_modules_mapping"] = dict(PACKED_MODULES)
    return QuarkConfig(quant_config=config, is_prequantized=True)


def expert_leaves(layer: int, experts) -> list[str]:
    return [
        EXPERT.format(layer, e, p)
        for e in experts
        for p in ("gate_proj", "up_proj", "down_proj")
    ]


class TestPtpcFp8Contract(CustomTestCase):
    """Only the release's PTPC scheme is accepted; anything else is refused by name."""

    def assertRejected(self, config: dict, fragment: str):
        reason = ptpc_fp8_rejection(quark(config), 256)
        self.assertIsNotNone(reason)
        self.assertIn(fragment, reason)

    def test_release_config_is_accepted(self):
        self.assertIsNone(ptpc_fp8_rejection(quark(release_quant_config()), 256))

    def test_quantizer_contract_rejections(self):
        def variant(role, **fields):
            config = release_quant_config()
            config["global_quant_config"][role].update(fields)
            return config

        cases = [
            (variant("input_tensors", ch_axis=0), "input_tensors.ch_axis"),
            (variant("weight", ch_axis=1), "weight.ch_axis"),
            (variant("weight", is_dynamic=True), "weight.is_dynamic"),
            (variant("input_tensors", is_dynamic=False), "input_tensors.is_dynamic"),
            (variant("weight", symmetric=False), "weight.symmetric"),
            (variant("input_tensors", qscheme="per_tensor"), "input_tensors.qscheme"),
            (variant("weight", dtype="fp4"), "weight.dtype"),
            (variant("weight", group_size=128), "weight.group_size"),
            (variant("weight", block_size=[128, 128]), "weight.block_size"),
            (variant("input_tensors", scale_format="e8m0"), "scale_format"),
            (variant("weight", mx_element_dtype="fp8_e4m3"), "mx_element_dtype"),
            (variant("input_tensors", scale_type="pof2"), "scale_type"),
        ]
        missing_axis = release_quant_config()
        del missing_axis["global_quant_config"]["input_tensors"]["ch_axis"]
        cases.append((missing_axis, "input_tensors.ch_axis"))
        for config, fragment in cases:
            with self.subTest(fragment=fragment):
                self.assertRejected(config, fragment)

    def test_overrides_and_extra_quantization_are_rejected(self):
        layer_type = release_quant_config()
        layer_type["layer_type_quant_config"] = {"Linear": {"weight": {"dtype": "fp4"}}}
        layer = release_quant_config()
        layer["layer_quant_config"] = {ATTN.format(0) + ".q_proj": {}}
        kv = release_quant_config()
        kv["kv_cache_quant_config"] = {"*k_proj": dict(_WEIGHT)}
        output = release_quant_config()
        output["global_quant_config"]["output_tensors"] = dict(_WEIGHT)
        fake = release_quant_config()
        fake["export"]["weight_format"] = "fake_quantized"
        for config, fragment in (
            (layer_type, "layer_type_quant_config"),
            (layer, "layer_quant_config"),
            (kv, "kv_cache_quant_config"),
            (output, "output_tensors"),
            (fake, "weight_format"),
        ):
            with self.subTest(fragment=fragment):
                self.assertRejected(config, fragment)

    def test_exclusions(self):
        """Whole-layer expert exclusion is served in BF16; partial or attention exclusions are refused."""
        whole = release_quant_config()
        whole["exclude"] += expert_leaves(0, range(256))
        whole["exclude"].append("model.language_model.layers.61.mlp.experts")
        self.assertIsNone(ptpc_fp8_rejection(quark(whole), 256))
        self.assertEqual(excluded_expert_layers(quark(whole), 256), {0, 61})

        partial = release_quant_config()
        partial["exclude"] += expert_leaves(3, range(255))
        self.assertRejected(partial, "layer 3 excludes")
        one_projection = release_quant_config()
        one_projection["exclude"] += [
            EXPERT.format(3, e, "up_proj") for e in range(256)
        ]
        self.assertRejected(one_projection, "layer 3 excludes")
        for entry in (
            ATTN.format(4) + ".gate_proj",  # the fused q/k/v/o-gate cannot split
            ATTN.format(4) + ".o_proj",
            "model.language_model.layers.4",
            "model.language_model",
            "re:.*self_attn.*",
            "model.language_model.layers.*.mlp.experts",
        ):
            with self.subTest(entry=entry):
                config = release_quant_config()
                config["exclude"].append(entry)
                self.assertIsNotNone(ptpc_fp8_rejection(quark(config), 256))

    def test_malformed_exclude_containers_are_rejected(self):
        for value in ("lm_head", {"lm_head": True}):
            with self.subTest(exclude=type(value).__name__):
                config = release_quant_config()
                config["exclude"] = value
                self.assertRejected(config, "exclude is a")

    def test_other_formats_keep_their_branches(self):
        mxfp4 = release_quant_config()
        for role in ("weight", "input_tensors"):
            mxfp4["global_quant_config"][role].update(
                dtype="fp4",
                qscheme="per_group",
                group_size=32,
                scale_format="e8m0",
                ch_axis=-1,
            )
        self.assertIsNotNone(ptpc_fp8_rejection(quark(mxfp4), 256))
        nvfp4 = SimpleNamespace(get_name=lambda: "modelopt_fp4")
        self.assertEqual(ptpc_fp8_rejection(nvfp4, 256), "not a Quark config")
        self.assertIsNotNone(ptpc_fp8_rejection(None, 256))


class TestAttentionProjectionLoaders(CustomTestCase):
    """The FP8 weight and per-row scale the Quark scheme registers go through the q/k/v/g-aware loader."""

    def layers(self, tp_size: int, rank: int, kv_heads: int):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=tp_size),
            role="test",
            ranks=SpawnRanks(world_rank=rank),
        )
        # QuarkW8A8Fp8 needs FP8-capable hardware; this is the device boundary.
        with patch(
            "sglang.srt.layers.quantization.quark.quark.get_device_capability",
            return_value=(9, 5),
        ):
            config = quark(release_quant_config(1))
            qkv = QKVOGateParallelLinear(
                hidden_size=HIDDEN,
                head_size=HEAD_DIM,
                total_num_heads=HEADS,
                total_num_kv_heads=kv_heads,
                prefix=ATTN.format(0) + ".qkv_proj",
                quant_config=config,
            )
            o_proj = RowParallelLinear(
                input_size=HEADS * HEAD_DIM,
                output_size=HIDDEN,
                bias=False,
                quant_config=config,
                prefix=ATTN.format(0) + ".o_proj",
                parallel_group="attn_tp",
            )
        return qkv, o_proj

    def test_rows_and_scales_per_rank_with_kv_replicas(self):
        gen = torch.Generator().manual_seed(0)
        for tp_size, kv_heads in ((2, 2), (4, 2), (8, 1)):
            shards = {
                name: quantize_rows(torch.randn(rows, HIDDEN, generator=gen))
                for name, rows in (
                    ("q", HEADS * HEAD_DIM),
                    ("k", kv_heads * HEAD_DIM),
                    ("v", kv_heads * HEAD_DIM),
                    ("g", HEADS * HEAD_DIM),
                )
            }
            o_codes, o_scale = quantize_rows(
                torch.randn(HIDDEN, HEADS * HEAD_DIM, generator=gen)
            )
            replicas = max(1, tp_size // kv_heads)
            q_rows = HEADS * HEAD_DIM // tp_size
            kv_rows = max(1, kv_heads // tp_size) * HEAD_DIM
            for rank in range(tp_size):
                with self.subTest(tp=tp_size, rank=rank):
                    qkv, o_proj = self.layers(tp_size, rank, kv_heads)
                    self.assertEqual(qkv.weight.dtype, FP8)
                    for param in (qkv.weight, qkv.weight_scale):
                        self.assertEqual(
                            param.weight_loader.__func__,
                            QKVOGateParallelLinear.weight_loader,
                        )
                    for name, (codes, scale) in shards.items():
                        qkv.weight.weight_loader(qkv.weight, codes, name)
                        qkv.weight_scale.weight_loader(qkv.weight_scale, scale, name)
                    q = slice(rank * q_rows, (rank + 1) * q_rows)
                    kv_start = (rank // replicas) * kv_rows
                    kv = slice(kv_start, kv_start + kv_rows)
                    pick = {"q": q, "k": kv, "v": kv, "g": q}
                    expected_codes = torch.cat([shards[n][0][pick[n]] for n in pick])
                    expected_scale = torch.cat([shards[n][1][pick[n]] for n in pick])
                    self.assertTrue(
                        torch.equal(
                            qkv.weight.view(torch.uint8),
                            expected_codes.view(torch.uint8),
                        )
                    )
                    self.assertTrue(torch.equal(qkv.weight_scale, expected_scale))
                    o_proj.weight.weight_loader(o_proj.weight, o_codes)
                    o_proj.weight_scale.weight_loader(o_proj.weight_scale, o_scale)
                    width = HEADS * HEAD_DIM // tp_size
                    self.assertTrue(
                        torch.equal(
                            o_proj.weight.view(torch.uint8),
                            o_codes[:, rank * width : (rank + 1) * width].view(
                                torch.uint8
                            ),
                        )
                    )
                    self.assertTrue(torch.equal(o_proj.weight_scale, o_scale))

    def test_wrong_row_counts_raise(self):
        qkv, _ = self.layers(4, 1, KV_HEADS)
        codes, scale = quantize_rows(torch.randn(KV_HEADS * HEAD_DIM, HIDDEN))
        for param, value in (
            (qkv.weight_scale, scale[:-1]),
            (qkv.weight_scale, scale[:1].reshape(())),
            (qkv.weight, codes[:-1]),
        ):
            with self.subTest(shape=tuple(value.shape)):
                with self.assertRaisesRegex(ValueError, "QKV gate shard v"):
                    param.weight_loader(param, value, "v")

    def test_rope_rows_need_one_per_head_channel(self):
        rows = HEADS * HEAD_DIM
        check_rope_rows(torch.ones(rows), HEADS, HEAD_DIM, "q_proj.weight_scale")
        check_rope_rows(torch.ones(rows, 4), HEADS, HEAD_DIM, "q_proj.weight")
        for value, name in (
            (torch.ones(()), "q_proj.weight_scale"),
            (torch.ones(rows - 1), "q_proj.weight_scale"),
            (torch.ones(rows, 1), "q_proj.weight_scale"),
            (torch.ones(rows), "q_proj.weight"),
        ):
            with self.subTest(shape=tuple(value.shape), name=name):
                with self.assertRaisesRegex(ValueError, "expected"):
                    check_rope_rows(value, HEADS, HEAD_DIM, name)


if __name__ == "__main__":
    unittest.main()
