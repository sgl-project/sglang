"""
CPU unit tests for Qwen3-Next weight loader v2 checkpoint name mapping and dispatch.

Regression coverage for https://github.com/sgl-project/sglang/issues/31051 PR6.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

import torch
from torch.nn import Parameter

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.model_loader.auto_loader import (
    QWEN3_NEXT_GDN_STACKED_MAPPING,
    STANDARD_QKV_MAPPING,
    ExpertParamsDispatch,
)
from sglang.srt.models.qwen3_next import (
    iter_qwen3_next_checkpoint_weights,
    remap_qwen3_next_checkpoint_name,
)


class TestQwen3NextCheckpointRemap(unittest.TestCase):
    def test_self_attn_strip_and_qkv_fuse(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        remapped = remap_qwen3_next_checkpoint_name(
            name,
            enable_shared_expert_fusion=False,
            num_experts=8,
        )
        self.assertEqual(remapped, "model.layers.0.q_proj.weight")

    def test_shared_expert_fusion_remap(self):
        name = "model.layers.3.mlp.shared_expert.gate_proj.weight"
        remapped = remap_qwen3_next_checkpoint_name(
            name,
            enable_shared_expert_fusion=True,
            num_experts=8,
        )
        self.assertEqual(remapped, "model.layers.3.mlp.experts.8.gate_proj.weight")

    def test_modelopt_kv_scale_remap(self):
        name = "model.layers.1.self_attn.k_proj.k_scale"
        remapped = remap_qwen3_next_checkpoint_name(
            name,
            enable_shared_expert_fusion=False,
            num_experts=4,
        )
        self.assertEqual(remapped, "model.layers.1.attn.k_scale")

    def test_mtp_filter_and_remap(self):
        weights = [
            ("mtp.fc.weight", torch.zeros(1)),
            ("model.embed_tokens.weight", torch.zeros(1)),
            ("mtp.layers.0.mlp.gate_proj.weight", torch.zeros(1)),
        ]
        out = list(
            iter_qwen3_next_checkpoint_weights(
                weights,
                is_mtp=True,
                enable_shared_expert_fusion=False,
                num_experts=4,
            )
        )
        names = [n for n, _ in out]
        self.assertEqual(names, ["fc.weight", "model.layers.0.mlp.gate_proj.weight"])

    def test_non_mtp_skips_mtp_tensors(self):
        weights = [
            ("mtp.fc.weight", torch.zeros(1)),
            ("model.norm.weight", torch.zeros(1)),
        ]
        out = list(
            iter_qwen3_next_checkpoint_weights(
                weights,
                is_mtp=False,
                enable_shared_expert_fusion=False,
                num_experts=4,
            )
        )
        self.assertEqual([n for n, _ in out], ["model.norm.weight"])


class TestQwen3NextDispatch(unittest.TestCase):
    def test_gdn_in_proj_qkvz_mapping(self):
        calls = []

        def loader(param, tensor, shard_id):
            calls.append((shard_id, tensor.shape))

        param = MagicMock(spec=Parameter)
        param.weight_loader = loader
        params = {"linear_attn.in_proj_qkvz.weight": param}

        tensor = torch.randn(8, 4)
        target = QWEN3_NEXT_GDN_STACKED_MAPPING.try_load(
            "linear_attn.in_proj_qkv.weight", tensor, params
        )
        self.assertEqual(target, "linear_attn.in_proj_qkvz.weight")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], (0, 1, 2))

    def test_expert_params_dispatch_gate_proj(self):
        calls = []

        def loader(param, tensor, qualname, shard_id=None, expert_id=None):
            calls.append((qualname, shard_id, expert_id))

        param = MagicMock(spec=Parameter)
        param.weight_loader = loader
        params = {"experts.w13_weight": param}

        mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=2,
        )
        dispatch = ExpertParamsDispatch.from_fused_moe_mapping(mapping)
        ckpt_name = "experts.1.gate_proj.weight"
        target = dispatch.try_load(ckpt_name, torch.randn(4, 8), params)
        self.assertEqual(target, "experts.w13_weight")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1:], ("w1", 1))

    def test_attention_qkv_mapping(self):
        calls = []

        def loader(param, tensor, shard_id):
            calls.append(shard_id)

        param = MagicMock(spec=Parameter)
        param.weight_loader = loader
        params = {"qkv_proj.weight": param}

        target = STANDARD_QKV_MAPPING.try_load(
            "k_proj.weight", torch.randn(2, 2), params
        )
        self.assertEqual(target, "qkv_proj.weight")
        self.assertEqual(calls, ["k"])


if __name__ == "__main__":
    unittest.main()
