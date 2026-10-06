"""Mixed replicated Q/KV and sharded gate checkpoints follow attention layout."""

import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import torch

from sglang.srt.configs.dots3 import Dots3Config
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.lora.layers import BaseLayerWithLoRA, unwrap_lora_layer
from sglang.srt.models.dots3_common.modeling import (
    Dots3AttentionMLA,
    Dots3LanguageModelForCausalLM,
)
from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.parallel_groups import parallel_scope, publish, rank_size
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


def loading_scope(changed):
    if not changed:
        return nullcontext()
    return parallel_scope(
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        attn_tp_size=1,
        attn_tp_rank=0,
        attn_tp_group=None,
        attn_dp_size=1,
        attn_dp_rank=0,
        attn_cp_size=1,
        attn_cp_rank=0,
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_ep_group=None,
        moe_dp_size=1,
        moe_dp_rank=0,
    )


def build_model(
    gate_type="headwise", layer_type="full_attention", fp8=False, nextn=False
):
    config = Dots3Config(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        max_position_embeddings=32,
        layer_types=[layer_type],
        attention_gate_type=gate_type,
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        num_attention_heads=8,
        num_key_value_heads=8,
        v_head_dim=128,
        swa_attention_gate_type=gate_type,
        swa_q_lora_rank=128,
        swa_kv_lora_rank=128,
        swa_qk_nope_head_dim=128,
        swa_qk_rope_head_dim=64,
        swa_num_attention_heads=8,
        swa_num_key_value_heads=8,
        swa_v_head_dim=128,
        n_routed_experts=0,
        n_shared_experts=0,
        num_nextn_predict_layers=1 if nextn else 0,
    )
    quant = (
        Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
        if fp8
        else None
    )
    model = Dots3LanguageModelForCausalLM.__new__(Dots3LanguageModelForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = config
    model.quant_config = quant
    model.num_fused_shared_experts = 0
    with torch.device("cuda"):
        attention = Dots3AttentionMLA(config, quant, layer_id=0)
    model.model = torch.nn.Module()
    model.model.start_layer = 0
    model.model.end_layer = 1
    if nextn:
        model.model.heads = torch.nn.ModuleList([torch.nn.Module()])
        model.model.heads[0].decoder = torch.nn.Module()
        model.model.heads[0].decoder.self_attn = attention
    else:
        model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        model.model.layers[0].self_attn = attention
    if fp8:
        assert isinstance(
            attention.fused_qkv_a_g_proj_with_mqa.quant_method, Fp8LinearMethod
        )
    return model, attention


def checkpoint_parts(attention, offset):
    module = attention.fused_qkv_a_g_proj_with_mqa
    widths = (
        attention.q_lora_rank,
        attention.kv_lora_rank + attention.qk_rope_head_dim,
        attention.num_heads
        * (1 if attention.attention_gate_type == "headwise" else attention.v_head_dim),
    )
    parts = {}
    for index, (name, width) in enumerate(
        zip(("q_a_proj", "kv_a_proj_with_mqa", "g_proj"), widths)
    ):
        weight = (
            (
                (
                    torch.arange(width * 128, device=module.weight.device)
                    + offset
                    + index * 11
                )
                % 31
                - 15
            )
            .reshape(width, 128)
            .to(module.weight.dtype)
        )
        parts[name + ".weight"] = weight
        if hasattr(module, "weight_scale_inv"):
            parts[name + ".weight_scale_inv"] = (
                (
                    torch.arange((width + 127) // 128, device=weight.device)
                    + offset
                    + index
                )
                % 3
                + 1
            ).float().reshape(-1, 1) / 128
    return parts


def load_fused(
    model, attention, *, changed=False, offset=0, reverse=False, nextn=False
):
    module = attention.fused_qkv_a_g_proj_with_mqa
    parts = checkpoint_parts(attention, offset)
    names = list(parts)
    if reverse:
        names.reverse()
    weights = [("model.layers.0.self_attn." + name, parts[name]) for name in names]
    # Keep this fixture focused on checkpoint fusion. The separate model hook
    # binds kv_b views and requantizes unrelated attention/MLP parameters.
    with loading_scope(changed), patch.object(model, "post_load_weights") as post:
        model.load_weights(weights, is_nextn=nextn)
        post.assert_called_once()
    owner = unwrap_lora_layer(attention.q_b_proj)
    gate = parts["g_proj.weight"].chunk(rank_size(owner)[1], 0)[rank_size(owner)[0]]

    def pad(t, target):
        return torch.cat((t, t.new_zeros(target - t.shape[0], t.shape[1])), 0)

    expected = torch.cat(
        (
            parts["q_a_proj.weight"],
            pad(
                parts["kv_a_proj_with_mqa.weight"],
                attention.kv_lora_rank + attention.qk_rope_head_dim_padded,
            ),
            pad(gate, attention.g_proj_local_dim_padded),
        ),
        0,
    )
    if module.weight.dtype == torch.float8_e4m3fn:
        torch.testing.assert_close(
            module.weight.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
        )
        start = rank_size(owner)[0] * gate.shape[0] // 128
        end = ((rank_size(owner)[0] + 1) * gate.shape[0] + 127) // 128
        expected_scale = torch.cat(
            (
                parts["q_a_proj.weight_scale_inv"],
                parts["kv_a_proj_with_mqa.weight_scale_inv"],
                parts["g_proj.weight_scale_inv"][start:end],
            ),
            0,
        )
        torch.testing.assert_close(
            module.weight_scale_inv, expected_scale, rtol=0, atol=0
        )
    else:
        torch.testing.assert_close(module.weight, expected, rtol=0, atol=0)
    return module


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestDotsFusedGateLoaderLayout(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        original = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, original)
        torch.set_default_dtype(torch.bfloat16)

    def check_loads(self, changed, nextn=False, wrapped=False):
        for dp in (1, 2):
            for rank in range(4):
                reset_context()
                publish(
                    ServerArgs(
                        model_path="dummy", device="cuda", tp_size=4, attn_dp_size=dp
                    ),
                    role="test",
                    ranks=SpawnRanks(world_rank=rank),
                )
                for gate in ("headwise", "elementwise"):
                    for layer in ("full_attention", "sliding_attention"):
                        for fp8 in (False, True):
                            model, attention = build_model(gate, layer, fp8, nextn)
                            if wrapped:
                                attention.q_b_proj = BaseLayerWithLoRA(
                                    attention.q_b_proj, Mock()
                                )
                            for offset in (0, 11):
                                for reverse in (False, True):
                                    with self.subTest(
                                        dp=dp,
                                        rank=rank,
                                        gate=gate,
                                        layer=layer,
                                        fp8=fp8,
                                        offset=offset,
                                        reverse=reverse,
                                    ):
                                        load_fused(
                                            model,
                                            attention,
                                            changed=changed,
                                            offset=offset,
                                            reverse=reverse,
                                            nextn=nextn,
                                        )

    def test_native_fused_parameters_in_construction_scope(self):
        self.check_loads(False)

    def test_native_fused_parameters_after_scope_exit(self):
        self.check_loads(True)

    def test_mtp_checkpoint_mapping_after_scope_exit(self):
        self.check_loads(True, nextn=True)

    def test_wrapped_gate_owner_after_scope_exit(self):
        self.check_loads(True, wrapped=True)

    def test_gate_keeps_rank_when_width_is_unchanged(self):
        publish(
            ServerArgs(model_path="dummy", device="cuda", tp_size=4, attn_dp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )
        for gate in ("headwise", "elementwise"):
            for fp8 in (False, True):
                model, attention = build_model(gate, fp8=fp8)
                with parallel_scope(
                    tp_rank=0, attn_tp_rank=0, attn_dp_rank=0, moe_tp_rank=0
                ):
                    load_fused(model, attention)

    def test_incomplete_checkpoint_still_reports_unresolved_parts(self):
        publish(
            ServerArgs(model_path="dummy", device="cuda", tp_size=4),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )
        model, attention = build_model()
        parts = checkpoint_parts(attention, 0)
        for changed in (False, True):
            with (
                loading_scope(changed),
                self.assertRaisesRegex(ValueError, "Unresolved fused q/kv/g"),
            ):
                model.load_weights(
                    [
                        (
                            "model.layers.0.self_attn.q_a_proj.weight",
                            parts["q_a_proj.weight"],
                        )
                    ]
                )


if __name__ == "__main__":
    unittest.main()
