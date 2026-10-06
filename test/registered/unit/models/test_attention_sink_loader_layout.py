"""Attention sink checkpoints follow the constructed head partitions."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models import gpt_oss
from sglang.srt.models.gpt_oss import GptOssAttention, GptOssForCausalLM
from sglang.srt.models.mimo_v2 import MiMoV2Attention, MiMoV2ForCausalLM
from sglang.srt.models.mimo_v2_nextn import MiMoV2MTP
from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.parallel_groups import parallel_scope, publish, rank_size
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


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


def build_model(kind, dtype=torch.bfloat16, kv_heads=1, device="cuda"):
    model_type = {
        "gpt": GptOssForCausalLM,
        "mimo": MiMoV2ForCausalLM,
        "mtp": MiMoV2MTP,
    }[kind]
    model = model_type.__new__(model_type)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_hidden_layers=1,
        n_routed_experts=0,
        tie_word_embeddings=False,
        encoder_only=False,
    )
    model.quant_config = None
    model._is_multimodal = False
    with torch.device(device):
        if kind == "gpt":
            attention = GptOssAttention(
                64,
                8,
                kv_heads,
                head_dim=8,
                max_position_embeddings=16,
                layer_type="full_attention",
                params_dtype=dtype,
            )
        else:
            attention = MiMoV2Attention(
                64,
                8,
                kv_heads,
                head_dim=8,
                v_head_dim=16,
                max_position_embeddings=16,
                attention_sink_bias=True,
            )
    model.model = torch.nn.Module()
    if kind == "mtp":
        model.model.mtp_block = torch.nn.Module()
        model.model.mtp_block.self_attn = attention
        name = "model.mtp.layers.0.self_attn.attention_sink_bias"
    else:
        model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        model.model.layers[0].self_attn = attention
        name = "model.layers.0.self_attn." + (
            "sinks" if kind == "gpt" else "attention_sink_bias"
        )
    return model, attention, name


def load_sinks(
    model,
    attention,
    name,
    *,
    changed=False,
    offset=0,
    source_heads=8,
    cpu_padding=False,
):
    param = (
        attention.sinks
        if isinstance(attention, GptOssAttention)
        else attention.attention_sink_bias
    )
    source = (
        torch.arange(source_heads, device=param.device, dtype=torch.float32) + offset
    ) / 16
    start = rank_size(attention.qkv_proj)[0] * param.numel()
    expected = source[start : start + param.numel()].to(param.dtype)
    if cpu_padding and expected.numel() < param.numel():
        expected = torch.cat(
            (expected, expected.new_zeros(param.numel() - expected.numel()))
        )
    with loading_scope(changed), patch.object(gpt_oss, "_is_cpu", cpu_padding):
        if isinstance(model, GptOssForCausalLM):
            model._load_normal_weights(
                [(name, source)], is_nextn=False, weight_name_mapping=None
            )
        else:
            model.load_weights([(name, source)])
    torch.testing.assert_close(param, expected, rtol=0, atol=0)
    return param


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestAttentionSinkLoaderLayout(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        original = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, original)

    def check_loads(self, changed):
        for dp in (1, 2):
            for rank in range(4):
                for dtype in (torch.float32, torch.bfloat16):
                    reset_context()
                    publish(
                        ServerArgs(
                            model_path="dummy",
                            device="cuda",
                            tp_size=4,
                            attn_dp_size=dp,
                            attention_backend="trtllm_mha"
                            if dtype == torch.float32
                            else "flashinfer",
                        ),
                        role="test",
                        ranks=SpawnRanks(world_rank=rank),
                    )
                    torch.set_default_dtype(dtype)
                    for kind in ("gpt", "mimo", "mtp"):
                        for kv_heads in (1, 4):
                            model, attention, name = build_model(kind, dtype, kv_heads)
                            for offset in (0, 11):
                                load_sinks(
                                    model,
                                    attention,
                                    name,
                                    changed=changed,
                                    offset=offset,
                                )

    def test_native_model_loaders_in_the_construction_scope(self):
        self.check_loads(False)

    def test_native_model_loaders_after_scope_exit(self):
        self.check_loads(True)

    def test_gpt_cpu_checkpoint_tail_padding(self):
        # Exercise the CPU checkpoint branch with real CPU parameters/tensors.
        # This tests loading, without requiring a CPU attention kernel.
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
                model, attention, name = build_model("gpt", device="cpu")
                for changed in (False, True):
                    for source_heads in (7, 8, 9):
                        load_sinks(
                            model,
                            attention,
                            name,
                            changed=changed,
                            source_heads=source_heads,
                            cpu_padding=True,
                        )


if __name__ == "__main__":
    unittest.main()
