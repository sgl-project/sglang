"""FOPE coefficient loading follows the destination attention partition."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from sglang.srt.models.interns1pro import (
    InternS1ProForConditionalGeneration,
    InternS1ProTextAttention,
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


def build_model(kv_heads, head_dim=16):
    model = InternS1ProForConditionalGeneration.__new__(
        InternS1ProForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_experts=0)
    with torch.device("cuda"):
        attention = InternS1ProTextAttention(
            128,
            8,
            kv_heads,
            head_dim=head_dim,
            max_position_embeddings=32,
            rope_scaling={
                "rope_type": "default",
                "fope_sep_head": True,
                "num_inv_freq": 3,
            },
            config=SimpleNamespace(),
        )
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    model.model.layers[0].self_attn = attention
    with torch.device("cuda"):
        second = InternS1ProTextAttention(
            128,
            8,
            kv_heads,
            layer_id=1,
            head_dim=head_dim,
            max_position_embeddings=32,
            rope_scaling={
                "rope_type": "default",
                "fope_sep_head": True,
                "num_inv_freq": 3,
            },
            config=SimpleNamespace(),
        )
    model.model.layers[1].self_attn = second
    assert attention.rotary_emb is second.rotary_emb
    return model, attention


def load_coefficients(model, attention, *, changed=False, offset=0, top_level=False):
    weights = []
    for index, coefficient in enumerate(("cos_coef", "sin_coef")):
        param = getattr(attention.rotary_emb, coefficient)
        shape = (attention.total_num_kv_heads, *param.shape[1:])
        full = (
            (
                torch.arange(shape[0] * shape[1] * shape[2], device=param.device)
                + offset
                + index * 11
            )
            % 37
            - 18
        ).float().reshape(shape) / 64
        owner = attention.qkv_proj
        first = (
            rank_size(owner)[0]
            // max(1, rank_size(owner)[1] // shape[0])
            * param.shape[0]
        )
        expected = full[first : first + param.shape[0]]
        name = "model.rotary_emb." + coefficient
        if top_level:
            weights.append(("model.language_model.rotary_emb." + coefficient, full))
        else:
            with loading_scope(changed):
                model._load_fope_weights(name, full, dict(model.named_parameters()))
            torch.testing.assert_close(param, expected, rtol=0, atol=0)
        weights.append((param, expected))
    if top_level:
        sources = [item for item in weights if isinstance(item[0], str)]
        with loading_scope(changed):
            model.load_weights(sources)
        for param, expected in weights:
            if not isinstance(param, str):
                torch.testing.assert_close(param, expected, rtol=0, atol=0)
    return attention.rotary_emb


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestInternS1ProFopeLoaderLayout(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        original = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, original)
        torch.set_default_dtype(torch.bfloat16)

    def check_loads(self, changed, top_level=False):
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
                for kv_heads in (1, 2, 8):
                    for head_dim in (8, 16):
                        model, attention = build_model(kv_heads, head_dim)
                        for offset in (0, 13):
                            with self.subTest(
                                dp=dp,
                                rank=rank,
                                kv_heads=kv_heads,
                                head_dim=head_dim,
                                offset=offset,
                            ):
                                load_coefficients(
                                    model,
                                    attention,
                                    changed=changed,
                                    offset=offset,
                                    top_level=top_level,
                                )

    def test_native_attention_coefficients_in_construction_scope(self):
        self.check_loads(False)

    def test_native_attention_coefficients_after_scope_exit(self):
        self.check_loads(True)

    def test_checkpoint_name_mapping_after_scope_exit(self):
        self.check_loads(True, top_level=True)

    def test_coefficient_shards_keep_rank_when_width_is_unchanged(self):
        publish(
            ServerArgs(model_path="dummy", device="cuda", tp_size=4, attn_dp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )
        for kv_heads in (1, 2, 8):
            model, attention = build_model(kv_heads)
            with parallel_scope(
                tp_rank=0, attn_tp_rank=0, attn_dp_rank=0, moe_tp_rank=0
            ):
                load_coefficients(model, attention)


if __name__ == "__main__":
    unittest.main()
