"""Model checkpoint preprocessing uses the constructed projection's layout."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.lora.layers import BaseLayerWithLoRA
from sglang.srt.models import inkling
from sglang.srt.models.inkling_common.attn import InklingAttention
from sglang.srt.models.inkling_common.dense_mlp import InklingDenseMLP
from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import parallel_scope, publish, rank_size
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, stage="weekly", runner_config="cpu")


def values(shape, offset=0, *, device="cpu", dtype=torch.float32):
    count = 1
    for size in shape:
        count *= size
    return (
        (((torch.arange(count, device=device) + offset) % 29 - 14) / 128)
        .reshape(shape)
        .to(dtype)
    )


def base(module):
    return module.base_layer if isinstance(module, BaseLayerWithLoRA) else module


def loading_scope(changed):
    if not changed:
        return nullcontext()
    return parallel_scope(
        tp_rank=0,
        tp_size=1,
        tp_group=None,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_tp_group=None,
        attn_dp_rank=0,
        attn_dp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        moe_tp_rank=0,
        moe_tp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        moe_ep_group=None,
        moe_dp_rank=0,
        moe_dp_size=1,
    )


def build_attention_model(
    kv_heads, *, local=False, wrapped=False, width=32, head_dim=8, quant_config=None
):
    model = inkling.InklingForConditionalGeneration.__new__(
        inkling.InklingForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.audio = model.visual = None
    model.text_config = SimpleNamespace(
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        swa_num_key_value_heads=kv_heads,
        swa_head_dim=head_dim,
        local_layer_ids=[0] if local else [],
        n_routed_experts=2,
    )
    model.llm = nn.Module()
    block = nn.Module()
    block.attn = InklingAttention(
        width,
        8,
        kv_heads,
        head_dim,
        head_dim,
        8,
        4,
        1e-6,
        local,
        0,
        kv_conv=True,
        sconv_kernel_size=3,
        quant_config=quant_config,
    )
    if wrapped:
        block.attn.qkvr = BaseLayerWithLoRA(block.attn.qkvr, Mock())
    model.llm.layers = nn.ModuleList([block])
    return model, block.attn


def load_attention(model, attention, *, changed=False, offset=0):
    qkvr = base(attention.qkvr)
    rank, size = rank_size(qkvr)
    device, dtype = qkvr.weight.device, qkvr.weight.dtype
    heads, kv_heads, head_dim = (
        qkvr.inkling_num_heads,
        qkvr.inkling_num_kv_heads,
        qkvr.inkling_head_dim,
    )
    weights, expected = [], []
    for i, (name, rows) in enumerate(
        (
            ("wq_du", heads * head_dim),
            ("wk_dv", kv_heads * head_dim),
            ("wv_dv", kv_heads * head_dim),
            ("wr_du", heads * qkvr.inkling_d_rel),
        )
    ):
        full = values((rows, qkvr.input_size), offset + i, device=device, dtype=dtype)
        weights.append((f"model.llm.layers.0.attn.{name}.weight", full))
        if name in ("wk_dv", "wv_dv") and size > kv_heads:
            begin = (rank // (size // kv_heads)) * head_dim
            expected.append(full[begin : begin + head_dim])
        else:
            expected.append(full.chunk(size)[rank])
    conv_expected = {}
    for i, name in enumerate(("k_sconv", "v_sconv")):
        full = values(
            (kv_heads * head_dim, 1, 3), offset + i + 4, device=device, dtype=dtype
        )
        weights.append((f"model.llm.layers.0.attn.{name}.weight", full))
        local_rows = attention.num_tp_kv_heads * head_dim
        begin = (
            (rank // (size // kv_heads)) * head_dim
            if size > kv_heads
            else rank * local_rows
        )
        conv_expected[name] = full[begin : begin + local_rows]
    # Checkpoints can include a layer absent from this model; keep ignoring it.
    weights.append(("model.llm.layers.999.attn.wk_dv.weight", weights[1][1]))
    with loading_scope(changed):
        loaded = model.load_weights(weights)
    assert loaded == {
        "llm.layers.0.attn.qkvr.weight",
        "llm.layers.0.attn.k_sconv.weight",
        "llm.layers.0.attn.v_sconv.weight",
    }
    expected = torch.cat(expected)
    torch.testing.assert_close(qkvr.weight, expected)
    for name, shard in conv_expected.items():
        torch.testing.assert_close(getattr(attention, name).weight, shard)
    return expected, conv_expected


def build_dense_model(
    *, mtp=False, wrapped=False, width=32, group="attn_tp", quant_config=None
):
    cls = (
        inkling.InklingForConditionalGenerationMTP
        if mtp
        else inkling.InklingForConditionalGeneration
    )
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    mlp = InklingDenseMLP(
        width,
        2 * width,
        False,
        0,
        fused=True,
        parallel_group=group,
        quant_config=quant_config,
    )
    if wrapped:
        mlp.gate_up_proj = BaseLayerWithLoRA(mlp.gate_up_proj, Mock())
    if mtp:
        model.model = nn.Module()
        model.model.mlp = mlp
    else:
        model.audio = model.visual = None
        model.text_config = SimpleNamespace(n_routed_experts=2)
        model.llm = nn.Module()
        block = nn.Module()
        block.mlp = mlp
        model.llm.layers = nn.ModuleList([block])
    return model, mlp


def load_dense(model, mlp, *, mtp=False, changed=False, compatible=False, offset=0):
    projection = base(mlp.gate_up_proj)
    rank, size = rank_size(projection)
    device, dtype = projection.weight.device, projection.weight.dtype
    full = values(
        (sum(projection.output_sizes), projection.input_size),
        offset,
        device=device,
        dtype=dtype,
    )
    expected = full.chunk(size)[rank]
    if compatible and not mtp:
        expected = torch.cat((expected[::2], expected[1::2]))
    down = base(mlp.down_proj)
    full_down = values(
        (down.output_size, down.input_size), offset + 1, device=device, dtype=dtype
    )
    prefix = "mtp.layers.0" if mtp else "llm.layers.0"
    with (
        loading_scope(changed),
        patch.object(
            inkling, "lora_compatible_layout_enabled", return_value=compatible
        ),
    ):
        loaded = model.load_weights(
            [
                (f"model.{prefix}.mlp.w13_dn.weight", full),
                (f"model.{prefix}.mlp.w2_md.weight", full_down),
            ]
        )
    torch.testing.assert_close(projection.weight, expected)
    down_rank, down_size = rank_size(down)
    expected_down = full_down.chunk(down_size, dim=1)[down_rank]
    torch.testing.assert_close(down.weight, expected_down)
    expected_prefix = "model" if mtp else "llm.layers.0"
    assert loaded == {
        f"{expected_prefix}.mlp.gate_up_proj.weight",
        f"{expected_prefix}.mlp.down_proj.weight",
    }
    return expected, expected_down


class TestInklingLinearLoaderLayout(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def exercise(self, changed, wrapped=False):
        for dp in (1, 2):
            for rank in range(4):
                reset_context()
                publish(
                    ServerArgs(
                        model_path="dummy", device="cpu", tp_size=4, attn_dp_size=dp
                    ),
                    role="test",
                    ranks=SpawnRanks(world_rank=rank),
                )
                for kv_heads in (1, 2, 4):
                    for local in (False, True):
                        model, attention = build_attention_model(
                            kv_heads, local=local, wrapped=wrapped
                        )
                        for offset in (0, 7):
                            expected, _ = load_attention(
                                model, attention, changed=changed, offset=offset
                            )
                            x = values((2, attention.hidden_size), 3)
                            actual = base(attention.qkvr)(x)[0]
                            torch.testing.assert_close(
                                actual, torch.nn.functional.linear(x, expected)
                            )
                for mtp in (False, True):
                    for group in ("attn_tp", "replicated"):
                        model, mlp = build_dense_model(
                            mtp=mtp, group=group, wrapped=wrapped
                        )
                        for compatible in (False, True):
                            for offset in (0, 7):
                                load_dense(
                                    model,
                                    mlp,
                                    mtp=mtp,
                                    changed=changed,
                                    compatible=compatible,
                                    offset=offset,
                                )

    def test_checkpoint_layout_after_scope_exit(self):
        self.exercise(True)

    def test_checkpoint_layout_through_lora_wrappers(self):
        self.exercise(True, wrapped=True)


if __name__ == "__main__":
    unittest.main()
