"""Inkling checkpoint preprocessing follows its constructed expert modules."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.lora.layers import BaseLayerWithLoRA
from sglang.srt.models import inkling
from sglang.srt.models.inkling_common.moe import InklingSharedFusedMoE
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def values(shape, offset=0, *, device="cpu", dtype=torch.float32):
    count = 1
    for size in shape:
        count *= size
    return (
        (((torch.arange(count, device=device) + offset) % 29 - 14) / 128)
        .reshape(shape)
        .to(dtype)
    )


def loading_scope(changed):
    if not changed:
        return nullcontext()
    return get_parallel().override(
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


def build_model(*, width=32, intermediate=64, wrapped=False):
    model = inkling.InklingForConditionalGeneration.__new__(
        inkling.InklingForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.text_config = SimpleNamespace(
        n_routed_experts=8, inference_moe_w13_interleaved=True
    )
    model.quant_config = model.audio = model.visual = None
    model.llm = nn.Module()
    block = nn.Module()
    block.mlp = nn.Module()
    routed = FusedMoE(
        num_experts=8,
        hidden_size=width,
        intermediate_size=intermediate,
        layer_id=0,
        top_k=1,
        use_weight_loader_fused=True,
    )
    shared = InklingSharedFusedMoE(
        2, width, intermediate, 0, "llm.layers.0.mlp.shared_experts", None, True
    )
    block.mlp.experts = BaseLayerWithLoRA(routed, Mock()) if wrapped else routed
    block.mlp.shared_experts = shared
    model.llm.layers = nn.ModuleList([block])
    return model, routed, shared


def load_per_expert(model, routed, *, changed=False, contiguous=False, offset=0):
    width = routed.hidden_size
    intermediate = routed.intermediate_size_per_partition * routed.moe_tp_size
    local = 8 // routed.moe_ep_size
    start = routed.moe_ep_rank * local
    shard = routed.moe_tp_rank * routed.intermediate_size_per_partition
    weights = []
    gates = []
    ups = []
    downs = []
    for expert in range(8):
        gate = values(
            (intermediate, width),
            offset + 3 * expert,
            device=routed.w13_weight.device,
            dtype=routed.w13_weight.dtype,
        )
        up = values(
            (intermediate, width),
            offset + 3 * expert + 1,
            device=gate.device,
            dtype=gate.dtype,
        )
        down = values(
            (width, intermediate),
            offset + 3 * expert + 2,
            device=gate.device,
            dtype=gate.dtype,
        )
        for name, weight in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
            weights.append(
                (f"model.llm.layers.0.mlp.experts.{expert}.{name}.weight", weight)
            )
        if start <= expert < start + local:
            gates.append(gate[shard : shard + routed.intermediate_size_per_partition])
            ups.append(up[shard : shard + routed.intermediate_size_per_partition])
            downs.append(
                down[:, shard : shard + routed.intermediate_size_per_partition]
            )
    with (
        loading_scope(changed),
        patch.object(
            inkling, "lora_compatible_layout_enabled", return_value=contiguous
        ),
    ):
        loaded = model.load_weights(weights)
    gate, up = torch.stack(gates), torch.stack(ups)
    expected_w13 = (
        torch.cat((gate, up), dim=1)
        if contiguous
        else torch.stack((gate, up), dim=2).flatten(1, 2)
    )
    expected_w2 = torch.stack(downs)
    suffix = (
        ".base_layer"
        if isinstance(model.llm.layers[0].mlp.experts, BaseLayerWithLoRA)
        else ""
    )
    assert loaded == {
        f"llm.layers.0.mlp.experts{suffix}.w13_weight",
        f"llm.layers.0.mlp.experts{suffix}.w2_weight",
    }
    torch.testing.assert_close(routed.w13_weight, expected_w13, rtol=0, atol=0)
    torch.testing.assert_close(routed.w2_weight, expected_w2, rtol=0, atol=0)
    return expected_w13, expected_w2


def load_fused(model, routed, shared, *, changed=False, offset=0):
    weights = []
    expected = []
    for kind, module, experts in (
        ("experts", routed, 8),
        ("shared_experts", shared, 2),
    ):
        width = module.hidden_size
        intermediate = module.intermediate_size_per_partition * module.moe_tp_size
        w13 = values(
            (experts, 2 * intermediate, width),
            offset,
            device=module.w13_weight.device,
            dtype=module.w13_weight.dtype,
        )
        w2 = values(
            (experts, width, intermediate),
            offset + 1,
            device=w13.device,
            dtype=w13.dtype,
        )
        prefix = f"model.llm.layers.0.mlp.{kind}."
        weights.extend(
            (
                (
                    prefix
                    + (
                        "shared_w13_weight"
                        if kind == "shared_experts"
                        else "w13_weight"
                    ),
                    w13,
                ),
                (
                    prefix
                    + ("shared_w2_weight" if kind == "shared_experts" else "w2_weight"),
                    w2,
                ),
            )
        )
        local = experts // module.moe_ep_size
        first = module.moe_ep_rank * local
        rank = module.moe_tp_rank
        local_w13 = w13[first : first + local].chunk(module.moe_tp_size, dim=1)[rank]
        local_w13 = torch.cat((local_w13[:, 0::2], local_w13[:, 1::2]), dim=1)
        local_w2 = w2[first : first + local].chunk(module.moe_tp_size, dim=2)[rank]
        expected.append((module, local_w13, local_w2))
    weights.append(("model.llm.layers.999.mlp.experts.w13_weight", weights[0][1]))
    with (
        loading_scope(changed),
        patch.object(inkling, "lora_compatible_layout_enabled", return_value=True),
        patch.object(inkling, "use_inkling_shared_fused_moe", return_value=True),
    ):
        loaded = model.load_weights(weights)
    assert loaded == {
        f"llm.layers.0.mlp.{kind}.{leaf}"
        for kind in ("experts", "shared_experts")
        for leaf in ("w13_weight", "w2_weight")
    }
    for module, w13, w2 in expected:
        torch.testing.assert_close(module.w13_weight, w13, rtol=0, atol=0)
        torch.testing.assert_close(module.w2_weight, w2, rtol=0, atol=0)
    return expected


class TestInklingMoeLoaderLayout(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def check_layouts(self, *, changed=False, wrapped=False):
        for ep in (1, 2, 4):
            for rank in range(4):
                publish(
                    ServerArgs(
                        model_path="dummy",
                        device="cpu",
                        tp_size=4,
                        ep_size=ep,
                        moe_runner_backend="triton",
                    ),
                    role="test",
                    ranks=SpawnRanks(world_rank=rank),
                )
                for contiguous in (False, True):
                    model, routed, _ = build_model(wrapped=wrapped)
                    for offset in (0, 7):
                        load_per_expert(
                            model,
                            routed,
                            changed=changed,
                            contiguous=contiguous,
                            offset=offset,
                        )
                if not wrapped:
                    model, routed, shared = build_model()
                    for offset in (0, 7):
                        load_fused(
                            model, routed, shared, changed=changed, offset=offset
                        )

    def test_native_expert_loaders_after_scope_exit(self):
        self.check_layouts(changed=True)

    def test_native_per_expert_loader_with_lora_parameter_aliases(self):
        self.check_layouts(changed=True, wrapped=True)


if __name__ == "__main__":
    unittest.main()
