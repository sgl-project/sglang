import importlib.util
from pathlib import Path
import sys
import types
from typing import NamedTuple

import torch


def _install_fake_modules():
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.eplb",
        "sglang.srt.layers",
        "sglang.srt.layers.moe",
        "sglang.srt.state_capturer",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    root = types.ModuleType("sgl_kernel_npu")
    norm = types.ModuleType("sgl_kernel_npu.norm")
    l1_norm_mod = types.ModuleType("sgl_kernel_npu.norm.l1_norm")

    def l1_norm(x):
        return x / x.sum(dim=-1, keepdim=True)

    l1_norm_mod.l1_norm = l1_norm
    sys.modules.setdefault("sgl_kernel_npu", root)
    sys.modules.setdefault("sgl_kernel_npu.norm", norm)
    sys.modules["sgl_kernel_npu.norm.l1_norm"] = l1_norm_mod

    expert_distribution = types.ModuleType("sglang.srt.eplb.expert_distribution")

    class Recorder:
        @staticmethod
        def on_select_experts(topk_ids):
            pass

    expert_distribution.get_global_expert_distribution_recorder = lambda: Recorder()
    sys.modules["sglang.srt.eplb.expert_distribution"] = expert_distribution

    expert_location = types.ModuleType("sglang.srt.eplb.expert_location_dispatch")
    expert_location.topk_ids_logical_to_physical = lambda topk_ids, info: topk_ids
    sys.modules["sglang.srt.eplb.expert_location_dispatch"] = expert_location

    moe_topk = types.ModuleType("sglang.srt.layers.moe.topk")

    class StandardTopKOutput(NamedTuple):
        topk_weights: torch.Tensor
        topk_ids: torch.Tensor
        router_logits: torch.Tensor

    def select_experts(*args, **kwargs):
        raise AssertionError("fallback select_experts should not be used")

    moe_topk.StandardTopKOutput = StandardTopKOutput
    moe_topk.select_experts = select_experts
    sys.modules["sglang.srt.layers.moe.topk"] = moe_topk

    routed_experts = types.ModuleType("sglang.srt.state_capturer.routed_experts")
    routed_experts.get_global_experts_capturer = lambda: None
    sys.modules["sglang.srt.state_capturer.routed_experts"] = routed_experts


def _load_npu_topk_module():
    _install_fake_modules()
    module_path = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/hardware_backend/npu/moe/topk.py"
    )
    spec = importlib.util.spec_from_file_location("_npu_topk_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_npu_sigmoid_topk_uses_sigmoid_semantics_and_output_scaling(monkeypatch):
    npu_topk = _load_npu_topk_module()

    router_logits = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    hidden_states = torch.zeros((1, 4), dtype=torch.bfloat16)
    routed_scaling_factor = 2.5

    class FakeNpuOps:
        @staticmethod
        def npu_moe_gating_top_k_softmax(*args, **kwargs):
            raise AssertionError("sigmoid routing must not use the softmax top-k op")

        @staticmethod
        def npu_moe_gating_top_k(
            router_logits,
            *,
            k,
            bias,
            k_group,
            group_count,
            group_select_mode,
            renorm,
            norm_type,
            routed_scaling_factor,
            eps,
        ):
            scores = router_logits.sigmoid()
            values, ids = torch.topk(scores, k=k, dim=-1)
            assert norm_type == 1
            assert renorm == 0
            assert routed_scaling_factor == 1
            return values, ids.to(torch.int32), None

    monkeypatch.setattr(torch.ops, "npu", FakeNpuOps, raising=False)

    topk_config = types.SimpleNamespace(
        top_k=2,
        use_grouped_topk=False,
        correction_bias=None,
        topk_group=None,
        num_expert_group=None,
        renormalize=True,
        scoring_func="sigmoid",
        num_fused_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=True,
    )

    topk_output = npu_topk.fused_topk_npu(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk_config=topk_config,
    )

    raw = router_logits.sigmoid().topk(2, dim=-1).values
    expected = raw / raw.sum(dim=-1, keepdim=True) * routed_scaling_factor
    torch.testing.assert_close(topk_output.topk_weights, expected)
