import importlib.util
import sys
import types
from pathlib import Path
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

    def capture_routed_experts_if_allowed(*args, **kwargs):
        return None

    moe_topk.StandardTopKOutput = StandardTopKOutput
    moe_topk.select_experts = select_experts
    moe_topk.capture_routed_experts_if_allowed = capture_routed_experts_if_allowed
    sys.modules["sglang.srt.layers.moe.topk"] = moe_topk

    routed_experts = types.ModuleType("sglang.srt.state_capturer.routed_experts")
    routed_experts.get_global_experts_capturer = lambda: None
    sys.modules["sglang.srt.state_capturer.routed_experts"] = routed_experts


def _load_npu_topk_module():
    _install_fake_modules()
    module_path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/hardware_backend/npu/moe/topk.py"
    )
    spec = importlib.util.spec_from_file_location("_npu_topk_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_topk_config(
    correction_bias, routed_scaling_factor, renormalize=True
) -> types.SimpleNamespace:
    """M3-shaped TopKConfig: sigmoid scoring, no grouped routing."""
    return types.SimpleNamespace(
        top_k=2,
        use_grouped_topk=False,
        correction_bias=correction_bias,
        topk_group=None,
        num_expert_group=None,
        renormalize=renormalize,
        scoring_func="sigmoid",
        num_fused_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=True,
    )


def _run_fused_topk_npu(npu_topk, topk_config, router_logits):
    return npu_topk.fused_topk_npu(
        hidden_states=torch.zeros((1, 4), dtype=torch.bfloat16),
        router_logits=router_logits,
        topk_config=topk_config,
    )


class _SigmoidFakeNpuOps:
    """npu_moe_gating_top_k mirroring the real sigmoid contract (norm_type=1)."""

    def __init__(self, router_logits, routed_scaling_factor, expect_bias=None):
        self.router_logits = router_logits
        self.routed_scaling_factor = routed_scaling_factor
        self.expect_bias = expect_bias

    def npu_moe_gating_top_k_softmax(self, *args, **kwargs):
        raise AssertionError("sigmoid routing must not use the softmax top-k op")

    def npu_moe_gating_top_k(
        self,
        router_logits,
        *,
        k,
        bias,
        renorm,
        norm_type,
        routed_scaling_factor,
        **kwargs,
    ):
        # Contract: sigmoid scoring -> norm_type=1; bias (if any) must reach the
        # op; renorm and the routed scaling factor are applied inside the op.
        assert norm_type == 1
        if self.expect_bias is not None:
            assert bias is not None and bias.shape == self.expect_bias.shape
        scores = (
            (router_logits + bias).sigmoid()
            if bias is not None
            else router_logits.sigmoid()
        )
        values, ids = torch.topk(scores, k=k, dim=-1)
        if renorm:
            values = values / values.sum(dim=-1, keepdim=True)
        values = values * routed_scaling_factor
        return values, ids.to(torch.int32), None


def test_npu_sigmoid_topk_without_bias_uses_sigmoid_op(monkeypatch):
    """Sigmoid routing without correction bias must NOT fall into the softmax fast path.

    Guards fused_topk_npu's fast-path branch: it previously matched
    ``not use_grouped_topk and correction_bias is None`` without excluding
    sigmoid scoring, routing sigmoid models through the softmax op.
    """
    npu_topk = _load_npu_topk_module()

    router_logits = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    routed_scaling_factor = 2.5
    fake = _SigmoidFakeNpuOps(router_logits, routed_scaling_factor, expect_bias=None)
    monkeypatch.setattr(torch.ops, "npu", fake, raising=False)

    topk_output = _run_fused_topk_npu(
        npu_topk,
        _make_topk_config(None, routed_scaling_factor),
        router_logits,
    )

    raw = router_logits.sigmoid().topk(2, dim=-1).values
    expected = raw / raw.sum(dim=-1, keepdim=True) * routed_scaling_factor
    torch.testing.assert_close(topk_output.topk_weights, expected)


def test_npu_sigmoid_topk_with_routing_bias_matches_m3_config(monkeypatch):
    """M3 real config (use_routing_bias=True): bias must reach the sigmoid op."""
    npu_topk = _load_npu_topk_module()

    router_logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
    routed_scaling_factor = 2.5
    correction_bias = torch.tensor([0.1, -0.2, 0.3, 0.05], dtype=torch.float32)
    fake = _SigmoidFakeNpuOps(
        router_logits, routed_scaling_factor, expect_bias=correction_bias
    )
    monkeypatch.setattr(torch.ops, "npu", fake, raising=False)

    topk_output = _run_fused_topk_npu(
        npu_topk,
        _make_topk_config(correction_bias, routed_scaling_factor),
        router_logits,
    )

    raw = (router_logits + correction_bias).sigmoid().topk(2, dim=-1).values
    expected = raw / raw.sum(dim=-1, keepdim=True) * routed_scaling_factor
    torch.testing.assert_close(topk_output.topk_weights, expected)
