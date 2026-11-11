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

    def _zero_topk_weights_padded_region(topk_weights, num_token_non_padded):
        if num_token_non_padded is None:
            return
        rows = torch.arange(topk_weights.shape[0])
        topk_weights.masked_fill_((rows >= num_token_non_padded)[:, None], 0.0)

    def is_routed_scaling_factor_fused_in_topk(topk_config):
        return topk_config.apply_routed_scaling_factor_on_output

    def fused_shared_expert_weight(*, routed_scaling_factor, fused_in_topk):
        if fused_in_topk or not routed_scaling_factor:
            return 1.0
        return 1.0 / routed_scaling_factor

    # Controllable knobs for the per-rank shared-slot layout tests.
    _fake_state = {"per_rank_slots": False, "moe_ep_size": 1, "moe_ep_rank": 0}

    def has_per_rank_fused_shared_slots(num_shared):
        return num_shared > 0 and _fake_state["per_rank_slots"]

    def remap_topk_for_per_rank_shared_slots(
        topk_ids,
        topk_weights,
        num_fused_shared_experts,
        num_physical_routed_experts,
        topk_config,
    ):
        ep_size = _fake_state["moe_ep_size"]
        ep_rank = _fake_state["moe_ep_rank"]
        num_local_routed = num_physical_routed_experts // ep_size
        num_local_experts = num_local_routed + num_fused_shared_experts
        routed_ids = (
            topk_ids + (topk_ids // num_local_routed) * num_fused_shared_experts
        )
        shared_ids = (
            ep_rank * num_local_experts
            + num_local_routed
            + torch.arange(
                num_fused_shared_experts, dtype=topk_ids.dtype, device=topk_ids.device
            )
        ).expand(topk_ids.shape[0], -1)
        shared_weights = topk_weights.new_full(
            (topk_weights.shape[0], num_fused_shared_experts),
            fused_shared_expert_weight(
                routed_scaling_factor=topk_config.routed_scaling_factor,
                fused_in_topk=is_routed_scaling_factor_fused_in_topk(topk_config),
            ),
        )
        return (
            torch.cat([routed_ids, shared_ids], dim=-1),
            torch.cat([topk_weights, shared_weights], dim=-1),
        )

    moe_topk.StandardTopKOutput = StandardTopKOutput
    moe_topk.select_experts = select_experts
    moe_topk.capture_routed_experts_if_allowed = capture_routed_experts_if_allowed
    moe_topk._zero_topk_weights_padded_region = _zero_topk_weights_padded_region
    moe_topk.is_routed_scaling_factor_fused_in_topk = (
        is_routed_scaling_factor_fused_in_topk
    )
    moe_topk.fused_shared_expert_weight = fused_shared_expert_weight
    moe_topk.has_per_rank_fused_shared_slots = has_per_rank_fused_shared_slots
    moe_topk.remap_topk_for_per_rank_shared_slots = remap_topk_for_per_rank_shared_slots
    moe_topk._fake_state = _fake_state
    sys.modules["sglang.srt.layers.moe.topk"] = moe_topk

    # `npu/moe/topk.py` now imports the helper from its defining module rather
    # than through `layers.moe.topk`.
    moe_utils = types.ModuleType("sglang.srt.layers.moe.utils")
    moe_utils.has_per_rank_fused_shared_slots = has_per_rank_fused_shared_slots
    sys.modules["sglang.srt.layers.moe.utils"] = moe_utils
    sys.modules["sglang.srt.layers.moe"].utils = moe_utils

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
    correction_bias,
    routed_scaling_factor,
    renormalize=True,
    num_shared_topk_slots=0,
    fused_shared_experts_scaling_factor=None,
) -> types.SimpleNamespace:
    """M3-shaped TopKConfig: sigmoid scoring, no grouped routing."""
    return types.SimpleNamespace(
        top_k=2 + num_shared_topk_slots,
        use_grouped_topk=False,
        correction_bias=correction_bias,
        topk_group=None,
        num_expert_group=None,
        renormalize=renormalize,
        scoring_func="sigmoid",
        num_shared_topk_slots=num_shared_topk_slots,
        fused_shared_experts_scaling_factor=fused_shared_experts_scaling_factor,
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


def test_npu_fused_shared_expert_is_appended_as_global_slot(monkeypatch):
    """SEF with a global shared slot (none + ep=1): gate selects routed only.

    The gate must receive k = top_k - S, and the shared column (id = R, weight
    from fused_shared_expert_weight) must be appended after the routed columns.
    """
    npu_topk = _load_npu_topk_module()
    fake_topk_mod = sys.modules["sglang.srt.layers.moe.topk"]
    fake_topk_mod._fake_state["per_rank_slots"] = False

    router_logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
    routed_scaling_factor = 2.5
    fake = _SigmoidFakeNpuOps(router_logits, routed_scaling_factor)
    fake.last_k = None
    orig_op = fake.npu_moe_gating_top_k

    def spy_op(*args, **kwargs):
        fake.last_k = kwargs.get("k")
        return orig_op(*args, **kwargs)

    fake.npu_moe_gating_top_k = spy_op
    monkeypatch.setattr(torch.ops, "npu", fake, raising=False)

    topk_output = _run_fused_topk_npu(
        npu_topk,
        _make_topk_config(None, routed_scaling_factor, num_shared_topk_slots=1),
        router_logits,
    )

    # Gate saw k = (2 + 1) - 1 = 2 routed experts.
    assert fake.last_k == 2
    # Routed ids [3, 2] (top sigmoid scores), then shared slot id = R = 4.
    assert topk_output.topk_ids.tolist() == [[3, 2, 4]]
    assert topk_output.topk_ids.dtype == torch.int32
    # Routed weights: renorm over the 2 routed experts, times rsf (fused in op).
    # Shared weight: fused_in_topk -> 1.0.
    raw = router_logits.sigmoid().topk(2, dim=-1).values
    expected_routed = raw / raw.sum(dim=-1, keepdim=True) * routed_scaling_factor
    torch.testing.assert_close(
        topk_output.topk_weights,
        torch.cat([expected_routed, torch.full((1, 1), 1.0)], dim=-1),
    )


def test_npu_fused_shared_expert_per_rank_slots_are_remapped(monkeypatch):
    """SEF with per-rank shared slots (deepep / ascend_* A2A backends).

    Routed ids are shifted into rank-major blocks with shared-slot gaps, and the
    shared column points at this rank's tail slot with weight 1/rsf when the
    routed scale is applied after the MoE.
    """
    npu_topk = _load_npu_topk_module()
    fake_topk_mod = sys.modules["sglang.srt.layers.moe.topk"]
    fake_topk_mod._fake_state["per_rank_slots"] = True
    fake_topk_mod._fake_state["moe_ep_size"] = 4
    fake_topk_mod._fake_state["moe_ep_rank"] = 2

    router_logits = torch.tensor(
        [[0.0, 1.0, 3.0, 2.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.5, 0.6]],
        dtype=torch.float32,
    )  # R = 12, ep = 4, L = 3, Eloc = 4
    routed_scaling_factor = 2.5
    fake = _SigmoidFakeNpuOps(router_logits, routed_scaling_factor)
    monkeypatch.setattr(torch.ops, "npu", fake, raising=False)

    try:
        topk_output = _run_fused_topk_npu(
            npu_topk,
            _make_topk_config(None, routed_scaling_factor, num_shared_topk_slots=1),
            router_logits,
        )

        # Top-2 sigmoid scores are experts 4 and 5 (rank 1's local experts 1, 2).
        # Per-rank dense ids: 4 -> 4 + 1 = 5, 5 -> 5 + 1 = 6 (one shared gap per
        # preceding rank block, S = 1).
        # Shared slot for ep_rank 2: 2 * 4 + 3 = 11.
        assert topk_output.topk_ids.tolist() == [[5, 6, 11]]
        assert topk_output.topk_ids.dtype == torch.int32
        # Shared weight: fused_in_topk -> 1.0; routed weights keep the gate
        # values (renorm over the 2 routed experts, times rsf fused in op).
        assert topk_output.topk_weights[0, -1].item() == 1.0
        raw = router_logits.sigmoid().topk(2, dim=-1).values
        expected_routed = raw / raw.sum(dim=-1, keepdim=True) * routed_scaling_factor
        torch.testing.assert_close(topk_output.topk_weights[0, :2], expected_routed[0])
    finally:
        fake_topk_mod._fake_state.update(
            {"per_rank_slots": False, "moe_ep_size": 1, "moe_ep_rank": 0}
        )
