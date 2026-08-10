import importlib.util
import os
from types import SimpleNamespace

os.environ["SGLANG_USE_AITER"] = "1"

import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from sglang.test.ci.ci_register import register_amd_ci  # noqa: E402

register_amd_ci(
    est_time=30,
    suite="stage-b-test-1-gpu-small-amd",
)

if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("AITER TopK requires ROCm and an AMD GPU.", allow_module_level=True)
if importlib.util.find_spec("aiter") is None:
    pytest.skip("AITER is not installed.", allow_module_level=True)

import sglang.srt.layers.moe.topk as topk_module  # noqa: E402
import sglang.srt.models.minimax_m3 as minimax_m3  # noqa: E402


class _FakeExperts(nn.Module):
    should_fuse_routed_scaling_factor_in_topk = False

    def __init__(self, **kwargs):
        super().__init__()


def test_aiter_minimax_topk_applies_routed_scaling_once(monkeypatch):
    assert topk_module._use_aiter
    assert minimax_m3._use_aiter

    exec_context = SimpleNamespace(
        moe=SimpleNamespace(ep_num_redundant_experts=0, enable_waterfill=False)
    )
    monkeypatch.setattr(minimax_m3, "get_parallel", lambda: SimpleNamespace(tp_size=1))
    monkeypatch.setattr(minimax_m3, "get_exec", lambda: exec_context)
    monkeypatch.setattr(topk_module, "get_exec", lambda: exec_context)
    monkeypatch.setattr(
        minimax_m3, "get_moe_impl_class", lambda quant_config: _FakeExperts
    )
    monkeypatch.setattr(
        minimax_m3, "is_shared_experts_fusion_disabled", lambda: True
    )
    monkeypatch.setattr(
        minimax_m3,
        "get_moe_a2a_backend",
        lambda: SimpleNamespace(is_deepep=lambda: False),
    )
    monkeypatch.setattr(
        minimax_m3, "ReplicatedLinear", lambda *args, **kwargs: nn.Identity()
    )

    config = SimpleNamespace(
        n_shared_experts=None,
        num_local_experts=4,
        num_experts_per_tok=2,
        hidden_size=4,
        intermediate_size=8,
        routed_scaling_factor=2.0,
        use_routing_bias=True,
        scoring_func="sigmoid",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
    )
    device = torch.device("cuda")
    moe = minimax_m3.MiniMaxM3MoE(
        config, layer_id=0, quant_config=object()
    ).to(device)
    topk_config = moe.topk.topk_config

    routing_logits = torch.tensor(
        [
            [2.0, 0.5, -1.0, -2.0],
            [-1.5, 1.5, 0.25, -0.75],
            [0.0, -2.0, 2.5, 0.75],
            [1.0, -0.5, -1.5, 2.0],
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    routing_bias = torch.tensor(
        [0.05, -0.10, 0.15, 0.0], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        moe.e_score_correction_bias.copy_(routing_bias)
    topk_config.correction_bias = moe.e_score_correction_bias

    actual_weights, actual_ids = topk_module.fused_topk(
        hidden_states=torch.zeros_like(routing_logits),
        gating_output=routing_logits,
        topk=topk_config.top_k,
        renormalize=topk_config.renormalize,
        correction_bias=topk_config.correction_bias,
        scoring_func=topk_config.scoring_func,
        routed_scaling_factor=topk_config.routed_scaling_factor,
        apply_routed_scaling_factor_on_output=(
            topk_config.apply_routed_scaling_factor_on_output
        ),
        num_fused_shared_experts=topk_config.num_fused_shared_experts,
    )

    scores = routing_logits.float().sigmoid()
    expected_ids = (scores + routing_bias).topk(topk_config.top_k, dim=-1).indices
    expected_weights = scores.gather(-1, expected_ids)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True) + 1e-20
    expected_weights *= config.routed_scaling_factor

    actual_dense = torch.zeros_like(scores)
    actual_dense.scatter_(1, actual_ids.long(), actual_weights)
    expected_dense = torch.zeros_like(scores)
    expected_dense.scatter_(1, expected_ids, expected_weights)
    torch.testing.assert_close(actual_dense, expected_dense, rtol=5e-3, atol=5e-3)

    assert actual_weights.sum(dim=-1).cpu().tolist() == pytest.approx(
        [config.routed_scaling_factor] * routing_logits.shape[0],
        rel=1e-5,
        abs=1e-5,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
