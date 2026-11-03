from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.moe import topk
from sglang.srt.layers.moe.token_dispatcher import moriep
from sglang.srt.models import mimo_v2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


@pytest.mark.parametrize(
    "configured,recording,capturing,expected",
    [
        (False, False, False, False),
        (False, False, True, False),
        (True, False, False, False),
        (True, False, True, True),
        (True, True, False, True),
        (True, True, True, True),
    ],
)
def test_mori_counts_only_for_configured_or_active_recorder(
    monkeypatch, configured, recording, capturing, expected
):
    recorder = (
        SimpleNamespace(recording=recording)
        if configured
        else moriep._ExpertDistributionRecorderNoop()
    )
    monkeypatch.setattr(
        moriep, "get_global_expert_distribution_recorder", lambda: recorder
    )
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda: SimpleNamespace(is_current_stream_capturing=lambda: capturing),
    )
    assert moriep._should_record_expert_distribution() is expected


def test_mori_moe_selects_a2a_forward(monkeypatch):
    backend = SimpleNamespace(
        is_deepep=lambda: False,
        is_mooncake=lambda: False,
        is_mori=lambda: True,
        is_ascend_fuseep=lambda: False,
    )
    monkeypatch.setattr(mimo_v2, "get_moe_a2a_backend", lambda: backend)
    monkeypatch.setattr(mimo_v2, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(mimo_v2, "get_moe_expert_parallel_world_size", lambda: 8)
    monkeypatch.setattr(
        mimo_v2,
        "get_global_server_args",
        lambda: SimpleNamespace(ep_num_redundant_experts=0),
    )
    monkeypatch.setattr(
        mimo_v2, "MoEGate", lambda **kw: SimpleNamespace(e_score_correction_bias=None)
    )
    monkeypatch.setattr(
        mimo_v2,
        "get_moe_impl_class",
        lambda _: lambda **kw: SimpleNamespace(
            should_fuse_routed_scaling_factor_in_topk=False
        ),
    )
    monkeypatch.setattr(mimo_v2, "TopK", lambda **kw: object())
    config = SimpleNamespace(
        n_routed_experts=256,
        hidden_act="silu",
        num_experts_per_tok=8,
        hidden_size=4096,
        moe_intermediate_size=2048,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        scoring_func="sigmoid",
    )
    model = mimo_v2.MiMoV2MoE(config, layer_id=1)
    model.forward_deepep = Mock(return_value="a2a")
    model.forward_normal = Mock(side_effect=AssertionError("TP path must not run"))
    hidden, batch = torch.empty((0, 4)), object()
    assert model.ep_size == 8
    assert model.forward(hidden, batch) == "a2a"
    model.forward_deepep.assert_called_once_with(hidden, batch)


def test_hip_masks_padded_expert_ids_without_eplb(monkeypatch):
    monkeypatch.setattr(topk, "_is_cuda", False)
    monkeypatch.setattr(topk, "_is_hip", True)
    monkeypatch.setattr(topk, "get_global_experts_capturer", lambda: None)
    monkeypatch.setattr(topk, "_eplb_remap_enabled", lambda: False)
    monkeypatch.setattr(
        topk,
        "_mask_topk_ids_padded_region",
        lambda ids, n, fill_value: ids[int(n) :].fill_(fill_value),
    )
    monkeypatch.setattr(
        topk,
        "_zero_topk_weights_padded_region",
        lambda weights, n: weights[int(n) :].zero_(),
    )
    ids = torch.tensor([[2, 3], [-1, 999]], dtype=torch.int32)
    weights = torch.ones((2, 2))
    config = SimpleNamespace(
        num_fused_shared_experts=0, fused_shared_experts_scaling_factor=1.0
    )
    result, weights, _ = topk._post_process_topk_ids(
        ids,
        weights,
        config,
        torch.empty((2, 4)),
        layer_id=0,
        num_token_non_padded=torch.tensor(1),
    )
    assert result.tolist() == [[2, 3], [0, 0]]
    assert weights.tolist() == [[1.0, 1.0], [0.0, 0.0]]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
