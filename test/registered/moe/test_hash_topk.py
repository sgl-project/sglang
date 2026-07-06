import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.moe import hash_topk as hash_topk_module
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


@pytest.fixture(autouse=True)
def _set_dummy_server_args():
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


def test_hash_topk_remaps_per_rank_fused_shared_slots(monkeypatch):
    monkeypatch.setattr(
        hash_topk_module, "has_per_rank_fused_shared_slots", lambda *_args: True
    )
    recorded = {}

    class FakeRecorder:
        def on_select_experts(self, *, topk_ids):
            recorded["topk_ids"] = topk_ids.clone()

    monkeypatch.setattr(
        hash_topk_module,
        "get_global_expert_distribution_recorder",
        lambda: FakeRecorder(),
    )

    topk = HashTopK(
        topk=3,
        num_experts=256,
        num_fused_shared_experts=1,
        vocab_size=2,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=2.5,
    )
    with torch.no_grad():
        topk.tid2eid.copy_(torch.tensor([[0, 65], [63, 127]], dtype=torch.int32))

    def fake_hash_topk(**kwargs):
        router_logits = kwargs["router_logits"]
        device = router_logits.device
        dtype = router_logits.dtype
        return (
            torch.tensor(
                [[1.0, 1.0, 0.4], [1.0, 1.0, 0.4]], dtype=dtype, device=device
            ),
            torch.tensor(
                [[0, 65, 256], [63, 127, 256]], dtype=torch.int32, device=device
            ),
        )

    monkeypatch.setattr("sglang.jit_kernel.dsv4.hash_topk", fake_hash_topk)

    info = ExpertLocationDispatchInfo(
        ep_dispatch_algorithm="static",
        partial_logical_to_rank_dispatch_physical_map=torch.arange(
            256, dtype=torch.int32
        ),
        partial_logical_to_all_physical_map=torch.arange(256, dtype=torch.int32).view(
            256, 1
        ),
        partial_logical_to_all_physical_map_num_valid=torch.ones(
            256, dtype=torch.int32
        ),
        num_physical_experts=256,
    )

    with get_parallel().override(moe_ep_size=4, moe_ep_rank=2):
        output = topk(
            hidden_states=torch.empty(2, 4),
            router_logits=torch.ones(2, 256),
            input_ids=torch.tensor([0, 1], dtype=torch.int64),
            expert_location_dispatch_info=info,
        )

    # Physical layout for EP=4 has 64 routed slots per rank plus one local
    # shared slot: [0..63, shared, 64..127, shared, ...].
    assert output.topk_ids.tolist() == [[0, 66, 194], [63, 128, 194]]
    assert torch.allclose(output.topk_weights[:, -1], torch.full((2,), 0.4))
    assert recorded["topk_ids"].tolist() == [[0, 65], [63, 127]]


def test_hash_topk_empty_output_keeps_per_rank_shared_slot(monkeypatch):
    monkeypatch.setattr(
        hash_topk_module, "has_per_rank_fused_shared_slots", lambda *_args: True
    )

    topk = HashTopK(
        topk=7,
        num_experts=256,
        num_fused_shared_experts=1,
        vocab_size=2,
        scoring_func="softmax",
    )

    output = topk.empty_topk_output(torch.device("cpu"))

    assert output.topk_ids.shape == (0, 7)
    assert output.topk_weights.shape == (0, 7)
    assert output.router_logits.shape == (0, 6)


def test_deepep_empty_forward_does_not_append_shared_slot_twice():
    captured = {}

    class FakeTopK:
        def empty_topk_output(self, device, *, layer_id=None):
            return StandardTopKOutput(
                topk_weights=torch.empty((0, 9), dtype=torch.float32, device=device),
                topk_ids=torch.empty((0, 9), dtype=torch.int32, device=device),
                router_logits=torch.empty((0, 8), dtype=torch.float32, device=device),
            )

    class FakeExperts:
        should_fuse_routed_scaling_factor_in_topk = True

        def __call__(self, hidden_states, topk_output):
            captured["topk_ids_shape"] = tuple(topk_output.topk_ids.shape)
            captured["topk_weights_shape"] = tuple(topk_output.topk_weights.shape)
            return hidden_states

    moe = SimpleNamespace(
        _fuse_shared_experts_inside_sbo=False,
        is_nextn=False,
        num_fused_shared_experts=1,
        layer_id=0,
        topk=FakeTopK(),
        experts=FakeExperts(),
        alt_stream=None,
        routed_scaling_factor=1.0,
    )

    hidden_states = torch.empty((0, 4), dtype=torch.float32)
    forward_batch = SimpleNamespace(num_token_non_padded=None)

    DeepseekV2MoE.forward_deepep(moe, hidden_states, forward_batch)

    assert captured["topk_ids_shape"] == (0, 9)
    assert captured["topk_weights_shape"] == (0, 9)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
