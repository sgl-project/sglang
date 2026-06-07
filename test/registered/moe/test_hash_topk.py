import sys

import pytest
import torch

from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.moe import hash_topk as hash_topk_module
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


def test_hash_topk_remaps_per_rank_fused_shared_slots(monkeypatch):
    monkeypatch.setattr(
        hash_topk_module, "uses_per_rank_fused_shared_slots", lambda *_args: True
    )
    monkeypatch.setattr(topk_module, "get_moe_expert_parallel_world_size", lambda: 4)
    monkeypatch.setattr(topk_module, "get_moe_expert_parallel_rank", lambda: 2)

    topk = HashTopK(
        topk=3,
        num_experts=256,
        num_fused_shared_experts=1,
        vocab_size=2,
        scoring_func="softmax",
        routed_scaling_factor=2.5,
    )
    with torch.no_grad():
        topk.tid2eid.copy_(torch.tensor([[0, 65], [63, 127]], dtype=torch.int32))

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
