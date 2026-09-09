from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.hisparse_coordinator import (
    HiSparseCoordinator,
    resolve_demand_group_roles,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.parametrize(
    "pattern,roles", [("NSSN", [1, 2, 2, 0]), ("NNS", [0, 1, 2]), ("N", [0])]
)
def test_actual_model_groups(pattern, roles):
    config = SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"],
        index_topk=2048,
        num_hidden_layers=len(pattern),
        index_topk_pattern=pattern,
    )
    assert resolve_demand_group_roles(config) == roles


def test_group_binding():
    c = HiSparseCoordinator.__new__(HiSparseCoordinator)
    c.mtp_demand_buffer_enabled = True
    c.mtp_demand_cache_rows = 4096
    c.mem_pool_device = SimpleNamespace(start_layer=0, layer_num=4)
    c.demand_group_roles = [1, 2, 2, 0]
    c.mtp_group_slots = torch.full((8, 2048), -1, dtype=torch.int32)
    c.mtp_demand_host_kv = torch.zeros((4, 64, 656), dtype=torch.uint8)
    c.mtp_demand_cache_tags = torch.zeros((4, 2, 4096), dtype=torch.int64)
    c.mtp_demand_decode_calls = torch.ones(2, dtype=torch.int32)
    c.mtp_demand_num_real_query_rows = torch.tensor([4], dtype=torch.int32)
    c.mtp_demand_device_locs = torch.zeros((2, 4102), dtype=torch.int64)
    c.top_k_host_locs_buffer = torch.zeros((8, 2048), dtype=torch.int32)
    c.mtp_demand_expanded_req_pool_indices = torch.ones(8, dtype=torch.int64)
    c.mtp_demand_expanded_committed_lens = torch.ones(8, dtype=torch.int32)
    bundles = [
        c.get_mtp_demand_attention_inputs(
            layer_id=i, seq_lens=torch.ones(4, dtype=torch.int32)
        )
        for i in range(4)
    ]
    assert [x.group_role for x in bundles] == [1, 2, 2, 0]
    assert len({x.group_slots.data_ptr() for x in bundles[:3]}) == 1
    assert len({x.cache_tags.data_ptr() for x in bundles}) == 4
    assert len({x.host_kv.data_ptr() for x in bundles}) == 4
    assert len({x.device_locs.data_ptr() for x in bundles}) == 1
    assert bundles[3].group_slots is None
