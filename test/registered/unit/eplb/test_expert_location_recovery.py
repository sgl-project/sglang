from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_location as location
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_recovery_preserves_graph_visible_tensors():
    fields = (
        "physical_to_logical_map",
        "logical_to_all_physical_map",
        "logical_to_all_physical_map_num_valid",
        "logical_to_rank_dispatch_physical_map",
    )
    tensors = [
        torch.zeros(shape, dtype=torch.int64)
        for shape in ((2, 4), (2, 3, 4), (2, 3), (2, 3))
    ]
    metadata = location.ExpertLocationMetadata(
        **dict(zip(fields, tensors)),
        ep_size=2,
        physical_to_logical_map_cpu=tensors[0].clone(),
        logical_to_all_physical_map_cpu=tensors[1].clone(),
    )
    pointers = [tensor.data_ptr() for tensor in tensors]

    def broadcast(tensor, src, group):
        assert (src, group) == (2, "recovery-group")
        tensor.fill_(7)

    with (
        patch.object(
            location, "get_global_expert_location_metadata", return_value=metadata
        ),
        patch("torch.distributed.broadcast", side_effect=broadcast) as send,
    ):
        assert (
            location.broadcast_global_expert_location_metadata_in_place(
                src_rank=2, group="recovery-group"
            )
            is None
        )
    assert send.call_count == 4
    assert [getattr(metadata, field).data_ptr() for field in fields] == pointers
    assert all(torch.all(getattr(metadata, field) == 7) for field in fields)
    assert torch.equal(metadata.physical_to_logical_map_cpu, tensors[0])
    assert torch.equal(metadata.logical_to_all_physical_map_cpu, tensors[1])
