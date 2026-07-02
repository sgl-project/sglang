from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed.device_communicators import triton_symm_mem_ag


def test_multimem_all_gatherer_skips_cross_node_tp_group():
    gatherer = triton_symm_mem_ag.MultimemAllGatherer(max_tokens=128)
    tp_group = SimpleNamespace(
        world_size=2,
        cpu_group=object(),
        device_group=object(),
        rank_in_group=0,
    )
    x = torch.empty((1, triton_symm_mem_ag._NUMEL_PER_THREAD), dtype=torch.bfloat16)

    with (
        patch("sglang.srt.distributed.get_tp_group", return_value=tp_group),
        patch(
            "sglang.srt.distributed.parallel_state.in_the_same_node_as",
            return_value=[True, False],
        ),
        patch.object(triton_symm_mem_ag, "create_state") as create_state,
    ):
        assert gatherer._build(x) is None
        create_state.assert_not_called()


def test_multimem_all_gatherer_keeps_single_node_tp_group():
    gatherer = triton_symm_mem_ag.MultimemAllGatherer(max_tokens=128)
    tp_group = SimpleNamespace(
        world_size=2,
        cpu_group=object(),
        device_group=object(),
        rank_in_group=0,
    )
    x = torch.empty((1, triton_symm_mem_ag._NUMEL_PER_THREAD), dtype=torch.bfloat16)
    state = SimpleNamespace(symm_mem_hdl=SimpleNamespace(multicast_ptr=1))

    with (
        patch("sglang.srt.distributed.get_tp_group", return_value=tp_group),
        patch(
            "sglang.srt.distributed.parallel_state.in_the_same_node_as",
            return_value=[True, True],
        ),
        patch.object(triton_symm_mem_ag, "create_state", return_value=state),
    ):
        assert gatherer._build(x) is state
