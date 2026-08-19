"""Unit tests for GroupCoordinator.send_tensor_dict / recv_tensor_dict with
`all_gather_exclude` (issue #30015).

When `all_gather_group` is passed, send/recv_tensor_dict send only a 1/N
slice of each tensor and reassemble it on the receiver with an all-gather
across the group. That optimization is only lossless for tensors replicated
across the all-gather group; a TP-sharded tensor (different data per rank)
gets reassembled from different ranks' slices into a corrupted tensor.
`all_gather_exclude` opts such tensors out per key so they are sent whole.

Runs 4 gloo CPU processes with tp=2 pp=2:
    TP groups: [[0, 1], [2, 3]]    PP groups: [[0, 2], [1, 3]]
PP stage-0 ranks (0, 1) send to stage-1 ranks (2, 3).
"""

import os
import unittest

import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, find_available_port

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

WORLD_SIZE = 4


def _make_group(group_ranks, rank, name):
    from sglang.srt.distributed.parallel_state import GroupCoordinator

    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=rank,
        torch_distributed_backend="gloo",
        use_pynccl=False,
        use_pymscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        use_message_queue_broadcaster=False,
        group_name=name,
    )


def _worker(rank, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group(
        backend="gloo", rank=rank, world_size=WORLD_SIZE
    )
    try:
        tp = _make_group([[0, 1], [2, 3]], rank, "tp_test")
        pp = _make_group([[0, 2], [1, 3]], rank, "pp_test")
        tp_rank = tp.rank_in_group
        is_sender = pp.rank_in_group == 0

        # Different data per TP rank vs. identical across the TP group.
        sharded = torch.full((2, 4), float(tp_rank + 1))
        replicated = torch.arange(8, dtype=torch.float32).reshape(2, 4)

        # Round 1: sharded tensor opted out via all_gather_exclude, replicated
        # tensor still using the slice/all-gather optimization. Both must
        # round-trip exactly.
        if is_sender:
            pp.send_tensor_dict(
                {"sharded": sharded, "replicated": replicated, "__msg_type__": "x"},
                all_gather_group=tp,
                all_gather_exclude={"sharded"},
            )
        else:
            recv = pp.recv_tensor_dict(
                all_gather_group=tp, all_gather_exclude={"sharded"}
            )
            assert torch.equal(recv["sharded"], sharded), recv["sharded"]
            assert torch.equal(recv["replicated"], replicated), recv["replicated"]
            assert recv["__msg_type__"] == "x"

        # Round 2: without the exclude, the sharded tensor is reassembled
        # from both TP ranks' slices — the documented corruption of #30015.
        if is_sender:
            pp.send_tensor_dict({"sharded": sharded}, all_gather_group=tp)
        else:
            recv = pp.recv_tensor_dict(all_gather_group=tp)
            corrupted = torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
            assert torch.equal(recv["sharded"], corrupted), recv["sharded"]

        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


class TestSendTensorDictAllGatherExclude(CustomTestCase):
    def test_all_gather_exclude_round_trip(self):
        port = find_available_port(23000)
        mp.spawn(_worker, args=(port,), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    unittest.main()
