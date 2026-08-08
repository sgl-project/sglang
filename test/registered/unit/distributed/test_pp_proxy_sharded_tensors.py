"""send/recv_tensor_dict with TP-sharded (non-replicated) tensors (#30015).

The PP proxy-tensor transfer sends a 1/N slice of every tensor and reassembles
it on the receiver with an all-gather over the attention-TP group. That is
lossless only for tensors replicated across the group; a TP-sharded tensor
(different data per rank) reassembles into slices from *different ranks*.

This tests the fix end to end at the primitive level:
  * `send_tensor_dict(..., all_gather_exclude={key})` sends the named tensors
    whole and records that in the wire metadata (`TensorMetadata.send_whole`),
  * `recv_tensor_dict` honors the metadata WITHOUT any extra argument (the
    sender is the single source of truth; the two sides cannot disagree),
  * replicated tensors in the same dict keep using the optimization,
  * without the exclusion the corruption is what #30015 describes (pinned so
    a behavior change is noticed).

Topology: 4 gloo CPU processes, tp=2 x pp=2.
    attention-TP groups: [[0, 1], [2, 3]]     PP groups: [[0, 2], [1, 3]]
Stage-0 ranks (0, 1) send to stage-1 ranks (2, 3).
"""

import os
from datetime import timedelta

import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, find_available_port

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

WORLD = 4
TP_GROUPS = [[0, 1], [2, 3]]
PP_GROUPS = [[0, 2], [1, 3]]


def _coordinator(group_ranks, rank, name):
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


def _run(rank: int, port: int):
    # CPU-only gloo test: hide CUDA so GroupCoordinator doesn't try to select a
    # per-rank device ordinal (world_size here exceeds the GPUs on a dev box).
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group(
        backend="gloo", rank=rank, world_size=WORLD, timeout=timedelta(seconds=60)
    )
    try:
        tp_group = _coordinator(TP_GROUPS, rank, "attn_tp_sharded_test")
        pp_group = _coordinator(PP_GROUPS, rank, "pp_sharded_test")
        tp_rank = tp_group.rank_in_group
        sending_stage = pp_group.rank_in_group == 0

        # v_first-style per-rank shard: every TP rank holds DIFFERENT data.
        # 8 elements, divisible by tp=2, so the all-gather path would engage.
        sharded = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 100.0 * (
            tp_rank + 1
        )
        # hidden_states-style tensor: identical on both TP ranks.
        replicated = torch.linspace(-1.0, 1.0, steps=8).reshape(2, 4)

        # --- with the fix: sharded key excluded on the SENDER only ---------
        if sending_stage:
            pp_group.send_tensor_dict(
                {"v_first": sharded, "hidden": replicated, "tag": "proxy"},
                all_gather_group=tp_group,
                all_gather_exclude={"v_first"},
            )
        else:
            # No all_gather_exclude here on purpose: the receiver must learn
            # the exclusion from the wire metadata alone.
            got = pp_group.recv_tensor_dict(all_gather_group=tp_group)
            torch.testing.assert_close(got["v_first"], sharded, rtol=0, atol=0)
            torch.testing.assert_close(got["hidden"], replicated, rtol=0, atol=0)
            assert got["tag"] == "proxy"

        torch.distributed.barrier()

        # --- without the fix: pin the #30015 corruption ---------------------
        if sending_stage:
            pp_group.send_tensor_dict({"v_first": sharded}, all_gather_group=tp_group)
        else:
            got = pp_group.recv_tensor_dict(all_gather_group=tp_group)
            # Each receiver reassembles half from rank 0's shard and half from
            # rank 1's shard - equal on both receivers, matching neither sender.
            first_half = torch.arange(4, dtype=torch.float32) + 100.0
            second_half = torch.arange(4, 8, dtype=torch.float32) + 200.0
            interleaved = torch.cat([first_half, second_half]).reshape(4, 2)
            torch.testing.assert_close(got["v_first"], interleaved, rtol=0, atol=0)
            assert not torch.equal(got["v_first"], sharded)

        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


class TestPPProxyShardedTensors(CustomTestCase):
    def test_sharded_proxy_tensor_round_trip(self):
        port = find_available_port(24000)
        mp.spawn(_run, args=(port,), nprocs=WORLD, join=True)


if __name__ == "__main__":
    import unittest

    unittest.main()
