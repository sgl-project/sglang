"""E8M0 expert scales must survive EPLB migration without numeric conversion."""

import os
import random
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.eplb import expert_location_updater
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=45, suite="base-a-test-cpu")


def _expert_weights(mapping, rank, device, layout):
    local_ids = torch.tensor(mapping[rank * 4 : (rank + 1) * 4], device=device)
    width = 5 if layout == "strided_byte" else 128
    bits = (
        torch.arange(2 * width, device=device).reshape(1, 2, width)
        + local_ids.reshape(-1, 1, 1) * 17
    ).to(torch.uint8)
    scales = bits.view(torch.float8_e8m0fnu)
    if layout != "contiguous":
        # Padded domain-major storage: each expert's two domains have a gap.
        storage = torch.empty((2, 5, width), dtype=scales.dtype, device=device)
        scales = storage[:, :4].transpose(0, 1).copy_(scales)
    return [local_ids.clone(), local_ids.float().reshape(-1, 1), scales]


def _run_rebalance_regression(rank, world_size, init_method, device_type="cpu"):
    device = torch.device(device_type, rank) if device_type == "cuda" else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl" if device_type == "cuda" else "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    batch_isend_irecv = dist.batch_isend_irecv

    def nccl_compatible_batch(ops):
        # Gloo accepts E8M0, but ProcessGroupNCCL has no mapping for it. Enforce
        # that external restriction while retaining real CPU data transfers.
        for op in ops:
            if op.tensor.dtype == torch.float8_e8m0fnu:
                raise TypeError("NCCL P2P does not support Float8_e8m0fnu")
        return batch_isend_irecv(ops)

    try:
        # Initialize NCCL collectively before a migration involves only a subset
        # of ranks (some ranks already hold the requested expert replicas).
        dist.barrier(device_ids=[rank] if device_type == "cuda" else None)
        # Every logical expert exists, with extra physical slots for replicas.
        initial = list(range(world_size * 3)) + list(range(world_size))
        local_reorder = [
            expert
            for start in range(0, len(initial), 4)
            for expert in reversed(initial[start : start + 4])
        ]
        cross_rank = initial[4:] + initial[:4]
        # Rank 0 must receive expert 4 once, then copy it into another slot.
        replicas = (
            [4, 4]
            + [expert for expert in range(world_size * 3) if expert != 4]
            + [0] * (world_size - 1)
        )
        rng = random.Random(0)
        shuffled = initial.copy()
        rng.shuffle(shuffled)
        for layout in ("contiguous", "strided_int64", "strided_byte"):
            for num_gpu_per_node in (world_size, world_size // 2):
                old = initial
                weights = _expert_weights(old, rank, device, layout)
                buffers = expert_location_updater.create_temp_buffers(weights)
                for new in (
                    local_reorder,
                    cross_rank,
                    initial,
                    replicas,
                    shuffled,
                    initial,
                ):
                    # CUDA runs exercise NCCL directly; CPU CI models only
                    # NCCL's unsupported-dtype boundary, not the migration.
                    transfer = (
                        batch_isend_irecv
                        if device_type == "cuda"
                        else nccl_compatible_batch
                    )
                    with patch.object(dist, "batch_isend_irecv", transfer):
                        expert_location_updater.update_expert_weights_single_layer(
                            routed_experts_weights=weights,
                            temp_buffers=buffers,
                            old_physical_to_logical_map=old,
                            new_physical_to_logical_map=new,
                            num_local_physical_experts=4,
                            num_gpu_per_node=num_gpu_per_node,
                            rank=rank,
                            world_size=world_size,
                        )
                    expected = _expert_weights(new, rank, device, layout)
                    for actual, wanted in zip(weights, expected, strict=True):
                        assert actual.dtype == wanted.dtype
                        assert torch.equal(
                            actual.view(torch.uint8), wanted.view(torch.uint8)
                        ), (rank, layout, num_gpu_per_node, old, new)
                    old = new
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestExpertLocationUpdater(CustomTestCase):
    def test_e8m0_rebalance_preserves_expert_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            init_method = Path(directory, "rendezvous").as_uri()
            with patch.dict(os.environ, {"GLOO_SOCKET_IFNAME": "lo"}):
                mp.spawn(
                    _run_rebalance_regression,
                    args=(2, init_method),
                    nprocs=2,
                    join=True,
                )


if __name__ == "__main__":
    unittest.main()
