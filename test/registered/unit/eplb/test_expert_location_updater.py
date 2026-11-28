"""Expert tensors must survive EPLB migration without numeric conversion."""

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

register_cpu_ci(est_time=75, suite="base-a-test-cpu")

_WEIGHT_DTYPES = (
    torch.float8_e8m0fnu,
    torch.int16,
    torch.uint16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.uint8,
    torch.int8,
    torch.int32,
    torch.int64,
    torch.uint32,
    torch.uint64,
)
# NCCL has no E8M0 or 16-bit integer datatype. Gloo does not enforce this boundary.
_UNSUPPORTED_NCCL_DTYPES = (torch.float8_e8m0fnu, torch.int16, torch.uint16)


def _expert_weights(mapping, rank, device, layout):
    local_ids = torch.tensor(mapping[rank * 4 : (rank + 1) * 4], device=device)
    width = 5 if layout == "strided_byte" else 128
    weights = [local_ids.clone()]
    for dtype in _WEIGHT_DTYPES:
        # Include arbitrary encodings (including NaNs), so numeric conversions
        # cannot silently pass a value-only comparison.
        itemsize = dtype.itemsize
        bits = (
            torch.arange(2 * width * itemsize, device=device).reshape(
                1, 2, width * itemsize
            )
            + local_ids.reshape(-1, 1, 1) * 17
        ).to(torch.uint8)
        tensor = bits.view(dtype)
        if layout == "transposed":
            tensor = tensor.transpose(-1, -2)
        elif layout != "contiguous":
            # Padded domain-major storage: each expert's two domains have a gap.
            storage = torch.empty((2, 5, width), dtype=dtype, device=device)
            tensor = storage[:, :4].transpose(0, 1).copy_(tensor)
        weights.append(tensor)
    return weights


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
        # Retain real Gloo transfers while modeling NCCL's unsupported-dtype
        # boundary. This catches missing send and receive reinterpretations.
        for op in ops:
            if op.tensor.dtype in _UNSUPPORTED_NCCL_DTYPES:
                raise TypeError(f"NCCL P2P does not support {op.tensor.dtype}")
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
                        ), (
                            rank,
                            actual.dtype,
                            layout,
                            num_gpu_per_node,
                            old,
                            new,
                        )
                    old = new
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestExpertLocationUpdater(CustomTestCase):
    def test_p2p_view_aliases_expert_storage(self):
        # Receive views must update the original buffers even for transposed
        # tensors; a cast or clone would leave the expert's storage unchanged.
        for layout in ("contiguous", "strided_int64", "strided_byte", "transposed"):
            for tensor in _expert_weights(list(range(8)), 0, "cpu", layout):
                with self.subTest(dtype=tensor.dtype, layout=layout):
                    before = tensor.contiguous().view(torch.uint8).clone()
                    view = expert_location_updater._p2p_tensor_view(tensor)
                    self.assertEqual(view.element_size(), tensor.element_size())
                    self.assertEqual(view.shape, tensor.shape)
                    self.assertEqual(view.stride(), tensor.stride())
                    self.assertEqual(view.storage_offset(), tensor.storage_offset())
                    self.assertEqual(view.data_ptr(), tensor.data_ptr())
                    self.assertTrue(
                        torch.equal(view.contiguous().view(torch.uint8), before)
                    )
                    view.zero_()
                    self.assertEqual(
                        torch.count_nonzero(
                            tensor.contiguous().view(torch.uint8)
                        ).item(),
                        0,
                    )

    def test_rebalance_preserves_expert_bytes_across_dtypes(self):
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
