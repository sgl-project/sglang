# SPDX-License-Identifier: Apache-2.0
"""Real NCCL layout validation; launch with torchrun on 2 or 4 CUDA ranks."""

import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.layers import usp


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        world = dist.get_world_size()
        assert world in (2, 4)
        group = SimpleNamespace(ulysses_group=dist.group.WORLD, ulysses_shm_handle=None)
        ipc = "sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a_multi.ipc_a2a_multi_ready"
        with (
            torch.no_grad(),
            patch.object(usp, "get_sp_group", return_value=group),
            patch.object(usp, "get_ulysses_parallel_world_size", return_value=world),
            patch.object(usp, "_UNCACHED_STAGING", False),
            patch(ipc, return_value=False),
        ):
            cases = 0
            for rows in (37, 9939):
                for dtype in (torch.bfloat16, torch.float16, torch.float32):
                    torch.manual_seed(42 + rank)
                    packed = torch.randn(rows, 3, 56, 128, device="cuda", dtype=dtype)
                    qkv = tuple(packed[:, i] for i in range(3))
                    local_heads = 56 // world
                    expected = []
                    for tensor in qkv:
                        received = [torch.empty_like(tensor) for _ in range(world)]
                        dist.all_gather(received, tensor.contiguous())
                        expected.append(
                            torch.cat(
                                [
                                    t[:, rank * local_heads : (rank + 1) * local_heads]
                                    for t in received
                                ]
                            )
                        )
                    cached_ptr = None
                    for plane in (False, True):
                        with patch.object(
                            dist, "all_to_all_single", wraps=dist.all_to_all_single
                        ) as collective:
                            actual = usp._usp_input_all_to_all_packed_qkv(
                                *qkv, plane_major=plane
                            )
                        assert collective.call_count == 1
                        for got, reference in zip(actual, expected):
                            assert torch.equal(got, reference), (
                                rank,
                                rows,
                                dtype,
                                plane,
                            )
                            assert got.stride() == (
                                3 * local_heads * 128,
                                128 if plane else 3 * 128,
                                1,
                            )
                        ptr = actual[0].data_ptr()
                        if cached_ptr is not None:
                            assert ptr == cached_ptr, (
                                "same-size layout switch should reuse receive staging"
                            )
                        cached_ptr = ptr
                        cases += 1
            dist.barrier()
            if rank == 0:
                print(
                    f"PASS: world={world}, {cases} cases/rank; real NCCL, bitwise all-gather reference, staging reuse",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
