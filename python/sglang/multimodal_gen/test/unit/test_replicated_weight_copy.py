# SPDX-License-Identifier: Apache-2.0
"""The replicated host-to-device copy gives every rank the bytes a plain copy gives."""

import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD_SIZE = 2


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _host_tensors() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    tensors = {
        # byte counts that do not split evenly across the ranks
        "bf16_odd": torch.randn(1031, 4099, generator=generator).to(torch.bfloat16),
        "fp32": torch.randn(2048, 1025, generator=generator),
        "fp8": torch.randn(3, 1024, 1367, generator=generator).to(torch.float8_e4m3fn),
        "int64": torch.randint(-(2**40), 2**40, (524289,), generator=generator),
        "bool": torch.rand(4 << 20, generator=generator) > 0.5,
    }
    # a strided view keeps the plain copy path
    tensors["bf16_transposed"] = tensors["bf16_odd"].t()
    return tensors


def _assert_same_bytes(got: torch.Tensor, want: torch.Tensor, name: str) -> None:
    assert got.dtype == want.dtype and got.shape == want.shape, name
    assert torch.equal(
        got.contiguous().reshape(-1).view(torch.uint8),
        want.contiguous().reshape(-1).view(torch.uint8),
    ), name


def _check(rank: int, port: int) -> None:
    from sglang.multimodal_gen.runtime.loader.replicated_weight_copy import (
        replicated_copy_to_device,
        uses_replicated_copy,
    )

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=WORLD_SIZE, device_id=torch.device("cuda", rank)
    )
    try:
        device = torch.device("cuda", rank)
        tensors = _host_tensors()
        for name, tensor in tensors.items():
            if not tensor.is_contiguous():
                continue
            got = replicated_copy_to_device(tensor, device, dist.group.WORLD)
            _assert_same_bytes(got, tensor.to(device), name)
        # strided and small tensors keep the per-rank copy
        assert not uses_replicated_copy(tensors["bf16_transposed"])
        assert not uses_replicated_copy(torch.ones(16))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= WORLD_SIZE,
    f"needs {WORLD_SIZE} GPUs",
)
class TestReplicatedWeightCopy(unittest.TestCase):
    def test_bytes_match_plain_copy(self):
        mp.spawn(_check, args=(_free_port(),), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    unittest.main()
