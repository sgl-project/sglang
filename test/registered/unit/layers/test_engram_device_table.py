"""Sharded device embeddings must reconstruct the checkpoint lookup on each rank."""

import os
import socket
import unittest
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="2-gpu-large")

ROWS = 2048
DIM = 256
BLK = 32


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init(rank: int, world: int, port: int, backend: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend=backend,
    )
    initialize_model_parallel(tensor_model_parallel_size=world, backend=backend)


def _teardown() -> None:
    destroy_model_parallel()
    destroy_distributed_environment()


def _checkpoint_table(seed: int = 0, rows: int = ROWS):
    """The checkpoint tensors as the loader hands them over: identical on every rank."""
    g = torch.Generator().manual_seed(seed)
    weight = (torch.randn(rows, DIM, generator=g) * 3).to(torch.float8_e4m3fn)
    scale = torch.randint(
        100, 140, (rows, DIM // BLK), dtype=torch.uint8, generator=g
    ).view(torch.float8_e8m0fnu)
    # Exponent byte zero represents 2**-127, not zero. Keep it in a real lookup.
    weight.view(torch.uint8)[0, :BLK] = (
        torch.tensor(1.0).to(torch.float8_e4m3fn).view(torch.uint8)
    )
    scale.view(torch.uint8)[0, 0] = 0
    return weight, scale


def _reference(weight, scale, ids):
    ids = ids.cpu()
    rows = weight[ids].float().unflatten(-1, (-1, BLK))
    return (rows * scale[ids].float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


def _build(rows: int):
    from sglang.srt.layers.engram import EngramEmbedding

    with (
        envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(False),
        torch.device("cuda"),
    ):
        return EngramEmbedding(rows, DIM, layer_id=1)


def _load(embed, weight, scale) -> None:
    embed.weight.weight_loader(embed.weight, weight)
    embed.scale.weight_loader(embed.scale, scale)
    embed.finish_load()


def _tp_worker(rank: int, world: int, port: int) -> None:
    torch.cuda.set_device(rank)
    _init(rank, world, port, "nccl")
    try:
        for rows in (2049, 2048, 1):
            weight, scale = _checkpoint_table(rows=rows)
            with (
                get_parallel().override(tp_size=world, tp_rank=rank),
                patch("sglang.srt.layers.engram.get_attention_dp_size", return_value=1),
            ):
                embed = _build(rows)
                # Each rank loads its own range, including uneven or empty shards.
                _load(embed, weight, scale)
                ids = torch.arange(rows, device="cuda", dtype=torch.int64).view(-1, 1)
                out = embed(ids)
            torch.cuda.synchronize()
            assert torch.equal(out.cpu(), _reference(weight, scale, ids)), (
                f"rank {rank}, rows {rows}: incorrect lookup"
            )
            assert out[0, 0, 0].item() == 2**-127, "zero exponent was decoded as zero"
            torch.distributed.barrier()
    finally:
        _teardown()


class TestEngramDeviceTable(CustomTestCase):
    def test_sharded_lookup_matches_checkpoint(self):
        world = min(2, torch.cuda.device_count())
        if world < 2:
            self.skipTest("needs two GPUs to run two TP ranks")
        mp.spawn(_tp_worker, args=(world, _free_port()), nprocs=world, join=True)


if __name__ == "__main__":
    unittest.main()
