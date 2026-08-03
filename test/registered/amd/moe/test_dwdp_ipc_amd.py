import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-b-test-2-gpu-large-amd")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _ipc_worker(rank: int, world_size: int, port: int):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        local = (
            torch.arange(32, device=f"cuda:{rank}", dtype=torch.int32).reshape(4, 8)
            + rank * 1000
        ).contiguous()
        metadata = {
            "handle": local.untyped_storage()._share_cuda_(),
            "shape": tuple(local.shape),
            "stride": tuple(local.stride()),
            "storage_offset": local.storage_offset(),
            "dtype": local.dtype,
        }
        all_metadata = [None] * world_size
        dist.all_gather_object(all_metadata, metadata)

        peer_rank = (rank + 1) % world_size
        peer_metadata = all_metadata[peer_rank]
        redirected_handle = (rank,) + tuple(peer_metadata["handle"])[1:]
        with torch.cuda.device(rank):
            peer_storage = torch.UntypedStorage._new_shared_cuda(*redirected_handle)
            peer = torch.empty(
                0,
                dtype=peer_metadata["dtype"],
                device=f"cuda:{rank}",
            ).set_(
                peer_storage,
                storage_offset=peer_metadata["storage_offset"],
                size=peer_metadata["shape"],
                stride=peer_metadata["stride"],
            )
            copied = peer.clone()
        torch.cuda.synchronize(rank)

        expected = (
            torch.arange(32, device=f"cuda:{rank}", dtype=torch.int32).reshape(4, 8)
            + peer_rank * 1000
        )
        torch.testing.assert_close(copied, expected, rtol=0, atol=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.version.hip is None,
    reason="ROCm HIP IPC test",
)
def test_rocm_ipc_peer_tensor_round_trip():
    world_size = int(os.environ.get("DWDP_TEST_WORLD_SIZE", "2"))
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} ROCm GPUs")
    mp.spawn(
        _ipc_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )
