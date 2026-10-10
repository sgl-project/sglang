"""Two-rank CPU check for a missing participant before custom-group metadata exchange."""

import socket
import time
from datetime import timedelta

import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.distributed.parallel_state import create_custom_parallel_group


def run_rank(rank: int, port: int) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    metadata_group = dist.new_group(
        [0, 1], backend="gloo", timeout=timedelta(seconds=15)
    )
    normal_group = create_custom_parallel_group(
        [0, 1],
        backend="gloo",
        metadata_group=metadata_group,
        timeout=timedelta(seconds=3),
    )
    assert normal_group is not None
    print(f"rank={rank} normal_group=ok", flush=True)
    dist.barrier(group=metadata_group)

    if rank == 1:
        time.sleep(7)
    started = time.monotonic()
    try:
        create_custom_parallel_group(
            [0, 1],
            backend="gloo",
            metadata_group=metadata_group,
            timeout=timedelta(seconds=3),
        )
    except RuntimeError as exc:
        elapsed = time.monotonic() - started
        print(
            f"rank={rank} missing_participant=detected elapsed={elapsed:.2f}s "
            f"error={str(exc).splitlines()[0]}",
            flush=True,
        )
        if rank == 0:
            assert 2.5 <= elapsed < 6
            assert "Rank 1" in str(exc) or "Ranks 1" in str(exc)
    else:
        raise AssertionError(
            f"rank {rank}: missing-participant call unexpectedly passed"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    mp.start_processes(run_rank, args=(port,), nprocs=2, start_method="fork")
