"""Eager aiter collectives on 8 ranks: an input the producer wrote into the registered pool
(GroupCoordinator.ca_input_buffer) gives the same bytes as the copied-in input (ROCm).

Each collective runs twice on the same values through GroupCoordinator: once from an
ordinary tensor (aiter copies it into its pool: custom_all_reduce / reduce_scatter
registered=False) and once from the pool base (all_reduce registered_input=True /
reduce_scatter registered=True), at the Muse Spark expert-output shapes:
  AR  all_reduce of [rows, 4096] bf16 (TP), also byte-identical across ranks;
  RS  reduce_scatter_tensor of [rows, 4096] bf16 (prefill sequence parallel).
Over aiter's eager cutoff the all-reduce would not take the custom all-reduce, so
ca_input_buffer must refuse.
"""

import multiprocessing
import socket
import time
import unittest
import zlib

import torch
import torch.distributed as dist

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=90, stage="stage-c", runner_config="large-8-gpu-amd-mi35x")

WORLD_SIZE = 8
HIDDEN = 4096


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _count_calls(ca, calls):
    def wrap(name, key):
        fn = getattr(ca, name)

        def counted(*args, **kwargs):
            calls.append(key(kwargs))
            return fn(*args, **kwargs)

        setattr(ca, name, counted)

    wrap("all_reduce", lambda kw: ("all_reduce", kw.get("registered_input", False)))
    wrap("custom_all_reduce", lambda kw: ("custom_all_reduce",))
    wrap("reduce_scatter", lambda kw: ("reduce_scatter", kw.get("registered", False)))


def _run(rank: int, port: int) -> None:
    from sglang.srt.distributed import parallel_state as ps

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    ps.init_distributed_environment(
        world_size=WORLD_SIZE,
        rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        local_rank=rank,
        backend="nccl",
    )
    group = ps.init_model_parallel_group(
        [list(range(WORLD_SIZE))], rank, "nccl", group_name="registered_input"
    )
    ca = group.ca_comm
    assert ca is not None and not ca.disabled, "no aiter custom all-reduce"
    assert group._ca_input_pool() is not None, "no aiter registered input pool"
    calls = []
    _count_calls(ca, calls)

    def check(tag, rows, run, copy_call, registered_call):
        seed = zlib.crc32(f"{tag}{rows}".encode()) + 7919 * rank
        generator = torch.Generator(device=device).manual_seed(seed)
        src = (
            torch.randn(rows, HIDDEN, generator=generator, device=device) * 0.5
        ).bfloat16()
        del calls[:]
        expected = run(src.clone())
        # custom_all_reduce itself calls all_reduce(registered_input=False)
        assert copy_call in calls and registered_call not in calls, (tag, rows, calls)
        buffer = group.ca_input_buffer((rows, HIDDEN), torch.bfloat16)
        assert buffer is not None, (tag, rows, "refused")
        buffer.copy_(src)
        del calls[:]
        got = run(buffer)
        assert calls == [registered_call], (tag, rows, calls)
        assert torch.equal(got.view(torch.int16), expected.view(torch.int16)), (
            tag,
            rows,
        )
        return got

    for rows in (8, 64, 512, 2048, 4096):
        out = check(
            "AR",
            rows,
            group.all_reduce,
            ("custom_all_reduce",),
            ("all_reduce", True),
        )
        checksums = [None] * WORLD_SIZE
        dist.all_gather_object(
            checksums, zlib.crc32(out.cpu().view(torch.int16).numpy())
        )
        assert len(set(checksums)) == 1, ("AR differs across ranks", rows)

    def reduce_scatter(x):
        out = x.new_empty((x.shape[0] // WORLD_SIZE, HIDDEN))
        group.reduce_scatter_tensor(out, x)
        return out

    for rows in (64, 512, 4096):
        check(
            "RS",
            rows,
            reduce_scatter,
            ("reduce_scatter", False),
            ("reduce_scatter", True),
        )

    over = ca._car_max_size // (2 * HIDDEN) + 16
    assert group.ca_input_buffer((over, HIDDEN), torch.bfloat16) is None
    torch.cuda.synchronize()
    dist.barrier()
    ca.close()
    group.destroy()
    ps.destroy_distributed_environment()


class TestAiterRegisteredInput(CustomTestCase):
    @unittest.skipUnless(
        is_hip() and torch.cuda.device_count() >= WORLD_SIZE, "needs 8 ROCm GPUs"
    )
    def test_registered_input_matches_copy_in(self):
        port = _get_open_port()
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(target=_run, args=(rank, port))
            for rank in range(WORLD_SIZE)
        ]
        for process in processes:
            process.start()
        deadline = time.monotonic() + 600
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        if any(process.is_alive() for process in processes):
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
            self.fail("registered input test timed out")
        for rank, process in enumerate(processes):
            self.assertEqual(process.exitcode, 0, f"rank {rank} failed")


if __name__ == "__main__":
    unittest.main()
