"""TP4 all-reduce fused with hc_post on AITER's custom all-reduce buffers: bitwise equal to
the unfused all-reduce followed by hc_post, eagerly and under CUDA-graph capture."""

import multiprocessing
import socket
import time
import unittest

import torch
import torch.distributed as dist

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# gfx950 only: DeepSeek-V4.1 fuses this all-reduce on MI35x
register_amd_ci(est_time=60, stage="stage-c", runner_config="large-8-gpu-amd-mi35x")

WORLD_SIZE = 4
HIDDEN = 5120
HC = 4


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _reference(inputs, residual, post, comb):
    # the all-reduce sums the ranks in order in fp32 and rounds to bf16; hc_post then
    # rounds every multiply and add in fp32
    reduced = inputs[0].float()
    for x in inputs[1:]:
        reduced = reduced + x.float()
    reduced = reduced.to(torch.bfloat16).float()
    out = []
    for o in range(HC):
        mixed = reduced * post[:, o, None]
        for i in range(HC):
            mixed = mixed + residual[:, i].float() * comb[:, i, o, None]
        out.append(mixed)
    return torch.stack(out, 1).to(torch.bfloat16)


def _run(rank: int, port: int) -> None:
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    from sglang.kernels.ops.communication.all_reduce_mhc_hip import (
        all_reduce_mhc_post,
    )

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    communicator = CustomAllreduce(dist.group.WORLD, device)
    try:
        assert not communicator.disabled
        for rows in (1, 3, 8):
            generator = torch.Generator(device=device).manual_seed(rows)
            inputs = [
                torch.randn(rows, HIDDEN, generator=generator, device=device).bfloat16()
                for _ in range(WORLD_SIZE)
            ]
            residual = torch.randn(
                rows, HC, HIDDEN, generator=generator, device=device
            ).bfloat16()
            post = torch.rand(rows, HC, generator=generator, device=device) * 2
            comb = torch.rand(rows, HC, HC, generator=generator, device=device)
            expected = _reference(inputs, residual, post, comb)

            out = all_reduce_mhc_post(inputs[rank], residual, post, comb, communicator)
            torch.cuda.synchronize()
            torch.testing.assert_close(out, expected, rtol=0, atol=0)

            # under capture the kernel reads the graph input through registered peer addresses
            static_input = torch.zeros_like(inputs[rank])
            graph = torch.cuda.CUDAGraph()
            with communicator.capture():
                with torch.cuda.graph(graph):
                    graph_out = all_reduce_mhc_post(
                        static_input, residual, post, comb, communicator
                    )
            static_input.copy_(inputs[rank])
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(graph_out, expected, rtol=0, atol=0)
            dist.barrier()
    finally:
        communicator.close()
        dist.destroy_process_group()


class TestAllReduceMhcHip(CustomTestCase):
    @unittest.skipUnless(
        is_gfx95_supported() and torch.cuda.device_count() >= WORLD_SIZE,
        "needs four gfx950 GPUs",
    )
    def test_matches_unfused_all_reduce_and_hc_post(self):
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
            self.fail("all_reduce_mhc_post test timed out")
        for rank, process in enumerate(processes):
            self.assertEqual(process.exitcode, 0, f"rank {rank} failed")


if __name__ == "__main__":
    unittest.main()
