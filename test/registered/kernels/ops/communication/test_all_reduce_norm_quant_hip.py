"""Fused TP4 all-reduce + Gemma norm + fp8 quant equals the all-reduce then gemma_add_rmsnorm_fp8."""

import multiprocessing
import time
import unittest

import torch
import torch.distributed as dist

from sglang.srt.utils import is_gfx95_supported
from sglang.srt.utils.network import get_open_port
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# gfx950 only: MiniMax-M3.1 fuses its TP4 decode boundaries on MI35x
register_amd_ci(est_time=60, stage="stage-c", runner_config="large-8-gpu-amd-mi35x")

WORLD_SIZE = 4
HIDDEN = 6144
EPS = 1e-6


def _check(out, expected, case):
    (quantized, residual_out), (ref_quantized, ref_residual) = out, expected
    # the sum is rounded to bf16 before the residual add, as the all-reduce output is
    torch.testing.assert_close(residual_out, ref_residual, rtol=0, atol=0, msg=case)
    # the norm's sum of squares is reduced in another order: a value next to a rounding
    # boundary may land on the neighbouring bf16 value, and its fp8 code with it
    if len(quantized) == 3:
        torch.testing.assert_close(
            quantized[2], ref_quantized[2], rtol=2**-7, atol=0, msg=case
        )
    torch.testing.assert_close(
        quantized[1], ref_quantized[1], rtol=2**-7, atol=0, msg=case
    )
    values = quantized[0].float() * quantized[1]
    ref_values = ref_quantized[0].float() * ref_quantized[1]
    torch.testing.assert_close(values, ref_values, rtol=2**-3, atol=0, msg=case)
    codes, ref_codes = (
        quantized[0].view(torch.uint8),
        ref_quantized[0].view(torch.uint8),
    )
    assert (codes != ref_codes).float().mean().item() < 1e-3, (
        f"{case}: fp8 codes differ"
    )


def _run(rank: int, port: int) -> None:
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    from sglang.kernels.ops.communication.all_reduce_norm_quant_hip import (
        all_reduce_gemma_norm_fp8,
    )
    from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import (
        gemma_add_rmsnorm_fp8,
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
        generator = torch.Generator(device=device).manual_seed(0)
        weight = (
            torch.randn(HIDDEN, generator=generator, device=device) * 0.1
        ).bfloat16()
        # more rows than AITER's 80 signal slots run the grid-stride loop
        for rows in (1, 8, 16, 100):
            inputs = [
                torch.randn(rows, HIDDEN, generator=generator, device=device).bfloat16()
                for _ in range(WORLD_SIZE)
            ]
            residual = (
                torch.randn(rows, HIDDEN, generator=generator, device=device) * 4
            ).bfloat16()
            # the all-reduce sums the ranks in order in fp32 and rounds to bf16
            reduced = inputs[0].float()
            for x in inputs[1:]:
                reduced = reduced + x.float()
            for keep_bf16 in (False, True):
                case = f"{rows=} {keep_bf16=}"
                expected = gemma_add_rmsnorm_fp8(
                    reduced.bfloat16(), residual, weight, EPS, keep_bf16=keep_bf16
                )
                run = lambda x: all_reduce_gemma_norm_fp8(  # noqa: E731
                    x, residual, weight, EPS, communicator, keep_bf16=keep_bf16
                )
                out = run(inputs[rank])
                torch.cuda.synchronize()
                _check(out, expected, case)

                # under capture the kernel reads the graph input through registered
                # peer addresses
                static_input = torch.zeros_like(inputs[rank])
                graph = torch.cuda.CUDAGraph()
                with communicator.capture():
                    with torch.cuda.graph(graph):
                        graph_out = run(static_input)
                static_input.copy_(inputs[rank])
                graph.replay()
                torch.cuda.synchronize()
                _check(graph_out, expected, f"graph {case}")
                dist.barrier()

            # consecutive captured boundaries: each graph input is written right before its
            # boundary while a peer may still read the last one
            sources = [torch.empty_like(inputs[rank]) for _ in range(6)]
            graph = torch.cuda.CUDAGraph()
            with communicator.capture():
                with torch.cuda.graph(graph):
                    graph_outs = []
                    for source in sources:
                        x = source.clone()
                        graph_outs.append(
                            all_reduce_gemma_norm_fp8(
                                x, residual, weight, EPS, communicator
                            )
                        )
            for replay in range(4):
                peers = [
                    [
                        torch.randn(
                            rows, HIDDEN, generator=generator, device=device
                        ).bfloat16()
                        for _ in range(WORLD_SIZE)
                    ]
                    for _ in sources
                ]
                for source, ranks in zip(sources, peers):
                    source.copy_(ranks[rank])
                graph.replay()
                torch.cuda.synchronize()
                for i, ranks in enumerate(peers):
                    reduced = ranks[0].float()
                    for x in ranks[1:]:
                        reduced = reduced + x.float()
                    expected = gemma_add_rmsnorm_fp8(
                        reduced.bfloat16(), residual, weight, EPS
                    )
                    _check(
                        graph_outs[i],
                        expected,
                        f"{rows=} replay {replay} call {i}",
                    )
                dist.barrier()
    finally:
        communicator.close()
        dist.destroy_process_group()


class TestAllReduceNormQuantHip(CustomTestCase):
    @unittest.skipUnless(
        is_gfx95_supported() and torch.cuda.device_count() >= WORLD_SIZE,
        "needs four gfx950 GPUs",
    )
    def test_matches_unfused_all_reduce_and_norm_quant(self):
        port = get_open_port()
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
            self.fail("all_reduce_gemma_norm_fp8 test timed out")
        for rank, process in enumerate(processes):
            self.assertEqual(process.exitcode, 0, f"rank {rank} failed")


if __name__ == "__main__":
    unittest.main()
