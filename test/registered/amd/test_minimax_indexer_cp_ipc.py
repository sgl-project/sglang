"""The IPC candidate exchange must deliver each head's native NVFP4 verify selection
on four ranks, across call counters, row-count changes and CUDA-graph replay."""

import multiprocessing
import socket
import types
import unittest

import torch
import torch.distributed as dist

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# backend-specific: the exchange is gfx950 HIP IPC; the suite's MI35x box supplies 4 GPUs
register_amd_ci(est_time=60, suite="stage-c-test-large-8-gpu-amd-mi35x")

WORLD = 4
NUM_DRAFT, SM_SCALE, K_SCALE = 8, 0.07, 0.5
CALLS = 5


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _run_exchange(rank: int, port: int) -> None:
    from test_minimax_indexer_cp import BLOCK, TOPK, _nvfp4_verify_inputs

    from sglang.kernels.ops.attention.minimax_sparse.decode import verify_topk_idx
    from sglang.kernels.ops.attention.minimax_sparse.decode.verify_topk_gfx950 import (
        CpExchange,
        cp_verify_exchange,
    )

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD,
    )
    try:
        group = types.SimpleNamespace(
            world_size=WORLD, rank_in_group=rank, cpu_group=dist.group.WORLD
        )
        exchange = CpExchange(group, device)
        # every rank builds the same pool; a call's row count changes between cases
        for seed, num_reqs in enumerate((64, 17, 48)):
            prefixes = [20000 - 97 * r - 1000 * seed for r in range(num_reqs)]
            q, cache, table, slots, lengths, _ = _nvfp4_verify_inputs(
                prefixes, NUM_DRAFT, seed=seed
            )
            max_len = table.shape[1]
            # distinct queries per call, so a word left from an earlier call never
            # passes for this one's
            gen = torch.Generator(device=device).manual_seed(100 + seed)
            queries = [q] + [
                torch.randn(q.shape, device=device, generator=gen).to(q.dtype)
                for _ in range(CALLS - 1)
            ]
            natives = [
                verify_topk_idx.verify_topk_idx(
                    q=query[:, rank : rank + 1].contiguous(),
                    k_cache=cache,
                    req_to_token=table,
                    seq_lens=lengths,
                    max_seqlen=max_len,
                    slot_ids=slots,
                    num_draft=NUM_DRAFT,
                    block_size=BLOCK,
                    topk=TOPK,
                    init_blocks=0,
                    local_blocks=1,
                    sm_scale=SM_SCALE,
                    q_scale=None,
                    k_scale=K_SCALE,
                    page_size=BLOCK,
                )
                for query in queries
            ]
            q_heads = torch.empty_like(q.permute(1, 0, 2).contiguous())

            def exchange_call():
                return cp_verify_exchange(
                    q_heads, cache, table, lengths, max_len, slots, NUM_DRAFT,
                    BLOCK, TOPK, 0, 1, SM_SCALE * K_SCALE, exchange,
                )  # fmt: skip

            # more calls than parities, so a stale parity or tag would be read back
            for call, (query, native) in enumerate(zip(queries, natives)):
                q_heads.copy_(query.permute(1, 0, 2))
                out = exchange_call()
                torch.cuda.synchronize()
                assert torch.equal(out, native), (
                    f"rank {rank} reqs {num_reqs} call {call}: "
                    f"{(out != native).sum()} IDs differ"
                )
            if seed == 0:
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    exchange_call()
                    torch.cuda.synchronize()
                    with torch.cuda.graph(graph, stream=stream):
                        replayed = exchange_call()
                for replay, (query, native) in enumerate(zip(queries, natives)):
                    q_heads.copy_(query.permute(1, 0, 2))
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(replayed, native), (
                        f"rank {rank} replay {replay}: "
                        f"{(replayed != native).sum()} IDs differ"
                    )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "MiniMax indexer CP is gfx950-only"
)
class TestMiniMaxIndexerCPExchange(CustomTestCase):
    def test_exchange_matches_the_native_selector_on_four_ranks(self):
        if torch.cuda.device_count() < WORLD:
            self.skipTest(f"needs {WORLD} GPUs")
        context = multiprocessing.get_context("spawn")
        port = _open_port()
        procs = [
            context.Process(target=_run_exchange, args=(rank, port))
            for rank in range(WORLD)
        ]
        for proc in procs:
            proc.start()
        # a buffer that serves a stale line spins the merge forever instead of failing
        for proc in procs:
            proc.join(timeout=600)
        hung = [proc for proc in procs if proc.is_alive()]
        for proc in hung:
            proc.kill()
        self.assertFalse(hung, "an exchange rank hung")
        self.assertEqual([proc.exitcode for proc in procs], [0] * WORLD)


if __name__ == "__main__":
    unittest.main()
