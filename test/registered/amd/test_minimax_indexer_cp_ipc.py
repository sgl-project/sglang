"""The IPC candidate exchange must deliver each head's native NVFP4 verify selection
on four ranks, for DSpark's 8-row and EAGLE's 4-row chains, tied scores and rows shorter
than top-k, across call counters and their wrap, row-count changes and graph replay."""

import multiprocessing
import types
import unittest

import torch
import torch.distributed as dist

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.srt.utils.network import get_open_port
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# backend-specific: the exchange is gfx950 HIP IPC; the suite's MI35x box supplies 4 GPUs
register_amd_ci(est_time=60, suite="stage-c-test-large-8-gpu-amd-mi35x")

WORLD = 4
BLOCK, TOPK, HEADS, DIM = 128, 16, 4, 128
SM_SCALE, K_SCALE = 0.07, 0.5
CALLS = 5
# (requests, draft rows, tied pages, init blocks, local blocks): init != local so a
# swapped argument shows; the second case also has rows shorter than top-k
CASES = ((64, 8, False, 2, 1), (17, 4, True, 0, 1))


def _nvfp4_verify_inputs(prefixes, num_draft, seed, tied_pages=False):
    """DSpark verify rows over a shuffled NVFP4 index-K pool of whole 128-token pages.

    With ``tied_pages`` most pages copy page 1, so most blocks tie.
    """
    from sglang.kernels.ops.attention.minimax_sparse.common.nvfp4 import Nvfp4Cache
    from sglang.test.quant_ref_utils import nvfp4_kv_quantize

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(seed)
    max_len = max(prefixes) + num_draft
    pages_per_req = -(-max_len // BLOCK)
    num_pages = len(prefixes) * pages_per_req
    pages = torch.randperm(num_pages, device=device, generator=gen) + 1
    offs = torch.arange(BLOCK, device=device)
    table = (
        (pages.view(len(prefixes), pages_per_req, 1) * BLOCK + offs)
        .view(len(prefixes), -1)[:, :max_len]
        .to(torch.int32)
    )
    k = torch.randn((num_pages + 1) * BLOCK, DIM, device=device, generator=gen)
    if tied_pages:
        k_pages = k.view(num_pages + 1, BLOCK * DIM)
        rare = torch.arange(num_pages + 1, device=device) % 13 == 0
        k_pages.copy_(k_pages[1 + rare.long()])
    payload, scales = nvfp4_kv_quantize(
        k.to(torch.bfloat16), torch.ones(1, device=device)
    )
    cache = Nvfp4Cache(
        payload=payload.view(num_pages + 1, BLOCK, DIM // 2),
        scales=scales.view(num_pages + 1, BLOCK, DIM // 16),
    )
    rows = len(prefixes) * num_draft
    q = torch.randn(rows, HEADS, DIM, device=device, generator=gen).to(torch.bfloat16)
    lengths = torch.tensor(
        [p + j + 1 for p in prefixes for j in range(num_draft)],
        dtype=torch.int64,
        device=device,
    )
    slots = torch.arange(len(prefixes), device=device).repeat_interleave(num_draft)
    # graph capture sizes the score columns past the real lengths
    return q, cache, table, slots, lengths, max_len + 4 * BLOCK


def _run_exchange(rank: int, port: int) -> None:
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
        for seed, (num_reqs, num_draft, tied, init, local) in enumerate(CASES):
            prefixes = [20000 - 97 * r - 1000 * seed for r in range(num_reqs)]
            if num_reqs > 5:
                prefixes[3], prefixes[5] = 3 * BLOCK - 2, 777  # <= top-k blocks
            q, cache, table, slots, lengths, _ = _nvfp4_verify_inputs(
                prefixes, num_draft, seed=seed, tied_pages=tied
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
                    num_draft=num_draft,
                    block_size=BLOCK,
                    topk=TOPK,
                    init_blocks=init,
                    local_blocks=local,
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
                    q_heads, cache, table, lengths, max_len, slots, num_draft,
                    BLOCK, TOPK, init, local, SM_SCALE * K_SCALE, exchange,
                )  # fmt: skip

            # seed 1 starts two calls short of the counters' wrap at 510
            if seed == 1:
                dist.barrier()
                exchange.write_calls.fill_(508)
                exchange.merge_calls.fill_(508)
                torch.cuda.synchronize()
                dist.barrier()
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


def _spawn(target, port):
    context = multiprocessing.get_context("spawn")
    procs = [context.Process(target=target, args=(rank, port)) for rank in range(WORLD)]
    for proc in procs:
        proc.start()
    return procs


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "MiniMax indexer CP is gfx950-only"
)
class TestMiniMaxIndexerCPExchange(CustomTestCase):
    def test_exchange_matches_the_native_selector_on_four_ranks(self):
        if torch.cuda.device_count() < WORLD:
            self.skipTest(f"needs {WORLD} GPUs")
        procs = _spawn(_run_exchange, get_open_port())
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
