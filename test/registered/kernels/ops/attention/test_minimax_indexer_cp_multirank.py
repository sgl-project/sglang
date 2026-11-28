"""On SM90, indexer CP over real ranks must select exactly the native selector's blocks,
with its gathers on multimem stores and, past the multimem row cap, on the communicator.

Run with ``python <this file>``; it relaunches itself under torchrun on 4 and 8 GPUs.
"""

import atexit
import os

import pytest
import torch

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (
    flash_decode_with_topk_idx,
)
from sglang.srt.distributed import parallel_state as ps
from sglang.srt.layers.attention.minimax_sparse_ops import indexer_cp
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

register_cuda_ci(est_time=120, stage="extra-b", runner_config="8-gpu-h200")

BLOCK, TOPK, HEADS, DIM = 128, 16, 4, 128
INIT_BLOCKS, LOCAL_BLOCKS = 0, 1


@cache_once
def _cp():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    ps.init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    publish_build_topology(tp_size=world, world_rank=rank)
    ps.initialize_model_parallel()
    atexit.register(ps.destroy_distributed_environment)
    atexit.register(ps.destroy_model_parallel)
    cp = indexer_cp.MiniMaxIndexerCP(ps.get_tp_group())
    cp.warmup()
    return cp


def _inputs(cp, batch, max_len, dtype, calls):
    """Per-call queries and slots; calls read disjoint, non-contiguous request rows."""
    gen = torch.Generator(device="cuda").manual_seed(20261107)
    rows = (calls + 1) * batch + 1
    # Shuffled pages, so a rank's strided blocks never land on contiguous memory.
    pages = torch.randperm(rows * max_len // BLOCK, generator=gen, device="cuda")
    table = (
        (pages.to(torch.int32)[:, None] * BLOCK + torch.arange(BLOCK, device="cuda"))
        .reshape(rows, max_len)
        .to(torch.int32)
    )
    cache = torch.randn(
        (rows * max_len, 1, DIM), generator=gen, device="cuda", dtype=torch.bfloat16
    ).to(dtype)
    q = torch.randn(
        (calls, batch, HEADS, DIM), generator=gen, device="cuda", dtype=torch.bfloat16
    )
    head = cp.rank * HEADS // cp.world
    qs = [q[call, :, head : head + 1].contiguous() for call in range(calls)]
    stride = (calls + 1) * torch.arange(batch, device="cuda")
    slot_sets = [(call + 1 + stride).flip(0) for call in range(calls)]
    lengths = max_len - (37 * torch.arange(batch, device="cuda")) % 200
    return qs, cache, table, slot_sets, lengths


def _native(q, cache, table, slots, lengths, max_len):
    return flash_decode_with_topk_idx(
        q=q,
        sink=None,
        k_cache=cache,
        v_cache=None,
        req_to_token=table,
        seq_lens=lengths,
        max_seqlen=max_len,
        slot_ids=slots,
        block_size=BLOCK,
        topk=TOPK,
        init_blocks=INIT_BLOCKS,
        local_blocks=LOCAL_BLOCKS,
        disable_index_value=True,
        page_size=BLOCK,
    )[1]


def _multimem_gathers(monkeypatch):
    calls = []
    inner = indexer_cp._multimem_all_gather

    def counted(state, local):
        calls.append(local.shape)
        return inner(state, local)

    monkeypatch.setattr(indexer_cp, "_multimem_all_gather", counted)
    return calls


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_multimem_gathers_select_native_blocks(dtype, monkeypatch):
    cp = _cp()
    if cp.multimem is None:
        pytest.skip("no NVLink multicast mapping for this group")
    batch, max_len = 4, 32768
    qs, cache, table, slot_sets, lengths = _inputs(cp, batch, max_len, dtype, calls=3)
    calls = _multimem_gathers(monkeypatch)

    def run_all():
        # Back-to-back calls reuse both symmetric buffers, as consecutive layers do.
        return [
            cp(q, cache, table, slots, lengths, max_len, INIT_BLOCKS, LOCAL_BLOCKS)
            for q, slots in zip(qs, slot_sets)
        ]

    def check(outs):
        for q, slots, out in zip(qs, slot_sets, outs):
            assert torch.equal(out, _native(q, cache, table, slots, lengths, max_len))

    check(run_all())
    assert len(calls) == 2 * len(qs)
    with cp.group.graph_capture() as ctx:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=ctx.stream):
            outs = run_all()
    for shrink in (300, 4097):
        offsets = (37 * torch.arange(batch, device="cuda")) % 200
        lengths.copy_(max_len - shrink - offsets)
        graph.replay()
        torch.cuda.synchronize()
        check(outs)


def test_batch_past_multimem_cap_uses_communicator(monkeypatch):
    cp = _cp()
    batch, max_len = indexer_cp.MULTIMEM_MAX_ROWS + 1, 2048
    qs, cache, table, slot_sets, lengths = _inputs(
        cp, batch, max_len, torch.bfloat16, calls=1
    )
    calls = _multimem_gathers(monkeypatch)
    out = cp(
        qs[0], cache, table, slot_sets[0], lengths, max_len, INIT_BLOCKS, LOCAL_BLOCKS
    )
    assert torch.equal(
        out, _native(qs[0], cache, table, slot_sets[0], lengths, max_len)
    )
    assert not calls


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4, 8))
