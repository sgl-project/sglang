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

register_cuda_ci(est_time=90, stage="extra-b", runner_config="4-gpu-h100")

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


def _inputs(cp, batch, max_len, dtype):
    gen = torch.Generator(device="cuda").manual_seed(20261107)
    # Spare rows and shuffled pages, so neither slots nor a rank's blocks are contiguous.
    rows = 3 * batch
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
        (batch, HEADS, DIM), generator=gen, device="cuda", dtype=torch.bfloat16
    )
    head = cp.rank * HEADS // cp.world
    slots = (batch + 1 + 2 * torch.arange(batch, device="cuda")).flip(0)
    lengths = max_len - (37 * torch.arange(batch, device="cuda")) % 200
    return q[:, head : head + 1].contiguous(), cache, table, slots, lengths


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
    max_len = 32768
    q, cache, table, slots, lengths = _inputs(cp, 4, max_len, dtype)
    calls = _multimem_gathers(monkeypatch)

    def run():
        return cp(q, cache, table, slots, lengths, max_len, INIT_BLOCKS, LOCAL_BLOCKS)

    assert torch.equal(run(), _native(q, cache, table, slots, lengths, max_len))
    assert len(calls) == 2
    with cp.group.graph_capture() as ctx:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=ctx.stream):
            out = run()
    # Replays read the symmetric buffers the eager calls also wrote.
    for shrink in (300, 4097):
        lengths.copy_(max_len - shrink - (37 * torch.arange(4, device="cuda")) % 200)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, _native(q, cache, table, slots, lengths, max_len))


def test_batch_past_multimem_cap_uses_communicator(monkeypatch):
    cp = _cp()
    batch, max_len = indexer_cp.MULTIMEM_MAX_ROWS + 1, 2048
    q, cache, table, slots, lengths = _inputs(cp, batch, max_len, torch.bfloat16)
    calls = _multimem_gathers(monkeypatch)
    out = cp(q, cache, table, slots, lengths, max_len, INIT_BLOCKS, LOCAL_BLOCKS)
    assert torch.equal(out, _native(q, cache, table, slots, lengths, max_len))
    assert not calls


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4, 8))
