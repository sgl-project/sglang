"""The DP-sharded Engram lookup gives the same rows whether or not it slices the
gathered rows to bound its buffer. CPU only: four gloo ranks, each owning a
quarter of a small fp8 table."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
from sglang.srt.layers import engram
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

WORLD = 4
NUM_ROWS = 64
DIM = 2 * FP8_BLOCK_SIZE
HEADS = 3


def _table():
    gen = torch.Generator().manual_seed(0)
    weight = (torch.rand(NUM_ROWS, DIM, generator=gen) * 4 - 2).to(torch.float8_e4m3fn)
    scale = torch.full((NUM_ROWS, DIM // FP8_BLOCK_SIZE), 2.0).to(torch.float8_e8m0fnu)
    return weight, scale


def _reference(indices, weight, scale):
    rows = weight[indices].float().unflatten(-1, (-1, FP8_BLOCK_SIZE))
    return (rows * scale[indices].float().unsqueeze(-1)).flatten(-2).bfloat16()


def _embedding(rank, weight, scale):
    emb = engram.EngramEmbedding.__new__(engram.EngramEmbedding)
    torch.nn.Module.__init__(emb)
    emb.dim = DIM
    emb.tp_size = WORLD
    emb.row_start = NUM_ROWS * rank // WORLD
    emb.rows = NUM_ROWS // WORLD
    emb.host_table = None
    rows = slice(emb.row_start, emb.row_start + emb.rows)
    emb.weight = torch.nn.Parameter(weight[rows].clone(), requires_grad=False)
    emb.scale = torch.nn.Parameter(scale[rows].clone(), requires_grad=False)
    return emb


def _run_rank(rank, port, counts, max_len, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    weight, scale = _table()
    slot = max(counts) if max_len else None
    offsets = [0]
    for n in counts:
        offsets.append(offsets[-1] + (slot if max_len else n))
    rows = offsets[-1]
    gen = torch.Generator().manual_seed(1)
    all_ids = [
        torch.randint(0, NUM_ROWS, (n, HEADS), generator=gen, dtype=torch.int64)
        for n in counts
    ]
    local_rows = slot if max_len else counts[rank]
    ids = torch.zeros(local_rows, HEADS, dtype=torch.int64)
    ids[: counts[rank]] = all_ids[rank]

    def gather(out, local, batch):
        out.zero_()
        parts = [torch.zeros_like(local) for _ in range(WORLD)] if max_len else None
        if max_len:
            dist.all_gather(parts, local)
            out.copy_(torch.cat(parts))
        else:
            for r, n in enumerate(counts):
                out[offsets[r] : offsets[r] + n] = all_ids[r]

    def all_reduce(t):
        dist.all_reduce(t)
        return t

    def reduce_scatter(out, inp):
        dist.all_reduce(inp)
        out.copy_(inp.narrow(0, rank * out.shape[0], out.shape[0]))

    def scatter(local, glob, batch):
        local.zero_()
        local[: counts[rank]] = glob[offsets[rank] : offsets[rank] + counts[rank]]

    padding = SimpleNamespace(is_max_len=lambda: max_len)
    batch = SimpleNamespace(dp_padding_mode=padding)
    parallel = SimpleNamespace(attn_dp_size=WORLD, attn_cp_size=1)
    emb = _embedding(rank, weight, scale)
    outs = {}
    with mock.patch.multiple(
        engram,
        get_parallel=lambda: parallel,
        get_global_dp_buffer_len=lambda: rows,
        dp_gather_replicate=gather,
        dp_reduce_scatter_tensor=reduce_scatter,
        dp_scatter=scatter,
        tensor_model_parallel_all_reduce=all_reduce,
        is_dp_gatherv_active=lambda: False,
        get_dp_local_info=lambda b: (
            torch.tensor(offsets[rank]),
            torch.tensor(counts[rank]),
        ),
    ):
        for budget in (1 << 30, DIM * HEADS * 2 * 5):  # one slice; 5-row slices
            with mock.patch.object(engram, "_DP_LOOKUP_CHUNK_BYTES", budget):
                outs[budget] = emb._dp_sharded_lookup(ids, batch)[: counts[rank]]
    expected = _reference(all_ids[rank], weight, scale)
    results[rank] = all(torch.equal(o, expected) for o in outs.values())
    dist.destroy_process_group()


class TestEngramDpLookupChunked(CustomTestCase):
    def _check(self, counts, max_len, port):
        manager = mp.Manager()
        results = manager.dict()
        mp.spawn(
            _run_rank, args=(port, counts, max_len, results), nprocs=WORLD, join=True
        )
        self.assertEqual(dict(results), {r: True for r in range(WORLD)})

    def test_max_len_slots_reduce_scatter(self):
        self._check([7, 3, 7, 0], max_len=True, port=29611)

    def test_variable_rows_all_reduce(self):
        self._check([9, 2, 0, 13], max_len=False, port=29612)


if __name__ == "__main__":
    unittest.main()
