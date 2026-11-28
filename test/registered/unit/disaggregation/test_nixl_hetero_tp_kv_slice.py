"""CPU tests for the NIXL heterogeneous-TP KV slice path.

The prepared dlists built by ``NixlKVManager._init_hetero_tp_prep_handle`` are
replayed against plain byte buffers laid out like the default MHA pool
(token-major NHD rows), and every decode rank's KV is compared with the heads
it should hold. This covers the cases the existing replication test does not:
both sides replicating KV heads (more ranks than heads on prefill AND decode),
where the head-group count used to evaluate to 0.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


import itertools
import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.nixl.conn import (
    NixlKVManager,
    expand_page_indices_for_slice,
)
from sglang.test.test_utils import CustomTestCase

BYTES_PER_HEAD_TOKEN = 4  # one head of one token, per K or V


class _CaptureAgent:
    def __init__(self):
        self.calls = []  # (peer_name, Nx5 runs)

    def prep_xfer_dlist(self, peer_name, array, mem_kind):
        self.calls.append((peer_name, np.asarray(array)))
        return len(self.calls) - 1


def _heads(tp, rank, num_heads):
    """(first, count) heads held by ``rank``: tp | heads, or heads | tp (replicated)."""
    per_rank = max(1, num_heads // tp)
    replicas = max(1, tp // num_heads)
    return (rank // replicas) * per_rank, per_rank


def _target_prefill_ranks(p_tp, d_tp, d_rank):
    """CommonKVManager._resolve_rank_mapping for non-MLA KV."""
    if d_tp >= p_tp:
        return [d_rank // (d_tp // p_tp)]
    k = p_tp // d_tp
    return list(range(d_rank * k, (d_rank + 1) * k))


class _Memory:
    """A flat byte arena; "pointers" are offsets into it."""

    def __init__(self):
        self.buf = np.zeros(0, dtype=np.uint8)

    def alloc(self, nbytes):
        ptr = len(self.buf) + 64  # leave a gap so overruns land in no region
        self.buf = np.concatenate([self.buf, np.zeros(nbytes + 64, dtype=np.uint8)])
        return ptr

    def view(self, ptr, nbytes):
        return self.buf[ptr : ptr + nbytes]


def _make_prefill(mem, p_tp, rank, num_heads, page_size, num_slots):
    mgr = object.__new__(NixlKVManager)
    mgr.agent = _CaptureAgent()
    mgr.attn_tp_size = p_tp
    mgr.prep_handle_slice_src = None
    mgr.prep_handles_slice_dst = {}
    local_heads = max(1, num_heads // p_tp)
    item_len = page_size * local_heads * BYTES_PER_HEAD_TOKEN
    ptrs = [mem.alloc(num_slots * item_len) for _ in range(2)]  # K, V
    mgr.kv_args = SimpleNamespace(
        gpu_id=0,
        engine_rank=rank,
        page_size=page_size,
        prefill_start_layer=0,
        total_kv_head_num=num_heads,
        kv_head_num=local_heads,
        kv_item_lens=[item_len, item_len],
        kv_data_ptrs=ptrs,
        kv_data_lens=[num_slots * item_len] * 2,
    )
    return mgr


def _make_decode(mem, d_tp, rank, num_heads, page_size, num_slots):
    local_heads = max(1, num_heads // d_tp)
    item_len = page_size * local_heads * BYTES_PER_HEAD_TOKEN
    return SimpleNamespace(
        agent_name=f"decode_{rank}",
        decode_tp_size=d_tp,
        decode_tp_rank=rank,
        dst_kv_item_len=item_len,
        dst_kv_ptrs=[mem.alloc(num_slots * item_len) for _ in range(2)],
        dst_num_slots=num_slots,
        gpu_id=0,
        local_heads=local_heads,
    )


def _fill_logical(mem, mgr, p_tp, num_heads, page_size, pages, logical):
    """Write logical[kv][token][head][byte] into a prefill rank's NHD pages."""
    rank = mgr.kv_args.engine_rank
    h0, hn = _heads(p_tp, rank, num_heads)
    item_len = mgr.kv_args.kv_item_lens[0]
    for kv, ptr in enumerate(mgr.kv_args.kv_data_ptrs):
        for i, page in enumerate(pages):
            rows = logical[kv, i * page_size : (i + 1) * page_size, h0 : h0 + hn]
            mem.view(ptr + page * item_len, item_len)[:] = rows.reshape(-1)


def _replay(mem, mgr, peer_name, src_pages, dst_pages):
    """Expand the prepared runs and copy what send_kvcache_slice would post."""
    src_handle, num_groups, num_ptr_pairs, num_slots_src = mgr.prep_handle_slice_src
    dst_handle, num_slots_dst, head_group_idx = mgr.prep_handles_slice_dst[peer_name]
    src = NixlKVManager._expand_stride_descs(mgr.agent.calls[src_handle][1])
    dst = NixlKVManager._expand_stride_descs(mgr.agent.calls[dst_handle][1])
    page_size = mgr.kv_args.page_size
    si = expand_page_indices_for_slice(
        np.asarray(src_pages, np.int32),
        num_ptr_pairs,
        num_slots_src,
        page_size,
        num_groups=num_groups,
        head_group_idx=head_group_idx,
    )
    di = expand_page_indices_for_slice(
        np.asarray(dst_pages, np.int32), num_ptr_pairs, num_slots_dst, page_size
    )
    assert len(si) == len(di)
    for (sa, sn, _), (da, dn, _) in zip(src[si].tolist(), dst[di].tolist()):
        assert sn == dn
        mem.view(da, dn)[:] = mem.view(sa, sn)


def run_case(num_heads, p_tp, d_tp, page_size, n_pages=3, num_slots=6, seed=0):
    rng = np.random.default_rng(seed)
    mem = _Memory()
    n_tok = n_pages * page_size
    logical = rng.integers(
        1, 255, size=(2, n_tok, num_heads, BYTES_PER_HEAD_TOKEN), dtype=np.uint8
    )
    src_pages = rng.permutation(num_slots)[:n_pages]
    dst_pages = rng.permutation(num_slots)[:n_pages]
    prefill = [
        _make_prefill(mem, p_tp, r, num_heads, page_size, num_slots)
        for r in range(p_tp)
    ]
    decode = [
        _make_decode(mem, d_tp, r, num_heads, page_size, num_slots) for r in range(d_tp)
    ]
    for mgr in prefill:
        _fill_logical(mem, mgr, p_tp, num_heads, page_size, src_pages, logical)
    for dec in decode:
        for p in _target_prefill_ranks(p_tp, d_tp, dec.decode_tp_rank):
            mgr = prefill[p]
            mgr._init_hetero_tp_prep_handle(dec.agent_name, dec)
            _replay(mem, mgr, dec.agent_name, src_pages, dst_pages)
    for dec in decode:
        h0, hn = _heads(d_tp, dec.decode_tp_rank, num_heads)
        for kv, ptr in enumerate(dec.dst_kv_ptrs):
            for i, page in enumerate(dst_pages):
                got = mem.view(ptr + page * dec.dst_kv_item_len, dec.dst_kv_item_len)
                want = logical[kv, i * page_size : (i + 1) * page_size, h0 : h0 + hn]
                np.testing.assert_array_equal(
                    got,
                    want.reshape(-1),
                    err_msg=f"decode rank {dec.decode_tp_rank} kv={kv} page={i}",
                )


def _shardable_tps(num_heads, candidates):
    return [t for t in candidates if num_heads % t == 0 or t % num_heads == 0]


class TestNixlHeteroTpBothSidesReplicate(CustomTestCase):
    """2 KV heads, prefill TP4 -> decode TP8: both sides replicate heads.

    Before the fix ``num_groups = d_tp // p_tp // dst_replication`` was
    ``8 // 4 // 4 == 0`` and the src-run geometry assert fired.
    """

    def test_num_groups_counts_heads_held_by_prefill_rank(self):
        mem = _Memory()
        mgr = _make_prefill(mem, 4, 0, 2, page_size=2, num_slots=4)
        dec = _make_decode(mem, 8, 1, 2, page_size=2, num_slots=4)
        mgr._init_hetero_tp_prep_handle(dec.agent_name, dec)
        _src_handle, num_groups, _pairs, _slots = mgr.prep_handle_slice_src
        self.assertEqual(num_groups, 1)
        self.assertEqual(mgr.prep_handles_slice_dst[dec.agent_name][2], 0)

    def test_data_lands_on_every_decode_rank(self):
        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                run_case(num_heads=2, p_tp=4, d_tp=8, page_size=page_size)


class TestNixlHeteroTpSliceGrid(CustomTestCase):
    """Every shardable (prefill TP, decode TP) pair moves the right NHD bytes."""

    def test_grid(self):
        seed = 0
        for num_heads, candidates in ((2, (1, 2, 4, 8)), (8, (1, 2, 4, 8, 16))):
            tps = _shardable_tps(num_heads, candidates)
            for p_tp, d_tp in itertools.product(tps, tps):
                if p_tp == d_tp:
                    continue  # equal TP never takes the slice path
                for page_size in (1, 2):
                    seed += 1
                    with self.subTest(
                        heads=num_heads, p_tp=p_tp, d_tp=d_tp, page_size=page_size
                    ):
                        run_case(num_heads, p_tp, d_tp, page_size, seed=seed)


class TestNixlHeteroTpRefusesDeclaredLayout(CustomTestCase):
    """The slice runs assume NHD rows; a pool declaring a layout is refused."""

    def test_declared_layout_raises(self):
        mem = _Memory()
        mgr = _make_prefill(mem, 1, 0, 2, page_size=1, num_slots=4)
        mgr.kv_layout = "some_head_major_layout"
        dec = _make_decode(mem, 2, 0, 2, page_size=1, num_slots=4)
        with self.assertRaisesRegex(NotImplementedError, "some_head_major_layout"):
            mgr._init_hetero_tp_prep_handle(dec.agent_name, dec)
        self.assertEqual(mgr.agent.calls, [])


if __name__ == "__main__":
    unittest.main()
