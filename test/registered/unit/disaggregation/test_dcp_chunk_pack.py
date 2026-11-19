"""PD DCP relayout (DCP-1 prefill to DCP-N decode) with one pack per chunk.

The chunk-pack path must move exactly the bytes the per-token path moves, to
exactly the same destination rows, for every decode DCP rank: the target KV, a
DCP-collapsed draft, a replicated draft and the DSA index-K keys and scales.
Host byte arrays stand in for device buffers; the gather and the Mooncake
writes are emulated on them.
"""

import bisect
import concurrent.futures
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.disaggregation.common import dcp_pack
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _Memory:
    """Fake address space: disjoint numpy byte arrays at fake base addresses."""

    def __init__(self):
        self.buffers = {}
        self._bases = []
        self._next = 1 << 32

    def alloc(self, nbytes, fill=None, rng=None):
        base = self._next
        self._next += ((nbytes + (1 << 20)) // (1 << 20) + 1) << 20
        if rng is not None:
            buf = rng.integers(0, 256, nbytes, dtype=np.uint8)
        else:
            buf = np.zeros(nbytes, dtype=np.uint8)
        self.buffers[base] = buf
        self._bases.append(base)
        return base

    def view(self, addr, nbytes):
        base = self._bases[bisect.bisect_right(self._bases, addr) - 1]
        buf = self.buffers[base]
        off = addr - base
        if off + nbytes > buf.size:
            raise AssertionError(f"access past buffer {base:#x}: {off}+{nbytes}")
        return buf[off : off + nbytes]

    def transfer(self, blocks):
        for src, dst, size in blocks:
            self.view(dst, size)[:] = self.view(src, size)
        return 0


class _FakePackBuffer:
    """StagingBuffer stand-in on host memory registered in a _Memory."""

    def __init__(self, memory, nbytes):
        self.base = memory.alloc(nbytes)
        self.buffer = torch.from_numpy(memory.buffers[self.base])
        self.memory = memory
        self.gathers = 0
        self.waits = 0
        stream = SimpleNamespace(
            wait_stream=lambda _: None, synchronize=self._synchronize
        )
        self.get_gather_stream = lambda: stream
        self.producer_stream = lambda: None

    def _synchronize(self):
        self.waits += 1

    def get_ptr(self):
        return self.base

    def get_size(self):
        return self.buffer.numel()

    def fits(self, nbytes):
        return nbytes <= self.buffer.numel()


def _emulated_gather(memory, pack_buffer):
    def copy_rows(kv_data_ptrs, row_indices, pack, token_item_lens, src_items=None):
        pack_buffer.gathers += 1
        out = pack.numpy()
        rows = row_indices.numpy()
        offset = 0
        for ptr, width in zip(kv_data_ptrs, token_item_lens):
            for i, row in enumerate(rows):
                out[offset + i * width : offset + (i + 1) * width] = memory.view(
                    ptr + int(row) * width, width
                )
            offset += rows.size * width

    return copy_rows


def _manager(memory, kv_ptrs, item_lens, num_draft, index_k_token_bytes, page_size):
    manager = object.__new__(MooncakeKVManager)
    manager.kv_args = SimpleNamespace(
        page_size=page_size,
        kv_layer_ids=[],
        kv_data_ptrs=kv_ptrs,
        kv_item_lens=item_lens,
        num_draft_entries=num_draft,
        engine_rank=0,
        dsa_index_k_token_bytes=index_k_token_bytes,
    )
    manager.is_mla_backend = True
    manager.attn_tp_size = 1
    manager.max_transfer_batch_indices = 0
    manager.enable_custom_mem_pool = True
    manager.enable_deferred_decode_kv_release = False
    manager.get_mla_kv_ptrs_with_pp = lambda src, dst, *_: (src, dst, len(src))
    manager.descriptors = 0
    manager.calls = 0

    def transfer(session, blocks):
        manager.descriptors += len(blocks)
        manager.calls += 1
        return memory.transfer(blocks)

    manager._transfer_data = transfer
    return manager


def _run_relayout(
    *,
    chunk_pack,
    seed,
    draft="collapsed",
    num_draft=2,
    num_index_k=2,
    total_tokens=None,
    chunk_tokens=4 * 64 * 8,
):
    """Send a request in chunks to every decode DCP rank; return the decode
    buffers and the manager, which counts the transfer descriptors."""
    rng = np.random.default_rng(seed)
    page_size, dcp = 64, 8
    target_width, draft_width = 576, 576
    num_target = 3
    value_bytes, scale_bytes = 128, 4
    index_k_page = page_size * (value_bytes + scale_bytes)
    decode_prefix_len = int(rng.integers(0, 3)) * page_size * dcp
    if total_tokens is None:
        total_tokens = int(rng.integers(1, 40 * page_size))
    src_pool_pages = 2 * (-(-total_tokens // page_size)) + 4
    dst_pool_pages = 2 * (-(-(decode_prefix_len + total_tokens) // (page_size * dcp)))
    dst_pool_pages += 4

    memory = _Memory()
    src_kv = [
        memory.alloc(src_pool_pages * page_size * target_width, rng=rng)
        for _ in range(num_target)
    ] + [
        memory.alloc(src_pool_pages * page_size * draft_width, rng=rng)
        for _ in range(num_draft)
    ]
    src_index_k = [
        memory.alloc(src_pool_pages * index_k_page, rng=rng) for _ in range(num_index_k)
    ]
    src_pages = rng.permutation(src_pool_pages)[: -(-total_tokens // page_size)]
    src_pages = src_pages.astype(np.int32)
    # Decode virtual pages: some runs, some scattered.
    dst_virtual_pages = np.sort(rng.permutation(dst_pool_pages)[: dst_pool_pages // 2])
    if rng.random() < 0.5:
        dst_virtual_pages = rng.permutation(dst_virtual_pages)
    need = -(-(decode_prefix_len + total_tokens) // (page_size * dcp))
    dst_virtual_pages = dst_virtual_pages[:need].astype(np.int32)

    collapsed = draft == "collapsed"
    draft_page_rows = page_size * (1 if collapsed else dcp)
    decode = []
    for _ in range(dcp):
        decode.append(
            {
                "kv": [
                    memory.alloc(dst_pool_pages * page_size * target_width)
                    for _ in range(num_target)
                ]
                + [
                    memory.alloc(dst_pool_pages * draft_page_rows * draft_width)
                    for _ in range(num_draft)
                ],
                "index_k": [
                    memory.alloc(dst_pool_pages * index_k_page)
                    for _ in range(num_index_k)
                ],
            }
        )

    manager = _manager(
        memory,
        src_kv,
        [page_size * target_width] * num_target + [page_size * draft_width] * num_draft,
        num_draft,
        (value_bytes, scale_bytes),
        page_size,
    )
    pack_bytes = dcp_pack.dcp_pack_buffer_bytes(
        [page_size * target_width] * num_target + [page_size * draft_width] * num_draft,
        page_size,
        chunk_tokens,
        dcp,
    )
    pack_bytes += (
        dcp * -(-chunk_tokens // dcp) * num_index_k * (value_bytes + scale_bytes)
    )
    pack_buffer = _FakePackBuffer(memory, pack_bytes) if chunk_pack else None
    dst_page_offset = decode_prefix_len // (page_size * dcp)

    with ExitStack() as stack:
        if pack_buffer is not None:
            stack.enter_context(
                patch.object(
                    dcp_pack,
                    "copy_mla_rows_into_pack",
                    _emulated_gather(memory, pack_buffer),
                )
            )
            stack.enter_context(patch.object(torch.cuda, "set_device", lambda _: None))
            stack.enter_context(
                patch.object(dcp_pack, "device_stream_context", lambda _: nullcontext())
            )
        executor = stack.enter_context(
            concurrent.futures.ThreadPoolExecutor(max_workers=3)
        )
        for start in range(0, total_tokens, chunk_tokens):
            end = min(total_tokens, start + chunk_tokens)
            page_slice = slice(start // page_size, -(-end // page_size))
            cache = {}
            for rank in range(dcp):
                ret = MooncakeKVManager.send_kvcache_dcp(
                    manager,
                    f"session-{rank}",
                    src_pages[page_slice],
                    decode[rank]["kv"],
                    dst_virtual_pages[dst_page_offset:],
                    dcp_token_item_lens=[target_width] * num_target
                    + [draft_width] * num_draft,
                    dst_dcp_size=dcp,
                    dst_dcp_rank=rank,
                    src_page_offset=page_slice.start,
                    decode_prefix_len=decode_prefix_len,
                    num_kv_tokens=end - start,
                    executor=executor,
                    dst_layer_ids=[],
                    pack_buffer=pack_buffer,
                    dst_kv_item_lens=[page_size * target_width] * num_target
                    + [draft_page_rows * draft_width] * num_draft,
                    draft_rows_collapsed=collapsed,
                    dsa_index_k=(src_index_k, decode[rank]["index_k"], index_k_page),
                    pack_cache=cache,
                )
                assert ret == 0, ret
    snapshot = [
        [memory.buffers[b].copy() for b in d["kv"] + d["index_k"]] for d in decode
    ]
    return snapshot, manager, pack_buffer, total_tokens


class TestDcpChunkPackRelayout(CustomTestCase):
    def test_chunk_pack_matches_per_token_relayout(self):
        for draft in ("collapsed", "replicated"):
            # Prefixes 512/1024; seeds 4 and 5 span 2 chunks (5's last has 13 tokens).
            for seed in (1, 4, 5):
                with self.subTest(draft=draft, seed=seed):
                    reference, ref_mgr, _, tokens = _run_relayout(
                        chunk_pack=False, seed=seed, draft=draft
                    )
                    packed, mgr, pack_buffer, _ = _run_relayout(
                        chunk_pack=True, seed=seed, draft=draft
                    )
                    for rank, (ref_bufs, new_bufs) in enumerate(zip(reference, packed)):
                        for i, (ref, new) in enumerate(zip(ref_bufs, new_bufs)):
                            self.assertTrue(
                                np.array_equal(ref, new),
                                f"rank {rank} buffer {i} differs ({tokens} tokens)",
                            )
                    # Source rows reach the decode at all: not a zero-vs-zero compare.
                    self.assertTrue(any(buf.any() for buf in packed[0]))
                    # One gather segment launch per entry group and one wait per chunk.
                    chunks = -(-tokens // (4 * 64 * 8))
                    self.assertEqual(pack_buffer.waits, chunks)
                    self.assertLessEqual(mgr.descriptors, ref_mgr.descriptors)
                    if draft == "collapsed" and tokens >= 64 * 8:
                        # The collapsed draft no longer moves token by token.
                        self.assertLess(mgr.descriptors * 4, ref_mgr.descriptors)

    def test_oversized_chunk_falls_back_to_per_rank_path(self):
        with (
            patch.object(dcp_pack, "pack_dcp_row_segments", return_value=None) as chunk,
            patch.object(dcp_pack, "try_pack_dcp_src", return_value=None) as per_rank,
        ):
            reference, _, _, _ = _run_relayout(chunk_pack=False, seed=4)
            packed, _, _, _ = _run_relayout(chunk_pack=True, seed=4)
        self.assertTrue(chunk.called and per_rank.called)
        for ref_bufs, new_bufs in zip(reference, packed):
            for ref, new in zip(ref_bufs, new_bufs):
                np.testing.assert_array_equal(ref, new)


if __name__ == "__main__":
    unittest.main()
