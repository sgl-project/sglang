# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Unit and multi-GPU bitwise tests for Zero-SM copy-engine DCP KV prefetch."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.dcp.comm import all_gather_kv_cache_for_mha_extend
from sglang.srt.layers.dcp.kv_prefetch import (
    CopyList,
    DcpKvPrefetcher,
    LayerOrder,
    MemcpyDriver,
    get_dcp_kv_prefetch,
    init_dcp_kv_prefetch,
    plan_dcp_chunk_layout,
    plan_hash,
    reset_dcp_kv_prefetch,
    staggered_pull_order,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _reference_reconstruct_chunk(
    seq_lens: list[int],
    starts: list[int],
    world_size: int,
) -> tuple[list[list[int]], list[int]]:
    """Reference token-interleaved DCP shard assignment and reconstruction."""
    # Assign a unique global token ID for every token in every request.
    global_tokens: list[int] = []
    per_rank_local: list[list[int]] = [[] for _ in range(world_size)]
    next_tok = 100
    for length, start in zip(seq_lens, starts):
        for offset in range(length):
            tok = next_tok
            next_tok += 1
            global_tokens.append(tok)
            owner = (start + offset) % world_size
            per_rank_local[owner].append(tok)
    return per_rank_local, global_tokens


def _simulate_pack_and_reorg(
    seq_lens: list[int],
    starts: list[int],
    world_size: int,
) -> tuple[list[list[int]], list[int]]:
    """Pure-Python simulation of _pack_dcp_kv_kernel + _reorg_dcp_kv_kernel."""
    per_rank_local, expected_global = _reference_reconstruct_chunk(
        seq_lens, starts, world_size
    )
    layouts = [
        plan_dcp_chunk_layout(seq_lens, starts, world_size, r)
        for r in range(world_size)
    ]
    # All ranks must agree on total_out and total_padded.
    total_out = layouts[0].total_out
    total_padded = layouts[0].total_padded
    for r, layout in enumerate(layouts):
        assert layout.total_out == total_out
        assert layout.total_padded == total_padded
        assert layout.total_local == len(per_rank_local[r])

    # Simulate pack into own_wire[r] of shape [total_padded].
    wires = [[-1] * total_padded for _ in range(world_size)]
    for r, layout in enumerate(layouts):
        local_vals = per_rank_local[r]
        for pid in range(layout.total_local):
            if layout.pack_single_span:
                dst_row = layout.pack_single_dst_base + pid
            else:
                lo, hi = 0, len(layout.pack_dst_base)
                while lo + 1 < hi:
                    mid = (lo + hi) >> 1
                    if layout.pack_local_cu[mid] <= pid:
                        lo = mid
                    else:
                        hi = mid
                dst_row = layout.pack_dst_base[lo] + (pid - layout.pack_local_cu[lo])
            wires[r][dst_row] = local_vals[pid]

    # Simulate reorg on every rank and confirm identical output.
    outputs: list[list[int]] = []
    for r, layout in enumerate(layouts):
        out = [-1] * layout.total_out
        for pid in range(layout.total_out):
            if layout.reorg_single_req:
                local_t = pid
                pad_base = 0
                start_mod = layout.reorg_single_start_mod
            else:
                lo, hi = 0, len(layout.reorg_pad_base)
                while lo + 1 < hi:
                    mid = (lo + hi) >> 1
                    if layout.reorg_out_cu[mid] <= pid:
                        lo = mid
                    else:
                        hi = mid
                local_t = pid - layout.reorg_out_cu[lo]
                pad_base = layout.reorg_pad_base[lo]
                start_mod = layout.reorg_start_mod[lo]
            idx = start_mod + local_t
            owner = idx % world_size
            padded_row = pad_base + (idx // world_size)
            out[pid] = wires[owner][padded_row]
        outputs.append(out)
    return outputs, expected_global


class TestDcpKvPrefetchUnit(unittest.TestCase):
    def test_staggered_pull_order_is_conflict_free(self):
        for world in (2, 4, 8):
            orders = [staggered_pull_order(world, r) for r in range(world)]
            for r, order in enumerate(orders):
                self.assertEqual(len(order), world - 1)
                self.assertNotIn(r, order)
                self.assertEqual(sorted(order), [p for p in range(world) if p != r])
            # At every step s, every rank pulls from a distinct source peer.
            for step in range(world - 1):
                sources = [orders[r][step] for r in range(world)]
                self.assertEqual(len(set(sources)), world)

    def test_plan_hash_consistency_and_sensitivity(self):
        h1 = plan_hash([1, 2, 3, 4])
        h2 = plan_hash([1, 2, 3, 4])
        self.assertEqual(h1, h2)
        self.assertIsInstance(h1, int)
        self.assertTrue(-(1 << 63) <= h1 < (1 << 63))
        # Sensitivity to single-element perturbations.
        self.assertNotEqual(h1, plan_hash([1, 2, 3, 5]))
        self.assertNotEqual(h1, plan_hash([1, 2, 4, 3]))

    def test_plan_dcp_chunk_layout_aligned_and_unaligned(self):
        cases = [
            # Aligned single and multi-request
            ([16], [0], 4),
            ([8, 12, 4], [0, 4, 8], 4),
            # Unaligned starts and lengths
            ([7], [3], 4),
            ([1], [2], 4),
            ([2, 5, 1, 9], [1, 3, 2, 0], 4),
            # Mixed with zero-length requests (common in multi-chunk prefix)
            ([0, 7, 0, 5, 0], [16, 16, 16, 16, 16], 4),
            ([0, 3, 0], [5, 7, 9], 4),
            # Single tiny request where some ranks own 0 local tokens
            ([2], [1], 4),
            ([1, 0, 2], [0, 3, 1], 8),
            # World size 2 and 8
            ([13, 6, 1], [1, 0, 1], 2),
            ([17, 0, 3, 11], [5, 2, 7, 4], 8),
        ]
        for seq_lens, starts, world in cases:
            with self.subTest(seq_lens=seq_lens, starts=starts, world=world):
                outputs, expected = _simulate_pack_and_reorg(seq_lens, starts, world)
                for rank_out in outputs:
                    self.assertEqual(rank_out, expected)

    def test_plan_dcp_chunk_layout_fast_paths(self):
        # Aligned multi-request takes single-span pack fast path.
        layout = plan_dcp_chunk_layout([16, 32], [0, 8], world_size=4, rank=2)
        self.assertTrue(layout.pack_single_span)
        self.assertEqual(layout.pack_single_dst_base, 0)
        self.assertFalse(layout.reorg_single_req)

        # Multi-request with only 1 active non-zero req takes scalar reorg fast path.
        layout = plan_dcp_chunk_layout([0, 13, 0], [4, 5, 8], world_size=4, rank=1)
        self.assertTrue(layout.reorg_single_req)
        self.assertEqual(layout.reorg_single_start_mod, 1)
        self.assertTrue(layout.pack_single_span)

    def test_layer_order_learn_freeze_and_recovery(self):
        order = LayerOrder()
        # Pass 1: learns [0, 1, 2]
        self.assertIsNone(order.advance(0, True))
        self.assertIsNone(order.advance(1, False))
        self.assertIsNone(order.advance(2, False))
        # Pass 2: frozen, returns 0, 1, 2
        self.assertEqual(order.advance(0, True), 0)
        self.assertEqual(order.advance(1, False), 1)
        self.assertEqual(order.advance(2, False), 2)
        # Pass 3: unexpected layer unfreezes and returns None
        self.assertEqual(order.advance(0, True), 0)
        self.assertIsNone(order.advance(5, False))
        self.assertFalse(order.frozen)
        # Pass 4: re-learns [0, 1]
        self.assertIsNone(order.advance(0, True))
        self.assertIsNone(order.advance(1, False))
        # Pass 5: frozen again with [0, 1]
        self.assertEqual(order.advance(0, True), 0)
        self.assertEqual(order.advance(1, False), 1)

        # Early stop invalidates frozen order on next forward
        self.assertEqual(order.advance(0, True), 0)
        self.assertIsNone(order.advance(0, True))
        self.assertTrue(order.learning)

    def test_copy_list_and_batch_fallback(self):
        copies = CopyList([(0x100, 0x900, 32), (0x200, 0xA00, 0), (0x300, 0xB00, 16)])
        self.assertEqual((copies.count, copies.nbytes), (2, 48))
        self.assertEqual(list(copies.destinations), [0x100, 0x300])
        self.assertEqual(list(copies.sources), [0x900, 0xB00])
        self.assertEqual(list(copies.sizes), [32, 16])

        def make_driver(batch_rc: int | None):
            drv = MemcpyDriver.__new__(MemcpyDriver)
            calls: list = []

            def one(dst, src, size, stream):
                calls.append(("one", dst, src, size, stream))
                return 0

            def batch(dsts, srcs, sizes, count, *rest):
                calls.append(
                    (
                        "batch",
                        list(zip(dsts[:count], srcs[:count], sizes[:count])),
                        rest[-1],
                    )
                )
                return batch_rc or 0

            drv._memcpy = one
            drv._memcpy_batch = None if batch_rc is None else batch
            drv._attributes_ref = drv._attribute_index = None
            return drv, calls

        # Batch succeeds for count > 1, single used for count == 1.
        drv, calls = make_driver(0)
        drv.memcpy_list(copies, 7)
        self.assertEqual(
            calls, [("batch", [(0x100, 0x900, 32), (0x300, 0xB00, 16)], 7)]
        )
        calls.clear()
        drv.memcpy_list(CopyList([(0x100, 0x900, 32)]), 7)
        drv.memcpy_list(CopyList([]), 7)
        self.assertEqual(calls, [("one", 0x100, 0x900, 32, 7)])

        # Non-zero batch return disables batch and falls back cleanly.
        drv, calls = make_driver(801)
        drv.memcpy_list(copies, 9)
        self.assertIsNone(drv._memcpy_batch)
        self.assertEqual(
            [c for c in calls if c[0] == "one"],
            [("one", 0x100, 0x900, 32, 9), ("one", 0x300, 0xB00, 16, 9)],
        )

    def test_state_machine_observe_parity_and_capture_skip(self):
        pf = DcpKvPrefetcher.__new__(DcpKvPrefetcher)
        pf.ready = True
        pf.orders = {}
        pf.order = LayerOrder()
        pf.forward_ref = None
        pf.position = None
        pf.step = None
        pf.pending = None
        pf.last_parity = 1

        launched: list[tuple[int, int]] = []
        pf._launch_item = lambda lp, ci: (
            launched.append((lp, ci)),
            setattr(
                pf,
                "pending",
                SimpleNamespace(
                    layer_id=pf.step.layer_ids[lp],
                    layer_pos=lp,
                    chunk_idx=ci,
                    parity=0,
                    total_out=16,
                    split_out=True,
                    dst_dtype=torch.bfloat16,
                ),
            ),
        )
        pf._abandon = lambda: (setattr(pf, "pending", None), setattr(pf, "step", None))

        class _Batch:
            def __init__(self, name: str):
                self.name = name

        # Chunk 0 (prefix == 0): observe_layer learns [0, 1].
        batch0 = _Batch("b0")
        with patch("torch.cuda.is_current_stream_capturing", return_value=False):
            pf.observe_layer(0, batch0)
            pf.observe_layer(1, batch0)
            self.assertEqual(pf.orders[0].layer_ids, [0, 1])

            # Chunk 1: freezes order immediately at layer 0!
            batch1 = _Batch("b1")
            pos0 = pf._advance_order(0, batch1)
            self.assertEqual(pos0, 0)
            pf.step = SimpleNamespace(layer_ids=(0, 1), chunks=(object(), object()))
            pf.start_first()
            self.assertEqual(launched, [(0, 0)])

            # Simulate join + start_next across 2 chunks x 2 layers.
            pf.pending = None
            pf.start_next(0, 0)
            self.assertEqual(launched, [(0, 0), (0, 1)])
            pf.pending = None
            pf.start_next(0, 1)
            self.assertEqual(launched, [(0, 0), (0, 1), (1, 0)])

            # Advance to layer 1, finish both chunks.
            self.assertEqual(pf._advance_order(1, batch1), 1)
            pf.pending = None
            pf.start_next(1, 0)
            self.assertEqual(launched, [(0, 0), (0, 1), (1, 0), (1, 1)])
            pf.pending = None
            pf.start_next(1, 1)
            self.assertIsNone(pf.step)

            # MTP / draft entry layer 47 gets its own isolated LayerOrder.
            draft_batch = _Batch("draft")
            pf.observe_layer(47, draft_batch)
            self.assertIn(47, pf.orders)
            self.assertTrue(pf.orders[0].frozen)

        # Verify maybe_start_dcp_mha_prefetch accepts a 2D CPU tensor for
        # prefix_chunk_seq_lens_cpu (shape [num_chunks, batch_size]).
        from sglang.srt.layers.dcp.kv_prefetch import maybe_start_dcp_mha_prefetch

        called = []
        pf.begin_layer = lambda *a, **kw: called.append(("begin", kw["mode"]))
        pf.start_first = lambda: called.append(("start",))
        chunk_batch = SimpleNamespace(
            prefix_chunk_kv_indices=[
                torch.zeros(4, dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
            ],
            prefix_chunk_seq_lens_cpu=torch.tensor(
                [[16, 16], [8, 0]], dtype=torch.int32
            ),
            prefix_chunk_starts_cpu=torch.tensor([[0, 0], [16, 16]], dtype=torch.int32),
        )
        with (
            patch("sglang.srt.layers.dcp.kv_prefetch._DCP_KV_PREFETCHER", pf),
            patch(
                "sglang.srt.layers.dcp.kv_prefetch._reads_are_translated",
                return_value=False,
            ),
        ):
            res = maybe_start_dcp_mha_prefetch(
                0,
                chunk_batch,
                object(),
                mode="mha_chunked",
                dst_dtype=torch.bfloat16,
                has_prefix=True,
            )
            self.assertIs(res, pf)
            self.assertEqual(called, [("begin", "mha_chunked"), ("start",)])

        # During CUDA graph capture, observe/join cleanly skip.
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            pf.pending = SimpleNamespace(
                layer_id=0,
                chunk_idx=0,
                split_out=True,
                parity=0,
                total_out=8,
                dst_dtype=torch.bfloat16,
            )
            self.assertIsNone(pf.join_split(0, 0))

    def test_derive_step_plan_defensive_guards(self):
        pf = DcpKvPrefetcher.__new__(DcpKvPrefetcher)
        pf.world_size = 4
        pf.rank = 0
        pf.kv_lora_rank = 512
        pf.qk_rope_head_dim = 64
        pf.kv_dim = 576
        pf.kv_cache_dtype = torch.bfloat16
        pf.out_dtype = torch.bfloat16
        pf.max_out_rows = 64
        pf.wire_rows = 32
        pf.order = LayerOrder()
        pf.order.layer_ids = [0]
        pf.step = None

        class _MockLayer:
            def __init__(self, view, buf):
                self.view = view
                self._buf = buf

            def key(self):
                return self._buf

        class _MockRouter:
            def __init__(self, layer):
                self._layer = layer

            def layer(self, lid):
                if lid != 0:
                    raise KeyError(lid)
                return self._layer

        good_buf = torch.empty((16, 1, 576), dtype=torch.bfloat16)
        good_ids = torch.arange(4, dtype=torch.int32)

        def run_with(
            *,
            pool_counter=None,
            view_counter=None,
            scale_buf=None,
            use_dsa=False,
            buf=good_buf,
            dst_dtype=torch.bfloat16,
            seq_lens=None,
            mode="mha_chunked",
        ):
            pf.step = None
            view = SimpleNamespace(
                layer_transfer_counter=view_counter,
                kv_scale_buffer=scale_buf,
                use_dsa=use_dsa,
            )
            pool = SimpleNamespace(
                layer_transfer_counter=pool_counter,
                layer_router=_MockRouter(_MockLayer(view, buf)),
            )
            pf._derive_step_plan(
                token_to_kv_pool=pool,
                mode=mode,
                dst_dtype=dst_dtype,
                kv_indices_list=[good_ids],
                seq_lens_list=[seq_lens or [16]],
                starts_list=[[0]],
            )
            return pf.step

        # Every guard condition must reject cleanly (step stays None -> serial fallback).
        self.assertIsNone(run_with(pool_counter=object()))
        self.assertIsNone(run_with(view_counter=object()))
        self.assertIsNone(run_with(scale_buf=torch.ones(1)))
        self.assertIsNone(run_with(use_dsa=True))
        self.assertIsNone(run_with(dst_dtype=torch.float32))
        self.assertIsNone(run_with(mode="unknown"))
        self.assertIsNone(run_with(seq_lens=[128]))  # exceeds max_out_rows=64
        self.assertIsNone(run_with(buf=torch.empty((16, 1, 576), dtype=torch.float16)))
        self.assertIsNone(run_with(buf=torch.empty((16, 2, 576), dtype=torch.bfloat16)))

    def test_interleave_prefix_and_extend_kv_with_tp_padding(self):
        # Verify FP8 prefix cast, multi-request interleaving, and TP padding drop.
        prefix_a = (
            torch.arange(5 * 4, dtype=torch.float32).view(5, 4).to(torch.float8_e4m3fn)
        )
        prefix_pe = (
            torch.arange(5 * 2, dtype=torch.float32)
            .view(5, 1, 2)
            .to(torch.float8_e4m3fn)
        )
        ext_a = torch.full(
            (6, 4), 99.0, dtype=torch.bfloat16
        )  # 4 real + 2 TP padding rows
        ext_a[:4] = torch.arange(100, 100 + 16, dtype=torch.bfloat16).view(4, 4)
        ext_pe = torch.full((6, 1, 2), 88.0, dtype=torch.bfloat16)
        ext_pe[:4] = torch.arange(200, 200 + 8, dtype=torch.bfloat16).view(4, 1, 2)

        out_a, out_pe = all_gather_kv_cache_for_mha_extend(
            None,
            None,
            None,
            [2, 3],
            [1, 3],
            ext_a,
            ext_pe,
            prefetched=(prefix_a, prefix_pe),
        )
        self.assertEqual(out_a.dtype, torch.bfloat16)
        self.assertEqual(out_pe.dtype, torch.bfloat16)
        self.assertEqual(out_a.shape, (9, 4))
        self.assertEqual(out_pe.shape, (9, 1, 2))
        expected_a = torch.cat(
            [
                prefix_a[:2].to(torch.bfloat16),
                ext_a[:1],
                prefix_a[2:5].to(torch.bfloat16),
                ext_a[1:4],
            ],
            dim=0,
        )
        expected_pe = torch.cat(
            [
                prefix_pe[:2].to(torch.bfloat16),
                ext_pe[:1],
                prefix_pe[2:5].to(torch.bfloat16),
                ext_pe[1:4],
            ],
            dim=0,
        )
        self.assertTrue(torch.equal(out_a, expected_a))
        self.assertTrue(torch.equal(out_pe, expected_pe))

    def test_singleton_disabled_for_world_size_one(self):
        reset_dcp_kv_prefetch()
        self.assertIsNone(get_dcp_kv_prefetch())
        res = init_dcp_kv_prefetch(
            SimpleNamespace(world_size=1),
            torch.device("cpu"),
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            kv_cache_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            max_chunk_capacity=1024,
        )
        self.assertIsNone(res)
        self.assertIsNone(get_dcp_kv_prefetch())

    def test_post_attention_start_next_and_mla_fill_helper(self):
        from sglang.srt.layers.dcp.kv_prefetch import (
            maybe_consume_dcp_kv_split,
            maybe_fill_dcp_mla_extend_buffer,
        )

        events: list[tuple] = []
        pf = DcpKvPrefetcher.__new__(DcpKvPrefetcher)
        pf.join_split = lambda lid, ci: (
            events.append(("join_split", lid, ci))
            or (torch.ones(2, 4), torch.ones(2, 1, 2))
        )
        pf.start_next = lambda lid, ci: events.append(("start_next", lid, ci))
        pf.observe_layer = lambda lid, fb: events.append(("observe", lid))
        pf.begin_layer = lambda lid, fb, pool, **kw: events.append(
            ("begin", lid, kw["mode"])
        )
        pf.start_first = lambda: events.append(("start_first",))
        pf.join_combined = lambda lid, ci: (
            events.append(("join_combined", lid, ci))
            or torch.full((3, 1, 6), 7.0, dtype=torch.bfloat16)
        )

        with patch("sglang.srt.layers.dcp.kv_prefetch._DCP_KV_PREFETCHER", pf):
            # start_next=False defers launching next layer until after attention.
            res = maybe_consume_dcp_kv_split(0, 0, start_next=False)
            self.assertIsNotNone(res)
            self.assertEqual(events, [("join_split", 0, 0)])

            events.clear()
            res = maybe_consume_dcp_kv_split(0, 0, start_next=True)
            self.assertIsNotNone(res)
            self.assertEqual(events, [("join_split", 0, 0), ("start_next", 0, 0)])

            # maybe_fill_dcp_mla_extend_buffer copies prefix + extend rows and defers
            # start_next to post-attn_mqa in forward_mla.py.
            events.clear()
            dcp_buf = torch.zeros((5, 1, 6), dtype=torch.bfloat16)
            k_nope = torch.full(
                (4, 1, 4), 2.0, dtype=torch.bfloat16
            )  # 2 real + 2 TP pad
            k_pe = torch.full((4, 1, 2), 3.0, dtype=torch.bfloat16)
            fb = SimpleNamespace(
                attn_dcp_metadata=SimpleNamespace(
                    dcp_extend_prefix_lens_sum=3,
                    dcp_kv_buffer=dcp_buf,
                    dcp_local_prefix_kv_indices=torch.zeros(1, dtype=torch.int32),
                ),
                extend_prefix_lens_cpu=[3],
            )
            with patch(
                "sglang.srt.layers.dcp.kv_prefetch._reads_are_translated",
                return_value=False,
            ):
                handled = maybe_fill_dcp_mla_extend_buffer(
                    object(), 0, fb, 4, k_nope, k_pe
                )
            self.assertTrue(handled)
            self.assertEqual(
                events,
                [("begin", 0, "mla_absorb"), ("start_first",), ("join_combined", 0, 0)],
            )
            self.assertTrue(torch.all(dcp_buf[:3] == 7.0))
            self.assertTrue(torch.all(dcp_buf[3:, :, :4] == 2.0))
            self.assertTrue(torch.all(dcp_buf[3:, :, 4:] == 3.0))

            # Translated KV reads must observe and fall back cleanly.
            events.clear()
            with patch(
                "sglang.srt.layers.dcp.kv_prefetch._reads_are_translated",
                return_value=True,
            ):
                self.assertFalse(
                    maybe_fill_dcp_mla_extend_buffer(object(), 0, fb, 4, k_nope, k_pe)
                )
            self.assertEqual(events, [("observe", 0)])


_GPU_WORKER_SCRIPT = r"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.kernels.ops.kvcache.mla_buffer import get_mla_kv_buffer_triton
from sglang.srt.layers.dcp.comm import all_gather_kv_cache_for_dcp
from sglang.srt.layers.dcp.kv_prefetch import DcpKvPrefetcher
from sglang.srt.layers.dcp.layout import filter_dcp_local_chunk_kv_indices


class _Batch:
    pass


class _FakeView:
    def __init__(self, buffers):
        self._buffers = buffers
        self.layer_transfer_counter = None
        self.kv_scale_buffer = None
        self.use_dsa = False

    def get_key_buffer(self, layer_id):
        return self._buffers[layer_id]


class _FakeKVLayer:
    def __init__(self, view, layer_id):
        self.view = view
        self.layer_id = layer_id

    def key(self):
        return self.view.get_key_buffer(self.layer_id)


class _FakeRouter:
    def __init__(self, view):
        self._view = view

    def layer(self, layer_id):
        return _FakeKVLayer(self._view, layer_id)


class _FakePool:
    def __init__(self, buffers):
        self.layer_transfer_counter = None
        self.layer_router = _FakeRouter(_FakeView(buffers))


class _FakeGroup:
    def __init__(self, device_group, cpu_group, world_size, rank):
        self.device_group = device_group
        self.cpu_group = cpu_group
        self.world_size = world_size
        self.rank_in_group = rank

    def all_gather_into_tensor(self, out, inp):
        dist.all_gather_into_tensor(out, inp, group=self.device_group)


def make_strided_kv_buffers(num_layers, num_slots, kv_dim, dtype, device, seed, strided):
    torch.manual_seed(seed)
    if strided:
        # Mimic SGLang token-major KVArena: slot stride = num_layers * kv_dim!
        arena = torch.empty(
            (num_slots, num_layers, 1, kv_dim), dtype=torch.bfloat16, device=device
        ).normal_(0, 1)
        if dtype != torch.bfloat16:
            arena = arena.to(dtype)
        return [arena[:, lid, :, :] for lid in range(num_layers)]
    bufs = []
    for _ in range(num_layers):
        t = torch.randn((num_slots, 1, kv_dim), dtype=torch.bfloat16, device=device)
        if dtype != torch.bfloat16:
            t = t.to(dtype)
        bufs.append(t)
    return bufs


def build_local_indices(seq_lens, starts, rank, world_size, num_slots, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    total = sum(seq_lens)
    global_ids = torch.randint(0, num_slots, (total,), generator=g, dtype=torch.int32, device="cpu").to(device)
    starts_t = torch.tensor(starts, dtype=torch.int32)
    lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    return filter_dcp_local_chunk_kv_indices(global_ids, starts_t, lens_t)


def reference_gather(kv_buf, local_ids, kv_lora_rank, qk_rope_dim, dst_dtype, seq_lens, starts):
    cache_nope = torch.empty((local_ids.numel(), 1, kv_lora_rank), dtype=dst_dtype, device=kv_buf.device)
    cache_rope = torch.empty((local_ids.numel(), 1, qk_rope_dim), dtype=dst_dtype, device=kv_buf.device)
    if local_ids.numel() > 0:
        get_mla_kv_buffer_triton(kv_buf, local_ids, cache_nope, cache_rope)
    gathered = all_gather_kv_cache_for_dcp(
        cache_nope,
        cache_rope,
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.tensor(starts, dtype=torch.int32),
    )
    kv_a, k_pe = gathered.split([kv_lora_rank, qk_rope_dim], dim=-1)
    return kv_a.squeeze(1).contiguous(), k_pe.contiguous(), gathered.contiguous()


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    cpu_group = dist.new_group(backend="gloo")
    device_group = dist.group.WORLD
    fake_group = _FakeGroup(device_group, cpu_group, world_size, rank)

    kv_lora_rank = 512
    qk_rope_dim = 64
    kv_dim = kv_lora_rank + qk_rope_dim
    num_layers = 4
    num_slots = 4096

    fake_parallel = SimpleNamespace(
        dcp_enabled=True,
        dcp_size=world_size,
        dcp_rank=rank,
        dcp_group=fake_group,
    )
    fake_platform = SimpleNamespace(is_sm100=False)

    test_configs = [
        # (kv_cache_dtype, out_dtype, dst_dtype, strided)
        (torch.bfloat16, torch.bfloat16, torch.bfloat16, True),
        (torch.float16, torch.float16, torch.float16, False),
        (torch.float8_e4m3fn, torch.bfloat16, torch.bfloat16, True),
        (torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, True),
        (torch.float8_e5m2, torch.bfloat16, torch.bfloat16, False),
    ]

    with (
        patch("sglang.srt.layers.dcp.comm.get_parallel", return_value=fake_parallel),
        patch("sglang.srt.layers.dcp.layout.get_parallel", return_value=fake_parallel),
        patch("sglang.srt.layers.dcp.comm.get_platform", return_value=fake_platform),
    ):
        for cfg_idx, (kv_dtype, out_dtype, dst_dtype, strided) in enumerate(test_configs):
            pf = DcpKvPrefetcher(
                fake_group,
                device,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_dim,
                kv_cache_dtype=kv_dtype,
                out_dtype=out_dtype,
                max_chunk_capacity=2048,
            )
            assert pf.ready, f"Prefetcher not ready for {kv_dtype} -> {dst_dtype}"
            assert pf.out_pool.shape == (2048 * kv_dim,), (
                f"Expected single-slot out_pool, got {pf.out_pool.shape}"
            )
            kv_buffers = make_strided_kv_buffers(
                num_layers, num_slots, kv_dim, kv_dtype, device, seed=1000 + rank * 37 + cfg_idx, strided=strided
            )
            pool = _FakePool(kv_buffers)

            # Warmup LayerOrder on Chunk 0 (prefix == 0).
            b0 = _Batch()
            for lid in range(num_layers):
                pf.observe_layer(lid, b0)

            # Case 1: MHA_CHUNKED_KV (2 chunks, multi-request, zero-len reqs in chunk 1, unaligned starts).
            if dst_dtype != torch.float8_e4m3fn:
                c0_lens, c0_starts = [37, 64, 13], [0, 0, 0]
                c1_lens, c1_starts = [0, 29, 2], [37, 64, 13]
                c0_ids = build_local_indices(c0_lens, c0_starts, rank, world_size, num_slots, device, 11 + cfg_idx)
                c1_ids = build_local_indices(c1_lens, c1_starts, rank, world_size, num_slots, device, 22 + cfg_idx)
                batch = _Batch()
                for lid in range(num_layers):
                    pf.begin_layer(
                        lid,
                        batch,
                        pool,
                        mode="mha_chunked",
                        dst_dtype=dst_dtype,
                        kv_indices_list=[c0_ids, c1_ids],
                        seq_lens_list=[c0_lens, c1_lens],
                        starts_list=[c0_starts, c1_starts],
                    )
                    pf.start_first()
                    for ci, (ids, lens, starts) in enumerate(
                        [(c0_ids, c0_lens, c0_starts), (c1_ids, c1_lens, c1_starts)]
                    ):
                        res = pf.join_split(lid, ci)
                        assert res is not None, f"missed prefetch layer={lid} chunk={ci}"
                        got_a, got_pe = res[0].clone(), res[1].clone()
                        pf.start_next(lid, ci)
                        ref_a, ref_pe, _ = reference_gather(
                            kv_buffers[lid], ids, kv_lora_rank, qk_rope_dim, dst_dtype, lens, starts
                        )
                        assert torch.equal(got_a, ref_a), (
                            f"kv_a mismatch cfg={cfg_idx} layer={lid} chunk={ci} "
                            f"max_diff={(got_a.float() - ref_a.float()).abs().max().item()}"
                        )
                        assert torch.equal(got_pe, ref_pe), (
                            f"k_pe mismatch cfg={cfg_idx} layer={lid} chunk={ci}"
                        )

                # Case 2: MHA_ONE_SHOT with tiny unaligned single request (some ranks own 0 local rows).
                os_lens, os_starts = [2], [0]
                os_ids = build_local_indices(os_lens, os_starts, rank, world_size, num_slots, device, 33 + cfg_idx)
                batch2 = _Batch()
                for lid in range(num_layers):
                    pf.begin_layer(
                        lid,
                        batch2,
                        pool,
                        mode="mha_one_shot",
                        dst_dtype=dst_dtype,
                        kv_indices_list=[os_ids],
                        seq_lens_list=[os_lens],
                    )
                    pf.start_first()
                    res = pf.join_split(lid, 0)
                    assert res is not None
                    got_a, got_pe = res[0].clone(), res[1].clone()
                    pf.start_next(lid, 0)
                    ref_a, ref_pe, _ = reference_gather(
                        kv_buffers[lid], os_ids, kv_lora_rank, qk_rope_dim, dst_dtype, os_lens, os_starts
                    )
                    assert torch.equal(got_a, ref_a)
                    assert torch.equal(got_pe, ref_pe)
            else:
                # Case 3: MLA absorbed path (FP8 -> FP8 combined output!).
                mla_lens, mla_starts = [41, 19], [0, 0]
                mla_ids = build_local_indices(mla_lens, mla_starts, rank, world_size, num_slots, device, 44 + cfg_idx)
                batch3 = _Batch()
                for lid in range(num_layers):
                    pf.begin_layer(
                        lid,
                        batch3,
                        pool,
                        mode="mla_absorb",
                        dst_dtype=dst_dtype,
                        kv_indices_list=[mla_ids],
                        seq_lens_list=[mla_lens],
                    )
                    pf.start_first()
                    got = pf.join_combined(lid, 0)
                    assert got is not None
                    got_copy = got.clone()
                    pf.start_next(lid, 0)
                    _, _, ref_comb = reference_gather(
                        kv_buffers[lid], mla_ids, kv_lora_rank, qk_rope_dim, dst_dtype, mla_lens, mla_starts
                    )
                    assert torch.equal(got_copy.view(torch.uint8), ref_comb.view(torch.uint8)), (
                        f"mla_absorb FP8 bitwise mismatch layer={lid}"
                    )

            torch.cuda.synchronize(device)
            dist.barrier(group=cpu_group)

    if rank == 0:
        print(f"BITWISE_ALL_PASS world_size={world_size} configs={len(test_configs)}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
"""


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestDcpKvPrefetchMultiGpuBitwise(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "requires at least 2 CUDA GPUs",
    )
    def test_multi_gpu_bitwise_parity(self):
        world_size = 4 if torch.cuda.device_count() >= 4 else 2
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(_GPU_WORKER_SCRIPT)
            script_path = f.name
        try:
            port = _find_free_port()
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={world_size}",
                f"--master_port={port}",
                script_path,
            ]
            res = subprocess.run(
                cmd,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(
                res.returncode,
                0,
                f"Multi-GPU bitwise test failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}",
            )
            self.assertIn(f"BITWISE_ALL_PASS world_size={world_size}", res.stdout)
        finally:
            if os.path.exists(script_path):
                os.unlink(script_path)


if __name__ == "__main__":
    unittest.main()
