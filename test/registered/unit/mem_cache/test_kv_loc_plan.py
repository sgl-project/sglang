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
"""`KVLocPlan`: one iteration's KV ids, translated once.

- the target and a fused draft read the same physical write ids; a
  pass-through reader (a private draft pool, a static pool) reads virtual;
- the virtual window is never mutated, and `cols` picks the same columns in
  both spaces;
- the sliding-window write ids come from the virtual window and agree with
  the derivation from the physical one;
- a forward whose `seq_lens` do not yet count the window it writes (a
  verify, a speculative draft decode, a draft extend) reads that window past
  them;
- the read table covers `seq_lens + read_extent`, is built once, and grows
  for a captured graph's padded lanes by copying (sink rows), never by
  building again;
- only a reader of the plan's own rows reads its table (the target, a fused
  draft); a pass-through reader gathers `req_to_token`, and a draft with a
  `req_to_token` of its own plans its own reads over the shared write ids
  (`reads_from`);
- one iteration's forwards -- a draft step, the verify, a draft extend, the
  readers of a target and a fused draft -- translate the window once and
  build each id space's table once;
- `bind` gives a batch its ids, with the virtual mirror only when they were
  translated;
- a runner's own write buffer (graph capture, warmup) is used as it is, its
  sliding-window ids naming the sink; a replayed graph's batch writes through
  the runner's padded buffer, its sliding-window ids padded with the sink.

  python -m pytest test/registered/unit/mem_cache/test_kv_loc_plan.py -v
"""

import gc
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch

from sglang.srt.mem_cache import kv_index_translator
from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.kv_loc_plan import IdSpaceKind, TokenSpan
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    FusedDraftPlacement,
)
from sglang.srt.mem_cache.unified_draft_pool import UnifiedDraftKVPool
from sglang.srt.mem_cache.unified_memory_pool import MHASubPoolSpec, UnifiedKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEV = "cpu"
_PS = 2
_FULL = IdSpaceKind.FULL
_SWA = IdSpaceKind.SLIDING_WINDOW


def _reference(req_to_token, rows, lens, v2p, width):
    """Independent derivation of a read table: each row's pages over `lens`,
    through `v2p`, the sink past them."""
    out = torch.zeros((rows.numel(), width), dtype=torch.int32)
    for b, row in enumerate(rows.tolist()):
        for c in range(min(-(-int(lens[b]) // _PS), width)):
            out[b, c] = max(int(v2p[int(req_to_token[row, c * _PS]) // _PS]), 0)
    return out


class _FakeKVCache:
    def __init__(self, max_slots):
        self.buf = torch.full((max_slots,), -1, dtype=torch.int64)
        self.allocator = None

    def attach_allocator(self, allocator):
        self.allocator = allocator


class _FakeUnifiedSWAKVPool:
    def __init__(self, shared_pool):
        self.full_kv_pool = _FakeKVCache(shared_pool.max_slots("full"))
        self.swa_kv_pool = _FakeKVCache(shared_pool.max_slots("swa"))
        self.full_to_swa_index_mapping = None

    def attach_allocators(self, *, full_allocator, swa_allocator):
        self._full_allocator = full_allocator
        self._swa_allocator = swa_allocator


class TestKVLocPlan(unittest.TestCase):
    def setUp(self):
        # `KVIndexTranslator.__init__` reads `attn_dcp_size`, a derived
        # parallel width that only exists once a config is published.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.bfloat16,
            grow_direction="down",
            draft_region=DenseDraftRegion(
                lane_num=1, head_num=1, head_dim=8, store_dtype=torch.bfloat16
            ),
        )
        swa_spec = MHASubPoolSpec(
            name="swa",
            layer_num=1,
            head_num=2,
            head_dim=4,
            store_dtype=torch.bfloat16,
            grow_direction="up",
        )
        n_full, n_swa = 32, 16
        pool = UnifiedKVPool(
            total_bytes=n_full * full_spec.entry_bytes()
            + n_swa * swa_spec.entry_bytes(),
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
            page_size=_PS,
            fused_draft=FusedDraftPlacement(
                region=full_spec.draft_region, runner_lane_counts=(1,)
            ),
        )
        kvcache = _FakeUnifiedSWAKVPool(pool)
        self.allocator = UnifiedSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            device=_DEV,
            full_max_total_num_tokens=n_full,
            swa_max_total_num_tokens=n_swa,
            page_size=_PS,
            need_sort=False,
            forward_stream=None,
        )
        draft_pool = UnifiedDraftKVPool(
            unified_buffer=pool,
            host_sub_pool_name="full",
            host_allocator=self.allocator,
            layer_lanes={0: 0},
            page_size=_PS,
        )
        # Every runner of one server shares the target's req_to_token.
        self.req_to_token = torch.zeros((4, 16), dtype=torch.int32)
        self.draft_pool = draft_pool
        self.target = self._translator(kvcache)
        self.fused_draft = self._translator(draft_pool)
        self.private_draft = self._translator(_FakeKVCache(64))

        # Two requests, two pages each; the second page of each is this
        # iteration's write window (one token per request).
        self.rpi = torch.tensor([1, 2], dtype=torch.int64)
        for row in (1, 2):
            ids = self.allocator.alloc(2 * _PS)
            self.assertIsNotNone(ids)
            self.req_to_token[row, : 2 * _PS] = ids.to(torch.int32)
        self.seq_lens = torch.tensor([3, 3], dtype=torch.int64)
        self.window = self.req_to_token[self.rpi, 2:4].to(torch.int64).reshape(-1)

    def _translator(self, pool):
        return KVIndexTranslator(
            req_to_token=self.req_to_token,
            token_to_kv_pool_allocator=self.allocator,
            token_to_kv_pool=pool,
            page_size=_PS,
            device=_DEV,
        )

    def _plan(self, source=None, read_extent=1):
        return (source or self.target).plan(
            req_pool_indices=self.rpi,
            seq_lens=self.seq_lens,
            seq_lens_cpu=self.seq_lens.clone(),
            write_virtual=self.window,
            read_extent=read_extent,
        )

    def test_readers_get_ids_in_their_own_space(self):
        window = self.window.clone()
        plan = self._plan()
        physical = self.allocator.translate_write_loc(window)
        self.assertTrue(torch.equal(plan.write_ids(self.target), physical))
        # A fused draft shares the target's pages: the same tensor, untranslated.
        self.assertIs(plan.write_ids(self.fused_draft), plan.write_physical)
        # A private draft pool indexes virtual ids.
        self.assertIs(plan.write_ids(self.private_draft), self.window)
        self.assertTrue(torch.equal(self.window, window))  # never mutated

    def test_cols_pick_the_same_columns_in_both_spaces(self):
        plan = self._plan()
        bs = int(self.rpi.numel())
        first = slice(0, 1)
        self.assertTrue(
            torch.equal(
                plan.write_ids(self.target, cols=first),
                plan.write_physical.view(bs, -1)[:, first].reshape(-1),
            )
        )
        self.assertTrue(
            torch.equal(
                plan.virtual_write_ids(cols=first),
                self.window.view(bs, -1)[:, first].reshape(-1),
            )
        )

    def test_swa_ids_come_from_the_virtual_window(self):
        plan = self._plan()
        self.assertTrue(
            torch.equal(
                plan.write_ids(self.target, kind=_SWA),
                self.allocator.translate_loc_from_full_to_swa(self.window),
            )
        )

    def test_each_sub_pool_is_translated_once_and_shared(self):
        """A sub-pool's ids are derived on first use and handed out again
        after; a reader whose pool shares the sub-pool (a fused draft's full
        one) gets the same tensor, and a pool without it gets None."""
        plan = self._plan()
        swa = plan.write_ids(self.target, kind=_SWA)
        self.assertIs(plan.write_ids(self.target, kind=_SWA), swa)
        self.assertIs(plan.write_ids(self.fused_draft), plan.write_ids(self.target))
        self.assertIsNone(plan.write_ids(self.fused_draft, kind=_SWA))
        self.assertIsNone(plan.write_ids(self.private_draft, kind=_SWA))

    def test_cols_slice_names_a_split_batch_s_tokens(self):
        """Two-batch overlap splits a forward's tokens; each half names its
        tokens of the window, whatever the forward's own columns were. A
        contiguous half is a run of tokens, read as a view of the window's ids;
        only columns spread over the rows become a flat index."""
        plan = self._plan()
        bs = int(self.rpi.numel())
        everything = plan.cols_slice(None, slice(1, None))
        self.assertEqual(everything, TokenSpan(1, self.window.numel()))
        self.assertEqual(plan.cols_slice(everything, slice(1, 3)), TokenSpan(2, 4))
        # A static plan's halves, and a translating one's, both read views.
        static = self._plan(self.private_draft)
        for p, reader in ((plan, self.target), (static, self.private_draft)):
            half = p.cols_slice(None, slice(0, 2))
            self.assertIsInstance(half, TokenSpan)
            whole = p.write_ids(reader)
            ids = p.write_ids(reader, cols=half)
            self.assertTrue(torch.equal(ids, whole[:2]))
            self.assertEqual(
                ids.untyped_storage().data_ptr(), whole.untyped_storage().data_ptr()
            )
        first = plan.cols_slice(slice(0, 1), slice(None))
        width = self.window.numel() // bs
        self.assertTrue(torch.equal(first, torch.arange(bs) * width))
        ragged = torch.tensor([3, -1, 0])
        self.assertTrue(torch.equal(plan.cols_slice(ragged, slice(1, 3)), ragged[1:]))
        for cols, tokens in (
            (None, slice(1, None)),
            (everything, slice(1, 3)),
            (slice(0, 1), slice(None)),
        ):
            self.assertTrue(
                torch.equal(
                    plan.write_ids(self.target, cols=plan.cols_slice(cols, tokens)),
                    plan.write_ids(self.target, cols=cols)[tokens],
                )
            )

    def test_one_iteration_translates_once(self):
        writes, builds = [], []
        real_write = self.allocator.translate_write_loc
        real_build = kv_index_translator.build_kv_read_table

        swa_writes = []
        real_swa_write = self.allocator.translate_loc_from_full_to_swa

        def counting_write(ids, *a, **kw):
            writes.append(ids)
            return real_write(ids, *a, **kw)

        def counting_swa_write(ids):
            swa_writes.append(ids)
            return real_swa_write(ids)

        def counting_build(**kwargs):
            builds.append(kwargs["v2p"])
            return real_build(**kwargs)

        with patch.object(kv_index_translator, "build_kv_read_table", counting_build):
            # The spaces hold the allocator's translates from when the
            # translator was built; count through them.
            for kind, write in ((_FULL, counting_write), (_SWA, counting_swa_write)):
                self.target._spaces[kind] = msgspec.structs.replace(
                    self.target.space(kind), write=write
                )
            plan = self._plan(read_extent=2)
            draft, verify, extend = (SimpleNamespace() for _ in range(3))
            plan.bind(draft, self.fused_draft, cols=slice(0, 1))
            plan.bind(verify, self.target)
            plan.bind(extend, self.fused_draft)
            bs = int(self.rpi.numel())
            for reader in (self.target, self.fused_draft):
                # What a packed stream gathers from, and a captured table.
                reader.read_source(plan, req_pool_indices=self.rpi, bs=bs + 1)
                reader.read_table(plan)
                reader.copy_page_table(
                    plan, out=torch.zeros((bs + 2, 4), dtype=torch.int32)
                )
            # The target's sliding-window layers, written and read twice.
            for _ in range(2):
                self.target.write_ids(verify, _SWA)
                self.target.read_table(plan, kind=_SWA)
                self.target.copy_page_table(
                    plan, out=torch.zeros((bs + 2, 4), dtype=torch.int32), kind=_SWA
                )
        self.assertEqual(len(writes), 1)
        self.assertEqual(len(swa_writes), 1)
        # The full and the sliding-window id space, one build each.
        self.assertEqual(len(builds), 2)
        self.assertTrue(torch.equal(verify.out_cache_loc, plan.write_physical))
        self.assertIs(extend.out_cache_loc, verify.out_cache_loc)

    def test_the_first_stream_reader_builds_the_table_in_the_same_launch(self):
        builds, fused = [], []
        real_build = kv_index_translator.build_kv_read_table
        real_fused = kv_index_translator.build_kv_read_table_and_stream

        def counting_build(**kwargs):
            builds.append(kwargs["v2p"])
            return real_build(**kwargs)

        def counting_fused(**kwargs):
            fused.append(kwargs["v2p"])
            return real_fused(**kwargs)

        bs = int(self.rpi.numel())
        # A captured wrapper's lanes: the batch's, then one padded lane.
        lens = torch.tensor([3, 2, 1], dtype=torch.int64)
        indptr = torch.tensor([0, 3, 5, 6], dtype=torch.int32)
        out = torch.full((8,), -7, dtype=torch.int32)
        with (
            patch.object(kv_index_translator, "build_kv_read_table", counting_build),
            patch.object(
                kv_index_translator, "build_kv_read_table_and_stream", counting_fused
            ),
        ):
            plan = self._plan(read_extent=1)
            self.assertTrue(
                self.fused_draft.pack_read_stream(
                    plan,
                    req_pool_indices=self.rpi,
                    seq_lens=lens,
                    indptr=indptr,
                    out=out,
                )
            )
            table = plan.read_table(rows=bs + 1)
        # One launch for the full space's table and stream; nothing read the
        # sliding-window table, so nothing built it.
        self.assertEqual(fused, [self.allocator.full_v2p_page_table])
        self.assertEqual(builds, [])
        self.assertTrue(plan.has_read_table())
        self.assertFalse(plan.has_read_table(_SWA))
        for b in range(bs):
            row = self.req_to_token[int(self.rpi[b]), : int(lens[b])].to(torch.int64)
            want = self.allocator.translate_write_loc(row)
            got = out[int(indptr[b]) : int(indptr[b + 1])].to(torch.int64)
            self.assertTrue(torch.equal(got, want))
        self.assertEqual(int(out[5]), 0)  # the padded lane reads the sink
        self.assertTrue(bool((out[6:] == -7).all()))
        reference = _reference(
            self.req_to_token,
            self.rpi,
            self.seq_lens + 1,
            self.allocator.full_v2p_page_table,
            table.ids.shape[1],
        )
        self.assertTrue(torch.equal(table.ids[:bs], reference))
        self.assertEqual(int(table.ids[bs:].abs().sum()), 0)

    def test_a_verify_reads_the_window_it_writes(self):
        def own_plan(mode, spec_info):
            return self.target.own_plan(
                SimpleNamespace(
                    forward_mode=mode,
                    spec_info=spec_info,
                    batch_size=int(self.rpi.numel()),
                    req_pool_indices=self.rpi,
                    seq_lens=self.seq_lens,
                    seq_lens_cpu=self.seq_lens.clone(),
                    out_cache_loc=self.window,
                )
            )

        width = self.window.numel() // self.rpi.numel()
        # A verify's rows read up to `draft_token_num` past their lengths, more
        # than the window when the verify is ragged.
        spec = SimpleNamespace(draft_token_num=width + 1)
        self.assertEqual(
            own_plan(ForwardMode.TARGET_VERIFY, spec).read_extent, width + 1
        )
        self.assertEqual(own_plan(ForwardMode.DECODE, spec).read_extent, width)
        # A draft extend's lengths take its window after its batch is built.
        self.assertEqual(own_plan(ForwardMode.DRAFT_EXTEND_V2, spec).read_extent, width)
        # Their lengths already count what an extend or a plain decode writes.
        self.assertEqual(own_plan(ForwardMode.EXTEND, spec).read_extent, 0)
        self.assertEqual(own_plan(ForwardMode.DECODE, None).read_extent, 0)

    def test_a_stream_only_iteration_builds_no_table(self):
        """A plan shares a table only when a reader of its rows reads the
        table form itself. Readers of streams and row gathers alone -- one
        runner, or a target and a draft over the same rows -- each translate
        in their own gather: no table, with or without host lengths. A runner
        that reads tables brings the shared build back, which every reader then
        takes, and its going away takes it out again; a table read after a
        stream-only one still gets its table."""
        calls = {"packed": 0, "fused": 0, "table": 0}

        def counting(name, real):
            def wrapped(**kwargs):
                calls[name] += 1
                return real(**kwargs)

            return wrapped

        def stream_only_backend():
            return SimpleNamespace(kv_index_translator=None, reads_kv_index_table=False)

        # A runner of rows no other translator here reads.
        self.req_to_token = self.req_to_token.clone()
        solo = self._translator(self.allocator.get_kvcache())
        # A multi-step draft container is bound with its step backends and
        # reads nothing itself.
        container = SimpleNamespace(kv_index_translator=None)
        solo.bind_and_verify_backends([container, stream_only_backend()])
        self.assertIs(container.kv_index_translator, solo)
        bs = int(self.rpi.numel())
        lens = torch.tensor([3, 2], dtype=torch.int64)
        indptr = torch.tensor([0, 3, 5], dtype=torch.int32)

        def pack(translator, plan, out):
            return translator.pack_read_stream(
                plan, req_pool_indices=self.rpi, seq_lens=lens, indptr=indptr, out=out
            )

        with (
            patch.object(
                kv_index_translator,
                "build_kv_read_table_packed",
                counting("packed", kv_index_translator.build_kv_read_table_packed),
            ),
            patch.object(
                kv_index_translator,
                "build_kv_read_table_and_stream",
                counting("fused", kv_index_translator.build_kv_read_table_and_stream),
            ),
            patch.object(
                kv_index_translator,
                "build_kv_read_table",
                counting("table", kv_index_translator.build_kv_read_table),
            ),
        ):
            plan = self._plan(solo, read_extent=1)
            out = torch.full((6,), -7, dtype=torch.int32)
            self.assertTrue(pack(solo, plan, out))
            self.assertEqual(calls, {"packed": 1, "fused": 0, "table": 0})
            self.assertFalse(plan.has_read_table())
            for b in range(bs):
                row = self.req_to_token[int(self.rpi[b]), : int(lens[b])]
                want = self.allocator.translate_write_loc(row.to(torch.int64))
                got = out[int(indptr[b]) : int(indptr[b + 1])].to(torch.int64)
                self.assertTrue(torch.equal(got, want))
            self.assertEqual(int(out[5]), -7)  # past the stream: untouched

            # A caller holding a padded graph buffer of request rows: the
            # lengths size the batch, the extra lanes are left out.
            padded = torch.cat([self.rpi, torch.zeros(5, dtype=torch.int64)])
            again = torch.full((6,), -7, dtype=torch.int32)
            solo.pack_read_stream(
                self._plan(solo, read_extent=1),
                req_pool_indices=padded,
                seq_lens=lens,
                indptr=indptr,
                out=again,
            )
            self.assertTrue(torch.equal(again, out))
            calls["packed"] -= 1

            # A table read after it still gets one.
            plan.read_table()
            self.assertEqual(calls["table"], 1)

            # Another stream reader of the same rows (a fused flashinfer
            # draft): still no table -- each translates its own gather.
            peer = self._translator(self.draft_pool)
            peer.bind_and_verify_backends([stream_only_backend()])
            self.assertFalse(solo.table_is_read(_FULL))
            for seq_lens_cpu in (self.seq_lens.clone(), None):
                both = solo.plan(
                    req_pool_indices=self.rpi,
                    seq_lens=self.seq_lens,
                    seq_lens_cpu=seq_lens_cpu,
                    write_virtual=self.window,
                    read_extent=1,
                )
                pack(solo, both, torch.empty(6, dtype=torch.int32))
                pack(peer, both, torch.empty(6, dtype=torch.int32))
                # The draft's row gather gets the virtual rows and the page
                # table, translated in its own kernel.
                src = peer.read_source(both, req_pool_indices=self.rpi, bs=bs)
                self.assertTrue(src.is_translated)
                self.assertIs(src.ids, self.req_to_token)
                self.assertIs(src.v2p, solo.space(_FULL).read_v2p)
                self.assertFalse(both.has_read_table())
            self.assertEqual(calls, {"packed": 5, "fused": 0, "table": 1})

            # A runner of these rows that reads the table form (a page-table
            # draft backend): the first stream reader builds the shared table
            # in its own launch, and the others read it.
            table_reader = self._translator(self.draft_pool)
            table_reader.bind_and_verify_backends(
                [SimpleNamespace(kv_index_translator=None, reads_kv_index_table=True)]
            )
            self.assertTrue(solo.table_is_read(_FULL))
            shared = self._plan(solo, read_extent=1)
            pack(solo, shared, torch.empty(6, dtype=torch.int32))
            self.assertEqual(calls["fused"], 1)
            self.assertTrue(shared.has_read_table())
            src = peer.read_source(shared, req_pool_indices=self.rpi, bs=bs)
            self.assertIsNone(src.v2p)
            self.assertIs(src.ids, shared.read_table(rows=bs).ids)
            # The drafts have no sliding-window sub-pool: that one the target
            # alone reads, so its stream still comes alone.
            self.assertFalse(solo.table_is_read(_SWA))
            solo.pack_read_stream(
                shared,
                req_pool_indices=self.rpi,
                seq_lens=lens,
                indptr=indptr,
                out=torch.empty(6, dtype=torch.int32),
                kind=_SWA,
            )
            self.assertEqual(calls["packed"], 6)
            self.assertFalse(shared.has_read_table(_SWA))

            # The table reader goes away: no shared table again.
            del table_reader
            gc.collect()
            self.assertFalse(solo.table_is_read(_FULL))

    def test_a_captured_first_reader_holds_the_table(self):
        """When the plan's first reader is a captured table, the table is built
        in that buffer -- no table of the plan's own, no copy -- and serves the
        plan's other readers; the same buffer again is left alone, and another
        captured reader copies it rather than building again; an eager reader
        of the batch takes the batch's rows of it."""
        builds = []
        real = kv_index_translator.build_kv_read_table

        def counting(**kwargs):
            builds.append(kwargs["out"])
            return real(**kwargs)

        bs = int(self.rpi.numel())
        captured = torch.full((bs + 1, 8), 7, dtype=torch.int32)
        with patch.object(kv_index_translator, "build_kv_read_table", counting):
            plan = self._plan(read_extent=1)
            self.target.copy_page_table(plan, out=captured)
            self.assertEqual(len(builds), 1)
            self.assertEqual(builds[0].data_ptr(), captured.data_ptr())
            table = plan.read_table(rows=bs + 1)
            self.assertEqual(table.ids.data_ptr(), captured.data_ptr())
            self.assertEqual(table.row_stride, captured.stride(0))
            # An eager reader of the batch (fa3, trtllm_mha) gets the batch's
            # rows of that table: its kernels size the batch by them.
            eager = self.target.read_table(plan, rows=bs)
            self.assertEqual(eager.ids.shape[0], bs)
            self.assertEqual(eager.row_ids.numel(), bs)
            self.assertEqual(eager.ids.data_ptr(), captured.data_ptr())
            width = table.ids.shape[1]
            reference = _reference(
                self.req_to_token,
                self.rpi,
                self.seq_lens + 1,
                self.allocator.full_v2p_page_table,
                width,
            )
            self.assertTrue(torch.equal(captured[:bs, :width], reference))
            self.assertEqual(int(captured[bs, :width].abs().sum()), 0)  # padded lane
            self.assertTrue(bool((captured[:, width:] == 7).all()))  # past the width
            self.target.copy_page_table(plan, out=captured)
            other = torch.full_like(captured, 5)
            self.target.copy_page_table(plan, out=other)
            self.assertEqual(len(builds), 1)
            self.assertTrue(torch.equal(other[:, :width], captured[:, :width]))

    def test_read_table_is_built_once_and_padded_by_copy(self):
        builds = []
        real = kv_index_translator.build_kv_read_table

        def counting(**kwargs):
            builds.append(kwargs["v2p"])
            return real(**kwargs)

        with patch.object(kv_index_translator, "build_kv_read_table", counting):
            plan = self._plan(read_extent=1)
            table = plan.read_table()
            self.assertIs(plan.read_table(), table)
            # Only the space that was read.
            self.assertEqual(len(builds), 1)
            swa = plan.read_table(kind=_SWA)
            self.assertEqual(len(builds), 2)
            padded = plan.read_table(rows=4)
            padded_swa = plan.read_table(kind=_SWA, rows=4)
            self.assertEqual(len(builds), 2)  # copied, not built again
            self.assertTrue(torch.equal(padded.ids[:2], table.ids))
            self.assertEqual(int(padded.ids[2:].abs().sum()), 0)  # the sink
            self.assertTrue(torch.equal(padded_swa.ids[:2], swa.ids))
            self.assertEqual(int(padded_swa.ids[2:].abs().sum()), 0)
        # The rows cover `seq_lens + read_extent`.
        reference = _reference(
            self.req_to_token,
            self.rpi,
            self.seq_lens + 1,
            self.allocator.full_v2p_page_table,
            table.ids.shape[1],
        )
        self.assertTrue(torch.equal(table.ids, reference))

    def test_only_readers_of_its_rows_read_its_table(self):
        plan = self._plan()
        self.assertTrue(plan.is_read_by(self.target))
        self.assertTrue(plan.is_read_by(self.fused_draft))
        self.assertFalse(plan.is_read_by(self.private_draft))
        compact = KVIndexTranslator(
            req_to_token=self.req_to_token.clone(),
            token_to_kv_pool_allocator=self.allocator,
            token_to_kv_pool=self.draft_pool,
            page_size=_PS,
            device=_DEV,
        )
        self.assertFalse(plan.is_read_by(compact))
        self.assertIs(
            self.fused_draft.read_source(plan, req_pool_indices=self.rpi, bs=2),
            plan.read_table(rows=2),
        )
        passthrough = self.private_draft.read_source(
            plan, req_pool_indices=self.rpi, bs=2
        )
        self.assertFalse(passthrough.is_translated)
        self.assertIs(passthrough.ids, self.req_to_token)

    def test_a_compact_draft_reads_its_own_rows_over_the_shared_writes(self):
        plan = self._plan()
        compact_rows = self.req_to_token.clone()
        compact = KVIndexTranslator(
            req_to_token=compact_rows,
            token_to_kv_pool_allocator=self.allocator,
            token_to_kv_pool=self.draft_pool,
            page_size=_PS,
            device=_DEV,
        )
        lens = torch.tensor([2, 1], dtype=torch.int64)
        derived = plan.reads_from(
            compact, seq_lens=lens, seq_lens_cpu=lens.clone(), read_extent=1
        )
        # The writes are the plan's, not translated again.
        self.assertIs(derived.write_physical, plan.write_physical)
        self.assertIs(derived.write_ids(compact), plan.write_physical)
        self.assertTrue(derived.is_read_by(compact))
        compact_rows[self.rpi, :2] = compact_rows[self.rpi, 2:4]
        reference = _reference(
            compact_rows,
            self.rpi,
            lens + 1,
            self.allocator.full_v2p_page_table,
            derived.read_table().ids.shape[1],
        )
        self.assertTrue(torch.equal(derived.read_table().ids, reference))

    def test_a_pass_through_plan_does_no_work(self):
        plan = self._plan(source=self.private_draft)
        self.assertIs(plan.write_physical, self.window)
        table = plan.read_table()
        self.assertFalse(table.is_translated)
        self.assertIs(table.ids, self.req_to_token)
        self.assertIs(table.row_ids, self.rpi)

    def test_bind_sets_the_write_fields(self):
        plan = self._plan()
        fused = SimpleNamespace()
        plan.bind(fused, self.fused_draft, cols=slice(0, 1))
        self.assertIs(fused.kv_loc_plan, plan)
        self.assertEqual(fused.kv_loc_cols, slice(0, 1))
        self.assertTrue(
            torch.equal(
                fused.out_cache_loc, plan.write_ids(self.target, cols=slice(0, 1))
            )
        )
        self.assertTrue(
            torch.equal(
                fused.out_cache_loc_virtual, plan.virtual_write_ids(cols=slice(0, 1))
            )
        )
        self.assertTrue(fused.out_cache_loc_is_physical)
        private = SimpleNamespace()
        plan.bind(private, self.private_draft)
        self.assertIs(private.out_cache_loc, self.window)
        self.assertIsNone(private.out_cache_loc_virtual)

    def test_an_npu_dcp_target_writes_rank_local_slots(self):
        """On NPU with DCP a target writes its rank's slots, while the
        scheduler, `req_to_token` and the replicated draft pool keep
        allocator-global ones: a plan's window stays global whichever runner
        built it (a target prefill handing its plan to the draft, a draft
        handing its window to the verify), the target binds it rank-local,
        and the draft binds it global."""
        parallel = SimpleNamespace(
            dcp_enabled=True, dcp_size=2, dcp_rank=1, attn_dcp_size=1
        )
        with (
            patch.object(kv_index_translator, "_is_npu", True),
            patch.object(kv_index_translator, "get_parallel", return_value=parallel),
        ):
            target, draft = (
                KVIndexTranslator(
                    req_to_token=self.req_to_token,
                    token_to_kv_pool_allocator=self.allocator,
                    token_to_kv_pool=_FakeKVCache(64),
                    page_size=4,
                    device=_DEV,
                    is_draft_worker=is_draft_worker,
                )
                for is_draft_worker in (False, True)
            )
        global_ids = torch.arange(8, 16)
        # Rank 1 of 2 owns every other 4-slot page, compacted.
        local_ids = torch.tensor([-1, -1, -1, -1, 4, 5, 6, 7])
        for source in (target, draft):
            plan = source.plan(
                req_pool_indices=self.rpi,
                seq_lens=self.seq_lens,
                seq_lens_cpu=self.seq_lens.clone(),
                write_virtual=global_ids,
            )
            target_batch, draft_batch = SimpleNamespace(), SimpleNamespace()
            plan.bind(target_batch, target)
            plan.bind(draft_batch, draft)
            self.assertTrue(torch.equal(target_batch.out_cache_loc, local_ids))
            self.assertTrue(torch.equal(draft_batch.out_cache_loc, global_ids))
            self.assertIs(plan.write_virtual, global_ids)

    def test_runner_slots_are_used_as_they_are(self):
        for translator, writes_swa in (
            (self.target, True),
            (self.private_draft, False),
        ):
            slots = torch.zeros(2, dtype=torch.int64)
            batch = SimpleNamespace(
                req_pool_indices=self.rpi,
                seq_lens=self.seq_lens,
                seq_lens_cpu=self.seq_lens.clone(),
                out_cache_loc=slots,
            )
            translator.bind_runner_slots(batch)
            # Kept by address: a captured graph writes this buffer on replay.
            self.assertIs(batch.out_cache_loc, slots)
            self.assertIsNone(batch.out_cache_loc_virtual)
            self.assertTrue(batch.out_cache_loc_is_physical)
            swa = translator.write_ids(batch, _SWA)
            if writes_swa:
                # The sink in the sliding-window sub-pool too.
                self.assertTrue(torch.equal(swa, torch.zeros_like(slots)))
            else:
                self.assertIsNone(swa)

    def test_a_replay_batch_writes_the_padded_slots(self):
        plan = self._plan()
        n = plan.write_physical.numel()
        slots = torch.zeros(n + 2, dtype=torch.int64)
        slots[:n].copy_(plan.write_physical)
        batch = SimpleNamespace()
        plan.bind_replay(batch, self.target, slots=slots)
        self.assertIs(batch.kv_loc_plan, plan)
        self.assertIs(batch.out_cache_loc, slots)
        self.assertIsNone(batch.out_cache_loc_virtual)
        swa = self.target.write_ids(batch, _SWA)
        self.assertEqual(swa.shape[0], n + 2)
        self.assertTrue(torch.equal(swa[:n], plan.write_ids(self.target, kind=_SWA)))
        self.assertEqual(int(swa[n:].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
