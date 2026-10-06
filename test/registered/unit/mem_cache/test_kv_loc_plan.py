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

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache import kv_index_translator
from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
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
                plan.swa_write_ids(),
                self.allocator.translate_loc_from_full_to_swa(self.window),
            )
        )

    def test_one_iteration_translates_once(self):
        writes, builds = [], []
        real_write = self.allocator.translate_write_loc
        real_build = kv_index_translator.build_kv_read_table

        def counting_write(ids, *a, **kw):
            writes.append(ids)
            return real_write(ids, *a, **kw)

        def counting_build(**kwargs):
            builds.append(kwargs["v2p"])
            return real_build(**kwargs)

        with (
            patch.object(self.allocator, "translate_write_loc", counting_write),
            patch.object(kv_index_translator, "build_kv_read_table", counting_build),
        ):
            # The translator binds the allocator's translate when built.
            self.target._translate_write_full = counting_write
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
        self.assertEqual(len(writes), 1)
        # The full and the sliding-window id space, one build each.
        self.assertEqual(len(builds), 2)
        self.assertTrue(torch.equal(verify.out_cache_loc, plan.write_physical))
        self.assertIs(extend.out_cache_loc, verify.out_cache_loc)

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
            # The full and the sliding-window space, one build each.
            self.assertEqual(len(builds), 2)
            padded = plan.read_table(rows=4)
            self.assertEqual(len(builds), 2)  # copied, not built again
            self.assertTrue(torch.equal(padded.ids[:2], table.ids))
            self.assertEqual(int(padded.ids[2:].abs().sum()), 0)  # the sink
            self.assertTrue(
                torch.equal(padded.sliding_window_ids[:2], table.sliding_window_ids)
            )
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
            if writes_swa:
                self.assertTrue(torch.equal(batch.out_cache_loc_swa, slots))
            else:
                self.assertIsNone(batch.out_cache_loc_swa)

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
        self.assertTrue(torch.equal(batch.out_cache_loc_swa[:n], plan.swa_write_ids()))
        self.assertEqual(int(batch.out_cache_loc_swa[n:].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
