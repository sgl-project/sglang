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
- the read table covers `seq_lens + read_extent`, is built once, and grows
  for a captured graph's padded lanes by copying (sink rows), never by
  building again;
- `bind` gives a batch its ids, with the virtual mirror only when they were
  translated.

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
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEV = "cpu"
_PS = 2


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
        # The same ids the physical-side derivation produces.
        self.assertTrue(
            torch.equal(
                plan.swa_write_ids(),
                self.target.sliding_window_write_loc_for(plan.write_physical),
            )
        )

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
        # The rows cover `seq_lens + read_extent`, as a table built for those
        # lengths does.
        reference = self.target.build_index_table(
            req_pool_indices=self.rpi,
            seq_lens=self.seq_lens + 1,
            max_pages=table.ids.shape[1],
        )
        self.assertTrue(torch.equal(table.ids, reference.ids))

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


if __name__ == "__main__":
    unittest.main()
