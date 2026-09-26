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
"""Graceful mamba slot-exhaustion (unified cache) unit tests.

When the mamba pool is exhausted AND eviction cannot reclaim a slot (e.g. every
tree state is transiently locked by an in-flight hicache write-through), the
optional caching/donation paths must SKIP the prefix instead of asserting and
killing the scheduler. This file locks in that behavior contract for
`MambaComponent`:

  - `_alloc_mamba_slot` returns None (no assert) when alloc fails twice;
  - `prepare_for_caching_req(is_finished=False)` returns 0 (caller runs its
    no-cache path) when the donation cannot allocate a slot;
  - the exhaustion is observable via `sglang:mamba_cache_skip_total`.

No server, no model weights.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_component(alloc_result):
    """A MambaComponent with a mocked cache, bypassing pool/tree plumbing.

    Only the attributes exercised by the graceful-skip paths are wired; the
    periodic reconcile canary and the reconcile-string dump are stubbed so the
    test does not need a real tree core.
    """
    cache = mock.MagicMock()
    cache.req_to_token_pool.mamba_allocator.alloc.return_value = alloc_result
    cache.req_to_token_pool.mamba_ckpt_pool = None  # int8 disabled
    cache.enable_mamba_extra_buffer = False

    comp = MambaComponent.__new__(MambaComponent)  # noqa: SLF001 - intentional
    comp.cache = cache
    comp._last_reconcile_ts = float("inf")  # never fire the periodic canary
    comp._reconcile_interval = 60.0
    comp._last_exhaust_report_ts = 0.0
    comp.mamba_reconcile_str = lambda: "<reconcile>"
    return comp


class TestMambaSlotGracefulSkip(CustomTestCase):
    def test_alloc_mamba_slot_returns_none_when_exhausted(self):
        # alloc always fails and eviction cannot reclaim -> None, no assert.
        comp = _make_component(alloc_result=None)
        slot = comp._alloc_mamba_slot()  # noqa: SLF001
        self.assertIsNone(slot)
        comp.cache.evict_for_alloc.assert_called_once()
        # Observable degradation: counter bumped once for pool="bf16".
        comp.cache.metrics_collector.increment_mamba_cache_skip.assert_called_once_with(
            "bf16"
        )

    def test_alloc_mamba_slot_returns_slot_when_available(self):
        # Normal path unchanged: a successful alloc returns the slot.
        fake_slot = torch.tensor([7])
        comp = _make_component(alloc_result=fake_slot)
        slot = comp._alloc_mamba_slot()  # noqa: SLF001
        self.assertIs(slot, fake_slot)
        comp.cache.evict_for_alloc.assert_not_called()
        comp.cache.metrics_collector.increment_mamba_cache_skip.assert_not_called()

    def test_prepare_for_caching_unfinished_returns_zero_when_exhausted(self):
        # Donation cannot allocate a slot -> return 0 so the caller runs its
        # no-cache path (effective_cache_len <= 0), instead of asserting.
        comp = _make_component(alloc_result=None)
        req = mock.MagicMock()
        insert_params = mock.MagicMock()
        insert_params.mamba_value = None  # initial state, like real InsertParams
        cache_len = comp.prepare_for_caching_req(
            req=req,
            insert_params=insert_params,
            token_ids_len=16,
            is_finished=False,
        )
        self.assertEqual(cache_len, 0)
        # No mamba_value was recorded for the tree insert.
        self.assertIsNone(insert_params.mamba_value)
        comp.cache.metrics_collector.increment_mamba_cache_skip.assert_called()


if __name__ == "__main__":
    unittest.main()
