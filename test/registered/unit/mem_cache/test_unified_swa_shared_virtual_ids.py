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
"""The unified SWA composite mints one virtual page id and binds it on both
sides, so the swa side's `virtual_to_physical` is indexed by the FULL side's
ids. Its own page count says nothing about how wide that has to be: which side
gets more pages out of a shared byte budget depends on the layer split, and a
model with few full-attention layers and many sliding ones (gemma-4: 10 and 50)
gives the full side an order of magnitude more.

Reaching an id that high takes cumulative allocation, so a narrow table fails
after churn rather than at once -- and on GPU it fails as a `tl.store` past the
end of the table: an unchecked write, not a raised index.

    python -m pytest test/registered/unit/mem_cache/test_unified_swa_shared_virtual_ids.py -v
"""

import unittest

import torch
from test_swa_locked_full_recover_unified import _DEV, _FakeUnifiedSWAKVPool

from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.unified_memory_pool import MHASubPoolSpec, UnifiedKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _build(n_full: int, n_swa: int, full_layers: int, swa_layers: int):
    """A composite whose sides have different per-page byte costs. `full_layers
    < swa_layers` is the gemma-4 shape: the cheap side is the id owner and ends
    up with far more pages than the side that has to address them."""
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=full_layers,
        head_num=2,
        head_dim=4,
        store_dtype=torch.float16,
        grow_direction="up",
    )
    swa_spec = MHASubPoolSpec(
        name="swa",
        layer_num=swa_layers,
        head_num=2,
        head_dim=4,
        store_dtype=torch.float16,
        grow_direction="down",
    )
    pool = UnifiedKVPool(
        total_bytes=n_full * full_spec.entry_bytes() + n_swa * swa_spec.entry_bytes(),
        sub_pool_specs=[full_spec, swa_spec],
        device=_DEV,
        enable_memory_saver=False,
    )
    return UnifiedSWATokenToKVPoolAllocator(
        unified_buffer=pool,
        kvcache=_FakeUnifiedSWAKVPool(pool),
        device=_DEV,
        full_max_total_num_tokens=n_full,
        swa_max_total_num_tokens=n_swa,
        need_sort=False,
        forward_stream=None,
    )


class TestSharedVirtualIdSpace(unittest.TestCase):
    def test_swa_table_spans_the_owners_id_space(self):
        """Static form: the table has to be wide enough before any alloc runs."""
        for full_layers, swa_layers in ((1, 5), (5, 1), (2, 2)):
            with self.subTest(full_layers=full_layers, swa_layers=swa_layers):
                alloc = _build(200, 20, full_layers, swa_layers)
                owner = alloc.full_attn_allocator
                swa = alloc.swa_attn_allocator
                self.assertGreaterEqual(
                    int(swa.virtual_to_physical.shape[0]),
                    owner.num_virtual_ids + 1,
                    "swa v2p cannot address every id the owner can mint",
                )
                # p2v stays this pool's own business: it is indexed by physical id.
                self.assertEqual(
                    int(swa.physical_to_virtual.shape[0]), swa.num_pages + 1
                )

    def test_churn_past_the_swa_page_count_binds_cleanly(self):
        """Dynamic form: alloc/free until the owner's cursor passes the swa
        side's page count, which is where the narrow table used to be written
        off the end."""
        alloc = _build(200, 20, full_layers=1, swa_layers=5)
        swa_pages = alloc.swa_attn_allocator.num_pages
        highest = 0
        for _ in range(40):
            v = alloc.alloc(4)
            if v is None:
                break
            highest = max(highest, int(v.max()) // alloc.page_size)
            alloc.free(v)
        self.assertGreater(
            highest,
            swa_pages,
            f"churn never reached past the swa side's {swa_pages} pages, so this "
            "test would pass on a narrow table too",
        )


if __name__ == "__main__":
    unittest.main()
