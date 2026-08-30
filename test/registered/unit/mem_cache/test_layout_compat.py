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
"""Unit tests for the page-major envelope byte layout.

The subject here is the ENVELOPE — the byte layout the unified pool stores its
KV in — pinned through ``MHASubPoolSpec``'s offset math. The dense 3-D views
the pool exposes over the same bytes are covered by
``test_unified_mha_views.py``, which also pins the view addressing
against the envelope formula byte for byte.

Verifies that:
1. ``MHASubPoolSpec.layer_k_offset_in_page`` / ``layer_v_offset_in_page`` math
   matches the layout intent at ``page_size == 1`` and ``> 1``.
2. ``move_kv_cache_native`` (the stock per-layer 3-D move) stays byte-exact.

CPU-only — no GPU / Triton needed.

    python -m pytest test/registered/unit/mem_cache/test_layout_compat.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.memory_pool import move_kv_cache_native
from sglang.srt.mem_cache.unified_memory_pool import MHASubPoolSpec

_DEV = "cpu"


def _make_mha_spec(name, grow, layer_num=2, head_num=2, head_dim=4):
    return MHASubPoolSpec(
        name=name,
        layer_num=layer_num,
        head_num=head_num,
        head_dim=head_dim,
        store_dtype=torch.float16,
        grow_direction=grow,
    )


class TestMHASpecLayerOffsets(unittest.TestCase):
    """Verify ``layer_k_offset_in_page`` / ``layer_v_offset_in_page`` math."""

    def test_offsets_at_page_size_1_match_envelope(self):
        spec = _make_mha_spec("full", "up", layer_num=3, head_num=2, head_dim=4)
        # At ps=1, layer-major within a 1-token page IS envelope-per-token.
        # Layer L's K offset = L * (k_row + v_row); V offset = +k_row.
        k_row = spec.k_row_bytes()
        v_row = spec.v_row_bytes()
        for L in range(spec.layer_num):
            self.assertEqual(
                spec.layer_k_offset_in_page(L, page_size=1),
                L * (k_row + v_row),
            )
            self.assertEqual(
                spec.layer_v_offset_in_page(L, page_size=1),
                L * (k_row + v_row) + k_row,
            )

    def test_offsets_at_page_size_gt_1(self):
        spec = _make_mha_spec("full", "up", layer_num=3, head_num=2, head_dim=4)
        ps = 8
        k_row = spec.k_row_bytes()
        v_row = spec.v_row_bytes()
        # Layer L's K block within the page starts at L * ps * (k_row+v_row).
        # V block starts at +ps * k_row.
        for L in range(spec.layer_num):
            self.assertEqual(
                spec.layer_k_offset_in_page(L, page_size=ps),
                L * ps * (k_row + v_row),
            )
            self.assertEqual(
                spec.layer_v_offset_in_page(L, page_size=ps),
                L * ps * (k_row + v_row) + ps * k_row,
            )

    def test_page_bytes(self):
        spec = _make_mha_spec("full", "up", layer_num=3, head_num=2, head_dim=4)
        # page_bytes = page_size * entry_bytes (preserved invariant)
        for ps in [1, 8, 64, 256]:
            self.assertEqual(spec.page_bytes(ps), ps * spec.entry_bytes())


class TestMoveKVCacheNative(unittest.TestCase):
    def test_move_kv_cache_3d_path_unchanged(self):
        """The stock per-layer 3-D move must relocate exactly the named token
        rows, byte-identically — compaction on static pools rides on it."""
        k = [torch.zeros((32, 2, 4), dtype=torch.float16) for _ in range(2)]
        v = [torch.zeros((32, 2, 4), dtype=torch.float16) for _ in range(2)]
        for L in range(2):
            k[L][5] = float(L + 1)
            v[L][5] = -float(L + 1)
        move_kv_cache_native(
            k,
            v,
            tgt_loc=torch.tensor([7], dtype=torch.int64),
            src_loc=torch.tensor([5], dtype=torch.int64),
        )
        for L in range(2):
            self.assertTrue(torch.all(k[L][7] == float(L + 1)))
            self.assertTrue(torch.all(v[L][7] == -float(L + 1)))


if __name__ == "__main__":
    unittest.main()
