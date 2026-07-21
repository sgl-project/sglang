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
"""Unit tests for the DCP extend-gather prefetcher (SGLANG_DCP_EXTEND_PREFETCH).

The prefetcher pipelines layer L+1's prefix KV all-gather on a side stream while
layer L computes. These tests pin its scheduling contract with a stubbed gather
(no distributed group needed): registry warm-up ordering, signature-mismatch
discard (chunk/batch change), last-layer wrap behavior, and that a consumed
prefetch is bit-identical to the inline gather of the same inputs.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="extra-a", runner_config="1-gpu-large")

from sglang.srt.layers.dcp import comm as dcp_comm


def _mk_attn(layer_id):
    return SimpleNamespace(layer_id=layer_id)


class _FakePool:
    """Pool whose per-layer KV is a deterministic function of (layer, indices)."""

    def get_mla_kv_buffer(self, attn, indices):
        base = float(attn.layer_id + 1)
        n = indices.shape[0]
        k_nope = torch.full((n, 1, 512), base, device=indices.device)
        k_rope = torch.full((n, 1, 64), -base, device=indices.device)
        return k_nope, k_rope


def _stub_gather(k_nope, k_rope, lens_cpu, prefix_starts_cpu=None):
    return torch.cat([k_nope, k_rope], dim=-1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDcpExtendPrefetcher(unittest.TestCase):
    def setUp(self):
        self.pf = dcp_comm._DcpExtendPrefetcher()
        self.pool = _FakePool()
        self.dev = torch.device("cuda")
        self.idx = torch.arange(16, dtype=torch.int64, device=self.dev)
        self.lens = [16]

    def _issue(self, layer_id):
        with mock.patch.object(dcp_comm, "all_gather_kv_cache_for_dcp", _stub_gather):
            self.pf.issue_next(layer_id, self.pool, self.lens, self.idx, 16)

    def test_registry_orders_layers(self):
        for lid in (5, 1, 3):
            self.pf.register(_mk_attn(lid))
        self.assertEqual(self.pf._order, [1, 3, 5])

    def test_consume_matches_inline_gather(self):
        for lid in (0, 1):
            self.pf.register(_mk_attn(lid))
        self._issue(0)  # prefetches layer 1
        sig = self.pf._sig(self.idx, 16)
        got = self.pf.consume(1, sig)
        self.assertIsNotNone(got)
        k_nope, k_rope = self.pool.get_mla_kv_buffer(_mk_attn(1), self.idx)
        want = _stub_gather(k_nope, k_rope, torch.tensor(self.lens))
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(got, want))

    def test_sig_mismatch_discards(self):
        for lid in (0, 1):
            self.pf.register(_mk_attn(lid))
        self._issue(0)
        other_idx = torch.arange(8, dtype=torch.int64, device=self.dev)
        self.assertIsNone(self.pf.consume(1, self.pf._sig(other_idx, 8)))
        # pending cleared — a second consume also returns None
        self.assertIsNone(self.pf.consume(1, self.pf._sig(self.idx, 16)))

    def test_wrong_layer_discards(self):
        for lid in (0, 1, 2):
            self.pf.register(_mk_attn(lid))
        self._issue(0)  # prefetches layer 1
        self.assertIsNone(self.pf.consume(2, self.pf._sig(self.idx, 16)))

    def test_last_layer_does_not_issue(self):
        for lid in (0, 1):
            self.pf.register(_mk_attn(lid))
        self._issue(1)  # layer 1 is last — nothing to prefetch
        self.assertIsNone(self.pf._pending)

    def test_unwarmed_layer_does_not_issue(self):
        self._issue(7)  # nothing registered
        self.assertIsNone(self.pf._pending)


if __name__ == "__main__":
    unittest.main()
