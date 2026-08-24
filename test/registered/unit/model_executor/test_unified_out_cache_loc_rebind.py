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
"""ForwardBatch construction wires the unified write-loc rebind (phase 1).

The write contract has two phases: `init_new` rebinds the FULL side once
(phase 1), and the sliding-window side derives at the per-batch build,
pointwise from the kernel-facing values (phase 2 — semantics pinned in
test_kv_index_translator.py over the real composite). Phase 2 needs no
ForwardBatch-side wiring at all: pads, slices, and buffer copies preserve
the values it derives from. So this file pins the ONE call site the
contract hangs on:

  `init_new` calls `kv_index_translator.rebind_write_loc` — a construction
  path that skipped it would ship VIRTUAL write ids to the kernels, a
  silent wrong-slot store under the unified pool;

plus an end-to-end run of the REAL `_pad_inputs_to_size` against a live
translator: pad lanes are zeros, and zeros derive to the slot-0 sink —
the property that lets the pad need no handover.

    python -m pytest test/registered/unit/model_executor/test_unified_out_cache_loc_rebind.py -v
"""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_DEV = "cpu"


def _make_fb(out_cache_loc, **kw):
    """Minimal ForwardBatch with only the required core fields."""
    n = 0 if out_cache_loc is None else out_cache_loc.shape[0]
    defaults = dict(
        forward_mode=ForwardMode.DECODE,
        batch_size=max(n, 1),
        input_ids=torch.zeros(max(n, 1), dtype=torch.int64),
        req_pool_indices=torch.zeros(max(n, 1), dtype=torch.int64),
        seq_lens=torch.ones(max(n, 1), dtype=torch.int64),
        out_cache_loc=out_cache_loc,
        seq_lens_sum=max(n, 1),
    )
    defaults.update(kw)
    return ForwardBatch(**defaults)


def _armed_source(v2p, swa_map):
    """A KVIndexTranslator hand-armed with fake translates: this file pins the
    ForwardBatch-side wiring, not the composite's formulas (those are pinned
    in test_kv_index_translator.py over the real allocator)."""
    src = KVIndexTranslator(
        req_to_token=torch.zeros((1, 4), dtype=torch.int64),
        token_to_kv_pool_allocator=SimpleNamespace(),
        token_to_kv_pool=SimpleNamespace(),
        page_size=1,
        device=_DEV,
    )
    src.is_translating = True
    src._translate_full = lambda t, out=None: v2p[t.to(torch.int64)]
    # Phase 2 derives from DENSE values through p2v + the swa v2p; arm the
    # inverse of the fake v2p (ps=1, both multipliers 1: dense == physical,
    # and the expected swa loc for virtual t is swa_map[t]).
    p2v = torch.zeros(int(v2p.max()) + 1, dtype=torch.int64)
    p2v[v2p] = torch.arange(v2p.numel(), dtype=torch.int64)
    src._full_p2v_table = p2v
    src._swa_v2p_table = swa_map
    src._full_page_multiplier = 1
    src._swa_page_multiplier = 1
    return src


def _call_names(func) -> list:
    """Dotted call targets appearing in `func`'s body, e.g.
    'model_runner.kv_index_translator.rebind_write_loc'."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            parts = []
            cur = node.func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            names.append(".".join(reversed(parts)))
    return names


class TestForwardBatchWiring(CustomTestCase):
    """Critical-path bookkeeping: the construction-time call sites."""

    def test_init_new_calls_the_rebind(self):
        self.assertIn(
            "model_runner.kv_index_translator.rebind_write_loc",
            _call_names(ForwardBatch.init_new.__func__),
            "init_new must rebind the write loc through the source; a batch "
            "built without it ships virtual ids to the kernels",
        )


class TestPadComposesWithDerivation(CustomTestCase):
    def _fake_runner_for_pad(self, src):
        return SimpleNamespace(
            attn_backend=SimpleNamespace(get_cuda_graph_seq_len_fill_value=lambda: 0),
            kv_index_translator=src,
        )

    def test_pad_lanes_derive_to_sink_and_slices_stay_pointwise(self):
        """The REAL `_pad_inputs_to_size` composes with phase 2: pad lanes are
        zeros, zeros derive to the slot-0 sink, and any slice of the padded
        tensor (the TBO-child shape) derives pointwise — no handover call
        exists for the pad to make."""
        n, padded = 3, 6
        v2p = torch.arange(64, dtype=torch.int64) * 3
        swa_map = torch.arange(64, dtype=torch.int64) * 5
        src = _armed_source(v2p, swa_map)
        virt = torch.tensor([11, 12, 13], dtype=torch.int64)
        fb = _make_fb(virt.clone())
        fb.positions = torch.arange(n, dtype=torch.int64)
        fb.lora_ids = [None] * fb.batch_size
        src.rebind_write_loc(fb)
        self.assertTrue(torch.equal(fb.out_cache_loc, v2p[virt]))

        fb._pad_inputs_to_size(self._fake_runner_for_pad(src), padded, fb.batch_size)

        self.assertEqual(fb.out_cache_loc.shape[0], padded)
        # Padded tail lanes go to slot 0 — the reserved dummy-write sink.
        self.assertTrue(bool((fb.out_cache_loc[n:] == 0).all()))
        loc = src._swa_write_loc_unified(fb.out_cache_loc)
        self.assertTrue(torch.equal(loc[:n], swa_map[virt]))
        self.assertTrue(bool((loc[n:] == 0).all()))
        self.assertEqual(loc.dtype, torch.int64)
        # The TBO-child shape: a slice of the PADDED tensor derives pointwise.
        sub = src._swa_write_loc_unified(fb.out_cache_loc[1:5])
        self.assertTrue(torch.equal(sub, loc[1:5]))

    def test_empty_loc_rebinds_to_empty(self):
        src = _armed_source(
            torch.arange(8, dtype=torch.int64), torch.arange(8, dtype=torch.int64)
        )
        fb = _make_fb(torch.empty(0, dtype=torch.int64))
        src.rebind_write_loc(fb)
        self.assertEqual(fb.out_cache_loc.numel(), 0)
        self.assertEqual(src._swa_write_loc_unified(fb.out_cache_loc).numel(), 0)


if __name__ == "__main__":
    unittest.main()
