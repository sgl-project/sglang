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
"""ForwardBatch construction wires the unified write-loc rebind.

`init_new` must call `kv_index_translator.rebind_write_loc`: a construction
path that skips it ships VIRTUAL write ids to the kernels, a silent
wrong-slot store. Also runs the REAL `_pad_inputs_to_size` against a live
translator, since pad lanes are zeros and zeros must derive to the slot-0
sink. Sliding-window semantics are pinned in test_kv_index_translator.py.

    python -m pytest test/registered/unit/model_executor/test_unified_out_cache_loc_rebind.py -v
"""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import create_autospec

import torch

from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

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
    # The WRITE loc has its own translate because under DCP it arrives widened;
    # at dcp_size == 1 it is the read translate, so arm it with the same fake.
    src._translate_write_full = src._translate_full
    # Phase 2 derives from kernel-facing values through p2v + the swa v2p; arm
    # the inverse of the fake v2p (ps=1, both multipliers 1: kernel == physical,
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
        tensor (the TBO-child shape) derives pointwise -- no handover call
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
        # Padded tail lanes go to slot 0 -- the reserved dummy-write sink.
        self.assertTrue(bool((fb.out_cache_loc[n:] == 0).all()))
        loc = src._swa_write_loc_unified(fb.out_cache_loc)
        self.assertTrue(torch.equal(loc[:n], swa_map[virt]))
        self.assertTrue(bool((loc[n:] == 0).all()))
        self.assertEqual(loc.dtype, torch.int64)
        # The TBO-child shape: a slice of the PADDED tensor derives pointwise.
        sub = src._swa_write_loc_unified(fb.out_cache_loc[1:5])
        self.assertTrue(torch.equal(sub, loc[1:5]))

    def test_the_probe_separates_kernel_facing_from_virtual_ids(self):
        """A skipped rebind is the failure mode this contract has no other
        guard against: virtual ids stay inside the OOB probe's bounds (they are
        `blocks_per_page` times SMALLER than a kernel-facing id), so the store lands on
        the wrong slots and only the output is wrong. The kernel-facing probe
        is what separates them -- the in-page offset of a kernel-facing id is always
        below page_size, and a virtual id's is not unless it happens to fall in
        the first block."""
        for page_size, blocks in ((1, 8), (4, 6)):
            with self.subTest(page_size=page_size, blocks=blocks):
                stride = page_size * blocks
                virt = torch.arange(1, 2 * stride, dtype=torch.int64)
                kernel = (virt // page_size) * stride + virt % page_size
                in_space = kernel % stride < page_size
                self.assertTrue(bool(in_space.all()), "kernel-facing ids must pass")
                # Virtual ids pass only in the first block; that is why the
                # probe needs a batch, not one id, to be conclusive.
                caught = ~(virt % stride < page_size)
                self.assertTrue(bool(caught.any()), "virtual ids must be caught")

    def test_empty_loc_rebinds_to_empty(self):
        src = _armed_source(
            torch.arange(8, dtype=torch.int64), torch.arange(8, dtype=torch.int64)
        )
        fb = _make_fb(torch.empty(0, dtype=torch.int64))
        src.rebind_write_loc(fb)
        self.assertEqual(fb.out_cache_loc.numel(), 0)
        self.assertEqual(src._swa_write_loc_unified(fb.out_cache_loc).numel(), 0)


class TestReadRailTranslatesAtProduction(CustomTestCase):
    """The model-door READ indices (req_to_token-derived, VIRTUAL under the
    unified pool) are translated at their PRODUCTION site -- the cache then
    holds the kernel-facing result and the pool door never translates."""

    def _fb_for_one_shot(self):
        fb = _make_fb(torch.tensor([1, 2], dtype=torch.int64))
        fb.batch_size = 2
        fb.seq_lens = torch.tensor([2, 3], dtype=torch.int64)
        fb.seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
        fb.req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
        return fb

    def test_one_shot_indices_translated_once_and_cached(self):
        from unittest.mock import patch

        from sglang.srt.model_executor import forward_batch_deepseek_mha_mixin as mix

        calls = []
        sentinel = torch.arange(5, dtype=torch.int64) + 5000

        def translate(t):
            calls.append(t)
            return sentinel

        fb = self._fb_for_one_shot()
        fake_pool = SimpleNamespace(
            req_to_token=torch.zeros((4, 16), dtype=torch.int32)
        )
        # autospec, not a bare namespace: setting a name the translator does
        # not have raises, so renaming the method breaks this test loudly.
        fake_translator = create_autospec(KVIndexTranslator, instance=True)
        fake_translator.translate_full_attn_ids = translate
        fake_backend = SimpleNamespace(kv_index_translator=fake_translator)
        with (
            patch.object(mix, "get_req_to_token_pool", return_value=fake_pool),
            patch.object(mix, "get_attn_backend", return_value=fake_backend),
            patch.object(mix, "create_flashinfer_kv_indices_triton"),
        ):
            r1 = fb.fetch_mha_one_shot_kv_indices()
            r2 = fb.fetch_mha_one_shot_kv_indices()

        self.assertIs(r1, sentinel)  # production site translated
        self.assertIs(r2, sentinel)  # cache holds the TRANSLATED result
        self.assertEqual(len(calls), 1)  # translated exactly once
        self.assertEqual(calls[0].dtype, torch.int32)  # raw producer output

    def test_one_shot_indices_noop_on_unmigrated_backend(self):
        from unittest.mock import patch

        from sglang.srt.model_executor import forward_batch_deepseek_mha_mixin as mix

        fb = self._fb_for_one_shot()
        fake_pool = SimpleNamespace(
            req_to_token=torch.zeros((4, 16), dtype=torch.int32)
        )
        # A backend that never set the attribute inherits the base-class None.
        fake_backend = SimpleNamespace(kv_index_translator=None)
        with (
            patch.object(mix, "get_req_to_token_pool", return_value=fake_pool),
            patch.object(mix, "get_attn_backend", return_value=fake_backend),
            patch.object(mix, "create_flashinfer_kv_indices_triton"),
        ):
            r = fb.fetch_mha_one_shot_kv_indices()
        # The raw int32 producer output passes through untouched.
        self.assertEqual(r.dtype, torch.int32)

    def test_get_mla_kv_buffer_door_passes_loc_untranslated(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        recorded = {}

        class _RecordingLeafPool:
            def get_mla_kv_buffer(self, layer, loc, dst_dtype):
                recorded["loc"] = loc
                return None, None

            def get_kv_size_bytes(self):
                return 0

        pool = HybridLinearKVPool(
            size=16,
            dtype=torch.float16,
            page_size=1,
            head_num=1,
            head_dim=8,
            full_attention_layer_ids=[0],
            device=_DEV,
            mamba_pool=SimpleNamespace(get_size_per_token=lambda: 0),
            enable_memory_saver=False,
            use_mla=True,
            start_layer=0,
            full_kv_pool=_RecordingLeafPool(),
        )
        loc = torch.tensor([9, 10], dtype=torch.int64)
        pool.get_mla_kv_buffer(SimpleNamespace(layer_id=0), loc, torch.float16)
        self.assertIs(recorded["loc"], loc)


if __name__ == "__main__":
    unittest.main()
