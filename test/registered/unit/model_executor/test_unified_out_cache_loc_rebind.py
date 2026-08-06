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
"""Unified-pool physical-loc contract: `forward_batch.out_cache_loc` is
REBOUND to a fresh PHYSICAL tensor exactly once, at ForwardBatch preparation
(`apply_unified_kv_loc_rebind`), and every downstream consumer — attention
backends, model-side pool doors — sees physical ids without translating again.

These tests pin the contract's load-bearing pieces:
  1. the rebind itself: fresh tensor (the ScheduleBatch's aliased tensor stays
     VIRTUAL for scheduler-thread readers), flag + read-rail callable set, and
     the ORDER-CRITICAL hybrid-SWA rule (one virtual id -> TWO physicals: the
     swa rail must be computed from the still-virtual loc BEFORE the full-side
     rebind replaces it);
  2. DP/cuda-graph padding and the spec draft-input backup cover the new swa
     rail (a shorter swa tensor would be silently sliced by
     KVWriteLoc.__post_init__ -> out-of-bounds store);
  3. the TBO split carries the new fields (its strict unknown-field check
     raises otherwise) and token-slices the swa rail;
  4. the pool's model-side write door (`set_mla_kv_buffer`) passes the loc
     through UNTRANSLATED — the "receives physical only" contract;
  5. per-step draft slicing commutes with the v2p map, so slicing the
     pre-rebound (physical) multi-step buffer equals the retired
     translate-then-slice;
  6. COMPLETENESS: every hand-built live forward calls the rebind. Standard
     batches get it from `init_new`, but a module that constructs its own
     ForwardBatch and runs it through `<runner>.forward()` must call the
     helper itself, or the backend tripwire fires at the first replay (and
     without the tripwire it would write VIRTUAL ids as if physical).
"""

import pathlib
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    apply_unified_kv_loc_rebind,
)
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


def _make_runner(v2p=None, swa_map=None):
    """Fake ModelRunner carrying just the two translate handles the hook probes."""
    allocator = SimpleNamespace()
    if v2p is not None:
        # Fresh-tensor gather, like MultiEndedAllocator.translate_kv_loc.
        allocator.translate_kv_loc = lambda t: v2p[t.to(torch.int64)]
    pool = SimpleNamespace()
    if swa_map is not None:
        pool.translate_loc_from_full_to_swa = swa_map
    return SimpleNamespace(token_to_kv_pool_allocator=allocator, token_to_kv_pool=pool)


class TestApplyUnifiedKvLocRebind(CustomTestCase):
    def test_rebind_is_fresh_physical_and_sets_contract(self):
        v2p = torch.arange(100, dtype=torch.int64) + 1000  # virtual v -> v+1000
        virtual = torch.tensor([3, 7, 42], dtype=torch.int64)
        virtual_copy = virtual.clone()
        fb = _make_fb(virtual)
        runner = _make_runner(v2p=v2p)

        apply_unified_kv_loc_rebind(fb, runner)

        # Fresh PHYSICAL tensor; the original (ScheduleBatch-aliased) tensor
        # object and content are untouched.
        self.assertIsNot(fb.out_cache_loc, virtual)
        self.assertTrue(torch.equal(fb.out_cache_loc, virtual + 1000))
        self.assertTrue(torch.equal(virtual, virtual_copy))
        self.assertTrue(fb.out_cache_loc_is_physical)
        self.assertIsNotNone(fb._unified_kv_loc_translate)
        # No SWA pool on this runner -> no swa rail.
        self.assertIsNone(fb.swa_out_cache_loc)

    def test_swa_rail_computed_from_virtual_before_rebind(self):
        v2p = torch.arange(100, dtype=torch.int64) * 2  # full: v -> 2v
        swa_inputs = []

        def swa_map(t):
            swa_inputs.append(t.clone())
            return (t * 3).to(torch.int32)  # swa: v -> 3v, int32 convention

        virtual = torch.tensor([1, 5, 9], dtype=torch.int64)
        fb = _make_fb(virtual)
        runner = _make_runner(v2p=v2p, swa_map=swa_map)

        apply_unified_kv_loc_rebind(fb, runner)

        # ORDER-CRITICAL: the swa map must have received the VIRTUAL ids, not
        # the full-physical result of the rebind.
        self.assertEqual(len(swa_inputs), 1)
        self.assertTrue(torch.equal(swa_inputs[0], virtual))
        self.assertTrue(
            torch.equal(fb.swa_out_cache_loc, (virtual * 3).to(torch.int32))
        )
        self.assertEqual(fb.swa_out_cache_loc.dtype, torch.int32)
        self.assertTrue(torch.equal(fb.out_cache_loc, virtual * 2))

    def test_non_unified_is_a_noop(self):
        virtual = torch.tensor([2, 4], dtype=torch.int64)
        fb = _make_fb(virtual)
        runner = _make_runner()  # allocator without translate_kv_loc

        apply_unified_kv_loc_rebind(fb, runner)

        self.assertIs(fb.out_cache_loc, virtual)  # identity alias preserved
        self.assertFalse(fb.out_cache_loc_is_physical)
        self.assertIsNone(fb.swa_out_cache_loc)
        self.assertIsNone(fb._unified_kv_loc_translate)

    def test_none_loc_is_a_noop(self):
        fb = _make_fb(None)
        runner = _make_runner(v2p=torch.arange(10, dtype=torch.int64))
        apply_unified_kv_loc_rebind(fb, runner)
        self.assertIsNone(fb.out_cache_loc)
        self.assertFalse(fb.out_cache_loc_is_physical)

    def test_empty_loc_rebinds_and_flags(self):
        # Idle/DP-idle batches: empty tensor still gets the contract flag so
        # backend tripwires see a consistent state.
        fb = _make_fb(torch.empty(0, dtype=torch.int64))
        fb.batch_size = 0
        runner = _make_runner(v2p=torch.arange(10, dtype=torch.int64))
        apply_unified_kv_loc_rebind(fb, runner)
        self.assertTrue(fb.out_cache_loc_is_physical)
        self.assertEqual(fb.out_cache_loc.numel(), 0)


class TestPaddingAndBackupCoverSwaRail(CustomTestCase):
    def _fake_runner_for_pad(self):
        return SimpleNamespace(
            attn_backend=SimpleNamespace(get_cuda_graph_seq_len_fill_value=lambda: 0)
        )

    def test_pad_inputs_pads_swa_rail_with_sink_zero(self):
        n, padded = 3, 6
        fb = _make_fb(torch.tensor([11, 12, 13], dtype=torch.int64))
        fb.positions = torch.arange(n, dtype=torch.int64)
        fb.lora_ids = [None] * fb.batch_size
        fb.swa_out_cache_loc = torch.tensor([21, 22, 23], dtype=torch.int32)

        fb._pad_inputs_to_size(self._fake_runner_for_pad(), padded, fb.batch_size)

        self.assertEqual(fb.out_cache_loc.shape[0], padded)
        self.assertEqual(fb.swa_out_cache_loc.shape[0], padded)
        # Padded tails go to slot 0 — the reserved dummy-write sink in both
        # sub-pools' physical spaces.
        self.assertTrue((fb.out_cache_loc[n:] == 0).all())
        self.assertTrue((fb.swa_out_cache_loc[n:] == 0).all())
        # dtype preserved (int32 = the shared read-index kernel convention).
        self.assertEqual(fb.swa_out_cache_loc.dtype, torch.int32)

    def test_draft_input_backup_covers_swa_rail(self):
        n, padded = 2, 4
        fb = _make_fb(torch.tensor([5, 6], dtype=torch.int64))
        fb.positions = torch.arange(n, dtype=torch.int64)
        fb.lora_ids = [None] * fb.batch_size
        fb.swa_out_cache_loc = torch.tensor([7, 8], dtype=torch.int32)
        fb.spec_info = SimpleNamespace(
            is_draft_input=lambda: True,
            hidden_states=torch.zeros(n, 4),
            topk_p=None,
            topk_index=None,
            draft_probs=None,
            num_correct_drafts=None,
        )

        fb._pad_inputs_to_size(self._fake_runner_for_pad(), padded, fb.batch_size)

        # Both write rails are backed up together, so the post-forward restore
        # rebinds them as a pair.
        self.assertTrue(hasattr(fb, "output_cache_loc_backup"))
        self.assertTrue(hasattr(fb, "swa_output_cache_loc_backup"))
        self.assertIs(fb.output_cache_loc_backup, fb.out_cache_loc)
        self.assertIs(fb.swa_output_cache_loc_backup, fb.swa_out_cache_loc)


class TestTboSplitCarriesContractFields(CustomTestCase):
    def test_filter_batch_strict_check_and_swa_slice(self):
        from sglang.srt.batch_overlap import two_batch_overlap as tbo

        n = 4
        fb = _make_fb(torch.tensor([10, 11, 12, 13], dtype=torch.int64))
        fb.batch_size = n
        fb.input_ids = torch.arange(n, dtype=torch.int64)
        fb.positions = torch.arange(n, dtype=torch.int64)
        fb.req_pool_indices = torch.arange(n, dtype=torch.int64)
        fb.seq_lens = torch.ones(n, dtype=torch.int64)
        fb.swa_out_cache_loc = torch.tensor([20, 21, 22, 23], dtype=torch.int32)
        fb.out_cache_loc_is_physical = True
        fb._unified_kv_loc_translate = lambda t: t

        with patch.object(
            tbo,
            "get_parallel",
            return_value=SimpleNamespace(
                attn_tp_size=1, tp_size=1, moe_dense_tp_size=1
            ),
        ):
            child = tbo.TboForwardBatchPreparer.filter_batch(
                fb,
                start_token_index=1,
                end_token_index=3,
                start_seq_index=1,
                end_seq_index=3,
                out_num_token_non_padded=torch.tensor([2], dtype=torch.int32),
            )

        # The strict unknown-field check did not raise, the swa rail is
        # token-sliced like out_cache_loc, and the contract flag + callable are
        # inherited by the child.
        self.assertTrue(
            torch.equal(child.out_cache_loc, torch.tensor([11, 12], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(
                child.swa_out_cache_loc, torch.tensor([21, 22], dtype=torch.int32)
            )
        )
        self.assertTrue(child.out_cache_loc_is_physical)
        self.assertIsNotNone(child._unified_kv_loc_translate)


class TestSetMlaKvBufferDoorContract(CustomTestCase):
    def test_write_door_passes_loc_untranslated(self):
        """The model-side write door receives PHYSICAL ids and must never
        translate: the exact loc object handed in reaches the leaf pool."""
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        recorded = {}

        class _RecordingLeafPool:
            def set_mla_kv_buffer(self, layer, loc, k_nope, k_rope):
                recorded["loc"] = loc

            def get_kv_size_bytes(self):
                return 0  # init computes mem_usage from this

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

        loc = torch.tensor([4, 5, 6], dtype=torch.int64)
        layer = SimpleNamespace(layer_id=0)
        pool.set_mla_kv_buffer(layer, loc, torch.zeros(3, 1, 8), torch.zeros(3, 1, 8))

        self.assertIs(recorded["loc"], loc)


class TestReadRailTranslatesAtProduction(CustomTestCase):
    """The model-door READ indices (req_to_token-derived, VIRTUAL under the
    unified pool) are translated at their PRODUCTION site — the cache then
    holds the physical result and the pool door never translates."""

    def _fb_for_one_shot(self, translate):
        fb = _make_fb(torch.tensor([1, 2], dtype=torch.int64))
        fb.batch_size = 2
        fb.seq_lens = torch.tensor([2, 3], dtype=torch.int64)
        fb.seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
        fb.req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
        fb._unified_kv_loc_translate = translate
        return fb

    def test_one_shot_indices_translated_once_and_cached(self):
        from sglang.srt.model_executor import forward_batch_deepseek_mha_mixin as mix

        calls = []
        sentinel = torch.arange(5, dtype=torch.int64) + 5000

        def translate(t):
            calls.append(t)
            return sentinel

        fb = self._fb_for_one_shot(translate)
        fake_pool = SimpleNamespace(
            req_to_token=torch.zeros((4, 16), dtype=torch.int32)
        )
        with (
            patch.object(mix, "get_req_to_token_pool", return_value=fake_pool),
            patch.object(mix, "create_flashinfer_kv_indices_triton") as kern,
        ):
            r1 = fb.fetch_mha_one_shot_kv_indices()
            r2 = fb.fetch_mha_one_shot_kv_indices()

        self.assertIs(r1, sentinel)  # production site translated
        self.assertIs(r2, sentinel)  # cache holds the TRANSLATED result
        self.assertEqual(len(calls), 1)  # translated exactly once
        self.assertEqual(calls[0].dtype, torch.int32)  # raw producer output

    def test_one_shot_indices_noop_when_non_unified(self):
        from sglang.srt.model_executor import forward_batch_deepseek_mha_mixin as mix

        fb = self._fb_for_one_shot(None)
        fake_pool = SimpleNamespace(
            req_to_token=torch.zeros((4, 16), dtype=torch.int32)
        )
        with (
            patch.object(mix, "get_req_to_token_pool", return_value=fake_pool),
            patch.object(mix, "create_flashinfer_kv_indices_triton") as kern,
        ):
            r = fb.fetch_mha_one_shot_kv_indices()
        # Non-unified: the raw int32 producer output passes through untouched.
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


class TestPerStepSlicingCommutesWithV2p(CustomTestCase):
    def test_slice_of_translated_equals_translate_of_slice(self):
        """Slicing the pre-rebound (physical) multi-step buffer equals the
        retired translate-then-slice: an elementwise v2p gather commutes with
        the per-step gather, including the non-contiguous permutation the
        `.contiguous()` guard exists for."""
        from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc

        bs, topk, num_steps = 3, 2, 4
        v2p = torch.randperm(1024, dtype=torch.int64)
        virtual = torch.arange(bs * topk * num_steps, dtype=torch.int64) + 100

        translated_then_sliced = per_step_draft_out_cache_loc(
            v2p[virtual], bs, topk, num_steps
        )
        sliced_then_translated_rows = [
            v2p[row]
            for row in per_step_draft_out_cache_loc(virtual, bs, topk, num_steps)
        ]
        for i in range(num_steps):
            self.assertTrue(
                torch.equal(translated_then_sliced[i], sliced_then_translated_rows[i])
            )


class TestHandBuiltForwardsRebind(CustomTestCase):
    """Completeness scan over the speculative workers.

    `init_new` rebinds every standard ForwardBatch, so the only way to reach a
    backend with a virtual write loc is a module that builds its own batch and
    runs it. Those are exactly the modules that both construct a
    `ForwardBatch(` and call `<runner>.forward(`; each must also call
    `apply_unified_kv_loc_rebind`. Cuda-graph runners fall out on their own —
    they build capture batches and never call a ModelRunner forward, and
    capture is exempt by contract (runner-built, zero-filled static buffers,
    and slot 0 is the reserved sink in every id space).

    Guards the failure mode this case was written for: a new speculative
    algorithm hand-builds a draft forward and forgets the rebind. That is how
    the DSPARK draft path shipped broken -- it inherited a translate the
    backends used to do for it, and nothing flagged its absence until a GPU
    run asserted at the first decode step.
    """

    def test_every_hand_built_live_forward_calls_the_rebind(self):
        # Anchor on a concrete module: `sglang.srt.speculative` is a namespace
        # package, so its own `__file__` is None.
        from sglang.srt.speculative import dflash_worker_v2 as _anchor_module

        root = pathlib.Path(_anchor_module.__file__).parent
        offenders = []
        checked = []
        for path in sorted(root.rglob("*.py")):
            text = path.read_text()
            if not re.search(r"\bForwardBatch\(", text):
                continue
            if not re.search(r"\w*model_runner\.forward\(", text):
                continue
            checked.append(path.name)
            if "apply_unified_kv_loc_rebind" not in text:
                offenders.append(str(path.relative_to(root)))

        self.assertEqual(
            offenders,
            [],
            "these modules hand-build a ForwardBatch and run it through a "
            "ModelRunner, but never call apply_unified_kv_loc_rebind, so under "
            "--enable-unified-memory their write loc reaches the attention "
            f"backend as VIRTUAL ids: {offenders}",
        )
        # The scan must actually be looking at something: if a refactor moves
        # these forwards elsewhere, an empty sweep would pass vacuously.
        self.assertGreaterEqual(
            len(checked),
            2,
            f"expected the DFLASH and DSPARK draft forwards; scanned {checked}",
        )


if __name__ == "__main__":
    unittest.main()
