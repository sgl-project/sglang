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
"""FlashInfer's and TRTLLM MHA's per-step sliding-window reads and writes for a
multi-step draft, and FlashInfer's draft-extend window.

A draft whose sliding-window layers live in the host's window sub-pool must
read each draft step's own window and write its own slots: rows past a step's
window are reclaimed and reused. Pinned, with stand-in wrappers:
  - FlashInfer's window wrapper takes the draft backend's window rail and its
    host indptr, untranslated, while the full-attention wrapper keeps the draft
    stream; with no rail the window wrapper plans the draft stream too;
  - the window rail's host indptr mirrors the kernel's clipped prefix sums;
  - a multi-step draft step's graph buffers keep the window wrapper apart from
    the full-attention one;
  - an EAGLE draft extend's window wrapper reads the tail that covers every
    row's window, from its start offset, under the plan-time window mask;
  - TRTLLM MHA binds each step's own write ids while it builds that step's
    metadata, eager and out of graph, and a static pool's captured window
    writes take that step's slots.

    python -m pytest test/registered/unit/layers/attention/test_draft_window_rails_flashinfer_trtllm.py -v
"""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import flashinfer_backend as fb
from sglang.srt.layers.attention import trtllm_mha_backend as tb
from sglang.srt.mem_cache.kv_loc_plan import IdSpaceKind
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Wrapper:
    """Records each plan call; ``fast`` installs a sync-free plan stand-in."""

    def __init__(self, *, fast: bool):
        self.calls = []
        self.is_cuda_graph_enabled = False
        self._custom_mask_buf = None
        self.begin_forward = partial(_fast_plan, self) if fast else self._plan

    def _plan(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def _fast_plan(wrapper, *args, **kwargs):
    wrapper.calls.append((args, kwargs))


class _NoTranslate:
    def translate_loc_from_full_to_swa(self, ids):
        raise AssertionError("the window rail is already in its storage's ids")


class TestFlashInferDecodeWindowRail(CustomTestCase):
    def _updater(self, *, swa_pool, translated):
        up = object.__new__(fb.FlashInferIndicesUpdaterDecode)
        up.attn_backend = SimpleNamespace(
            kv_index_translator=SimpleNamespace(reads_are_translated=translated)
        )
        up.num_qo_heads, up.num_kv_heads, up.head_dim = 4, 1, 8
        up.data_type = up.q_data_type = torch.bfloat16
        up.kv_last_page_len = torch.ones(8, dtype=torch.int32)
        up._swa_kv_pool = swa_pool
        up.sliding_window_size = 3
        up.kv_indptr = [torch.zeros(9, dtype=torch.int32) for _ in range(2)]
        return up

    def _run(self, up, spec_info):
        wrappers = [_Wrapper(fast=True), _Wrapper(fast=True)]
        seq_lens = torch.tensor([10, 2], dtype=torch.int32)
        full_host = torch.tensor([0, 11, 14], dtype=torch.int32)
        with (
            patch.object(fb, "fast_decode_plan", _fast_plan, create=True),
            patch.object(fb, "global_override_indptr_cpu", full_host),
        ):
            up.update_sliding_window(
                seq_lens,
                seq_lens.clone(),
                int(seq_lens.sum()),
                wrappers,
                None,
                spec_info,
                plan=None,
                req_pool_indices=torch.arange(2),
            )
        return wrappers, full_host, spec_info.window_kv_indptr_cpu

    def _spec(self, *, rail: bool):
        return SimpleNamespace(
            kv_indptr=torch.tensor([0, 11, 14], dtype=torch.int32),
            kv_indices=torch.arange(14, dtype=torch.int32),
            window_kv_indptr=(
                torch.tensor([0, 4, 7], dtype=torch.int32) if rail else None
            ),
            window_kv_indices=(
                torch.arange(100, 107, dtype=torch.int32) if rail else None
            ),
            window_kv_indptr_cpu=(
                torch.tensor([0, 4, 7], dtype=torch.int32) if rail else None
            ),
        )

    def test_the_window_wrapper_reads_the_rail(self):
        for translated in (True, False):
            with self.subTest(translated=translated):
                spec = self._spec(rail=True)
                wrappers, full_host, window_host = self._run(
                    self._updater(swa_pool=_NoTranslate(), translated=translated),
                    spec,
                )
                ((window_args, window_kw),) = wrappers[0].calls
                self.assertIs(window_args[0], spec.window_kv_indptr)
                self.assertIs(window_args[1], spec.window_kv_indices)
                self.assertIs(window_kw["global_override_indptr_cpu"], window_host)
                ((full_args, full_kw),) = wrappers[1].calls
                self.assertIs(full_args[0], spec.kv_indptr)
                self.assertIs(full_args[1], spec.kv_indices)
                self.assertIs(full_kw["global_override_indptr_cpu"], full_host)

    def test_without_a_rail_the_window_wrapper_keeps_the_draft_stream(self):
        spec = self._spec(rail=False)
        swa_pool = SimpleNamespace(translate_loc_from_full_to_swa=lambda ids: ids)
        wrappers, full_host, _ = self._run(
            self._updater(swa_pool=swa_pool, translated=False), spec
        )
        ((window_args, window_kw),) = wrappers[0].calls
        self.assertIs(window_args[0], spec.kv_indptr)
        self.assertIs(window_kw["global_override_indptr_cpu"], full_host)


class TestDraftWindowIndptrHost(CustomTestCase):
    def test_host_indptr_matches_the_clipped_prefix_sums(self):
        lengths = torch.tensor([5, 1, 300, 0])
        topk, num_steps, window = 2, 3, 129
        got = fb._build_draft_window_indptr_host(
            lengths, topk=topk, num_steps=num_steps, window=window
        )
        for step in range(num_steps):
            per_branch = [
                min(length + step + 1, window)
                for length in lengths.tolist()
                for _ in range(topk)
            ]
            expected = torch.cumsum(torch.tensor([0] + per_branch), 0).to(torch.int32)
            torch.testing.assert_close(got[step], expected, rtol=0, atol=0)


class TestFlashInferStepGraphBuffers(CustomTestCase):
    def _backend(self, dispatch_reason, num_wrappers):
        be = object.__new__(fb.FlashInferAttnBackend)
        be.dispatch_reason = dispatch_reason
        be.num_wrappers = num_wrappers
        be.max_context_len = 16
        be.use_sliding_window_kv_pool = False
        be.skip_prefill = True
        return be

    def test_a_window_step_keeps_its_wrappers_apart(self):
        be = self._backend(fb.WrapperDispatch.SLIDING_WINDOW, 2)
        full = torch.zeros(64, dtype=torch.int32)
        window = torch.zeros(32, dtype=torch.int32)
        be.init_cuda_graph_state(
            4, 4, kv_indices_buf=full, window_kv_indices_buf=window
        )
        self.assertIs(be.cuda_graph_kv_indices[0], window)
        self.assertIs(be.cuda_graph_kv_indices[1], full)

    def test_a_single_wrapper_step_is_unchanged(self):
        be = self._backend(None, 1)
        full = torch.zeros(64, dtype=torch.int32)
        be.init_cuda_graph_state(4, 4, kv_indices_buf=full)
        self.assertEqual(len(be.cuda_graph_kv_indices), 1)
        self.assertIs(be.cuda_graph_kv_indices[0], full)


class _DraftExtendStub(SpecInput):
    def __init__(self):
        super().__init__(SpecInputType.EAGLE_DRAFT_EXTEND)
        self.num_tokens_per_req = 4
        self.got = {}

    def generate_attn_arg_prefill(self, **kwargs):
        self.got = kwargs
        return (
            torch.zeros(16, dtype=torch.int32),
            torch.zeros(3, dtype=torch.int32),
            torch.zeros(3, dtype=torch.int32),
            None,
        )


class TestFlashInferDraftExtendWindow(CustomTestCase):
    W = 8

    def _updater(self):
        up = object.__new__(fb.FlashInferIndicesUpdaterPrefill)
        up.sliding_window_size = self.W
        up.prefill_wrapper_ragged = None
        up._swa_kv_pool = None
        up.kv_indptr = [torch.zeros(4, dtype=torch.int32) for _ in range(2)]
        up.qo_indptr = [torch.zeros(4, dtype=torch.int32) for _ in range(2)]
        return up

    def _calls(self, spec_info, seq_lens):
        up = self._updater()
        calls = []
        up.call_begin_forward = lambda *args, **kwargs: calls.append((args, kwargs))
        up.update_sliding_window(
            torch.arange(seq_lens.numel()),
            seq_lens,
            seq_lens.clone(),
            int(seq_lens.sum()),
            None,
            [None, None],
            False,
            None,
            spec_info,
            plan=None,
        )
        return calls

    def test_the_window_covers_every_row_of_the_extend(self):
        # Rows are laid out accepted first: a rejected tail still queries, so
        # the reads cover W keys before the first row of the whole width.
        seq_lens = torch.tensor([30, 5, 12])
        spec = SimpleNamespace(
            spec_input_type=SpecInputType.EAGLE_DRAFT_EXTEND,
            num_tokens_per_req=4,
            num_accept_tokens=torch.tensor([1, 4, 2]),
        )
        (window_args, window_kw), (full_args, full_kw) = self._calls(spec, seq_lens)
        lens = torch.clamp(seq_lens, max=self.W + 4)
        torch.testing.assert_close(window_args[3], lens, rtol=0, atol=0)
        self.assertEqual(window_args[4], int(lens.sum()))
        torch.testing.assert_close(window_args[7], seq_lens - lens, rtol=0, atol=0)
        self.assertEqual(window_kw["window_left"], self.W)
        torch.testing.assert_close(full_args[3], seq_lens, rtol=0, atol=0)
        self.assertEqual(full_kw["window_left"], -1)

    def test_the_extend_window_leaves_a_verify_plan_alone(self):
        """Scope only: a target verify keeps the generic plan, whose window
        reads start at position 0 (an inherited issue, apart from the drafts)."""
        seq_lens = torch.tensor([30, 5])
        spec = SimpleNamespace(
            spec_input_type=SpecInputType.EAGLE_VERIFY, num_accept_tokens=None
        )
        (window_args, window_kw), _ = self._calls(spec, seq_lens)
        torch.testing.assert_close(
            window_args[3], torch.clamp(seq_lens, max=self.W), rtol=0, atol=0
        )
        self.assertEqual(window_kw["window_left"], -1)

    def test_the_start_offset_reaches_the_draft_extend_args(self):
        up = self._updater()
        up.attn_backend = SimpleNamespace(
            kv_index_translator=SimpleNamespace(reads_are_translated=True)
        )
        up.num_qo_heads, up.num_kv_heads, up.head_dim = 4, 1, 8
        up.q_data_type = up.data_type = torch.bfloat16
        up.kv_last_page_len = torch.ones(8, dtype=torch.int32)
        spec = _DraftExtendStub()
        start = torch.tensor([18, 0])
        wrapper = _Wrapper(fast=False)
        up.call_begin_forward(
            None,
            wrapper,
            torch.arange(2),
            torch.tensor([12, 5]),
            17,
            torch.tensor([30, 5]),
            None,
            start,
            up.kv_indptr[0],
            up.qo_indptr[0],
            False,
            spec,
            use_sliding_window_kv_pool=True,
            window_left=self.W,
            plan=None,
        )
        self.assertIs(spec.got["kv_start_idx"], start)
        self.assertEqual(spec.got["kind"], IdSpaceKind.SLIDING_WINDOW)
        ((_, plan_kw),) = wrapper.calls
        self.assertEqual(plan_kw["window_left"], self.W)


class TestEagleDraftExtendStartOffset(CustomTestCase):
    def test_the_translator_gathers_from_the_start_offset(self):
        spec = EagleDraftExtendInput(
            num_correct_drafts=torch.tensor([0, 1]), num_tokens_per_req=4
        )
        got = {}
        translator = SimpleNamespace(
            pack_read_stream=lambda plan, **kwargs: got.update(kwargs)
        )
        start = torch.tensor([1, 2])
        spec.generate_attn_arg_prefill(
            req_pool_indices=torch.arange(2),
            paged_kernel_lens=torch.tensor([5, 6]),
            paged_kernel_lens_sum=11,
            translator=translator,
            plan=None,
            kind=IdSpaceKind.SLIDING_WINDOW,
            kv_start_idx=start,
        )
        self.assertIs(got["kv_start_idx"], start)
        self.assertEqual(got["kind"], IdSpaceKind.SLIDING_WINDOW)


class _Step:
    use_sliding_window_kv_pool = True

    def __init__(self, seen):
        self.seen = seen

    def _record(self, forward_batch):
        self.seen.append(
            (forward_batch.out_cache_loc.clone(), forward_batch.kv_loc_cols)
        )

    def init_forward_metadata(self, forward_batch):
        self._record(forward_batch)

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        self._record(forward_batch)


class TestTRTLLMDraftStepWrites(CustomTestCase):
    bs, topk, num_steps = 3, 1, 3

    def _backend(self, seen):
        be = object.__new__(tb.TRTLLMHAAttnMultiStepDraftBackend)
        be.topk, be.speculative_num_steps = self.topk, self.num_steps
        be.attn_backends = [_Step(seen) for _ in range(self.num_steps - 1)]
        return be

    def _batch(self):
        loc = torch.arange(100, 100 + self.bs * self.topk * self.num_steps)
        return SimpleNamespace(
            batch_size=self.bs,
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(self.bs),
            seq_lens=torch.full((self.bs,), 7),
            seq_lens_sum=7 * self.bs,
            seq_lens_cpu=None,
            encoder_lens=None,
            out_cache_loc=loc,
            kv_loc_plan=None,
            kv_loc_cols=None,
            spec_info=SimpleNamespace(is_draft_input=lambda: True),
        )

    def _check(self, seen, loc):
        per_step = per_step_draft_out_cache_loc(loc, self.bs, self.topk, self.num_steps)
        self.assertEqual(len(seen), self.num_steps - 1)
        for step, (step_loc, cols) in enumerate(seen):
            torch.testing.assert_close(step_loc, per_step[step], rtol=0, atol=0)
            self.assertEqual(cols, slice(step, None, self.num_steps))

    def test_each_step_binds_its_own_writes(self):
        seen = []
        batch = self._batch()
        loc = batch.out_cache_loc
        self._backend(seen).init_forward_metadata(batch)
        self._check(seen, loc)
        self.assertIs(batch.out_cache_loc, loc)
        self.assertIsNone(batch.kv_loc_cols)

    def test_each_step_binds_its_own_writes_out_of_graph(self):
        seen = []
        batch = self._batch()
        self._backend(seen).init_forward_metadata_out_graph(batch)
        self._check(seen, batch.out_cache_loc)

    def _apply(self, *, translating):
        be = object.__new__(tb.TRTLLMHAAttnBackend)
        metadata = tb.TRTLLMMHAMetadata()
        metadata.cache_seqlens_int32 = torch.zeros(self.bs, dtype=torch.int32)
        metadata.cu_seqlens_k = torch.zeros(self.bs + 1, dtype=torch.int32)
        metadata.page_table = torch.zeros((self.bs, 4), dtype=torch.int32)
        metadata.swa_page_table = torch.zeros((self.bs, 4), dtype=torch.int32)
        metadata.swa_out_cache_loc = torch.zeros(self.bs, dtype=torch.int64)
        be.decode_cuda_graph_metadata = {self.bs: metadata}
        be.speculative_step_id = 1
        be.topk = self.topk
        be.kv_index_translator = SimpleNamespace(is_translating=translating)
        be.max_num_pages = 4
        be.req_to_token = torch.zeros((self.bs, 16), dtype=torch.int32)
        be.page_size = 1
        be._swa_full_to_swa_mapping = None if translating else torch.arange(128)
        be.expand_encoder_only_verify = False
        loc = torch.arange(100, 100 + self.bs * self.topk * self.num_steps)
        captured = {}
        with (
            patch.object(
                tb,
                "update_trtllm_mha_graph_metadata",
                lambda **kwargs: captured.update(kwargs),
            ),
            patch.object(
                tb,
                "get_spec",
                return_value=SimpleNamespace(speculative_num_steps=self.num_steps),
            ),
        ):
            be._apply_cuda_graph_metadata(
                self.bs,
                torch.arange(self.bs),
                torch.full((self.bs,), 7),
                ForwardMode.DECODE,
                spec_info=SimpleNamespace(),
                out_cache_loc=loc,
            )
        return captured["out_cache_loc"], loc

    def test_a_static_pool_captures_this_steps_window_writes(self):
        """The fused metadata kernel reads its loc as a contiguous row, so a
        step's strided slice of the batch's slots reaches it compacted."""
        out_cache_loc, loc = self._apply(translating=False)
        self.assertTrue(out_cache_loc.is_contiguous())
        torch.testing.assert_close(
            out_cache_loc,
            per_step_draft_out_cache_loc(loc, self.bs, self.topk, self.num_steps)[1],
            rtol=0,
            atol=0,
        )

    def test_a_translating_pool_leaves_the_captured_writes_alone(self):
        out_cache_loc, loc = self._apply(translating=True)
        self.assertIs(out_cache_loc, loc)


if __name__ == "__main__":
    unittest.main()
