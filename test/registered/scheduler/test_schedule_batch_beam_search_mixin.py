# Copyright 2026 SGLang Team
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
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.beam_search_type import BeamSearchList
from sglang.srt.managers.schedule_batch_beam_search_mixin import (
    BeamSearchAdmissionError,
    ReqBeamSearchMixin,
    ScheduleBatchBeamSearchMixin,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small")

CPU = torch.device("cpu")


def _t(values):
    return torch.tensor(values, device=CPU)  # CPU int64 tensor helper


def make_req(beam_width=None, start_idx=None, finished=None, last_tokens=None):
    """Mock beam-search req; only the passed attributes are set on it."""
    req = Mock()
    if beam_width is not None:
        req.beam_width = beam_width
    req.beam_list = Mock()
    if start_idx is not None:
        req.beam_list.batch_slot_start_idx = start_idx
    if last_tokens is not None:
        req.beam_list.last_tokens = _t(last_tokens)
    if finished is not None:
        req.finished = Mock(return_value=finished)
    return req


_SENTINEL = object()


def make_batch(
    reqs,
    *,
    req_pool_indices=None,
    seq_lens=None,
    orig_seq_lens=None,
    req_to_token=None,
    alloc_return=_SENTINEL,
    available_size=None,
):
    """Build a Mock ScheduleBatch wired with CPU tensors. A req_to_token_pool with
    tracked alloc_by_count/available_size mocks is created when any pool field is set.
    """
    batch = Mock()
    batch.device = CPU
    batch.reqs = reqs
    if req_pool_indices is not None:
        batch.req_pool_indices = _t(req_pool_indices)
    if seq_lens is not None:
        batch.seq_lens = _t(seq_lens)
        batch.seq_lens_cpu = torch.tensor(seq_lens, device=CPU)
    if orig_seq_lens is not None:
        batch.orig_seq_lens = _t(orig_seq_lens)
    want_alloc = alloc_return is not _SENTINEL
    if req_to_token is not None or want_alloc or available_size is not None:
        pool = Mock()
        if req_to_token is not None:
            pool.req_to_token = req_to_token
        pool.alloc_by_count = Mock(return_value=alloc_return if want_alloc else None)
        if available_size is not None:
            pool.available_size = Mock(return_value=available_size)
        batch.req_to_token_pool = pool
    return batch


class TensorTestCase(unittest.TestCase):
    def assertTensorEqual(self, actual, expected):
        if not torch.is_tensor(expected):
            expected = _t(expected)
        self.assertTrue(
            torch.equal(actual, expected), f"{actual.tolist()} != {expected.tolist()}"
        )


class TestPrepareForBeamSearchDecode(TensorTestCase):
    """Test prepare_for_beam_search_decode (happy path + pre-flight gate)."""

    @patch("sglang.srt.managers.schedule_batch_beam_search_mixin.alloc_for_decode")
    def test_happy_path_different_beam_widths(self, mock_alloc_for_decode):
        req1 = make_req(beam_width=2, last_tokens=[100, 200])
        req2 = make_req(beam_width=3, last_tokens=[300, 400, 500])
        batch = make_batch(
            [req1, req2],
            seq_lens=[5, 5, 6, 6, 6],
            orig_seq_lens=[5, 5, 6, 6, 6],
        )
        mock_alloc_for_decode.return_value = _t([10, 11, 12, 13, 14])
        ScheduleBatchBeamSearchMixin.prepare_for_beam_search_decode(batch)
        # forward_mode flipped to DECODE; input_ids gathered from beam last_tokens
        self.assertEqual(batch.forward_mode, ForwardMode.DECODE)
        self.assertTensorEqual(
            batch.input_ids, torch.tensor([100, 200, 300, 400, 500], dtype=torch.int32)
        )
        self.assertIsNone(batch.output_ids)
        # new-beam init delegated, out_cache_loc allocated
        batch._prepare_for_new_beam_search.assert_called_once()
        mock_alloc_for_decode.assert_called_once_with(batch, token_per_req=1)
        self.assertTensorEqual(batch.out_cache_loc, [10, 11, 12, 13, 14])
        # seq_lens bumped by 1
        self.assertTensorEqual(batch.seq_lens, [6, 6, 7, 7, 7])
        self.assertTensorEqual(batch.seq_lens_cpu, [6, 6, 7, 7, 7])
        self.assertTensorEqual(batch.orig_seq_lens, [6, 6, 7, 7, 7])

    def test_preflight_raises_before_mutation_when_pool_too_small(self):
        """Pending new-beam slots > available -> raise before any mutation."""
        req = make_req(beam_width=100, start_idx=-1)  # still in prefill
        batch = make_batch([req], available_size=10)
        batch.forward_mode = "SENTINEL"
        with self.assertRaises(BeamSearchAdmissionError) as ctx:
            ScheduleBatchBeamSearchMixin.prepare_for_beam_search_decode(batch)
        # Structured context for the scheduler's skip-tick handler.
        self.assertEqual(ctx.exception.total_slots, 100)
        self.assertEqual(ctx.exception.available, 10)
        self.assertEqual(ctx.exception.failing_reqs, [req])
        # Raise happens before any batch-state mutation or delegated init.
        self.assertEqual(batch.forward_mode, "SENTINEL")
        batch._prepare_for_new_beam_search.assert_not_called()


class TestFilterBeamSearchBatch(TensorTestCase):
    """Test filter_beam_search_batch keep-subset remapping."""

    def test_exclude_finished(self):
        req1 = make_req(beam_width=2, start_idx=0, finished=False)
        req2 = make_req(beam_width=2, start_idx=2, finished=True)
        batch = make_batch(
            [req1, req2],
            req_pool_indices=[0, 1, 2, 3],
            seq_lens=[7, 7, 8, 8],
            orig_seq_lens=[7, 7, 8, 8],
            req_to_token=torch.arange(40, device=CPU).reshape(4, 10),
        )
        batch.has_stream = False
        batch.has_grammar = False
        ScheduleBatchBeamSearchMixin.filter_beam_search_batch(batch)
        self.assertEqual(batch.reqs, [req1])
        self.assertEqual(req1.beam_list.batch_slot_start_idx, 0)
        self.assertTensorEqual(batch.req_pool_indices, [0, 1])
        self.assertTensorEqual(batch.seq_lens, [7, 7])
        self.assertTensorEqual(batch.seq_lens_cpu, [7, 7])
        self.assertEqual(batch.seq_lens_sum, 14)
        self.assertTensorEqual(batch.orig_seq_lens, [7, 7])
        # req1 keeps pool rows [0, 1]; req2's rows [2, 3] dropped.
        actual = batch.req_to_token_pool.req_to_token[batch.req_pool_indices]
        self.assertTensorEqual(actual, torch.arange(20, device=CPU).reshape(2, 10))

    def test_mixed_beam_widths_explicit_keep_indices(self):
        """Mixed beam widths + a prefill (start_idx=-1) req, kept via keep_indices."""
        req1 = make_req(beam_width=2, start_idx=0)
        req2 = make_req(beam_width=1, start_idx=-1)
        req3 = make_req(beam_width=3, start_idx=3)
        batch = make_batch(
            [req1, req2, req3],
            req_pool_indices=[0, 1, 2, 3, 4, 5],
            seq_lens=[7, 7, 8, 9, 9, 9],
            orig_seq_lens=[7, 7, 8, 9, 9, 9],
            req_to_token=torch.arange(60, device=CPU).reshape(6, 10),
        )
        batch.has_stream = False
        batch.has_grammar = False
        ScheduleBatchBeamSearchMixin.filter_beam_search_batch(
            batch, keep_indices=[0, 2]
        )
        self.assertEqual(batch.reqs, [req1, req3])
        self.assertTensorEqual(batch.req_pool_indices, [0, 1, 3, 4, 5])
        self.assertTensorEqual(batch.seq_lens, [7, 7, 9, 9, 9])
        self.assertTensorEqual(batch.seq_lens_cpu, [7, 7, 9, 9, 9])
        self.assertEqual(batch.seq_lens_sum, 41)
        self.assertTensorEqual(batch.orig_seq_lens, [7, 7, 9, 9, 9])
        # batch_slot_start_idx remapped: req1 stays 0, req3 shifts 3 -> 2.
        self.assertEqual(req1.beam_list.batch_slot_start_idx, 0)
        self.assertEqual(req3.beam_list.batch_slot_start_idx, 2)
        # Kept pool rows still map to their original req_to_token rows (0,1,3,4,5).
        full = torch.arange(60, device=CPU).reshape(6, 10)
        actual = batch.req_to_token_pool.req_to_token[batch.req_pool_indices]
        self.assertTensorEqual(actual, full[[0, 1, 3, 4, 5]])

    def test_empty_result(self):
        batch = make_batch([make_req(finished=True)])
        ScheduleBatchBeamSearchMixin.filter_beam_search_batch(batch)
        self.assertEqual(len(batch.reqs), 0)

    def test_no_change_keeps_all(self):
        reqs = [
            make_req(beam_width=2, finished=False),
            make_req(beam_width=2, finished=False),
        ]
        batch = make_batch(list(reqs))
        ScheduleBatchBeamSearchMixin.filter_beam_search_batch(batch)
        self.assertEqual(batch.reqs, reqs)


class TestPrepareForNewBeamSearch(TensorTestCase):
    """Test _prepare_for_new_beam_search alloc path + alloc-failure."""

    def test_alloc_paths(self):
        # Only the new (start_idx == -1) req gets beam slots allocated; old reqs are
        # skipped. KV cache at the prefill src_row is copied to each new beam row dst.
        cases = [
            # single new req: prefill row 0, seq_len 5, alloc beams [1,2,3]
            dict(
                name="single_request",
                reqs_spec=[(3, -1)],
                pool=[0],
                seqs=[5],
                alloc=[1, 2, 3],
                rows=5,
                exp_pool=[1, 2, 3],
                exp_seq=[5, 5, 5],
                new_start=0,
                src_row=0,
                dsts=[1, 2, 3],
                copy_len=5,
            ),
            # old req (kept) + new req: skip old's 2 slots, alloc beams [3,4,5]
            dict(
                name="multiple_requests_skip_old",
                reqs_spec=[(2, 0), (3, -1)],
                pool=[0, 1, 2],
                seqs=[7, 7, 8],
                alloc=[3, 4, 5],
                rows=6,
                exp_pool=[0, 1, 3, 4, 5],
                exp_seq=[7, 7, 8, 8, 8],
                new_start=2,
                src_row=2,
                dsts=[3, 4, 5],
                copy_len=8,
            ),
        ]
        for c in cases:
            with self.subTest(c["name"]):
                reqs = [
                    make_req(beam_width=bw, start_idx=si) for bw, si in c["reqs_spec"]
                ]
                new_req = reqs[-1]
                pool = torch.arange(c["rows"] * 10, device=CPU).reshape(c["rows"], 10)
                batch = make_batch(
                    reqs,
                    req_pool_indices=c["pool"],
                    seq_lens=c["seqs"],
                    orig_seq_lens=c["seqs"],
                    req_to_token=pool,
                    alloc_return=c["alloc"],
                )
                ScheduleBatchBeamSearchMixin._prepare_for_new_beam_search(batch)
                batch.req_to_token_pool.alloc_by_count.assert_called_once_with(
                    new_req.beam_width
                )
                self.assertTensorEqual(batch.req_pool_indices, c["exp_pool"])
                self.assertTensorEqual(batch.seq_lens, c["exp_seq"])
                self.assertTensorEqual(batch.orig_seq_lens, c["exp_seq"])
                self.assertEqual(new_req.beam_list.batch_slot_start_idx, c["new_start"])
                # KV cache copied from the prefill row to each new beam row.
                src = pool[c["src_row"], : c["copy_len"]]
                for dst in c["dsts"]:
                    self.assertTensorEqual(pool[dst, : c["copy_len"]], src)

    def test_no_new_requests_is_noop(self):
        req = make_req(beam_width=2, start_idx=0)
        batch = make_batch([req], req_pool_indices=[0, 1], alloc_return=None)
        original = batch.req_pool_indices.clone()
        ScheduleBatchBeamSearchMixin._prepare_for_new_beam_search(batch)
        batch.req_to_token_pool.alloc_by_count.assert_not_called()
        self.assertTensorEqual(batch.req_pool_indices, original)

    def test_alloc_failure_raises_admission_error(self):
        req = make_req(beam_width=3, start_idx=-1)
        batch = make_batch(
            [req], req_pool_indices=[0], alloc_return=None, available_size=0
        )
        with self.assertRaises(BeamSearchAdmissionError) as ctx:
            ScheduleBatchBeamSearchMixin._prepare_for_new_beam_search(batch)
        self.assertIsInstance(ctx.exception, RuntimeError)
        self.assertEqual(ctx.exception.total_slots, 3)
        self.assertEqual(ctx.exception.available, 0)
        self.assertEqual(ctx.exception.failing_reqs, [req])


class TestInitBeamSearchAttributes(unittest.TestCase):
    """Test ReqBeamSearchMixin._init_beam_search_attributes."""

    def test_enabled_sets_full_state(self):
        req = Mock()
        ReqBeamSearchMixin._init_beam_search_attributes(
            req, is_beam_search=True, sampling_params=Mock(n=3)
        )
        self.assertTrue(req.is_beam_search)
        self.assertEqual(req.beam_width, 3)
        self.assertEqual(req.beam_candidates, 6)  # beam_width * 2
        self.assertIsInstance(req.beam_list, BeamSearchList)
        self.assertIsNone(req._stop_token_ids_cache)

    def test_disabled_is_minimal(self):
        req = Mock(spec=[])  # spec=[] so hasattr reflects real assignment
        ReqBeamSearchMixin._init_beam_search_attributes(
            req, is_beam_search=False, sampling_params=Mock(n=3)
        )
        self.assertFalse(req.is_beam_search)
        self.assertFalse(hasattr(req, "beam_width"))
        self.assertFalse(hasattr(req, "beam_candidates"))
        self.assertFalse(hasattr(req, "beam_list"))
        self.assertIsNone(req._stop_token_ids_cache)


class TestStopTokenIds(unittest.TestCase):
    """Test stop_token_ids property."""

    def test_collection_dedup_and_caching(self):
        # (name, stop_token_ids, eos_token_ids, tokenizer(eos, extra) or None, expected)
        cases = [
            (
                "basic",
                [100, 200],
                [300, 400],
                (50256, [500, 600]),
                {100, 200, 300, 400, 50256, 500, 600},
            ),
            ("empty_sources", None, None, (None, None), set()),
            ("no_tokenizer", [100, 200], [300], None, {100, 200, 300}),
            (
                "deduplication",
                [100, 200],
                [200, 300],
                (300, [100, 400]),
                {100, 200, 300, 400},
            ),
        ]
        for name, stop, eos, tok, expected in cases:
            with self.subTest(name):
                req = Mock()
                req._stop_token_ids_cache = None
                req.sampling_params = Mock(stop_token_ids=stop)
                req.eos_token_ids = eos
                if tok is None:
                    req.tokenizer = None
                else:
                    req.tokenizer = Mock(
                        eos_token_id=tok[0], additional_stop_token_ids=tok[1]
                    )
                result = ReqBeamSearchMixin.stop_token_ids.fget(req)
                self.assertEqual(result, expected)
                # result is cached back onto the req
                self.assertEqual(req._stop_token_ids_cache, expected)

    def test_returns_cache_without_recompute(self):
        req = Mock()
        cached = {100, 200, 300}
        req._stop_token_ids_cache = cached
        self.assertEqual(ReqBeamSearchMixin.stop_token_ids.fget(req), cached)


if __name__ == "__main__":
    unittest.main()
