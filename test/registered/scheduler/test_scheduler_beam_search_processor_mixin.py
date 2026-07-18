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
"""Unit tests for the composed SchedulerBeamSearchProcessor component.

The processor was refactored from a Scheduler mixin into a standalone
``@dataclass(kw_only=True)`` whose only field is ``scheduler``. It reaches
scheduler state via ``self.scheduler.<attr>`` for exactly five attrs:
tree_cache, token_to_kv_pool_allocator, req_to_token_pool, output_streamer,
metrics_reporter. Each test builds a Mock() scheduler carrying those attrs, then
``proc = SchedulerBeamSearchProcessor(scheduler=mock)`` and calls methods.
"""

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.beam_search_type import BeamSearchList, BeamSearchSequence
from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    FINISH_MATCHED_STR,
    FINISH_MATCHED_TOKEN,
    FINISHED_MATCHED_REGEX,
)
from sglang.srt.managers.scheduler_components.beam_search_processor import (
    SchedulerBeamSearchProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CPU = torch.device("cpu")
RELEASE_KV = (
    "sglang.srt.managers.scheduler_components.beam_search_processor.release_kv_cache"
)
P = SchedulerBeamSearchProcessor  # patch.object target shorthand


def T(data, dtype=None):
    return torch.tensor(data, dtype=dtype, device=CPU)


def make_proc():
    """Build a processor wrapping a mock scheduler with the five required attrs."""
    s = Mock()
    s.req_to_token_pool = Mock()
    s.token_to_kv_pool_allocator = Mock()
    s.tree_cache = Mock()
    s.output_streamer = Mock()
    s.metrics_reporter = Mock()
    s.metrics_reporter.num_generated_tokens = 0
    s.metrics_reporter.forward_ct_decode = 0
    s.metrics_reporter.current_scheduler_metrics_enabled = False
    s.server_args = Mock()
    s.server_args.decode_log_interval = 10
    return SchedulerBeamSearchProcessor(scheduler=s)


def make_req(**overrides):
    """Build a beam-search request Mock with common sampling defaults."""
    req = Mock(
        is_retracted=False,
        is_beam_search=True,
        beam_width=2,
        beam_candidates=4,
        origin_input_ids=[1, 2, 3],
        stop_token_ids=set(),
        custom_logit_processor=None,
        finished_reason=None,
        to_finish=None,
        tokenizer=Mock(),
        finished=Mock(return_value=False),
    )
    req.sampling_params = Mock(
        max_new_tokens=10,
        ignore_eos=False,
        stop_strs=[],
        stop_regex_strs=[],
        length_penalty=1.0,
        custom_params=None,
    )
    for k, v in overrides.items():
        setattr(req, k, v)
    return req


def patch_methods(testcase, *attrs):
    """Patch the named processor methods; return dict {attr: mock}."""
    mocks = {}
    for attr in attrs:
        p = patch.object(P, attr)
        mocks[attr] = p.start()
        testcase.addCleanup(p.stop)
    return mocks


class TestStaticMethods(CustomTestCase):
    """sum_beam_completion_tokens + convert_beam_sequences_to_output."""

    def test_sum_beam_completion_tokens(self):
        req = Mock(
            beam_list=Mock(
                completed=[
                    BeamSearchSequence(tokens=[1, 2, 3]),
                    BeamSearchSequence(tokens=[4, 5]),
                    BeamSearchSequence(tokens=[6, 7, 8, 9]),
                ]
            )
        )
        self.assertEqual(P.sum_beam_completion_tokens(req), 9)

    def test_sum_beam_completion_tokens_empty(self):
        self.assertEqual(
            P.sum_beam_completion_tokens(Mock(beam_list=Mock(completed=[]))), 0
        )

    def test_convert_beam_sequences_to_output(self):
        req = Mock(
            beam_list=Mock(
                completed=[
                    BeamSearchSequence(
                        tokens=[1, 2, 3],
                        cum_logprob=-5.0,
                        beam_score=-1.67,
                        finish_reason=FINISH_LENGTH(length=3),
                    ),
                    BeamSearchSequence(
                        tokens=[4, 5],
                        cum_logprob=-3.0,
                        beam_score=-1.5,
                        finish_reason=FINISH_MATCHED_TOKEN(matched=50256),
                    ),
                ]
            )
        )
        out = P.convert_beam_sequences_to_output(req)
        self.assertEqual(len(out.sequences), 2)
        self.assertEqual(out.sequences[0].tokens, [1, 2, 3])
        self.assertEqual(out.sequences[0].cum_logprob, -5.0)
        self.assertEqual(out.sequences[0].finish_reason["type"], "length")
        self.assertEqual(out.sequences[1].tokens, [4, 5])
        self.assertEqual(out.sequences[1].finish_reason["type"], "stop")


class TestProcessPrefillSingleReq(CustomTestCase):
    """_process_beam_search_prefill_result_single_req routing."""

    def setUp(self):
        self.proc = make_proc()
        self.m = patch_methods(
            self,
            "_create_initial_beam_sequences",
            "_batch_check_prefill_generated_tokens_stop_conditions",
            "_create_completed_beams_for_insufficient_candidates",
        )

    def _run(self, req, mask_fn=None):
        if mask_fn is not None:
            self.m[
                "_batch_check_prefill_generated_tokens_stop_conditions"
            ].side_effect = mask_fn
        self.proc._process_beam_search_prefill_result_single_req(
            req, Mock(device=CPU), torch.randn(100, device=CPU), CPU
        )

    def test_normal_path_creates_initial_beams(self):
        req = make_req(beam_list=BeamSearchList())
        self._run(req, lambda *_: T([True, False, False, False], dtype=torch.bool))
        self.m["_create_initial_beam_sequences"].assert_called_once()
        args = self.m["_create_initial_beam_sequences"].call_args[0]
        self.assertEqual(args[0], req)
        self.assertEqual(len(args[1]), req.beam_candidates)  # top_logprobs_val
        self.assertEqual(len(args[2]), req.beam_candidates)  # top_logprobs_idx
        self.assertEqual(len(args[3]), req.beam_candidates)  # finish_mask_cpu
        self.assertEqual(args[4], CPU)

    def test_finish_by_len_when_max_new_tokens_le_1(self):
        req = make_req(beam_list=BeamSearchList())
        req.sampling_params.max_new_tokens = 1
        self._run(req)  # no mask: max_new_tokens<=1 marks all finished
        args = self.m["_create_completed_beams_for_insufficient_candidates"].call_args[
            0
        ]
        self.assertTrue(all(args[3]))  # finish_mask all True
        self.assertTrue(args[4])  # finish_by_len True

    def test_insufficient_candidates_routes_to_completed(self):
        req = make_req(beam_list=BeamSearchList())
        self._run(req, lambda *_: T([True, True, True, False], dtype=torch.bool))
        args = self.m["_create_completed_beams_for_insufficient_candidates"].call_args[
            0
        ]
        self.assertEqual(args[3], [True, True, True, False])
        self.assertFalse(args[4])


class TestBatchCheckPrefillStopConditions(CustomTestCase):
    def setUp(self):
        self.proc = make_proc()

    def test_stop_tokens(self):
        req = make_req(stop_token_ids={50256, 50257})
        r = self.proc._batch_check_prefill_generated_tokens_stop_conditions(
            req, T([100, 50256, 200, 50257], dtype=torch.int64), CPU
        )
        self.assertEqual(r.cpu().tolist(), [False, True, False, True])

    def test_stop_strings(self):
        req = make_req()
        req.sampling_params.stop_strs = ["STOP"]
        req.tokenizer.decode = Mock(
            side_effect=lambda t, **k: "STOP" if t[0] == 100 else "x"
        )
        r = self.proc._batch_check_prefill_generated_tokens_stop_conditions(
            req, T([100, 200, 300], dtype=torch.int64), CPU
        )
        self.assertEqual(r.cpu().tolist(), [True, False, False])


class TestCreateCompletedAndInitialBeams(CustomTestCase):
    """_create_completed_beams_for_insufficient_candidates + _create_initial_beam_sequences."""

    def setUp(self):
        self.proc = make_proc()

    def test_insufficient_candidates_sets_finished_reason(self):
        """Crash-fix coverage: top candidate finished -> finished_reason set."""
        req = make_req(beam_list=Mock(), finished_reason=None)
        self.proc._create_completed_beams_for_insufficient_candidates(
            req,
            [-1.0, -2.0, -3.0, -4.0],
            [100, 200, 300, 400],
            [True, True, True, False],
            False,
        )
        self.assertEqual(len(req.beam_list.completed), 2)
        self.assertEqual(len(req.beam_list.incomplete), 0)
        first = req.beam_list.completed[0]
        self.assertEqual(first.tokens, [100])
        self.assertEqual(first.cum_logprob, -1.0)
        self.assertIsInstance(first.finish_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(first.finish_reason.matched, 100)
        self.assertIsNotNone(req.finished_reason)
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_reason.matched, 100)

    def test_insufficient_candidates_unfinished_top_still_sets_reason(self):
        """Crash-fix: top candidate unfinished, but req.finished_reason must still
        be non-None (adopts first real finish reason from a later beam)."""
        req = make_req(beam_list=Mock(), finished_reason=None)
        self.proc._create_completed_beams_for_insufficient_candidates(
            req,
            [-1.0, -2.0, -3.0, -4.0],
            [100, 200, 300, 400],
            [False, True, True, True],
            False,  # top candidate unfinished
        )
        self.assertIsNone(req.beam_list.completed[0].finish_reason)
        self.assertIsNotNone(req.finished_reason)
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_reason.matched, 200)

    def test_create_initial_beam_sequences_basic(self):
        req = make_req(beam_list=BeamSearchList())
        self.proc._create_initial_beam_sequences(
            req,
            [-1.0, -2.0, -3.0, -4.0, -5.0],
            [100, 200, 300, 400, 500],
            [False] * 5,
            CPU,
        )
        bl = req.beam_list
        self.assertEqual(len(bl.incomplete), 2)
        self.assertEqual(len(bl.completed), 0)
        self.assertEqual([b.tokens for b in bl.incomplete], [[100], [200]])
        self.assertEqual(bl.prompt_lens.tolist(), [3, 3])
        self.assertEqual(bl.last_tokens.tolist(), [100, 200])
        self.assertEqual(bl.cum_logprobs.tolist(), [-1.0, -2.0])

    def test_create_initial_beam_sequences_with_finished_beam(self):
        req = make_req(beam_width=3, beam_candidates=6, beam_list=BeamSearchList())
        self.proc._create_initial_beam_sequences(
            req,
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
            [100, 200, 300, 400, 500, 600],
            [True, False, False, False, False, False],
            CPU,
        )
        bl = req.beam_list
        self.assertEqual(len(bl.incomplete), 3)
        self.assertEqual(len(bl.completed), 1)
        self.assertIsNotNone(bl.completed[0].beam_score)
        self.assertEqual(bl.completed[0].tokens, [100])
        self.assertIsInstance(bl.completed[0].finish_reason, FINISH_MATCHED_TOKEN)
        for beam in bl.incomplete:
            self.assertIsNone(beam.beam_score)
            self.assertIsNone(beam.finish_reason)
        self.assertEqual([b.tokens for b in bl.incomplete], [[200], [300], [400]])


class TestCheckBeamFinished(CustomTestCase):
    """_check_beam_finished stop-condition cases."""

    def setUp(self):
        self.proc = make_proc()
        self.tail = patch_methods(self, "_tail_str")["_tail_str"]

    def _beam(self, tokens):
        return BeamSearchSequence(tokens=tokens, cum_logprob=-5.0)

    def test_stop_token(self):
        req = make_req(stop_token_ids={50256, 50257})
        beam = self._beam([1, 2, 3, 50256])
        self.assertTrue(self.proc._check_beam_finished(req, beam))
        self.assertIsInstance(beam.finish_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(beam.finish_reason.matched, 50256)

    def test_stop_string(self):
        self.tail.return_value = "This is STOP"
        req = make_req()
        req.sampling_params.ignore_eos = True
        req.sampling_params.stop_strs = ["STOP", "END"]
        beam = self._beam([1, 2, 3, 4])
        self.assertTrue(self.proc._check_beam_finished(req, beam))
        self.assertIsInstance(beam.finish_reason, FINISH_MATCHED_STR)
        self.assertEqual(beam.finish_reason.matched, "STOP")

    def test_regex(self):
        self.tail.return_value = "Call 123-4567 now"
        req = make_req()
        req.sampling_params.ignore_eos = True
        req.sampling_params.stop_strs = ["STOP"]
        req.sampling_params.stop_regex_strs = [r"\d{3}-\d{4}"]
        beam = self._beam([1, 2, 3, 4, 5])
        self.assertTrue(self.proc._check_beam_finished(req, beam))
        self.assertIsInstance(beam.finish_reason, FINISHED_MATCHED_REGEX)

    def test_not_finished(self):
        self.tail.return_value = "Continue"
        req = make_req(stop_token_ids={50256})
        req.sampling_params.stop_strs = ["STOP"]
        beam = self._beam([1, 2, 3, 4])
        self.assertFalse(self.proc._check_beam_finished(req, beam))
        self.assertIsNone(beam.finish_reason)

    def test_ignore_eos_skips_stop_token(self):
        self.tail.return_value = "no stop"
        req = make_req(stop_token_ids={50256})
        req.sampling_params.ignore_eos = True
        req.sampling_params.stop_strs = ["STOP"]
        beam = self._beam([1, 2, 3, 50256])
        self.assertFalse(self.proc._check_beam_finished(req, beam))
        self.assertIsNone(beam.finish_reason)


class TestTailStr(CustomTestCase):
    def test_tail_str_decodes_tail(self):
        proc = make_proc()
        req = make_req()
        req.sampling_params.stop_strs = ["STOP"]
        req.sampling_params.stop_str_max_len = 2
        req.sampling_params.stop_regex_max_len = 0
        req.tokenizer.decode = Mock(return_value="tail text")
        self.assertEqual(proc._tail_str(req, [1, 2, 3, 4, 5]), "tail text")
        self.assertTrue(req.tokenizer.decode.called)


class TestProcessBeamSearchExpansion(CustomTestCase):
    """_process_beam_search_expansion routing across finish conditions."""

    def setUp(self):
        self.proc = make_proc()
        self.m = patch_methods(
            self,
            "_expand_and_prune_beams",
            "_create_completed_beams_for_finished_request",
        )
        self.top_tokens = T([[100, 200, 300, 400], [500, 600, 700, 800]])
        self.top_logprobs = T([[-1.0, -2.0, -3.0, -4.0], [-1.5, -2.5, -3.5, -4.5]])

    def _req(self, tokens_len=2, **kw):
        req = make_req(beam_list=Mock(), **kw)
        req.beam_list.incomplete = [
            BeamSearchSequence(tokens=list(range(tokens_len)), cum_logprob=-2.0),
            BeamSearchSequence(tokens=list(range(tokens_len)), cum_logprob=-3.0),
        ]
        req.beam_list.cum_logprobs = T([-2.0, -3.0])
        req.beam_list.batch_slot_start_idx = 0
        return req

    def _run(self, req):
        return self.proc._process_beam_search_expansion(
            req, Mock(device=CPU), 2, 4, self.top_tokens, self.top_logprobs
        )

    def test_normal_path_returns_slot_indices(self):
        self.m["_expand_and_prune_beams"].return_value = [0, 1]
        result = self._run(self._req())
        self.m["_expand_and_prune_beams"].assert_called_once()
        self.m["_create_completed_beams_for_finished_request"].assert_not_called()
        self.assertEqual(result.cpu().tolist(), [0, 1])

    def test_finished_by_length(self):
        req = self._req(tokens_len=9)  # +1 reaches max_new_tokens=10
        result = self._run(req)
        self.m["_create_completed_beams_for_finished_request"].assert_called_once()
        self.m["_expand_and_prune_beams"].assert_not_called()
        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertIsNone(result)

    def test_to_finish_preset(self):
        req = self._req(to_finish=FINISH_MATCHED_TOKEN(matched=50256))
        result = self._run(req)
        call = self.m["_create_completed_beams_for_finished_request"].call_args[0]
        self.assertIsInstance(call[6], FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_reason, req.to_finish)
        self.assertIsNone(result)

    def test_expand_returns_none(self):
        self.m["_expand_and_prune_beams"].return_value = None
        result = self._run(self._req())
        self.m["_expand_and_prune_beams"].assert_called_once()
        self.m["_create_completed_beams_for_finished_request"].assert_not_called()
        self.assertIsNone(result)


class TestExpandAndPruneBeams(CustomTestCase):
    """_expand_and_prune_beams: fast / eos / stop-str / insufficient paths."""

    def setUp(self):
        self.proc = make_proc()
        ps = patch.object(P, "_calculate_beam_score", return_value=0.5)
        pc = patch.object(P, "_check_beam_finished")
        self.score = ps.start()
        self.check = pc.start()
        self.addCleanup(ps.stop)
        self.addCleanup(pc.stop)

    def _req(self, incomplete, last_tokens, cum_logprobs, **sp):
        req = make_req()
        for k, v in sp.items():
            setattr(req.sampling_params, k, v)
        req.beam_list = Mock(
            incomplete=incomplete,
            completed=[],
            last_tokens=T(last_tokens),
            cum_logprobs=T(cum_logprobs),
        )
        return req

    @staticmethod
    def _two_beams():
        return [
            BeamSearchSequence(tokens=[1, 2], cum_logprob=-2.0),
            BeamSearchSequence(tokens=[3, 4], cum_logprob=-3.0),
        ]

    def test_fast_path_no_stop_conditions(self):
        req = self._req(
            self._two_beams(), [2, 4], [-2.0, -3.0], ignore_eos=True, stop_strs=[]
        )
        result = self.proc._expand_and_prune_beams(
            req,
            2,
            4,
            torch.arange(8, device=CPU),
            T([-2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5, -6.0]),
            T([100, 200, 300, 400, 500, 600, 700, 800]),
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(req.beam_list.incomplete[0].tokens, [1, 2, 100])
        self.assertAlmostEqual(req.beam_list.incomplete[0].cum_logprob, -2.5)
        self.assertEqual(req.beam_list.incomplete[1].tokens, [1, 2, 200])
        self.assertEqual(len(req.beam_list.completed), 0)
        self.assertEqual(req.beam_list.last_tokens.tolist(), [100, 200])

    def test_eos_vectorized_path(self):
        req = self._req(
            self._two_beams(), [2, 4], [-2.0, -3.0], ignore_eos=False, stop_strs=[]
        )
        req.stop_token_ids = {50256}
        result = self.proc._expand_and_prune_beams(
            req,
            2,
            4,
            T([0, 1, 4, 5]),
            T([-2.5, -3.0, -3.5, -4.0]),
            T([50256, 200, 300, 400, 500, 600, 700, 800]),
        )
        self.assertEqual(result, [0, 1])
        self.assertEqual(len(req.beam_list.completed), 1)
        self.assertEqual(req.beam_list.completed[0].tokens, [1, 2, 50256])
        self.assertIsInstance(
            req.beam_list.completed[0].finish_reason, FINISH_MATCHED_TOKEN
        )
        self.assertEqual(req.beam_list.completed[0].beam_score, 0.5)
        self.assertEqual(len(req.beam_list.incomplete), 2)
        self.assertEqual(req.beam_list.incomplete[1].tokens, [3, 4, 500])
        self.score.assert_called_once()

    def test_stop_str_sequential_path(self):
        self.score.return_value = 0.8
        req = self._req(
            self._two_beams(),
            [2, 4],
            [-2.0, -3.0],
            ignore_eos=False,
            stop_strs=["STOP"],
        )

        def check(_req, beam):
            if beam.tokens[-1] in (100, 200, 300, 400, 500, 600):
                beam.finish_reason = FINISH_MATCHED_STR(matched="STOP")
                return True
            return False

        self.check.side_effect = check
        result = self.proc._expand_and_prune_beams(
            req,
            2,
            4,
            torch.arange(8, device=CPU),
            T([-2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5, -6.0]),
            T([100, 200, 300, 400, 500, 600, 700, 800]),
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(self.check.call_count, 8)
        self.assertEqual(len(req.beam_list.completed), 6)
        self.assertEqual(len(req.beam_list.incomplete), 2)
        self.assertEqual(req.beam_list.incomplete[0].tokens, [3, 4, 700])
        self.assertEqual(self.score.call_count, 6)

    def test_insufficient_candidates_returns_none(self):
        self.score.return_value = -1.0
        req = self._req(
            [BeamSearchSequence(tokens=[1, 2], cum_logprob=-2.0)],
            [2],
            [-2.0],
            ignore_eos=False,
            stop_strs=[],
        )
        req.stop_token_ids = {50256}
        result = self.proc._expand_and_prune_beams(
            req,
            3,
            6,
            torch.arange(6, device=CPU),
            T([-2.5, -3.0, -3.5, -4.0, -4.5, -5.0]),
            T([50256] * 6),
        )
        self.assertIsNone(result)
        self.assertEqual(len(req.beam_list.completed), 6)
        self.assertEqual(len(req.beam_list.incomplete), 0)


class TestCreateCompletedBeamsForFinishedRequest(CustomTestCase):
    def test_appends_completed_with_finish_reason(self):
        proc = make_proc()
        existing = BeamSearchSequence(
            tokens=[5, 6, 7],
            cum_logprob=-1.5,
            finish_reason=FINISH_MATCHED_TOKEN(matched=50256),
            beam_score=-0.5,
        )
        req = Mock(
            beam_list=Mock(
                incomplete=[
                    BeamSearchSequence(tokens=[1, 2], cum_logprob=-2.0),
                    BeamSearchSequence(tokens=[3, 4], cum_logprob=-3.0),
                ],
                completed=[existing],
                cum_logprobs=T([-2.0, -3.0]),
            )
        )
        reason = FINISH_LENGTH(length=10)
        proc._create_completed_beams_for_finished_request(
            req,
            2,
            4,
            T([0, 1, 2, 3]),
            T([-2.5, -3.0, -3.5, -4.0]),
            T([100, 200, 300, 400, 500, 600, 700, 800]),
            reason,
        )
        self.assertEqual(len(req.beam_list.completed), 3)
        self.assertEqual(len(req.beam_list.incomplete), 0)
        self.assertEqual(req.beam_list.completed[0], existing)
        for beam in req.beam_list.completed[1:]:
            self.assertEqual(beam.finish_reason, reason)


class TestCalculateBeamScore(CustomTestCase):
    def test_beam_score_with_penalties(self):
        f = P._calculate_beam_score
        self.assertAlmostEqual(f(-10.0, 5, 1.0), -2.0, places=5)
        self.assertAlmostEqual(f(-10.0, 4, length_penalty=2.0), -0.625, places=5)
        self.assertAlmostEqual(f(-10.0, 4, length_penalty=0.5), -5.0, places=5)


if __name__ == "__main__":
    unittest.main()
