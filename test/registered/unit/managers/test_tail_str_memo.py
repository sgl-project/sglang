"""Req.tail_str memoization: with stop strings configured the same output_ids
tail is consumed twice per decode step (_check_str_based_finish via
update_finish_state, then the streaming check_match_stop_str_prefix inside
SchedulerOutputStreamer.accept), so the second decode must be served from the
memo — and the memo must never serve a stale string across append,
retract-style regrow (same length, different content), or window-size changes
(speculative multi-token accepts). Whole-output windows (unquantified stop
regexes) must bypass the memo entirely: they grow every step, so there is no
reuse to gain and caching would pin the full output on the Req. Pure CPU:
drives the real Req (and the real accept() streaming route) against a counting
fake tokenizer, cross-checking every value against an uncached reference
decode."""

import unittest
from array import array

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ID_TO_TEXT = {i: chr(ord("a") + i % 26) for i in range(10, 40)}


class CountingTokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None

    def __init__(self):
        self.decode_calls = 0

    def decode(self, ids):
        self.decode_calls += 1
        return "".join(ID_TO_TEXT[int(i)] for i in ids)


class MockTokForNormalize:
    def encode(self, s, add_special_tokens=False):
        return list(range(len(s)))


def _ref_decode(ids):
    return "".join(ID_TO_TEXT[int(i)] for i in ids)


def _ref_tail(req, new_accepted_len=1):
    tail_len = req._stop_match_tail_len(new_accepted_len)
    return _ref_decode(req.output_ids[len(req.output_ids) - tail_len :])


def _make_req(output_ids, stop=None, stop_regex=None):
    sp = SamplingParams(max_new_tokens=1000, stop=stop, stop_regex=stop_regex)
    sp.normalize(tokenizer=MockTokForNormalize())  # char-based stop_str_max_len
    req = Req(
        rid="t",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=sp,
        eos_token_ids=frozenset(),
        vocab_size=10_000,
    )
    req.tokenizer = CountingTokenizer()
    req.output_ids = array("q", output_ids)
    return req


def _make_streamer():
    """Minimal _GenerationStreamAccumulator (the class whose accept() is the
    production streaming route) that runs the happy path: every return-*
    switch off, no spec, rust payload block skipped. Fields are populated by
    reflection over the dataclass so a field rename fails loudly instead of
    silently skipping coverage."""
    import dataclasses

    from sglang.srt.managers.scheduler_components.output_streamer import (
        _GenerationStreamAccumulator,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    overrides = {
        "spec_algorithm": SpeculativeAlgorithm.NONE,
        "disaggregation_mode": None,
        "default_stream_interval": 1,
        "default_force_stream_interval": 1,
        "get_cached_tokens_details": lambda req: None,
        "rust_server_mode": True,
        "return_logprob": False,
        "return_hidden_states": False,
        "return_routed_experts": False,
        "return_indexer_topk": False,
        "return_sampling_mask": False,
        "has_input_top_logprobs_flat": False,
    }
    kwargs = {}
    for f in dataclasses.fields(_GenerationStreamAccumulator):
        if f.name in overrides:
            kwargs[f.name] = overrides[f.name]
        elif f.default is not dataclasses.MISSING:
            kwargs[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            kwargs[f.name] = f.default_factory()  # type: ignore[misc]
    return _GenerationStreamAccumulator(**kwargs)


class TestTailStrMemo(unittest.TestCase):
    def test_both_call_sites_share_one_decode_per_step(self):
        # The exact sequence batch_result_processor runs per still-alive
        # streaming request per step: update_finish_state (which decodes the
        # new tail) then check_match_stop_str_prefix (same window, must hit
        # the memo instead of decoding again).
        req = _make_req([10, 11, 12, 13, 14], stop=["zzz"])
        next_id = 20
        for _ in range(3):
            req.output_ids.append(next_id)  # step commits one token
            next_id += 1
            before = req.tokenizer.decode_calls
            req.update_finish_state(1)
            self.assertFalse(req.finished())
            self.assertFalse(req.check_match_stop_str_prefix())
            self.assertEqual(req.tokenizer.decode_calls - before, 1)
            self.assertEqual(req.tail_str(), _ref_tail(req))

    def test_accept_streaming_route_hits_memo(self):
        # Same per-step sequence, but the second consumer is the real
        # production streaming route: SchedulerOutputStreamer.accept() ->
        # check_match_stop_str_prefix() (stream_interval=1, so every step
        # streams). One decode per step across both entry points.
        req = _make_req([10, 11, 12, 13, 14], stop=["zzz"])
        req.stream = True
        streamer = _make_streamer()
        next_id = 20
        for step in range(3):
            req.output_ids.append(next_id)  # step commits one token
            next_id += 1
            before = req.tokenizer.decode_calls
            req.update_finish_state(1)
            self.assertFalse(req.finished())
            streamer.accept(req=req)
            self.assertEqual(req.tokenizer.decode_calls - before, 1)
            self.assertEqual(len(streamer.output_ids), step + 1)

    def test_append_invalidates(self):
        req = _make_req([10, 11, 12, 13, 14], stop=["zzz"])
        self.assertEqual(req.tail_str(), _ref_tail(req))
        req.output_ids.append(30)
        before = req.tokenizer.decode_calls
        self.assertEqual(req.tail_str(), _ref_tail(req))
        self.assertEqual(req.tokenizer.decode_calls - before, 1)

    def test_retract_regrow_same_length_is_not_stale(self):
        # Retraction discards generated tokens and later decode steps grow
        # output_ids back to a previously seen length with DIFFERENT content
        # (here: pop one id, append another). A length-only cache key would
        # serve the old window; the content key must miss. The stop string is
        # short so the window is a bounded sliding tail of a longer output.
        req = _make_req([10, 11, 12, 13, 14, 15, 16], stop=["zzz"])
        self.assertEqual(req.tail_str(), _ref_tail(req))
        req.output_ids.pop()
        req.output_ids.append(30)
        before = req.tokenizer.decode_calls
        self.assertEqual(req.tail_str(), _ref_tail(req))
        self.assertEqual(req.tokenizer.decode_calls - before, 1)

    def test_interleaved_window_sizes_stay_correct(self):
        # Speculative accepts widen the finish-check window
        # (new_accepted_len > 1) while the streaming prefix check keeps
        # new_accepted_len=1: two different windows consulted per step. The
        # memo must answer each with its own value; being single-entry it
        # thrashes on interleaves (no savings under spec decoding — 5 decodes
        # for 6 queries here, 3-per-3 in steady state — the count below
        # documents that), but never returns a wrong string.
        req = _make_req([10, 11, 12, 13, 14, 15, 16, 17], stop=["zzz"])
        for _ in range(2):
            self.assertEqual(req.tail_str(1), _ref_tail(req, 1))
            self.assertEqual(req.tail_str(3), _ref_tail(req, 3))
            self.assertEqual(req.tail_str(1), _ref_tail(req, 1))
        self.assertEqual(req.tokenizer.decode_calls, 5)

    def test_unbounded_regex_window_bypasses_memo(self):
        # stop_regex=".*END" has an unquantified repeat, so
        # stop_regex_max_len is 2**30 and _stop_match_tail_len clamps to the
        # whole output: the window grows on every step and a memo hit is
        # impossible. The memo must stay untouched (no whole-output id copy
        # or text pinned on the Req) and every call must still decode the
        # correct value.
        req = _make_req([10, 11, 12, 13], stop_regex=[".*END"])
        self.assertEqual(
            req._stop_match_tail_len(1), len(req.output_ids)
        )  # whole-output window, as constructed
        self.assertEqual(req.tail_str(), _ref_tail(req))
        self.assertIsNone(req._tail_str_cache_ids)
        self.assertEqual(req.tail_str(), _ref_tail(req))  # decodes again
        self.assertIsNone(req._tail_str_cache_ids)
        req.output_ids.append(20)
        self.assertEqual(req.tail_str(), _ref_tail(req))
        self.assertIsNone(req._tail_str_cache_ids)

    def test_no_stop_config_never_decodes(self):
        req = _make_req([10, 11, 12, 13])
        self.assertEqual(req.tail_str(), "")
        self.assertFalse(req.check_match_stop_str_prefix())
        self.assertEqual(req.tokenizer.decode_calls, 0)

    def test_values_match_uncached_reference_over_trace(self):
        # Append/retract/append/spec trace; every query equals the reference.
        # The trace crosses both the whole-output regime (output shorter than
        # the stop tail) and the bounded sliding-window regime.
        req = _make_req([10, 11, 12], stop=["zzz"])
        for action in ("append 20", "append 21", "retract", "append 22", "spec3"):
            if action == "retract":
                req.output_ids.pop()
            elif action == "spec3":
                self.assertEqual(req.tail_str(3), _ref_tail(req, 3))
                continue
            else:
                req.output_ids.append(int(action.split()[1]))
            self.assertEqual(req.tail_str(), _ref_tail(req))


if __name__ == "__main__":
    unittest.main()
