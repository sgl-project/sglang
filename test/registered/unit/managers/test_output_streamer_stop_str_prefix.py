import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_STR
from sglang.srt.managers.scheduler_components.output_streamer import (
    _GenerationStreamAccumulator,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _StubTokenizer:
    """One token id -> one character, so decoding is a plain concat."""

    def decode(self, ids):
        # Token 999 decodes to "\n" so a stop string can start with it while
        # all token ids stay in the valid non-negative range production uses.
        return "".join(chr(10) if t == 999 else chr(97 + (t % 26)) for t in ids)


STOP_STR = "\nzzz"
FILLER = list(range(0, 49))  # 49 filler tokens -> "abc..."
STOP_TOKENS = [999, 25, 25, 25]  # "\nzzz"; first token lands at position 50


class _FakeReq:
    """Minimal Req stub driving _GenerationStreamAccumulator.accept offline."""

    def __init__(self, rid, output_ids, stream, matched=None):
        self.rid = rid
        self.http_worker_ipc = None
        self.finished_reason = matched
        self.finished_output = False
        self.finished_len = None
        self.stream = stream
        self.tokenizer = _StubTokenizer()
        self.sampling_params = SimpleNamespace(
            stream_interval=None,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            no_stop_trim=False,
            stop_strs=[STOP_STR],
            stop_regex_strs=[],
            stop_str_max_len=8,
            stop_regex_max_len=0,
        )
        self.output_ids = output_ids
        self.output_ids_through_stop = output_ids
        self.send_token_offset = 0
        self.send_output_token_logprobs_offset = 0
        self.send_decode_id_offset = 0
        self.decoded_text = ""
        self.origin_input_ids = []
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.retraction_count = 0
        self.time_stats = None
        self.mm_image_tokens = 0
        self.mm_audio_tokens = 0
        self.mm_video_tokens = 0
        self.multimodal_inputs = None
        self.customized_info = None
        self._matched = matched

    def finished(self):
        return self._matched is not None

    def init_incremental_detokenize(self):
        return self.output_ids_through_stop, 0

    def check_match_stop_str_prefix(self):
        # Mirrors Req.check_match_stop_str_prefix: does the decoded tail's
        # suffix overlap a prefix of any stop string?
        tail = self.tokenizer.decode(self.output_ids[-10:])
        for i in range(1, min(len(tail), len(STOP_STR)) + 1):
            if tail[-i:] == STOP_STR[:i]:
                return True
        return False


def _make_accumulator(stream_interval=1, force_stream_interval=50):
    return _GenerationStreamAccumulator(
        return_logprob=False,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        disaggregation_mode=DisaggregationMode.NULL,
        default_stream_interval=stream_interval,
        default_force_stream_interval=force_stream_interval,
        get_cached_tokens_details=lambda req: None,
    )


def _client_visible_text(req, payloads):
    """Replay the DetokenizerManager / TokenizerManager text pipeline.

    Intermediate chunks append their decoded delta to ReqState.text
    verbatim. On finish, detokenizer_manager trims the full text at
    output.find(matched) and appends output_str[sent_offset:] -- which is
    empty when the trim point fell behind the already-sent offset, leaving
    a leaked stop-string prefix in the returned text.
    """
    text = ""
    sent_offset = 0
    for payload in payloads:
        for i, rid in enumerate(payload.rids):
            assert rid == req.rid
            delta = req.tokenizer.decode(payload.decode_ids[i])
            if payload.finished_reasons[i] is not None:
                full = req.tokenizer.decode(req.output_ids_through_stop)
                trimmed = full[: full.find(payload.finished_reasons[i]["matched"])]
                text += trimmed[sent_offset:]
                sent_offset = len(trimmed)
            else:
                text += delta
                sent_offset += len(delta)
    return text


class TestOutputStreamerStopStrPrefix(unittest.TestCase):
    def test_nonstream_boundary_holds_stop_str_prefix(self):
        # Stop string straddles the forced 50-token intermediate-output
        # boundary: after 50 tokens the output ends mid-stop-string.
        req = _FakeReq("r1", FILLER + STOP_TOKENS[:1], stream=False)
        matched = FINISH_MATCHED_STR(STOP_STR)

        # At len(output_ids) == 50 the intermediate output must be HELD;
        # emitting it would commit the stop-string prefix ahead of the final
        # trim point, leaking it into the returned text.
        acc = _make_accumulator()
        acc.accept(req=req)
        self.assertIsNone(acc.to_payload(dp_rank=0, is_idle_batch=False))

        # Finish: the final chunk carries everything since offset 0, so the
        # client-visible text excludes the stop string and its prefix.
        req.output_ids = FILLER + STOP_TOKENS
        req.output_ids_through_stop = req.output_ids
        req._matched = matched
        req.finished_reason = matched
        acc = _make_accumulator()
        acc.accept(req=req)
        final_payload = acc.to_payload(dp_rank=0, is_idle_batch=False)
        self.assertIsNotNone(final_payload)

        text = _client_visible_text(req, [final_payload])
        self.assertEqual(text, req.tokenizer.decode(FILLER))

    def test_nonstream_boundary_without_stop_prefix_still_emits(self):
        # Control: boundary hit with no stop-string overlap -> the forced
        # intermediate output is emitted exactly as before.
        boundary_ids = FILLER + [7]  # 50 tokens, tail has no stop-string prefix
        req = _FakeReq("r2", boundary_ids, stream=False)
        acc = _make_accumulator()
        acc.accept(req=req)
        payload = acc.to_payload(dp_rank=0, is_idle_batch=False)
        self.assertIsNotNone(payload)
        self.assertEqual(payload.rids, ["r2"])
        self.assertEqual(payload.decode_ids[0], boundary_ids)

    def test_stream_branch_parity(self):
        # Parity: the stream branch (stream_interval=1 would emit on every
        # token) already applies the same hold.
        req = _FakeReq("r3", FILLER + STOP_TOKENS[:1], stream=True)
        acc = _make_accumulator(stream_interval=1)
        acc.accept(req=req)
        self.assertIsNone(acc.to_payload(dp_rank=0, is_idle_batch=False))


if __name__ == "__main__":
    unittest.main()
