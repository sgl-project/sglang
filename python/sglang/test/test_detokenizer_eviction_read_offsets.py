"""Streaming must not lose text when detokenizer state is evicted mid-request.

The scheduler sends ``read_offsets`` as an absolute offset into the request's
cumulative surrogate+decode id list (a constant ``INIT_INCREMENTAL_DETOKENIZATION_OFFSET``
from the first chunk on), while ``decode_ids`` is sliced to only the ids since
the last send. When the detokenizer re-initializes a request's ``DecodeStatus``
after an eviction, it consumes ``read_offsets`` as the count of already-committed
tokens inside *this chunk*, so the constant points past ids that are not in the
chunk and the text of up to 5 newly generated tokens is silently dropped from
the stream.
"""

from array import array
from types import SimpleNamespace

from sglang.srt.managers.detokenizer_manager import (
    DetokenizerManager,
    LimitedCapacityDict,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.output_streamer import (
    _GenerationStreamAccumulator,
)

# Token ids -> text: prompt tail (15..19) must never appear in output deltas;
# output ids 20..27 map to distinct letters.
_TOK_TEXT = {i: chr(ord("a") + i - 10) for i in range(10, 20)}
_TOK_TEXT.update({i: chr(ord("A") + i - 20) for i in range(20, 28)})


class _FakeTokenizer:
    is_fast = True

    def decode(
        self, ids, skip_special_tokens=False, spaces_between_special_tokens=False
    ):
        return "".join(_TOK_TEXT[i] for i in ids)

    def batch_decode(
        self, ids_list, skip_special_tokens=False, spaces_between_special_tokens=False
    ):
        return [self.decode(ids) for ids in ids_list]


def _make_streaming_req() -> Req:
    req = object.__new__(Req)
    req.rid = "r-evict"
    req.beam_group = None
    req.finished_reason = None
    req.finished_len = None
    req.finished_output = False
    req.stream = True
    req.sampling_params = SimpleNamespace(
        stream_interval=1,
        stop_strs=[],
        stop_regex_strs=[],
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        no_stop_trim=False,
    )
    req.origin_input_ids_unpadded = array("i", range(10, 20))
    req.origin_input_ids = list(range(10, 20))
    req.output_ids = array("i", [])
    req.send_token_offset = 0
    req.send_decode_id_offset = 0
    req.send_output_token_logprobs_offset = 0
    req.decoded_text = ""
    req.http_worker_ipc = None
    req.surr_offset = None
    req.read_offset = None
    req.reasoning_tokens = 0
    req.cached_tokens = 0
    req.customized_info = None
    req.return_hidden_states = False
    req.return_routed_experts = False
    req.return_indexer_topk = False
    req.return_sampling_mask = False
    req.mm_image_tokens = 0
    req.mm_audio_tokens = 0
    req.mm_video_tokens = 0
    req.multimodal_inputs = None
    req.retraction_count = 0
    req.time_stats = None
    return req


def _make_accumulator() -> _GenerationStreamAccumulator:
    return _GenerationStreamAccumulator(
        return_logprob=False,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        return_sampling_mask=False,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        disaggregation_mode=None,
        default_stream_interval=1,
        default_force_stream_interval=1,
        get_cached_tokens_details=lambda req: None,
        rust_server_mode=False,
        current_weight_version=None,
    )


def _stream_one_step(req: Req):
    acc = _make_accumulator()
    acc.accept(req=req)
    return acc.to_payload(dp_rank=0, is_idle_batch=False)


def _make_detokenizer() -> DetokenizerManager:
    d = object.__new__(DetokenizerManager)
    d.decode_status = LimitedCapacityDict(capacity=4)
    d.tokenizer = _FakeTokenizer()
    d.vocab_size = None
    d.disable_tokenizer_batch_decode = False
    d.is_tool_call_parser_gpt_oss = False
    return d


def test_streamer_sends_chunk_relative_read_offsets():
    req = _make_streaming_req()

    req.output_ids.extend(array("i", [20]))
    p1 = _stream_one_step(req)
    # First chunk carries the 5 prompt-tail surrogate tokens inside decode_ids;
    # read_offset counts them as the committed prefix.
    assert p1.read_offsets == [5]
    assert list(p1.decode_ids[0]) == [15, 16, 17, 18, 19, 20]

    # Later chunks carry only new output ids, so their committed-prefix count
    # inside the chunk must be 0, not the absolute offset from the first chunk.
    req.output_ids.extend(array("i", [21]))
    p2 = _stream_one_step(req)
    assert list(p2.decode_ids[0]) == [21]
    assert p2.read_offsets == [0]

    req.output_ids.extend(array("i", [22]))
    p3 = _stream_one_step(req)
    assert list(p3.decode_ids[0]) == [22]
    assert p3.read_offsets == [0]


def test_evicted_stream_reinit_keeps_all_text():
    req = _make_streaming_req()
    detok = _make_detokenizer()

    deltas = []
    for step, new_ids in enumerate(([20], [21], [22], [23])):
        req.output_ids.extend(array("i", list(new_ids)))
        payload = _stream_one_step(req)
        if step == 1:
            # Simulate LimitedCapacityDict evicting the in-flight request's
            # state under pressure (the SGLANG_DETOKENIZER_MAX_STATES scenario).
            detok.decode_status.pop(req.rid, None)
        deltas.extend(detok._decode_batch_token_id_output(payload))

    assert "".join(deltas) == "ABCD"
