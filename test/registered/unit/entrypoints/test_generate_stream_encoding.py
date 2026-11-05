"""Native /generate logprob streams and the cached chunk encoder: every chunk
the TokenizerManager streams must encode exactly like dumps_json, and the route
must use the encoder only for cumulative logprob streams."""

import asyncio
import random
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import orjson
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints import http_server  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    BatchStrOutput,
    GenerateReqInput,
    RequestOutputStats,
    flatten_output_ids,
)
from sglang.srt.managers.tokenizer_manager import (  # noqa: E402
    ReqState,
    TokenizerManager,
)
from sglang.srt.observability.req_time_stats import (  # noqa: E402
    APIServerReqTimeStats,
)
from sglang.srt.runtime_context import get_context  # noqa: E402
from sglang.srt.utils.json_response import (  # noqa: E402
    CUMULATIVE_STREAM_ENCODER_AVAILABLE,
    CumulativeStreamEncoder,
    dumps_json,
)

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

# Token strings for return_text_in_logprobs, including characters JSON escapes.
_TOKEN_TEXTS = (' "q"', "a\\b", "\n", "\u00e9", "\U0001f600", "\x00", "<|im_end|>")
_PROMPT_TOKEN_IDS = [101, 102, 103, 104, 105, 106]
_LOGPROB_FIELDS = tuple(
    f"{side}_{kind}_{part}"
    for side in ("input", "output")
    for kind in ("token_logprobs", "top_logprobs", "token_ids_logprobs")
    for part in ("val", "idx")
)


class _Tokenizer:
    def batch_decode(self, token_ids):
        return [_TOKEN_TEXTS[ids[0] % len(_TOKEN_TEXTS)] for ids in token_ids]


def _make_tokenizer_manager(case) -> TokenizerManager:
    """A TokenizerManager with what its output path reads, bypassing __init__."""
    override = get_context().override_server_args(speculative_algorithm=None)
    override.install()
    case.addCleanup(override.restore)
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.rid_to_state = {}
    tm.enable_metrics = False
    tm.enable_lora = False
    tm.incremental_streaming_output = False
    tm.skip_tokenizer_init = False
    tm.dump_requests_folder = ""
    tm.crash_dump_folder = ""
    tm.tokenizer = _Tokenizer()
    tm.request_logger = Mock()
    tm.request_metrics_exporter_manager = Mock()
    tm.request_metrics_exporter_manager.exporter_enabled.return_value = False
    return tm


def _generate_obj(rid, *, top_k, probe_ids, logprob_start_len, text_in_logprobs):
    obj = GenerateReqInput(
        text="prompt",
        rid=rid,
        stream=True,
        return_logprob=True,
        logprob_start_len=logprob_start_len,
        top_logprobs_num=top_k,
        token_ids_logprob=probe_ids,
        return_text_in_logprobs=text_in_logprobs,
        return_prompt_token_ids=True,
    )
    obj.normalize_batch_and_arguments()
    return obj


def _logprobs(rng, n):
    # Some -inf, which JSON carries as null.
    return [
        float("-inf") if rng.random() < 0.05 else -8 * rng.random() for _ in range(n)
    ]


def _token_rows(rng, n, first_is_none):
    vals = [
        None if first_is_none and p == 0 else _logprobs(rng, 1)[0] for p in range(n)
    ]
    return vals, [rng.randrange(150000) for _ in range(n)]


def _position_rows(rng, n, width, first_is_none):
    vals = [
        None if first_is_none and p == 0 else _logprobs(rng, width) for p in range(n)
    ]
    idxs = [None if v is None else [rng.randrange(150000) for _ in v] for v in vals]
    return vals, idxs


def _request_rows(rng, rid, *, step_sizes, top_k, probe_ids, input_len):
    """One request's detokenizer rows, one per step with that many new tokens.
    The first row also carries `input_len` prompt positions of logprobs."""
    rows = []
    num_tokens = 0
    for step, size in enumerate(step_sizes):
        first, last = step == 0, step == len(step_sizes) - 1
        ids = [rng.randrange(150000) for _ in range(size)]
        num_tokens += size
        row = {
            "rid": rid,
            "finished_reason": (
                {"type": "length", "length": num_tokens} if last else None
            ),
            "text": "".join(_TOKEN_TEXTS[i % len(_TOKEN_TEXTS)] for i in ids),
            "ids": ids,
            "completion_tokens": num_tokens,
            "request_stats": (
                RequestOutputStats(
                    prompt_tokens=len(_PROMPT_TOKEN_IDS),
                    cached_tokens=0,
                    cached_tokens_details=None,
                    retraction_count=0,
                    image_tokens=0,
                    audio_tokens=0,
                    video_tokens=0,
                    time_stats=None,
                )
                if first or last
                else None
            ),
        }
        for side, n in (("input", input_len if first else 0), ("output", size)):
            # The first prompt position has no logprob.
            first_is_none = side == "input"
            row[f"{side}_token_logprobs_val"], row[f"{side}_token_logprobs_idx"] = (
                _token_rows(rng, n, first_is_none)
            )
            row[f"{side}_top_logprobs_val"], row[f"{side}_top_logprobs_idx"] = (
                _position_rows(rng, n if top_k else 0, top_k, first_is_none)
            )
            (
                row[f"{side}_token_ids_logprobs_val"],
                row[f"{side}_token_ids_logprobs_idx"],
            ) = _position_rows(
                rng, n if probe_ids else 0, len(probe_ids or ()), first_is_none
            )
        rows.append(row)
    return rows


def _batch_str_output(rows) -> BatchStrOutput:
    output_ids_flat, output_ids_offsets = flatten_output_ids(
        [row["ids"] for row in rows]
    )
    fields = {
        "rids": [row["rid"] for row in rows],
        "finished_reasons": [row["finished_reason"] for row in rows],
        "output_strs": [row["text"] for row in rows],
        "output_ids_flat": output_ids_flat,
        "output_ids_offsets": output_ids_offsets,
        "completion_tokens": [row["completion_tokens"] for row in rows],
        "reasoning_tokens": [0] * len(rows),
        "request_stats": [row["request_stats"] for row in rows],
    }
    fields.update({name: [row[name] for row in rows] for name in _LOGPROB_FIELDS})
    # Per-row outputs these requests did not ask for.
    for field in msgspec.structs.fields(BatchStrOutput):
        if (
            field.name not in fields
            and field.default is msgspec.NODEFAULT
            and field.default_factory is msgspec.NODEFAULT
        ):
            fields[field.name] = [None] * len(rows)
    return BatchStrOutput(**fields)


async def _stream_chunks(tm, objs, steps):
    """Feeds `steps` of (rows, drain) through `_handle_batch_output` while the
    requests' stream is consumed and each chunk encoded both ways, as
    `stream_results` would. Without a drain, the next step's rows queue up."""
    for obj in objs:
        state = ReqState(
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            time_stats=APIServerReqTimeStats(),
        )
        state.prompt_token_ids = list(_PROMPT_TOKEN_IDS)
        tm.rid_to_state[obj.rid] = state
    waiters = [tm._wait_one_response(obj) for obj in objs]
    if len(objs) == 1:
        stream = waiters[0]
    else:
        stream = tm._stream_batch_responses(waiters, [obj.rid for obj in objs])
    encoder = CumulativeStreamEncoder()
    chunks = []

    async def consume():
        async for out in stream:
            chunks.append((encoder.encode(out), dumps_json(out)))

    consumer = asyncio.create_task(consume())
    for rows, drain in steps:
        await tm._handle_batch_output(_batch_str_output(rows))
        if drain:
            for _ in range(8):
                await asyncio.sleep(0)
    await asyncio.wait_for(consumer, timeout=30)
    return encoder, chunks


@unittest.skipUnless(CUMULATIVE_STREAM_ENCODER_AVAILABLE, "requires orjson.Fragment")
class TestTokenizerManagerStreamEncoding(
    unittest.IsolatedAsyncioTestCase, CustomTestCase
):
    def _assert_chunks_encode_like_dumps_json(self, encoder, chunks):
        for i, (cached, full) in enumerate(chunks):
            self.assertEqual(cached, full, f"chunk {i}")
        self.assertEqual(encoder._requests, {})
        return [orjson.loads(full) for _, full in chunks]

    async def test_single_request_chunks(self):
        """One request with prompt, top-k, token-id and text logprobs; its last
        chunk carries a copy of output_ids."""
        tm = _make_tokenizer_manager(self)
        rng = random.Random(0)
        probe_ids = [7, 11, 13]
        obj = _generate_obj(
            "single",
            top_k=5,
            probe_ids=probe_ids,
            logprob_start_len=0,
            text_in_logprobs=True,
        )
        rows = _request_rows(
            rng,
            obj.rid,
            step_sizes=[1, 2, 1, 1, 3, 1, 1, 2],
            top_k=5,
            probe_ids=probe_ids,
            input_len=len(_PROMPT_TOKEN_IDS),
        )
        steps = [([row], step not in (2, 5)) for step, row in enumerate(rows)]

        encoder, chunks = await _stream_chunks(tm, [obj], steps)

        parsed = self._assert_chunks_encode_like_dumps_json(encoder, chunks)
        self.assertLess(len(parsed), len(steps))  # queued steps coalesced
        last = parsed[-1]
        meta_info = last["meta_info"]
        self.assertEqual(len(last["output_ids"]), 12)
        self.assertEqual(last["prompt_token_ids"], _PROMPT_TOKEN_IDS)
        self.assertEqual(len(meta_info["input_token_logprobs"]), 6)
        self.assertEqual(len(meta_info["output_top_logprobs"]), 12)
        self.assertEqual(len(meta_info["output_token_ids_logprobs"]), 12)
        self.assertIn(meta_info["output_top_logprobs"][0][0][2], _TOKEN_TEXTS)

    async def test_batch_request_chunks(self):
        """A batch stream, as n > 1 sampling or a prompt list gives, interleaves
        requests that differ in logprob options and finish at different steps."""
        tm = _make_tokenizer_manager(self)
        rng = random.Random(1)
        options = (
            dict(top_k=0, probe_ids=None, logprob_start_len=-1, text_in_logprobs=False),
            dict(top_k=2, probe_ids=[5], logprob_start_len=0, text_in_logprobs=True),
            dict(top_k=5, probe_ids=None, logprob_start_len=2, text_in_logprobs=False),
        )
        objs = [_generate_obj(f"batch-{i}", **o) for i, o in enumerate(options)]
        requests = [
            _request_rows(
                rng,
                obj.rid,
                step_sizes=[1] * num_steps,
                top_k=o["top_k"],
                probe_ids=o["probe_ids"],
                input_len=0 if o["logprob_start_len"] < 0 else 4,
            )
            for obj, o, num_steps in zip(objs, options, (4, 9, 6))
        ]
        steps = [
            ([rows[step] for rows in requests if step < len(rows)], step % 3 != 1)
            for step in range(9)
        ]

        encoder, chunks = await _stream_chunks(tm, objs, steps)

        parsed = self._assert_chunks_encode_like_dumps_json(encoder, chunks)
        last_chunks = {
            out["index"]: out
            for out in parsed
            if out["meta_info"]["finish_reason"] is not None
        }
        self.assertEqual(
            {i: len(out["output_ids"]) for i, out in last_chunks.items()},
            {0: 4, 1: 9, 2: 6},
        )
        self.assertEqual(
            len(last_chunks[1]["meta_info"]["output_token_ids_logprobs"]), 9
        )


@unittest.skipUnless(CUMULATIVE_STREAM_ENCODER_AVAILABLE, "requires orjson.Fragment")
class TestGenerateStreamRoute(CustomTestCase):
    def _post(self, body, *, incremental_streaming_output=False):
        """The SSE body of a three-chunk cumulative stream, the body a full
        encode of each chunk gives, and the chunks the cached encoder wrote."""
        expected = []

        class FakeTokenizerManager:
            async def generate_request(self, obj, request):
                output_ids, logprobs = [], []
                for step in range(3):
                    output_ids.append(step)
                    logprobs.append((-0.5 * step, step, None))
                    last = step == 2
                    out = {
                        "text": "t" * step,
                        "output_ids": output_ids.copy() if last else output_ids,
                        "meta_info": {
                            "id": "r",
                            "finish_reason": {"type": "length"} if last else None,
                            "output_token_logprobs": logprobs,
                        },
                    }
                    expected.append(b"data: " + dumps_json(out) + b"\n\n")
                    yield out

            def create_abort_task(self, obj):
                return None

        tokenizer_manager = FakeTokenizerManager()
        tokenizer_manager.incremental_streaming_output = incremental_streaming_output
        app = FastAPI()
        app.add_api_route("/generate", http_server.generate_request, methods=["POST"])
        prior_state = http_server.get_global_state()
        http_server.set_global_state(
            SimpleNamespace(tokenizer_manager=tokenizer_manager)
        )
        encode = CumulativeStreamEncoder.encode
        try:
            with patch.object(
                CumulativeStreamEncoder,
                "encode",
                autospec=True,
                side_effect=encode,
            ) as cached_encode:
                response = TestClient(app).post("/generate", json=body)
        finally:
            http_server._global_state = prior_state
        expected.append(b"data: [DONE]\n\n")
        return response.content, b"".join(expected), cached_encode.call_count

    def test_only_cumulative_logprob_streams_use_the_cached_encoder(self):
        logprob_stream = {"text": "hi", "stream": True, "return_logprob": True}
        cases = (
            ("logprob stream", logprob_stream, False, 3),
            ("no logprobs", {"text": "hi", "stream": True}, False, 0),
            ("incremental streaming output", logprob_stream, True, 0),
        )
        for name, body, incremental, num_cached in cases:
            with self.subTest(name):
                content, expected, calls = self._post(
                    body, incremental_streaming_output=incremental
                )
                self.assertEqual(content, expected)
                self.assertEqual(calls, num_cached)
        with self.subTest("turned off by env"):
            with envs.SGLANG_ENABLE_CACHED_STREAM_ENCODING.override(False):
                content, expected, calls = self._post(logprob_stream)
            self.assertEqual(content, expected)
            self.assertEqual(calls, 0)


if __name__ == "__main__":
    unittest.main()
