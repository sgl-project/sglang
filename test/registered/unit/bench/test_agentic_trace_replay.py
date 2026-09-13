"""Unit tests for agentic-trace replay: WekaTrace conversion and per-turn decode.

Both halves only otherwise run on an 8-GPU MI35x nightly, so the arithmetic is
pinned here. The turn plan has to make the *replayed* history match the recorded
prompt at every turn -- which it can only do if each round also generates the
recorded number of reply tokens, so the two are tested together.
"""

import contextlib
import json
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sglang.test.agentic_trace_replay import load_conversations, run_agentic_replay
from sglang.test.agentic_trace_utils import (
    DEFAULT_MIN_DELTA_TOKENS,
    RecordedTurn,
    build_agentic_trace,
    conversation_turns,
    iter_weka_records,
    plan_replay,
    recommended_output_len,
    weka_trace_url,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _build_tokenizer() -> PreTrainedTokenizerFast:
    """Whitespace WordLevel tokenizer: one word in, one token out."""
    vocab = {f"w{i}": i for i in range(512)}
    vocab["[UNK]"] = len(vocab)
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")


def _weka_record(record_id: str, main_requests, subagent_groups=()) -> dict:
    requests = [
        {"t": float(i), "type": "s", "in": in_len, "out": out_len, "hash_ids": []}
        for i, (in_len, out_len) in enumerate(main_requests)
    ]
    for group_index, inner in enumerate(subagent_groups):
        requests.append(
            {
                "t": float(1000 + group_index),
                "type": "subagent",
                "agent_id": f"subagent_{group_index}",
                "requests": [
                    {"t": 0.0, "type": "n", "in": in_len, "out": out_len}
                    for in_len, out_len in inner
                ],
            }
        )
    return {
        "id": record_id,
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": requests,
    }


class TestTurnPlan(CustomTestCase):
    def test_replayed_history_matches_recorded_prompts(self):
        # Real prefix growth from the published corpus' first trajectory.
        recorded = [
            RecordedTurn(640, 20),
            RecordedTurn(63744, 395),
            RecordedTurn(65408, 265),
            RecordedTurn(66880, 257),
        ]
        segments = plan_replay(recorded)
        self.assertEqual(len(segments), 1)

        replayed = 0
        for planned, want in zip(segments[0], recorded):
            replayed += planned.new_tokens
            self.assertEqual(replayed, want.prompt_tokens)
            self.assertEqual(planned.prompt_tokens, want.prompt_tokens)
            self.assertEqual(planned.output_tokens, want.output_tokens)
            replayed += planned.output_tokens

    def test_uniform_output_len_overrides_the_recording(self):
        recorded = [RecordedTurn(4096, 20), RecordedTurn(8192, 3693)]
        (plan,) = plan_replay(recorded, output_len=256)
        self.assertEqual([p.output_tokens for p in plan], [256, 256])
        self.assertEqual(plan[1].new_tokens, 8192 - 4096 - 256)

    def test_small_shortfall_clamps_then_resyncs(self):
        # A trimmed message leaves replay slightly ahead of the recording; that
        # is noise, so the conversation continues and re-syncs.
        recorded = [
            RecordedTurn(4096, 256),
            RecordedTurn(4224, 128),
            RecordedTurn(8192, 64),
        ]
        (plan,) = plan_replay(recorded)
        self.assertEqual(plan[0].new_tokens, 4096)
        self.assertEqual(plan[1].new_tokens, DEFAULT_MIN_DELTA_TOKENS)
        self.assertEqual(plan[1].prompt_tokens, 4096 + 256 + DEFAULT_MIN_DELTA_TOKENS)
        # The next round that grows past the offset lands back on the recording.
        self.assertEqual(plan[2].prompt_tokens, 8192)

    def test_context_reset_starts_a_new_conversation(self):
        recorded = [
            RecordedTurn(40960, 256),
            RecordedTurn(41984, 128),
            RecordedTurn(3840, 64),  # agent compacted its context
            RecordedTurn(5120, 32),
        ]
        first, second = plan_replay(recorded)

        self.assertEqual([p.prompt_tokens for p in first], [40960, 41984])
        # The fresh segment replays the small prompt the agent actually sent.
        self.assertEqual([p.prompt_tokens for p in second], [3840, 5120])

    def test_max_turns_truncates(self):
        recorded = [RecordedTurn(100 * (i + 1), 16) for i in range(10)]
        (plan,) = plan_replay(recorded, max_turns=3)
        self.assertEqual(len(plan), 3)

    def test_empty_trace_yields_no_segments(self):
        self.assertEqual(plan_replay([]), [])


class TestRecordParsing(CustomTestCase):
    def setUp(self):
        self.record = _weka_record(
            "abc",
            main_requests=[(640, 20), (63744, 395)],
            subagent_groups=[[(35008, 153), (35456, 126)]],
        )

    def test_main_agent_stream_only_by_default(self):
        self.assertEqual(
            conversation_turns(self.record),
            [[RecordedTurn(640, 20), RecordedTurn(63744, 395)]],
        )

    def test_subagent_groups_become_extra_conversations(self):
        self.assertEqual(
            conversation_turns(self.record, include_subagents=True),
            [
                [RecordedTurn(640, 20), RecordedTurn(63744, 395)],
                [RecordedTurn(35008, 153), RecordedTurn(35456, 126)],
            ],
        )

    def test_recommended_output_len_is_the_recorded_mean(self):
        self.assertEqual(recommended_output_len([self.record]), round((20 + 395) / 2))
        self.assertEqual(
            recommended_output_len([self.record], include_subagents=True),
            round((20 + 395 + 153 + 126) / 4),
        )
        self.assertIsNone(recommended_output_len([]))


class TestBuildAgenticTrace(CustomTestCase):
    def setUp(self):
        self.tokenizer = _build_tokenizer()
        self.records = [
            _weka_record("aaa", [(64, 10), (256, 30), (512, 40)]),
            _weka_record("bbb", [(128, 20), (640, 50)]),
        ]

    def test_turn_content_hits_the_planned_token_counts(self):
        document, stats = build_agentic_trace(self.records, self.tokenizer)

        self.assertEqual(stats.num_conversations, 2)
        self.assertEqual(stats.num_turns, 5)
        self.assertEqual(stats.max_prompt_tokens, 640)
        self.assertEqual(stats.total_output_tokens, 10 + 30 + 40 + 20 + 50)

        for conversation, record in zip(document["conversations"], self.records):
            replayed = 0
            for turn, recorded in zip(conversation, conversation_turns(record)[0]):
                content = turn["messages"][0]["content"]
                delta = len(self.tokenizer.encode(content, add_special_tokens=False))
                replayed += delta
                self.assertEqual(replayed, recorded.prompt_tokens)
                self.assertEqual(turn["prompt_tokens"], recorded.prompt_tokens)
                self.assertEqual(turn["output_tokens"], recorded.output_tokens)
                replayed += recorded.output_tokens

    def test_conversations_do_not_share_a_prefix(self):
        # AgentX pins --cache-bust first_turn_prefix, so trajectories must not
        # collide in the radix cache.
        document, _ = build_agentic_trace(self.records, self.tokenizer)
        first_turns = {
            conversation[0]["messages"][0]["content"]
            for conversation in document["conversations"]
        }
        self.assertEqual(len(first_turns), len(document["conversations"]))

    def test_generation_is_deterministic(self):
        first, _ = build_agentic_trace(self.records, self.tokenizer)
        second, _ = build_agentic_trace(self.records, self.tokenizer)
        self.assertEqual(first, second)

        reseeded, _ = build_agentic_trace(self.records, self.tokenizer, seed=7)
        self.assertNotEqual(first["conversations"], reseeded["conversations"])

    def test_max_conversations_caps_output(self):
        document, stats = build_agentic_trace(
            self.records, self.tokenizer, max_conversations=1
        )
        self.assertEqual(len(document["conversations"]), 1)
        self.assertEqual(stats.num_conversations, 1)

    def test_empty_corpus_raises(self):
        with self.assertRaises(ValueError):
            build_agentic_trace([], self.tokenizer)

    def test_recorded_output_lengths_survive_a_round_trip(self):
        document, _ = build_agentic_trace(self.records, self.tokenizer, max_turns=2)
        conversations = _load_written_trace(document)

        self.assertEqual(len(conversations), 2)
        self.assertTrue(all(len(c) == 2 for c in conversations))
        self.assertEqual([t.output_len for t in conversations[0]], [10, 30])
        self.assertEqual([t.output_len for t in conversations[1]], [20, 50])
        self.assertEqual(conversations[0][0].messages[0]["role"], "user")

    def test_uniform_output_len_overrides_the_recorded_lengths(self):
        document, _ = build_agentic_trace(self.records, self.tokenizer, max_turns=2)
        conversations = _load_written_trace(document, output_len=24)

        self.assertTrue(all(t.output_len == 24 for c in conversations for t in c))

    def test_loading_caps_conversations_and_turns(self):
        document, _ = build_agentic_trace(self.records, self.tokenizer)
        conversations = _load_written_trace(document, num_conversations=1, max_turns=2)

        self.assertEqual(len(conversations), 1)
        self.assertEqual(len(conversations[0]), 2)


class TestRecordStreaming(CustomTestCase):
    def test_local_jsonl_is_streamed_and_limited(self):
        records = [_weka_record(f"id{i}", [(64, 8)]) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "traces.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
                f.write("\n")  # blank lines are skipped, not parsed

            self.assertEqual(len(list(iter_weka_records(str(path)))), 5)
            limited = list(iter_weka_records(str(path), limit=2))

        self.assertEqual([r["id"] for r in limited], ["id0", "id1"])

    def test_default_source_is_the_published_corpus(self):
        self.assertTrue(weka_trace_url().endswith("/traces.jsonl"))
        self.assertIn("cc-traces-weka", weka_trace_url())


def _load_written_trace(document, **kwargs):
    """Round-trip a converted document through the replay loader."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "trace.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(document, f)
        return load_conversations(str(path), **kwargs)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _ChatHandler(BaseHTTPRequestHandler):
    """Minimal streaming OpenAI chat endpoint that records what it was asked for.

    Streams as many tokens as the request asked for, so a caller can check both
    the requested decode length and how history accumulates, and reports a
    fixed cached-token count so the cache report has something to aggregate.
    """

    request_bodies: list = []
    cached_tokens_per_turn = 4

    def _respond(self, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _stream(self, chunks: list):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler interface)
        # /v1/models is the readiness probe; /server_info is where the replay
        # reads the speculative accept length from.
        self._respond(
            {
                "data": [{"id": "dummy-model"}],
                "internal_states": [{"avg_spec_accept_length": 2.5}],
            }
        )

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler interface)
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length)) if length else {}
        if "chat/completions" not in self.path:  # /flush_cache and friends
            self._respond({})
            return

        self.request_bodies.append(body)
        num_tokens = body.get("max_completion_tokens", 1)
        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "w1 "}}]}
            for _ in range(num_tokens)
        ]
        final = {"choices": [], "usage": {"completion_tokens": num_tokens}}
        # The real endpoint reports its cache details only when the request asks
        # for them, so a replay that forgets to ask must see zeros here too.
        if body.get("return_cached_tokens_details"):
            final["sglext"] = {
                "cached_tokens_details": {"device": self.cached_tokens_per_turn}
            }
        chunks.append(final)
        self._stream(chunks)

    def log_message(self, fmt, *args):
        return


class _FailingSecondTurnHandler(_ChatHandler):
    """Fails every round that carries history, i.e. every round but the first."""

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler interface)
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length)) if length else {}
        if "chat/completions" not in self.path:
            self._respond({})
            return
        self.request_bodies.append(body)
        if len(body.get("messages", [])) > 1:
            self.send_error(500, "boom")
            return
        self._stream(
            [
                {"choices": [{"index": 0, "delta": {"content": "w1 "}}]},
                {"choices": [], "usage": {"completion_tokens": 1}},
            ]
        )


@contextlib.contextmanager
def _mock_chat_server(handler_cls=_ChatHandler):
    class Handler(handler_cls):
        request_bodies = []

    server = HTTPServer(("127.0.0.1", _free_port()), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", Handler.request_bodies
    finally:
        server.shutdown()
        server.server_close()


class TestAgenticReplayDriver(CustomTestCase):
    """The replay must ask each round for the length its turn recorded.

    This is the chain the MI35x nightly depends on and the reason the driver
    exists: trace conversion, per-turn decode lengths, history accumulation and
    the metrics hand-off, none of which can run without an MI35x node otherwise.
    """

    def setUp(self):
        self.tokenizer = _build_tokenizer()
        self.records = [
            _weka_record("aaa", [(64, 10), (256, 30)]),
            _weka_record("bbb", [(128, 20), (640, 50)]),
        ]

    @contextlib.contextmanager
    def _replay(self, handler_cls=_ChatHandler, **kwargs):
        document, stats = build_agentic_trace(self.records, self.tokenizer)
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            _mock_chat_server(handler_cls) as (base_url, request_bodies),
        ):
            trace_path = Path(tmpdir) / "trace.json"
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(document, f)

            output_file = kwargs.pop("output_file", str(Path(tmpdir) / "result.jsonl"))
            kwargs.setdefault("flush_cache", False)
            result = run_agentic_replay(
                base_url=base_url,
                model="dummy-model",
                tokenizer=self.tokenizer,
                trace_path=str(trace_path),
                warmup=False,
                output_file=output_file,
                **kwargs,
            )
            yield result, stats, list(request_bodies), Path(output_file)

    def test_every_round_asks_for_its_recorded_length(self):
        # One lane, so the two trajectories replay one after the other and the
        # recorded bodies are in trace order.
        with self._replay(max_concurrency=1) as (result, stats, bodies, _):
            self.assertEqual(result["completed"], stats.num_turns)
            self.assertEqual(result["failed"], 0)
            self.assertEqual(
                [b["max_completion_tokens"] for b in bodies], [10, 30, 20, 50]
            )
            self.assertEqual(result["total_output_tokens"], stats.total_output_tokens)
            # Round two of each conversation carries round one plus its reply.
            self.assertEqual([len(b["messages"]) for b in bodies], [1, 3, 1, 3])
            self.assertEqual(bodies[1]["messages"][1]["role"], "assistant")

    def test_uniform_output_len_overrides_the_recording(self):
        with self._replay(output_len=7, max_concurrency=2) as (_, _, bodies, _):
            self.assertEqual([b["max_completion_tokens"] for b in bodies], [7] * 4)

    def test_metrics_cover_both_sides_of_the_workload(self):
        with self._replay(max_concurrency=2, flush_cache=True) as (
            result,
            stats,
            bodies,
            output_file,
        ):
            # The prompt side is summed per turn, so a trajectory's growing
            # context is counted once for every round that re-sends it.
            self.assertEqual(result["total_input_tokens"], stats.total_prompt_tokens)
            self.assertGreater(result["input_throughput"], 0)
            self.assertGreater(result["output_throughput"], 0)
            self.assertGreater(result["mean_ttft_ms"], 0)
            self.assertEqual(result["accept_length"], 2.5)

            cache = result["cache_report"]
            self.assertEqual(
                cache["total_cached_tokens"],
                stats.num_turns * _ChatHandler.cached_tokens_per_turn,
            )
            self.assertTrue(
                all(b["return_cached_tokens_details"] for b in bodies),
                "every round must opt into the server's cache-detail reporting",
            )
            self.assertEqual(cache["total_prompt_tokens"], result["total_input_tokens"])
            self.assertEqual(
                cache["device_cached_tokens"], cache["total_cached_tokens"]
            )

            with open(output_file, encoding="utf-8") as f:
                self.assertEqual(json.loads(f.readlines()[-1]), result)

    def test_a_failed_turn_stops_its_trajectory(self):
        with self._replay(handler_cls=_FailingSecondTurnHandler, max_concurrency=2) as (
            result,
            stats,
            bodies,
            _,
        ):
            # Each conversation gets through its first round and stops on the
            # second; the turns it never reached still count as not completed.
            self.assertEqual(result["completed"], 2)
            self.assertEqual(result["failed"], stats.num_turns - 2)
            self.assertEqual(len(bodies), 4)
            self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()
