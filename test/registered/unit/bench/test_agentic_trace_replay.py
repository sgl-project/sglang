"""Unit tests for agentic-trace replay: WekaTrace conversion and per-turn decode.

Both halves only otherwise run on an 8-GPU MI35x nightly, so the arithmetic is
pinned here. The turn plan has to make the *replayed* history match the recorded
prompt at every turn -- which it can only do if each round also generates the
recorded number of reply tokens, so the two are tested together.
"""

import asyncio
import json
import socket
import tempfile
import threading
import unittest
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sglang.benchmark.datasets.agentic_trace import AgenticTraceDataset
from sglang.benchmark.serving import (
    RequestFuncInput,
    async_request_openai_chat_completions,
    set_global_args,
    wrap_multi_turn_request_func,
)
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

register_cpu_ci(est_time=16, suite="base-a-test-cpu")


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
        rows = self._load_dataset(document, fixed_output_len=None)

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(len(row.prompt) == 2 for row in rows))
        self.assertEqual(rows[0].output_lens, [10, 30])
        self.assertEqual(rows[1].output_lens, [20, 50])
        self.assertEqual(rows[0].output_len, 10)
        self.assertEqual(rows[0].prompt[0][0]["role"], "user")

    def test_fixed_output_len_overrides_the_recorded_lengths(self):
        document, _ = build_agentic_trace(self.records, self.tokenizer, max_turns=2)
        rows = self._load_dataset(document, fixed_output_len=24)

        self.assertTrue(all(row.output_lens is None for row in rows))
        self.assertTrue(all(row.output_len == 24 for row in rows))

    def _load_dataset(self, document, fixed_output_len):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(document, f)

            return AgenticTraceDataset(
                dataset_path=str(path),
                num_requests=10,
                fixed_output_len=fixed_output_len,
                offset=0,
                max_turns=None,
            ).load(self.tokenizer)


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


class _ChatHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI chat endpoint that records what it was asked for."""

    request_bodies: list = []

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler interface)
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length)) if length else {}
        self.request_bodies.append(body)

        reply = " ".join(["tok"] * body.get("max_completion_tokens", 1))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": reply},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {"completion_tokens": len(reply.split())},
                }
            ).encode()
        )
        self.wfile.flush()

    def log_message(self, fmt, *args):
        return


class TestMultiTurnDecodeLengths(CustomTestCase):
    """``wrap_multi_turn_request_func`` must honour per-round output lengths."""

    @classmethod
    def setUpClass(cls):
        set_global_args(
            Namespace(
                disable_stream=True,
                disable_ignore_eos=False,
                print_requests=False,
                tokenizer="",
                header=None,
            )
        )

    def _replay(self, prompt, output_len, output_lens):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        class Handler(_ChatHandler):
            request_bodies = []

        server = HTTPServer(("127.0.0.1", port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            wrapped = wrap_multi_turn_request_func(
                async_request_openai_chat_completions, backend="sglang-oai-chat"
            )
            outputs = asyncio.run(
                wrapped(
                    RequestFuncInput(
                        prompt=prompt,
                        api_url=f"http://127.0.0.1:{port}/v1/chat/completions",
                        prompt_len=1,
                        output_len=output_len,
                        output_lens=output_lens,
                        model="dummy-model",
                        lora_name="",
                        image_data=None,
                        extra_request_body={},
                    )
                )
            )
            return outputs, Handler.request_bodies
        finally:
            server.shutdown()
            server.server_close()

    def test_each_round_requests_its_recorded_length(self):
        prompt = [
            [{"role": "user", "content": "turn one"}],
            [{"role": "user", "content": "turn two"}],
            [{"role": "user", "content": "turn three"}],
        ]
        outputs, bodies = self._replay(prompt, output_len=7, output_lens=[3, 11, 5])

        self.assertTrue(all(o.success for o in outputs), [o.error for o in outputs])
        self.assertEqual([b["max_completion_tokens"] for b in bodies], [3, 11, 5])
        # History still accumulates: user turn, reply, user turn, ...
        self.assertEqual([len(b["messages"]) for b in bodies], [1, 3, 5])
        self.assertEqual(bodies[1]["messages"][1]["role"], "assistant")

    def test_without_recorded_lengths_every_round_uses_output_len(self):
        prompt = [
            [{"role": "user", "content": "turn one"}],
            [{"role": "user", "content": "turn two"}],
        ]
        _, bodies = self._replay(prompt, output_len=7, output_lens=None)

        self.assertEqual([b["max_completion_tokens"] for b in bodies], [7, 7])

    def test_short_length_list_falls_back_for_the_remaining_rounds(self):
        prompt = [
            [{"role": "user", "content": "turn one"}],
            [{"role": "user", "content": "turn two"}],
        ]
        _, bodies = self._replay(prompt, output_len=7, output_lens=[3])

        self.assertEqual([b["max_completion_tokens"] for b in bodies], [3, 7])


if __name__ == "__main__":
    unittest.main()
