"""Unit tests for the agentic multi-turn benchmark helpers.

The nightly perf tests that use these helpers need an MI35x node and a
multi-hundred-GB checkpoint, so the parts that can go wrong quietly -- the
shape of the synthesized corpus and the parsing of a bench_serving record --
are pinned here instead.
"""

import argparse
import inspect
import json
import os
import tempfile
import unittest
from argparse import Namespace

from sglang.benchmark import serving as bench_serving
from sglang.benchmark.datasets.agentic_trace import (
    DEFAULT_AGENTIC_OUTPUT_LEN,
    AgenticTraceDataset,
)
from sglang.benchmark.serving import MULTI_TURN_BACKENDS, _normalize_round_messages
from sglang.test.agentic_bench_utils import (
    AGENTIC_TRACE_PATH_ENV,
    AgenticTraceSpec,
    _parse_bench_record,
    build_agentic_bench_command,
    calibrate_tokens_per_part,
    generate_agentic_markdown_report,
    resolve_agentic_trace,
    write_agentic_coding_trace,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

SMALL_SPEC = AgenticTraceSpec(
    num_conversations=5,
    turns_per_conversation=4,
    system_prompt_tokens=64,
    repo_context_tokens=256,
    turn_tokens=32,
)


class _StubTokenizer:
    """Splits on a fixed rule so calibration is testable without the Hub."""

    def __init__(self, tokens_per_char_group=4):
        self.tokens_per_char_group = tokens_per_char_group

    def encode(self, text, add_special_tokens=False):
        # One token per fixed-size chunk: a stand-in for subword splitting that
        # stays proportional to text length, which is all calibration needs.
        return [0] * (len(text) // self.tokens_per_char_group + 1)


def _bench_record(**overrides):
    record = {
        "completed": 24,
        "duration": 100.0,
        "output_throughput": 512.0,
        "mean_ttft_ms": 900.0,
        "p99_ttft_ms": 1800.0,
        "mean_itl_ms": 22.0,
        "p99_itl_ms": 40.0,
        "mean_e2e_latency_ms": 6000.0,
        "concurrency": 3.9,
        "accept_length": 3.42,
        "cache_report": {"cache_hit_rate_pct": 71.25, "host_cached_tokens": 120345},
        "errors": [None] * 24,
    }
    record.update(overrides)
    return record


class TestAgenticTraceSynthesis(CustomTestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _write(self, name="trace.json", spec=SMALL_SPEC, seed=42):
        path = write_agentic_coding_trace(
            os.path.join(self.tmpdir.name, name), spec=spec, seed=seed
        )
        with open(path, "r", encoding="utf-8") as f:
            return path, json.load(f)

    def test_trace_has_requested_shape(self):
        _, data = self._write()
        conversations = data["conversations"]
        self.assertEqual(len(conversations), SMALL_SPEC.num_conversations)
        for conversation in conversations:
            self.assertEqual(len(conversation), SMALL_SPEC.turns_per_conversation)

        first_turn, second_turn = conversations[0][0], conversations[0][1]
        self.assertEqual(
            [m["role"] for m in first_turn["messages"]], ["system", "user"]
        )
        # Later turns carry only the new delta; the replay rebuilds the rest.
        self.assertEqual([m["role"] for m in second_turn["messages"]], ["user"])
        self.assertEqual(first_turn["prompt_tokens"], SMALL_SPEC.first_turn_tokens())
        self.assertGreater(
            conversations[0][-1]["prompt_tokens"], first_turn["prompt_tokens"]
        )

    def test_prefix_structure(self):
        """The scaffold is shared; the repository context is not.

        Both halves matter: the shared scaffold is the cross-session prefix a
        coding agent re-sends, and the unique context is what each session's
        own later turns hit in cache.
        """
        _, data = self._write()
        systems = {c[0]["messages"][0]["content"] for c in data["conversations"]}
        self.assertEqual(len(systems), 1)

        contexts = {c[0]["messages"][1]["content"] for c in data["conversations"]}
        self.assertEqual(len(contexts), SMALL_SPEC.num_conversations)

    def test_deterministic_for_a_fixed_seed(self):
        path_a, _ = self._write("a.json", seed=7)
        path_b, _ = self._write("b.json", seed=7)
        path_c, _ = self._write("c.json", seed=8)
        with open(path_a) as a, open(path_b) as b, open(path_c) as c:
            text_a, text_b, text_c = a.read(), b.read(), c.read()
        self.assertEqual(text_a, text_b)
        self.assertNotEqual(text_a, text_c)

    def test_longer_context_spec_produces_longer_prompts(self):
        _, small = self._write("small.json")
        _, large = self._write(
            "large.json",
            spec=AgenticTraceSpec(
                num_conversations=2,
                turns_per_conversation=2,
                system_prompt_tokens=64,
                repo_context_tokens=4096,
                turn_tokens=32,
            ),
        )
        small_len = len(small["conversations"][0][0]["messages"][1]["content"])
        large_len = len(large["conversations"][0][0]["messages"][1]["content"])
        self.assertGreater(large_len, small_len * 4)


class TestTokenBudgetCalibration(CustomTestCase):
    """The declared budgets decide how much context each session carries, so
    they have to land near the truth, not just be documented."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _turn_one_tokens(self, tokenizer, spec, calibrate):
        path = write_agentic_coding_trace(
            os.path.join(self.tmpdir.name, "trace.json"),
            spec=spec,
            tokenizer=tokenizer if calibrate else None,
        )
        with open(path, "r", encoding="utf-8") as f:
            first_turn = json.load(f)["conversations"][0][0]
        return sum(
            len(tokenizer.encode(m["content"], add_special_tokens=False))
            for m in first_turn["messages"]
        )

    def test_calibration_reports_tokens_per_fragment(self):
        dense, sparse = _StubTokenizer(2), _StubTokenizer(8)
        self.assertGreater(
            calibrate_tokens_per_part(dense), calibrate_tokens_per_part(sparse)
        )

    def test_calibrated_budget_is_close_to_declared(self):
        spec = AgenticTraceSpec(
            num_conversations=1,
            turns_per_conversation=2,
            system_prompt_tokens=512,
            repo_context_tokens=4096,
            turn_tokens=256,
        )
        tokenizer = _StubTokenizer(3)
        measured = self._turn_one_tokens(tokenizer, spec, calibrate=True)
        declared = spec.first_turn_tokens()
        self.assertAlmostEqual(measured / declared, 1.0, delta=0.1)

    def test_calibration_adapts_to_the_tokenizer(self):
        """A tokenizer that splits twice as finely must yield half the text, so
        both land on the same budget."""
        spec = AgenticTraceSpec(
            num_conversations=1,
            turns_per_conversation=1,
            system_prompt_tokens=256,
            repo_context_tokens=2048,
            turn_tokens=128,
        )
        for chunk in (2, 6):
            tokenizer = _StubTokenizer(chunk)
            ratio = self._turn_one_tokens(tokenizer, spec, calibrate=True) / (
                spec.first_turn_tokens()
            )
            self.assertAlmostEqual(ratio, 1.0, delta=0.1, msg=f"chunk={chunk}")


class TestAgenticTraceLoads(CustomTestCase):
    """The synthesized corpus has to satisfy the loader it was written for."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.path = write_agentic_coding_trace(
            os.path.join(self.tmpdir.name, "trace.json"), spec=SMALL_SPEC
        )

    def _load(self, **overrides):
        args = Namespace(
            dataset_path=self.path,
            num_prompts=3,
            sharegpt_output_len=None,
            dataset_offset=0,
            agentic_max_turns=None,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return AgenticTraceDataset.from_args(args).load(tokenizer=None)

    def test_rows_are_multi_turn(self):
        rows = self._load()
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(len(row.prompt), SMALL_SPEC.turns_per_conversation)
            self.assertEqual(row.output_len, DEFAULT_AGENTIC_OUTPUT_LEN)

    def test_turns_satisfy_the_multi_turn_detector(self):
        """`benchmark()` routes on this predicate; a shape it rejects would
        silently replay the corpus as single-shot prompts."""
        rows = self._load()
        for turn in rows[0].prompt:
            normalized = _normalize_round_messages(turn)
            self.assertIsNotNone(normalized)
            for message in normalized:
                self.assertIn(message["role"], ("system", "user"))
                self.assertTrue(message["content"])

    def test_sweep_knobs_are_honored(self):
        rows = self._load(agentic_max_turns=2, sharegpt_output_len=220, num_prompts=2)
        self.assertEqual([len(row.prompt) for row in rows], [2, 2])
        self.assertEqual({row.output_len for row in rows}, {220})

        rotated = self._load(dataset_offset=1, num_prompts=1)
        unrotated = self._load(dataset_offset=0, num_prompts=1)
        self.assertNotEqual(rotated[0].prompt[0], unrotated[0].prompt[0])


class TestResolveAgenticTrace(CustomTestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self._saved = os.environ.pop(AGENTIC_TRACE_PATH_ENV, None)
        if self._saved is not None:
            self.addCleanup(os.environ.__setitem__, AGENTIC_TRACE_PATH_ENV, self._saved)

    def test_synthesizes_when_unset(self):
        path = resolve_agentic_trace(self.tmpdir.name, spec=SMALL_SPEC)
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(os.path.dirname(path), self.tmpdir.name)

    def test_prefers_the_override(self):
        real = write_agentic_coding_trace(
            os.path.join(self.tmpdir.name, "real.json"), spec=SMALL_SPEC
        )
        os.environ[AGENTIC_TRACE_PATH_ENV] = real
        self.addCleanup(os.environ.pop, AGENTIC_TRACE_PATH_ENV, None)
        self.assertEqual(resolve_agentic_trace(self.tmpdir.name), real)

    def test_rejects_a_missing_override(self):
        os.environ[AGENTIC_TRACE_PATH_ENV] = os.path.join(self.tmpdir.name, "nope.json")
        self.addCleanup(os.environ.pop, AGENTIC_TRACE_PATH_ENV, None)
        with self.assertRaises(FileNotFoundError):
            resolve_agentic_trace(self.tmpdir.name)


class TestParseBenchRecord(CustomTestCase):
    def test_maps_a_clean_record(self):
        point = _parse_bench_record(_bench_record(), concurrency=4, conversations=8)
        self.assertEqual(point.concurrency, 4)
        self.assertEqual(point.conversations, 8)
        self.assertEqual(point.total_turns, 24)
        self.assertEqual(point.completed_turns, 24)
        self.assertEqual(point.failed_turns, 0)
        self.assertEqual(point.output_throughput, 512.0)
        self.assertEqual(point.accept_length, 3.42)
        self.assertEqual(point.cache_hit_rate_pct, 71.25)
        self.assertEqual(point.host_cached_tokens, 120345)

    def test_counts_dropped_turns(self):
        record = _bench_record(completed=20, errors=[None] * 20 + ["boom"] * 4)
        point = _parse_bench_record(record, concurrency=4, conversations=8)
        self.assertEqual(point.total_turns, 24)
        self.assertEqual(point.failed_turns, 4)

    def test_drops_per_turn_arrays_from_raw(self):
        record = _bench_record(generated_texts=["x"] * 24, itls=[[1.0]] * 24)
        point = _parse_bench_record(record, concurrency=4, conversations=8)
        self.assertNotIn("generated_texts", point.raw)
        self.assertNotIn("errors", point.raw)
        self.assertIn("output_throughput", point.raw)

    def test_tolerates_a_record_without_cache_or_spec_fields(self):
        record = _bench_record()
        del record["cache_report"]
        del record["accept_length"]
        point = _parse_bench_record(record, concurrency=1, conversations=2)
        self.assertIsNone(point.cache_hit_rate_pct)
        self.assertIsNone(point.accept_length)


def _bench_serving_parser():
    """Rebuild bench_serving's parser without running a benchmark.

    ``cli_main`` constructs the parser and immediately parses and runs, so the
    parser is only reachable by replaying the construction up to that point.
    """
    lines = inspect.getsource(bench_serving.cli_main).splitlines()
    body = []
    for line in lines[1:]:
        if "args = parser.parse_args()" in line:
            break
        body.append(line[4:])
    namespace = {}
    exec("\n".join(body), {"argparse": argparse, **vars(bench_serving)}, namespace)
    return namespace["parser"]


class TestBuildAgenticBenchCommand(CustomTestCase):
    """The sweep shells out, so a renamed flag surfaces as a nightly failure a
    day later unless the command is parsed against the real CLI here."""

    def setUp(self):
        self.parser = _bench_serving_parser()

    def _parse(self, **overrides):
        kwargs = dict(
            base_url="http://127.0.0.1:30000",
            model_path="amd/GLM-5.2-MXFP4",
            tokenizer="/models/glm52",
            trace_path="/tmp/trace.json",
            conversations=24,
            concurrency=12,
            output_file="/tmp/out.jsonl",
        )
        kwargs.update(overrides)
        command = build_agentic_bench_command(**kwargs)
        self.assertEqual(command[:3], ["python3", "-m", "sglang.benchmark.serving"])
        return self.parser.parse_args(command[3:])

    def test_required_flags_parse(self):
        args = self._parse()
        self.assertIn(args.backend, MULTI_TURN_BACKENDS)
        self.assertEqual(args.dataset_name, "agentic-trace")
        self.assertEqual(args.dataset_path, "/tmp/trace.json")
        self.assertEqual(args.tokenizer, "/models/glm52")
        # For this dataset --num-prompts counts conversations, not turns.
        self.assertEqual(args.num_prompts, 24)
        self.assertEqual(args.max_concurrency, 12)
        self.assertTrue(args.cache_report)
        self.assertTrue(args.output_details)

    def test_optional_flags_are_omitted_when_unset(self):
        args = self._parse()
        self.assertIsNone(args.agentic_max_turns)
        self.assertIsNone(args.sharegpt_output_len)

    def test_optional_flags_parse_when_set(self):
        args = self._parse(
            max_turns=6, output_len=220, extra_bench_args=["--seed", "7"]
        )
        self.assertEqual(args.agentic_max_turns, 6)
        self.assertEqual(args.sharegpt_output_len, 220)
        self.assertEqual(args.seed, 7)


class TestAgenticMarkdownReport(CustomTestCase):
    def test_renders_a_row_per_point(self):
        points = [
            _parse_bench_record(_bench_record(), concurrency=c, conversations=c * 2)
            for c in (1, 4, 12)
        ]
        report = generate_agentic_markdown_report(points, "model (arm) [MI35x]")

        self.assertIn("### model (arm) [MI35x]", report)
        rows = [line for line in report.splitlines() if line.startswith("| ")]
        header, separator, *body = rows
        self.assertEqual(len(body), 3)
        self.assertEqual(header.count("|"), separator.count("|"))
        for row in body:
            self.assertEqual(row.count("|"), header.count("|"))
        self.assertIn("24/24", body[0])
        # Input throughput is meaningless for multi-turn replay, so it must not
        # appear as a column that readers would compare against other runs.
        self.assertNotIn("input throughput", header)

    def test_renders_missing_values_as_na(self):
        record = _bench_record()
        del record["cache_report"]
        record["accept_length"] = None
        point = _parse_bench_record(record, concurrency=1, conversations=2)
        report = generate_agentic_markdown_report([point], "no-extras")
        self.assertIn("n/a", report)


if __name__ == "__main__":
    unittest.main()
