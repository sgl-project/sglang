"""Unit tests for the sgl-eval transport in ``sglang.test.sgl_eval``.

Hermetic: sgl-eval is replaced by fakes at the import boundary, so these run on
CPU CI whether or not the optional dependency is installed. What they guard is
the contract the AMD accuracy, disaggregation and perf tests now depend on:

  1. ``api_base_url`` accepts the ``host="http://127.0.0.1"`` form those tests
     pass. Building ``http://http://...`` instead would fail every one of them.
  2. The metrics dict carries ``accuracy``, ``invalid``, ``latency`` and
     ``output_throughput``. Callers read those names directly, so dropping one
     surfaces as a KeyError rather than a score change.
  3. Per-run generation settings (token budget, sampling, thinking) reach the
     generation config handed to the eval.
  4. An incomplete, unparsable or erroring run raises instead of reporting a low
     score, which would otherwise read as a model regression.
  5. A call from a worker thread is handed off rather than graded in place:
     math_verify arms SIGALRM, which only a main thread can do.
"""

import json
import sys
import tempfile
import threading
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.sgl_eval import api_base_url, run_sgl_eval
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@dataclass
class FakeGenConfig:
    """Mirrors the fields the transport reads back off the resolved config."""

    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: Optional[int] = None
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    reasoning_effort: Optional[str] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    extra_body: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    system_message: Optional[str] = None


@dataclass
class FakeSample:
    text: str = "answer"
    completion_tokens: int = 7
    finish_reason: str = "stop"


@dataclass
class FakeExampleResult:
    example: Any
    samples: List[FakeSample]
    scores: List[float]
    extracted: List[Optional[str]]


@dataclass
class FakeRunResult:
    aggregate: Dict[str, float]
    per_example: List[FakeExampleResult]
    latency: float = 2.0
    output_throughput: float = 21.0
    num_examples: int = 2
    n_repeats: int = 1
    partial: bool = False


class FakeSampler:
    def __init__(self, base_url=None, model=None, api_key=None):
        self.base_url = base_url
        self.model = model or "resolved-model"
        self.api_key = api_key
        self.aborted = False

    def abort(self):
        self.aborted = True


class FakePredictionsWriter:
    def __init__(self, out_dir, repeats, schema):
        self.closed = False

    def close(self):
        self.closed = True


@dataclass
class FakeSpec:
    """Stands in for a registered eval; records what it was run with."""

    name: str = "gsm8k"
    default_gen: FakeGenConfig = field(default_factory=FakeGenConfig)
    default_num_threads: int = 64
    pred_schema: str = "schema"
    result: Optional[FakeRunResult] = None
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _example_results(scores, extracted):
    return [
        FakeExampleResult(
            example=SimpleNamespace(id=f"gsm8k-{i}"),
            samples=[FakeSample()],
            scores=[score],
            extracted=[answer],
        )
        for i, (score, answer) in enumerate(zip(scores, extracted))
    ]


def _fake_sgl_eval_modules(spec):
    """Build the ``sgl_eval`` submodules the transport imports."""
    predictions = types.ModuleType("sgl_eval.predictions")
    predictions.PredictionsWriter = FakePredictionsWriter

    registry = types.ModuleType("sgl_eval.registry")
    registry.get = lambda name: spec

    sampler = types.ModuleType("sgl_eval.sampler")
    sampler.ChatCompletionSampler = FakeSampler

    root = types.ModuleType("sgl_eval")
    root.__path__ = []

    return {
        "sgl_eval": root,
        "sgl_eval.predictions": predictions,
        "sgl_eval.registry": registry,
        "sgl_eval.sampler": sampler,
    }


class TestApiBaseUrl(CustomTestCase):
    def test_url_forms(self):
        # `host` carrying a scheme is the form every migrated AMD test passes.
        cases = [
            (dict(base_url="http://127.0.0.1:30000"), "http://127.0.0.1:30000/v1"),
            (dict(base_url="http://127.0.0.1:30000/"), "http://127.0.0.1:30000/v1"),
            (dict(base_url="http://127.0.0.1:30000/v1"), "http://127.0.0.1:30000/v1"),
            (dict(host="http://127.0.0.1", port=8123), "http://127.0.0.1:8123/v1"),
            (dict(host="127.0.0.1", port=8123), "http://127.0.0.1:8123/v1"),
            (dict(host="https://gw.example", port=443), "https://gw.example:443/v1"),
            (dict(host="127.0.0.1", port=None), "http://127.0.0.1:30000/v1"),
        ]
        for kwargs, expected in cases:
            with self.subTest(**kwargs):
                self.assertEqual(api_base_url(SimpleNamespace(**kwargs)), expected)

    def test_base_url_wins_over_host_and_port(self):
        args = SimpleNamespace(base_url="http://lb:9000", host="127.0.0.1", port=1)
        self.assertEqual(api_base_url(args), "http://lb:9000/v1")


class TestRunSglEval(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def _run(self, spec, **overrides):
        args = SimpleNamespace(
            **{
                "eval_name": spec.name,
                "host": "http://127.0.0.1",
                "port": 8123,
                "model": "test-model",
                "num_examples": 2,
                "num_threads": 8,
                "sgl_eval_out_dir": self.tmpdir,
                **overrides,
            }
        )
        with patch.dict(sys.modules, _fake_sgl_eval_modules(spec)):
            return run_sgl_eval(args)

    @staticmethod
    def _passing_spec(scores=(1.0, 0.0), extracted=("18", None), **aggregate):
        return FakeSpec(
            result=FakeRunResult(
                aggregate={"score": 0.5, "error_rate": 0.0, **aggregate},
                per_example=_example_results(list(scores), list(extracted)),
            )
        )

    def test_metrics_expose_the_names_callers_read(self):
        metrics = self._run(self._passing_spec())

        self.assertEqual(metrics["score"], 0.5)
        # `accuracy` aliases `score`; both names are read across the AMD tests.
        self.assertEqual(metrics["accuracy"], 0.5)
        # One of the two answers did not parse.
        self.assertEqual(metrics["invalid"], 0.5)
        self.assertEqual(metrics["latency"], 2.0)
        self.assertEqual(metrics["output_throughput"], 21.0)
        self.assertTrue(metrics["sgl_eval_metrics_path"].endswith("metrics.json"))

    def test_metrics_json_artifact_records_the_run(self):
        metrics = self._run(self._passing_spec(), max_tokens=4096)
        payload = json.loads(Path(metrics["sgl_eval_metrics_path"]).read_text())

        self.assertEqual(payload["name"], "gsm8k")
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["generation"]["max_tokens"], 4096)
        self.assertEqual(payload["latency_seconds"], 2.0)
        self.assertEqual(payload["output_throughput_tps"], 21.0)

    def test_generation_overrides_reach_the_eval(self):
        spec = self._passing_spec()
        self._run(
            spec,
            max_tokens=2048,
            temperature=0.6,
            top_p=0.9,
            top_k=20,
            seed=42,
            sgl_eval_thinking=True,
        )

        gen = spec.calls[0]["gen"]
        self.assertEqual(gen.max_tokens, 2048)
        self.assertEqual(gen.temperature, 0.6)
        self.assertEqual(gen.top_p, 0.9)
        self.assertEqual(gen.seed, 42)
        # top_k is not an OpenAI field, so it rides along in extra_body.
        self.assertEqual(gen.extra_body["top_k"], 20)
        # Reasoning models need thinking on, or the answer stays in the
        # reasoning channel and nothing parses.
        self.assertTrue(gen.chat_template_kwargs["thinking"])
        self.assertTrue(gen.chat_template_kwargs["enable_thinking"])

    def test_token_budget_defaults_to_2048(self):
        # The eval's own default is None, which lets the server pick; an
        # unbounded budget lets a long-reasoning model stall the run.
        spec = self._passing_spec()
        self._run(spec)
        self.assertEqual(spec.calls[0]["gen"].max_tokens, 2048)

    def test_example_and_thread_counts_are_forwarded(self):
        spec = self._passing_spec()
        self._run(spec, num_examples=200, num_threads=128, repeat=3)

        call = spec.calls[0]
        self.assertEqual(call["num_examples"], 200)
        self.assertEqual(call["num_threads"], 128)
        self.assertEqual(call["n_repeats"], 3)

    def test_thread_count_falls_back_to_the_eval_default(self):
        spec = self._passing_spec()
        self._run(spec, num_threads=None)
        self.assertEqual(spec.calls[0]["num_threads"], spec.default_num_threads)

    def test_incomplete_run_raises(self):
        spec = self._passing_spec()
        spec.result.partial = True
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            self._run(spec)

    def test_run_with_no_graded_answers_raises(self):
        spec = FakeSpec(
            result=FakeRunResult(
                aggregate={"score": 0.0, "error_rate": 0.0}, per_example=[]
            )
        )
        with self.assertRaisesRegex(RuntimeError, "Empty"):
            self._run(spec)

    def test_request_errors_raise_instead_of_scoring_low(self):
        # A server dropping requests must not read as a model regression.
        spec = self._passing_spec(error_rate=0.2)
        with self.assertRaisesRegex(RuntimeError, "error_rate"):
            self._run(spec)

    def test_sampler_and_predictions_writer_are_torn_down(self):
        # An unclosed writer leaves the prediction artifact truncated, and that
        # artifact is the only record of what the model actually answered.
        spec = self._passing_spec()
        self._run(spec)
        self.assertTrue(spec.calls[0]["sampler"].aborted)
        self.assertTrue(spec.calls[0]["predictions_writer"].closed)

    def test_main_thread_call_grades_in_place(self):
        spec = self._passing_spec()
        metrics = self._run(spec)

        self.assertEqual(len(spec.calls), 1)
        self.assertEqual(metrics["score"], 0.5)

    def test_worker_thread_call_is_handed_off_for_grading(self):
        # test_qwen35_fp8_ar_fusion_mi35x.py evaluates two servers from a
        # ThreadPoolExecutor. math_verify arms SIGALRM, which only a main thread
        # can do, so a worker-thread call has to be handed to another process
        # instead of graded where it was called. The stand-in below captures the
        # hand-off; that the real spawn supplies a usable main thread is covered
        # by live runs.
        spec = self._passing_spec()
        handed_off = {"score": "graded elsewhere"}

        class StandInProcessPool:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def submit(self, fn, *fn_args):
                return SimpleNamespace(result=lambda: handed_off)

        results = {}

        def worker():
            with patch("sglang.test.sgl_eval.ProcessPoolExecutor", StandInProcessPool):
                results["metrics"] = self._run(spec)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(60)

        self.assertEqual(
            spec.calls,
            [],
            "graded on the calling worker thread, where math_verify cannot arm SIGALRM",
        )
        self.assertIs(results.get("metrics"), handed_off)


if __name__ == "__main__":
    unittest.main()
