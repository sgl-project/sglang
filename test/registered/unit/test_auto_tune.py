"""Unit tests for sglang.auto_tune — no server, no model loading."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.auto_tune import (
    CandidateResult,
    TuneResult,
    WorkloadConfig,
    build_server_command,
    get_auth_headers,
    parse_backends,
    pick_best_backend,
    tune_attention_backends,
    write_result,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestParseBackends(CustomTestCase):
    def test_comma_separated_string(self):
        self.assertEqual(
            parse_backends("triton,flashinfer"),
            ["triton", "flashinfer"],
        )

    def test_strips_whitespace(self):
        self.assertEqual(
            parse_backends(" triton , flashinfer "),
            ["triton", "flashinfer"],
        )

    def test_single_backend(self):
        self.assertEqual(parse_backends("triton"), ["triton"])

    def test_list_input(self):
        self.assertEqual(
            parse_backends(["triton", "flashinfer"]),
            ["triton", "flashinfer"],
        )

    def test_drops_empty_segments(self):
        self.assertEqual(
            parse_backends("triton,,flashinfer,"),
            ["triton", "flashinfer"],
        )

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            parse_backends("")

    def test_only_commas_raises(self):
        with self.assertRaises(ValueError):
            parse_backends(",,,")

    def test_empty_list_raises(self):
        with self.assertRaises(ValueError):
            parse_backends([])

    def test_list_of_empty_strings_raises(self):
        with self.assertRaises(ValueError):
            parse_backends(["", "  "])


class TestPickBestBackend(CustomTestCase):
    def test_higher_is_better_picks_max(self):
        candidates = [
            CandidateResult(
                backend="triton",
                ok=True,
                metrics={"output_throughput": 100.0},
            ),
            CandidateResult(
                backend="flashinfer",
                ok=True,
                metrics={"output_throughput": 150.0},
            ),
        ]
        self.assertEqual(
            pick_best_backend(candidates, primary_metric="output_throughput"),
            "flashinfer",
        )

    def test_lower_is_better_picks_min(self):
        candidates = [
            CandidateResult(
                backend="triton",
                ok=True,
                metrics={"median_tpot_ms": 30.0},
            ),
            CandidateResult(
                backend="flashinfer",
                ok=True,
                metrics={"median_tpot_ms": 20.0},
            ),
        ]
        self.assertEqual(
            pick_best_backend(candidates, primary_metric="median_tpot_ms"),
            "flashinfer",
        )

    def test_all_failed_returns_none(self):
        candidates = [
            CandidateResult(backend="triton", ok=False, error="boom"),
            CandidateResult(backend="flashinfer", ok=False, error="boom"),
        ]
        self.assertIsNone(pick_best_backend(candidates))

    def test_unsupported_metric_raises(self):
        with self.assertRaises(ValueError):
            pick_best_backend([], primary_metric="not_a_real_metric")


class TestBuildServerCommand(CustomTestCase):
    def test_includes_backend_and_extra_args(self):
        cmd = build_server_command(
            model_path="Qwen/Qwen3.5-9B",
            backend="triton",
            tp=1,
            host="127.0.0.1",
            port=30000,
            extra_args=["--disable-cuda-graph"],
        )
        self.assertIn("--attention-backend", cmd)
        self.assertIn("triton", cmd)
        self.assertIn("--disable-cuda-graph", cmd)


class TestGetAuthHeaders(CustomTestCase):
    def test_returns_empty_when_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(get_auth_headers(), {})


class TestWriteResult(CustomTestCase):
    def test_writes_json_to_output_dir(self):
        result = TuneResult(
            model_path="m",
            device_name="unknown",
            tp=1,
            workload=WorkloadConfig(),
            primary_metric="output_throughput",
            candidates=[],
            best_attention_backend="triton",
            recommended_args=["--attention-backend", "triton"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_result(result, tmp)
            self.assertTrue(path.exists())
            self.assertIn("triton", path.read_text(encoding="utf-8"))

    def test_raises_when_output_dir_is_a_file(self):
        result = TuneResult(
            model_path="m",
            device_name="unknown",
            tp=1,
            workload=WorkloadConfig(),
            primary_metric="output_throughput",
            candidates=[],
            best_attention_backend=None,
            recommended_args=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "not_a_dir"
            blocker.write_text("x", encoding="utf-8")
            with self.assertRaises(OSError):
                write_result(result, blocker)


class TestTuneAttentionBackends(CustomTestCase):
    def test_selects_best_backend_from_candidates(self):
        def fake_run_one(**kwargs):
            return CandidateResult(
                backend=kwargs["backend"],
                ok=True,
                metrics={
                    "output_throughput": (
                        10.0 if kwargs["backend"] == "triton" else 20.0
                    )
                },
            )

        with tempfile.TemporaryDirectory() as tmp:
            result = tune_attention_backends(
                model_path="Qwen/Qwen3.5-9B",
                backends=["triton", "flashinfer"],
                output_dir=tmp,
                run_one_fn=fake_run_one,
                device_name="test-gpu",
            )
        self.assertEqual(result.best_attention_backend, "flashinfer")
        self.assertEqual(
            result.recommended_args,
            ["--attention-backend", "flashinfer"],
        )

    def test_returns_no_recommendation_when_all_fail(self):
        def fake_run_one(**kwargs):
            return CandidateResult(
                backend=kwargs["backend"],
                ok=False,
                error="failed filter",
            )

        with tempfile.TemporaryDirectory() as tmp:
            result = tune_attention_backends(
                model_path="Qwen/Qwen3.5-9B",
                backends=["triton", "flashinfer"],
                output_dir=tmp,
                run_one_fn=fake_run_one,
                device_name="test-gpu",
            )
        self.assertIsNone(result.best_attention_backend)
        self.assertEqual(result.recommended_args, [])
        self.assertTrue(all(not c.ok for c in result.candidates))


if __name__ == "__main__":
    unittest.main()
