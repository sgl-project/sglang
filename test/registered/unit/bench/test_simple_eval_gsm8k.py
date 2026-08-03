import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.run_eval import _run_sgl_eval
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


def _write_fake_metrics(out_parent: Path, eval_name: str, payload: dict) -> None:
    run_dir = out_parent / f"sgl_eval_{eval_name}_20260101-000000"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(payload))


class TestRunSglEval(CustomTestCase):
    """sgl-eval is a black box, so these mock subprocess.run and assert the shim
    builds the CLI and parses metrics.json's aggregate.score (not top-level)."""

    def _args(self, out_dir: str, **overrides):
        defaults = dict(
            base_url="http://127.0.0.1:30000",
            model="test-model",
            num_examples=7,
            num_threads=8,
            temperature=0.0,
            sgl_eval_out_dir=out_dir,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _fake_run_factory(self, out_dir: Path, eval_name: str, payload: dict):
        def fake_run(cmd, **kwargs):
            _write_fake_metrics(out_dir, eval_name, payload)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        return fake_run

    def test_parses_aggregate_score_and_maps_latency(self):
        payload = {
            "name": "gsm8k",
            "model": "test-model",
            "num_examples": 7,
            "n_repeats": 1,
            "latency_seconds": 12.5,
            "output_throughput_tps": 34.0,
            "aggregate": {"score": 0.75, "no_answer": 0.1},
        }
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            args = self._args(td)
            with patch(
                "sglang.test.run_eval.subprocess.run",
                side_effect=self._fake_run_factory(out_dir, "gsm8k", payload),
            ):
                metrics = _run_sgl_eval("gsm8k", args)

        self.assertAlmostEqual(metrics["score"], 0.75)
        self.assertAlmostEqual(metrics["latency"], 12.5)
        self.assertAlmostEqual(metrics["output_throughput"], 34.0)
        self.assertEqual(metrics["no_answer"], 0.1)
        self.assertTrue(metrics["sgl_eval_metrics_path"].endswith("metrics.json"))

    def test_builds_cli_with_required_flags(self):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            out_dir = Path(captured["cmd"][captured["cmd"].index("--out-dir") + 1])
            _write_fake_metrics(
                out_dir,
                "gsm8k",
                {
                    "name": "gsm8k",
                    "model": "test-model",
                    "latency_seconds": 1.0,
                    "output_throughput_tps": 1.0,
                    "aggregate": {"score": 0.5},
                },
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as td:
            args = self._args(td)
            with patch("sglang.test.run_eval.subprocess.run", side_effect=fake_run):
                _run_sgl_eval("gsm8k", args)

        cmd = captured["cmd"]
        self.assertEqual(cmd[0:3], ["sgl-eval", "run", "gsm8k"])
        self.assertIn("--base-url", cmd)
        self.assertIn("http://127.0.0.1:30000/v1", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("test-model", cmd)
        self.assertIn("--num-threads", cmd)
        self.assertIn("8", cmd)
        self.assertIn("--temperature", cmd)
        self.assertIn("0.0", cmd)
        self.assertIn("--num-examples", cmd)
        self.assertIn("7", cmd)

    def test_omits_num_examples_when_none(self):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            _write_fake_metrics(
                out_dir,
                "gsm8k",
                {
                    "model": "test-model",
                    "latency_seconds": 1.0,
                    "output_throughput_tps": 1.0,
                    "aggregate": {"score": 0.5},
                },
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as td:
            args = self._args(td, num_examples=None)
            with patch("sglang.test.run_eval.subprocess.run", side_effect=fake_run):
                _run_sgl_eval("gsm8k", args)

        self.assertNotIn("--num-examples", captured["cmd"])

    def test_raises_on_nonzero_exit(self):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="boom")

        with tempfile.TemporaryDirectory() as td:
            args = self._args(td)
            with patch("sglang.test.run_eval.subprocess.run", side_effect=fake_run):
                with self.assertRaises(RuntimeError) as cm:
                    _run_sgl_eval("gsm8k", args)
            self.assertIn("exit code 2", str(cm.exception))

    def test_raises_when_metrics_json_missing(self):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as td:
            args = self._args(td)
            with patch("sglang.test.run_eval.subprocess.run", side_effect=fake_run):
                with self.assertRaises(FileNotFoundError):
                    _run_sgl_eval("gsm8k", args)

    def test_raises_when_aggregate_score_missing(self):
        payload = {
            "name": "gsm8k",
            "latency_seconds": 1.0,
            "output_throughput_tps": 1.0,
            "aggregate": {"no_answer": 0.5},  # no score key
        }
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            args = self._args(td)
            with patch(
                "sglang.test.run_eval.subprocess.run",
                side_effect=self._fake_run_factory(out_dir, "gsm8k", payload),
            ):
                with self.assertRaises(KeyError):
                    _run_sgl_eval("gsm8k", args)


if __name__ == "__main__":
    unittest.main()
