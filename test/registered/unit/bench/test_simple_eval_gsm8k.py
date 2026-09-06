import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.run_eval import _run_sgl_eval, run_eval
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="stage-a-test-cpu-intel")


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

    def _capture_cmd(self, eval_name="gsm8k", **overrides):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            _write_fake_metrics(
                out_dir,
                eval_name,
                {
                    "model": "test-model",
                    "latency_seconds": 1.0,
                    "output_throughput_tps": 1.0,
                    "aggregate": {"score": 0.5},
                },
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as td:
            args = self._args(td, **overrides)
            with patch("sglang.test.run_eval.subprocess.run", side_effect=fake_run):
                _run_sgl_eval(eval_name, args)
        return captured["cmd"]

    def test_omits_sampling_flags_when_unset(self):
        """Unset top_p / seed / repeat must not reach the CLI -- sgl-eval's own
        defaults differ from a forced value (seed unset != seed 0)."""
        cmd = self._capture_cmd()
        for flag in ("--top-p", "--seed", "--n-repeats"):
            self.assertNotIn(flag, cmd)

    def test_forwards_sampling_flags_when_set(self):
        cmd = self._capture_cmd(top_p=0.95, seed=0, repeat=1)
        for flag, value in (("--top-p", "0.95"), ("--seed", "0"), ("--n-repeats", "1")):
            self.assertIn(flag, cmd)
            self.assertEqual(cmd[cmd.index(flag) + 1], value)

    def test_model_preset_owns_model_and_sampling_defaults(self):
        cmd = self._capture_cmd(
            eval_name="mmmu_pro",
            model=None,
            num_examples=300,
            num_threads=None,
            temperature=None,
            load_preset_from_model_id="moonshotai/Kimi-K3",
        )

        self.assertEqual(cmd[:3], ["sgl-eval", "run", "mmmu_pro"])
        self.assertIn("--load-preset-from-model-id", cmd)
        self.assertEqual(
            cmd[cmd.index("--load-preset-from-model-id") + 1],
            "moonshotai/Kimi-K3",
        )
        self.assertEqual(cmd[cmd.index("--num-examples") + 1], "300")
        for flag in (
            "--model",
            "--num-threads",
            "--temperature",
            "--top-p",
            "--max-tokens",
            "--thinking",
        ):
            self.assertNotIn(flag, cmd)

    def test_non_preset_cli_keeps_legacy_top_p_default(self):
        cmd = self._capture_cmd(top_p=None, _sgl_eval_from_cli=True)

        self.assertIn("--top-p", cmd)
        self.assertEqual(cmd[cmd.index("--top-p") + 1], "1.0")

    @patch("sglang.test.run_eval._run_sgl_eval", return_value={"score": 0.8})
    def test_run_eval_dispatches_hyphenated_mmmu_pro_name(self, mock_sgl_eval):
        args = SimpleNamespace(
            base_url="http://127.0.0.1:30000",
            eval_name="mmmu-pro",
        )

        try:
            result = run_eval(args)
        except ValueError as exc:
            self.fail(f"mmmu-pro must dispatch to sgl-eval: {exc}")
        self.assertEqual(result, {"score": 0.8})
        mock_sgl_eval.assert_called_once_with("mmmu_pro", args)

    def test_thinking_auto_detected_from_model_name(self):
        self.assertIn(
            "--thinking", self._capture_cmd(model="Qwen/Qwen3.5-397B-A17B-FP8")
        )

    def test_explicit_thinking_false_suppresses_auto_detect(self):
        """A caller matching a harness that sent no chat_template_kwargs has to be
        able to turn the model-name heuristic off."""
        cmd = self._capture_cmd(
            model="Qwen/Qwen3.5-397B-A17B-FP8", sgl_eval_thinking=False
        )
        self.assertNotIn("--thinking", cmd)

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
