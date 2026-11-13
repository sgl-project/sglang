"""CPU contracts for the opt-in cuTile service benchmark driver."""

import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase

_DRIVER = Path(__file__).resolve().parents[4] / (
    "benchmark/kernels/flashinfer_cutile/bench_cutile_service.py"
)
_SPEC = importlib.util.spec_from_file_location("cutile_service_benchmark", _DRIVER)
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


def _response(tokens=64):
    output_ids = list(range(7, 7 + tokens))
    return {
        "text": "answer",
        "output_ids": output_ids,
        "meta_info": {
            "finish_reason": {"type": "length", "length": tokens},
            "completion_tokens": tokens,
            "output_token_logprobs": [
                [-0.2, token_id, None] for token_id in output_ids
            ],
        },
    }


def _arguments(tokenizer="/already/downloaded", output_dir="/results/new"):
    return [
        "--base-url",
        "http://localhost:30000",
        "--model",
        "test",
        "--tokenizer",
        tokenizer,
        "--output-dir",
        output_dir,
        "--implementation-label",
        "candidate",
        "--server-revision",
        "a" * 40,
        "--checkpoint-revision",
        "b" * 40,
        "--flashinfer-variant",
        "stock-0.7.3",
    ]


class _Server:
    """Track the resource state, rather than mocking successful method calls."""

    def __init__(self, fail_released_info=False):
        self.released = False
        self.release_count = 0
        self.generate_count = 0
        self.flush_count = 0
        self.fail_released_info = fail_released_info

    def call(self, path, payload=None):
        if path.startswith("/flush_cache"):
            if self.released:
                raise AssertionError("Attempted a flush while resources were released")
            self.flush_count += 1
        elif path == "/generate":
            if self.released:
                raise AssertionError("Attempted generation before resume")
            self.generate_count += 1
            return _response()
        elif path == "/release_memory_occupation":
            if self.released:
                raise AssertionError("Double release")
            self.released = True
            self.release_count += 1
        elif path == "/resume_memory_occupation":
            if not self.released:
                raise AssertionError("Resume without release")
            self.released = False
        elif path == "/server_info":
            if self.fail_released_info and self.released:
                raise RuntimeError("Diagnostic request failed")
            return {}


class TestCuTileServiceBenchmark(CustomTestCase):
    def test_rejects_missing_backup_before_lifecycle(self):
        args = SimpleNamespace(mode="lifecycle", expected_backend="flashinfer_cutile")
        info = {"moe_runner_backend": "flashinfer_cutile", "enable_memory_saver": True}
        with self.assertRaisesRegex(ValueError, "enable_weights_cpu_backup"):
            driver._validate_server(info, args)
        info["enable_weights_cpu_backup"] = True
        info["dp_size"] = 1
        driver._validate_server(info, args)
        info["dp_size"] = 2
        with self.assertRaisesRegex(ValueError, "dp_size=1"):
            driver._validate_server(info, args)

    def test_rejects_wrong_backend(self):
        args = SimpleNamespace(mode="serving", expected_backend="flashinfer_cutile")
        with self.assertRaisesRegex(ValueError, "backend"):
            driver._validate_server({"moe_runner_backend": "marlin"}, args)

    def test_lifecycle_regenerates_after_resume_with_cold_cache(self):
        server = _Server()
        with tempfile.TemporaryDirectory() as directory:
            driver._lifecycle(server, SimpleNamespace(logprob_atol=0), Path(directory))
            self.assertTrue((Path(directory) / "lifecycle-result.json").is_file())
        self.assertFalse(server.released)
        self.assertEqual(server.release_count, 1)
        self.assertEqual(server.generate_count, 3 * len(driver.DEFAULT_PROMPTS))
        self.assertEqual(server.flush_count, server.generate_count + 1)

    def test_failed_diagnostics_still_restore_resources(self):
        server = _Server(fail_released_info=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "Diagnostic"):
                driver._lifecycle(
                    server, SimpleNamespace(logprob_atol=0), Path(directory)
                )
            self.assertFalse((Path(directory) / "lifecycle-result.json").exists())
        self.assertFalse(server.released)

    def test_lifecycle_requires_complete_generation_before_release(self):
        for problem, message in (
            ("abort", "finish by length"),
            ("missing_finish_reason", "finish by length"),
            ("wrong_finish_reason", "finish by length"),
            ("short_output", "expected 64"),
            ("completion_count", "completion_tokens"),
            ("logprob_count", "one logprob per token"),
        ):
            response = _response(63 if problem == "short_output" else 64)
            meta = response["meta_info"]
            if problem == "abort":
                # The native endpoint can return this abort as HTTP 200.
                meta["finish_reason"] = {"type": "abort", "message": "cancelled"}
            elif problem == "missing_finish_reason":
                del meta["finish_reason"]
            elif problem == "wrong_finish_reason":
                meta["finish_reason"] = {"type": "stop"}
            elif problem == "completion_count":
                meta["completion_tokens"] = 63
            elif problem == "logprob_count":
                meta["output_token_logprobs"].pop()

            class InvalidGenerationServer(_Server):
                def call(self, path, payload=None):
                    result = super().call(path, payload)
                    return response if path == "/generate" else result

            server = InvalidGenerationServer()
            with (
                self.subTest(problem=problem),
                tempfile.TemporaryDirectory() as directory,
            ):
                with self.assertRaisesRegex(ValueError, message):
                    driver._lifecycle(
                        server, SimpleNamespace(logprob_atol=0), Path(directory)
                    )
                self.assertEqual(server.release_count, 0)
                self.assertFalse((Path(directory) / "lifecycle-result.json").exists())

    def test_lifecycle_checks_complete_generation_after_resume(self):
        class ShortRestoredServer(_Server):
            def call(self, path, payload=None):
                result = super().call(path, payload)
                if path == "/generate" and self.release_count:
                    return _response(63)
                return result

        server = ShortRestoredServer()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "expected 64"):
                driver._lifecycle(
                    server, SimpleNamespace(logprob_atol=0), Path(directory)
                )
            self.assertFalse(server.released)
            self.assertEqual(server.release_count, 1)
            self.assertFalse((Path(directory) / "lifecycle-result.json").exists())

    def test_confirmed_release_rejection_does_not_resume(self):
        for status, body in (
            (401, "Unauthorized"),
            (
                400,
                "Memory was not released because requests are still running or queued.",
            ),
        ):

            class RejectedServer(_Server):
                def call(self, path, payload=None):
                    if path == "/release_memory_occupation":
                        raise driver.ServerResponseError(status, body)
                    return super().call(path, payload)

            server = RejectedServer()
            with (
                self.subTest(status=status),
                tempfile.TemporaryDirectory() as directory,
            ):
                with self.assertRaises(driver.ServerResponseError):
                    driver._lifecycle(
                        server,
                        SimpleNamespace(logprob_atol=0),
                        Path(directory),
                        server_info={"dp_size": 1},
                    )
                cycle = json.loads(
                    (Path(directory) / "lifecycle-resource-cycle.json").read_text()
                )
                self.assertEqual(cycle["release"], "rejected")
                self.assertEqual(cycle["resume"], "not_attempted")
                self.assertEqual(server.release_count, 0)

    def test_partial_http_failure_requires_manual_recovery(self):
        class PartialServer(_Server):
            def call(self, path, payload=None):
                result = super().call(path, payload)
                if path == "/release_memory_occupation":
                    raise driver.ServerResponseError(400, "weight pause failed")
                return result

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(driver.ServerResponseError):
                driver._lifecycle(
                    PartialServer(),
                    SimpleNamespace(logprob_atol=0),
                    Path(directory),
                    server_info={"dp_size": 1},
                )
            cycle = json.loads(
                (Path(directory) / "lifecycle-resource-cycle.json").read_text()
            )
            self.assertEqual(cycle["release"], "failed_state_unknown")
            self.assertEqual(cycle["resume"], "not_attempted")
            self.assertIn("recovery", cycle)

    def test_model_identity_and_recorded_case_backend_are_checked(self):
        args = SimpleNamespace(
            model="expected", mode="serving", expected_backend="flashinfer_cutile"
        )
        info = {
            "model_path": "/models/expected",
            "served_model_name": "expected",
            "weight_version": "v1",
        }
        self.assertEqual(
            driver._validate_model(info, args), ("/models/expected", "expected", "v1")
        )
        with self.assertRaisesRegex(ValueError, "model identity"):
            driver._validate_model(
                {"model_path": "wrong", "served_model_name": "wrong"}, args
            )
        with self.assertRaisesRegex(ValueError, "backend"):
            driver._validate_result(
                {"server_info": {"moe_runner_backend": "marlin"}},
                {"input_len": 128, "output_len": 64},
                2,
                args,
            )

    def test_ambiguous_release_still_attempts_resume(self):
        class TimeoutServer(_Server):
            def call(self, path, payload=None):
                result = super().call(path, payload)
                if path == "/release_memory_occupation":
                    raise TimeoutError("release response timed out")
                return result

        server = TimeoutServer()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TimeoutError, "release response"):
                driver._lifecycle(
                    server, SimpleNamespace(logprob_atol=0), Path(directory)
                )
            cycle = json.loads(
                (Path(directory) / "lifecycle-resource-cycle.json").read_text()
            )
            self.assertEqual(cycle["resume"], "confirmed")
        self.assertFalse(server.released)

    def test_dual_failure_keeps_both_errors_and_recovery_request(self):
        class FailedServer(_Server):
            def call(self, path, payload=None):
                if path == "/resume_memory_occupation":
                    raise TimeoutError("resume response timed out")
                result = super().call(path, payload)
                if path == "/release_memory_occupation":
                    raise TimeoutError("release response timed out")
                return result

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "resume failed"):
                driver._lifecycle(
                    FailedServer(), SimpleNamespace(logprob_atol=0), Path(directory)
                )
            cycle = json.loads(
                (Path(directory) / "lifecycle-resource-cycle.json").read_text()
            )
            self.assertIn("release response", cycle["release_or_diagnostic_error"])
            self.assertIn("resume response", cycle["resume_error"])
            self.assertEqual(
                cycle["recovery"]["body"], {"tags": ["weights", "kv_cache"]}
            )

    def test_http_flush_requires_positive_confirmation(self):
        server = driver.Server("http://localhost:30000", 1)
        for body in (b"Flush cache failed.", b'{"success": false}', b"null"):
            with self.subTest(body=body):
                with patch.object(
                    driver.urllib.request, "urlopen", return_value=io.BytesIO(body)
                ):
                    with self.assertRaises(RuntimeError):
                        server.call("/flush_cache?timeout=60", {})
        with patch.object(
            driver.urllib.request,
            "urlopen",
            return_value=io.BytesIO(b"Cache flushed.\n"),
        ):
            self.assertEqual(
                server.call("/flush_cache?timeout=60", {}), "Cache flushed.\n"
            )

    def test_token_or_logprob_changes_fail(self):
        before = _response()
        after = _response()
        after["output_ids"][0] = 9
        after["meta_info"]["output_token_logprobs"][0][1] = 9
        with self.assertRaisesRegex(AssertionError, "token IDs"):
            driver._compare_generations([before], [after], 0)
        after = _response()
        after["meta_info"]["output_token_logprobs"][0][0] = -0.21
        with self.assertRaisesRegex(AssertionError, "logprob error"):
            driver._compare_generations([before], [after], 0)
        driver._compare_generations([before], [after], 0.02)

    def test_nonfinite_logprobs_are_rejected(self):
        response = _response()
        response["meta_info"]["output_token_logprobs"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            driver._signature(response)

    def test_fixed_workload_and_full_success_are_required(self):
        case = {"input_len": 128, "output_len": 64}
        result = {key: 1 for key in driver.METRICS}
        result.update(
            completed=2, input_lens=[128, 128], output_lens=[64, 64], errors=["", ""]
        )
        driver._validate_result(result, case, 2)
        result["completed"] = 1
        with self.assertRaisesRegex(AssertionError, "failed"):
            driver._validate_result(result, case, 2)
        result["completed"] = 2
        result["output_lens"] = [64, 32]
        with self.assertRaisesRegex(AssertionError, "output_lens"):
            driver._validate_result(result, case, 2)

    def test_pilot_is_validated_but_excluded_from_summary(self):
        info = {
            "model_path": "/models/expected",
            "served_model_name": "expected",
            "weight_version": "v1",
        }

        class BenchmarkServer:
            def call(self, path):
                if path == "/model_info":
                    return info
                return {"moe_runner_backend": "flashinfer_cutile"}

        args = SimpleNamespace(
            model="expected",
            mode="serving",
            expected_backend="flashinfer_cutile",
            num_prompts=2,
            case_timeout=12,
        )
        for pilot_fails in (False, True):
            with (
                self.subTest(pilot_fails=pilot_fails),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                case = {
                    "name": "case",
                    "input_len": 128,
                    "output_len": 64,
                    "concurrency": 1,
                    "argv": ["benchmark", str(root / "case.jsonl")],
                    "pilot_argv": ["benchmark", str(root / "case-pilot.jsonl")],
                }
                runs = []

                def run(argv, **kwargs):
                    runs.append(argv[-1])
                    self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
                    self.assertEqual(kwargs["timeout"], args.case_timeout)
                    pilot = argv[-1].endswith("-pilot.jsonl")
                    result = {key: 1 for key in driver.METRICS}
                    result.update(
                        completed=1 if pilot and pilot_fails else 2,
                        input_lens=[128, 128],
                        output_lens=[64, 64],
                        errors=["", ""],
                        server_info={"moe_runner_backend": "flashinfer_cutile"},
                        output_throughput=111 if pilot else 222,
                    )
                    Path(argv[-1]).write_text(json.dumps(result) + "\n")

                with patch.object(driver.subprocess, "run", side_effect=run):
                    if pilot_fails:
                        with self.assertRaisesRegex(AssertionError, "requests failed"):
                            driver._serving(args, root, [case], BenchmarkServer(), info)
                        self.assertEqual(len(runs), 1)
                        self.assertFalse((root / "serving-summary.csv").exists())
                    else:
                        driver._serving(args, root, [case], BenchmarkServer(), info)
                        self.assertEqual(len(runs), 2)
                        with (root / "serving-summary.csv").open() as handle:
                            rows = list(csv.DictReader(handle))
                        self.assertEqual(len(rows), 1)
                        self.assertEqual(rows[0]["output_throughput"], "222")

    def test_case_timeout_preserves_stage_and_stops_matrix(self):
        info = {
            "model_path": "/models/test",
            "served_model_name": "test",
            "weight_version": "v1",
        }

        class BenchmarkServer:
            def call(self, path):
                if path == "/model_info":
                    return info
                if path == "/server_info":
                    return {"moe_runner_backend": "flashinfer_cutile"}
                if path != "/health":
                    raise AssertionError(f"Unexpected server operation: {path}")

        for timeout_stage in ("pilot", "measured"):
            with (
                self.subTest(stage=timeout_stage),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                output = root / "results"
                argv = _arguments(str(root), str(output)) + [
                    "--input-lens",
                    "128",
                    "256",
                    "--output-lens",
                    "64",
                    "--concurrencies",
                    "1",
                    "--num-prompts",
                    "2",
                    "--case-timeout",
                    "12",
                ]
                calls = []

                def run(command, **kwargs):
                    calls.append(command)
                    self.assertEqual(kwargs["timeout"], 12)
                    stage = (
                        "pilot" if command[-1].endswith("-pilot.jsonl") else "measured"
                    )
                    if stage == timeout_stage:
                        kwargs["stdout"].write("Diagnostic before client timeout\n")
                        raise driver.subprocess.TimeoutExpired(
                            command, kwargs["timeout"]
                        )
                    result = {key: 1 for key in driver.METRICS}
                    result.update(
                        completed=2,
                        input_lens=[128, 128],
                        output_lens=[64, 64],
                        errors=["", ""],
                        server_info={"moe_runner_backend": "flashinfer_cutile"},
                    )
                    Path(command[-1]).write_text(json.dumps(result) + "\n")

                with (
                    patch.object(driver.sys, "argv", ["benchmark"] + argv),
                    patch.object(driver, "_metadata", return_value={}),
                    patch.object(driver, "Server", return_value=BenchmarkServer()),
                    patch.object(driver.subprocess, "run", side_effect=run),
                ):
                    with self.assertRaisesRegex(TimeoutError, timeout_stage) as caught:
                        driver.main()
                self.assertIsInstance(
                    caught.exception.__cause__, driver.subprocess.TimeoutExpired
                )
                self.assertEqual(len(calls), 1 if timeout_stage == "pilot" else 2)
                status = json.loads((output / "status.json").read_text())
                self.assertEqual(status["status"], "failed")
                self.assertIn(timeout_stage, status["error"])
                self.assertIn("in128-out64-c1", status["error"])
                timeout_files = list(output.glob("*-timeout.json"))
                self.assertEqual(len(timeout_files), 1)
                diagnostic = json.loads(timeout_files[0].read_text())
                self.assertEqual(diagnostic["case"], "in128-out64-c1")
                self.assertEqual(diagnostic["stage"], timeout_stage)
                self.assertEqual(diagnostic["timeout_seconds"], 12)
                self.assertIn("TimeoutExpired", diagnostic["error"])
                self.assertIn(
                    "Diagnostic before client timeout",
                    Path(diagnostic["log_path"]).read_text(),
                )
                self.assertFalse((output / "serving-summary.csv").exists())

    def test_case_timeout_must_be_finite_and_positive(self):
        for value in ("0", "-1", "nan", "inf"):
            with (
                self.subTest(value=value),
                patch.object(
                    driver.sys,
                    "argv",
                    ["benchmark"]
                    + _arguments()
                    + ["--dry-run", "--case-timeout", value],
                ),
                patch.object(driver.sys, "stderr", new_callable=io.StringIO) as stderr,
            ):
                with self.assertRaises(SystemExit) as caught:
                    driver.main()
                self.assertEqual(caught.exception.code, 2)
                self.assertIn(
                    "--case-timeout must be finite and positive", stderr.getvalue()
                )

    def test_commands_use_current_module_and_no_download_dataset(self):
        args = driver._parser().parse_args(_arguments())
        self.assertEqual(args.case_timeout, 1800)
        commands = driver._commands(args, Path(args.output_dir))
        self.assertEqual(len(commands), 18)
        argv = commands[0]["argv"]
        self.assertEqual(argv[2], "sglang.benchmark.serving")
        self.assertEqual(argv[argv.index("--dataset-name") + 1], "random-ids")
        self.assertEqual(argv[argv.index("--random-range-ratio") + 1], "1")
        self.assertIn("--tokenize-prompt", argv)
        self.assertIn("--output-details", argv)
        self.assertIn("--flush-cache", argv)
        last = commands[-1]
        warmup = int(last["argv"][last["argv"].index("--warmup-requests") + 1])
        self.assertGreaterEqual(warmup, last["concurrency"])
        self.assertNotEqual(last["argv"][-1], last["pilot_argv"][-1])
        self.assertTrue(last["pilot_argv"][-1].endswith("-pilot.jsonl"))
        self.assertEqual(last["argv"][:-1], last["pilot_argv"][:-1])


if __name__ == "__main__":
    unittest.main()
