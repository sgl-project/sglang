"""Smoke tests for the PD-disaggregation benchmark harness.

Wires mocked `requests` responses through the bench scripts to verify: CLI
parsing, `capture` JSON schema, `compare` exit codes, and argument validators
rejecting malformed input (input lengths, rtol).

Runs on CPU only and is meant for CI gating on the bench tooling itself. For
real benchmarks run the scripts manually per ``benchmark/disaggregation/README.md``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmark" / "disaggregation"


def _load_module(name: str):
    """Import a bench script by file path (it lives outside the `sglang` package)."""
    path = BENCH_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestBenchPdTtftSmoke(unittest.TestCase):
    """Validate bench_pd_ttft.py against mocked /generate, /health, /get_model_info."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module("bench_pd_ttft")

    def _mock_post(self, *args, **kwargs):
        """Stand-in for `requests.post(.../generate)`: a deterministic payload with
        server-reported + client-derivable latency fields so percentile/throughput math runs."""
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = lambda: None
        resp.json = lambda: {
            "text": "hello",
            "meta_info": {
                "first_token_latency": 0.012,  # seconds → 12ms
                "e2e_latency": 0.020,
                "prompt_tokens": 16,
                "completion_tokens": 1,
            },
        }
        return resp

    def _mock_get(self, url, *args, **kwargs):
        """Stand-in for `requests.get`: responds for /health and /get_model_info, 404 otherwise."""
        resp = MagicMock()
        if url.endswith("/health"):
            resp.status_code = 200
            resp.raise_for_status = lambda: None
            return resp
        if url.endswith("/get_model_info"):
            resp.status_code = 200
            resp.raise_for_status = lambda: None
            resp.json = lambda: {"vocab_size": 32000}
            return resp
        resp.status_code = 404
        resp.raise_for_status = lambda: None
        return resp

    def test_capture_writes_documented_json_schema(self):
        """`capture` must produce a JSON file with the documented metadata + results structure."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "capture.json"
            args = argparse.Namespace(
                cmd="capture",
                lb_url="http://mock:30011",
                output=str(output_path),
                input_lengths="16,32",
                profile="kpi",
                concurrency=1,
                num_requests=2,
                warmup=1,
                max_new_tokens=1,
                vocab_size=None,  # → triggers /get_model_info probe
                seed=42,
                timeout=5.0,
                kv_dtype="bf16",
                label="smoke",
                branch="test",
                commit="deadbeef",
                func=self.mod._capture,
            )
            with patch.object(self.mod.requests, "post", side_effect=self._mock_post), \
                 patch.object(self.mod.requests, "get", side_effect=self._mock_get):
                rc = self.mod._capture(args)
            self.assertEqual(rc, 0, "capture must succeed against a healthy mock")
            self.assertTrue(output_path.exists())
            with output_path.open() as f:
                payload = json.load(f)
            self.assertIn("metadata", payload)
            self.assertIn("results", payload)
            meta = payload["metadata"]
            for k in (
                "lb_url", "label", "captured_at", "input_lengths",
                "vocab_size", "vocab_size_source",
            ):
                self.assertIn(k, meta, f"metadata missing required key {k!r}")
            self.assertEqual(meta["vocab_size"], 32000)
            self.assertEqual(meta["vocab_size_source"], "probed")
            self.assertEqual(len(payload["results"]), 2)  # 2 input lengths
            for r in payload["results"]:
                self.assertIn("input_length", r)
                self.assertIn("client_e2e_ms", r)
                self.assertIn("server_first_token_latency_ms", r)

    def test_capture_rejects_malformed_input_lengths(self):
        args = argparse.Namespace(
            cmd="capture",
            lb_url="http://mock:30011",
            output="/tmp/should-not-write.json",
            input_lengths="16,not-an-int,32",
            profile="kpi",
            concurrency=None,
            num_requests=None,
            warmup=1,
            max_new_tokens=1,
            vocab_size=32000,
            seed=42,
            timeout=5.0,
            kv_dtype=None,
            label=None,
            branch=None,
            commit=None,
            func=self.mod._capture,
        )
        rc = self.mod._capture(args)
        self.assertEqual(
            rc, 2, "malformed --input-lengths must exit with code 2",
        )

    def test_capture_aborts_when_vocab_unavailable(self):
        """No --vocab-size + a failing probe must abort loudly rather than use a wrong vocab range."""
        def _failing_get(url, *args, **kwargs):
            resp = MagicMock()
            resp.status_code = 500
            resp.raise_for_status = MagicMock(
                side_effect=self.mod.requests.RequestException("503")
            )
            return resp

        args = argparse.Namespace(
            cmd="capture",
            lb_url="http://mock:30011",
            output="/tmp/should-not-write.json",
            input_lengths="16",
            profile="kpi",
            concurrency=None,
            num_requests=None,
            warmup=1,
            max_new_tokens=1,
            vocab_size=None,  # neither CLI nor probe → abort
            seed=42,
            timeout=5.0,
            kv_dtype=None,
            label=None,
            branch=None,
            commit=None,
            func=self.mod._capture,
        )
        with patch.object(self.mod.requests, "get", side_effect=_failing_get):
            rc = self.mod._capture(args)
        self.assertEqual(rc, 2)

    def test_capture_normalizes_trailing_slash_urls(self):
        """A trailing-slash --lb-url must not hit //generate or //health (some proxies treat those as distinct routes)."""
        get_urls = []
        post_urls = []

        def _record_get(url, *args, **kwargs):
            get_urls.append(url)
            return self._mock_get(url, *args, **kwargs)

        def _record_post(url, *args, **kwargs):
            post_urls.append(url)
            return self._mock_post(url, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "capture.json"
            args = argparse.Namespace(
                cmd="capture",
                lb_url="http://mock:30011/",
                output=str(output_path),
                input_lengths="16",
                profile="kpi",
                concurrency=1,
                num_requests=1,
                warmup=0,
                max_new_tokens=1,
                vocab_size=None,
                seed=42,
                timeout=5.0,
                kv_dtype="bf16",
                label="slash",
                branch="test",
                commit="deadbeef",
                func=self.mod._capture,
            )
            with patch.object(self.mod.requests, "post", side_effect=_record_post), \
                 patch.object(self.mod.requests, "get", side_effect=_record_get):
                rc = self.mod._capture(args)

        self.assertEqual(rc, 0)
        self.assertIn("http://mock:30011/get_model_info", get_urls)
        self.assertIn("http://mock:30011/health", get_urls)
        self.assertEqual(post_urls, ["http://mock:30011/generate"])

    def test_throughput_counts_successful_requests_only(self):
        """Failed measured requests must bump num_errors, not inflate throughput_req_per_sec."""
        calls = {"n": 0}

        def _flaky_post(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise self.mod.requests.RequestException("simulated")
            return self._mock_post(*args, **kwargs)

        with patch.object(self.mod.requests, "post", side_effect=_flaky_post):
            result = self.mod._run_one_length(
                lb_url="http://mock:30011",
                length=16,
                num_requests=2,
                concurrency=1,
                warmup=0,
                max_new_tokens=1,
                vocab_size=32000,
                base_seed=42,
                timeout=5.0,
            )

        self.assertEqual(result["num_errors"], 1)
        self.assertEqual(len(result["samples"]), 1)
        self.assertAlmostEqual(
            result["throughput_req_per_sec"] * result["wall_elapsed_s"],
            1.0,
            places=6,
        )

    def test_compare_runs_against_smoke_captures(self):
        """End-to-end: capture two JSONs, compare them; compare must exit 0 and write the table."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = tmp_path / "baseline.json"
            experimental = tmp_path / "exp.json"

            def _capture_to(path: Path):
                args = argparse.Namespace(
                    cmd="capture",
                    lb_url="http://mock:30011",
                    output=str(path),
                    input_lengths="16",
                    profile="kpi",
                    concurrency=1,
                    num_requests=2,
                    warmup=0,
                    max_new_tokens=1,
                    vocab_size=32000,
                    seed=42,
                    timeout=5.0,
                    kv_dtype="bf16",
                    label=path.stem,
                    branch="test",
                    commit="deadbeef",
                    func=self.mod._capture,
                )
                with patch.object(self.mod.requests, "post",
                                  side_effect=self._mock_post), \
                     patch.object(self.mod.requests, "get",
                                  side_effect=self._mock_get):
                    return self.mod._capture(args)

            self.assertEqual(_capture_to(baseline), 0)
            self.assertEqual(_capture_to(experimental), 0)

            args = argparse.Namespace(
                cmd="compare",
                baseline=str(baseline),
                experimental=str(experimental),
                func=self.mod._compare,
            )
            rc = self.mod._compare(args)
            self.assertEqual(rc, 0)


class TestBenchPdQualitySmoke(unittest.TestCase):
    """Validate bench_pd_quality.py compare logic + config validator.

    Capture mode needs lm-eval-harness (not a runtime dep here); compare and
    the YAML schema validator are pure-Python and can be smoke-tested.
    """

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module("bench_pd_quality")

    def _write_capture(self, path: Path, label: str, metrics: dict):
        path.write_text(json.dumps({
            "metadata": {
                "lb_url": "http://mock",
                "label": label,
                "config_path": "test.yaml",
                "captured_at": "2026-01-01T00:00:00+00:00",
                "captured_finished_at": "2026-01-01T00:01:00+00:00",
                "duration_seconds": 60.0,
            },
            "results": metrics,
        }))

    def test_compare_pass_within_rtol(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = tmp_path / "baseline.json"
            experimental = tmp_path / "exp.json"
            self._write_capture(baseline, "baseline", {
                "gsm8k": {"exact_match,strict-match": 0.500},
            })
            self._write_capture(experimental, "experimental", {
                # 0.51 vs 0.50 → +2% → within default rtol=5%
                "gsm8k": {"exact_match,strict-match": 0.510},
            })
            args = argparse.Namespace(
                baseline=str(baseline),
                experimental=str(experimental),
                rtol=0.05,
                func=self.mod._compare,
            )
            self.assertEqual(self.mod._compare(args), 0)

    def test_compare_fail_outside_rtol(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = tmp_path / "baseline.json"
            experimental = tmp_path / "exp.json"
            self._write_capture(baseline, "baseline", {
                "gsm8k": {"exact_match,strict-match": 0.500},
            })
            self._write_capture(experimental, "experimental", {
                # 0.450 vs 0.500 → -10% → outside default rtol=5%
                "gsm8k": {"exact_match,strict-match": 0.450},
            })
            args = argparse.Namespace(
                baseline=str(baseline),
                experimental=str(experimental),
                rtol=0.05,
                func=self.mod._compare,
            )
            self.assertEqual(self.mod._compare(args), 1)

    def test_compare_rejects_out_of_range_rtol(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for bad_rtol in (-0.1, 0.0, 1.5, 100.0):
                with self.subTest(rtol=bad_rtol):
                    baseline = tmp_path / "baseline.json"
                    self._write_capture(baseline, "b", {
                        "gsm8k": {"exact_match,strict-match": 0.5}
                    })
                    args = argparse.Namespace(
                        baseline=str(baseline),
                        experimental=str(baseline),
                        rtol=bad_rtol,
                        func=self.mod._compare,
                    )
                    self.assertEqual(
                        self.mod._compare(args), 2,
                        f"rtol={bad_rtol} must be rejected (exit 2)",
                    )

    def test_config_validator_accepts_well_formed_config(self):
        cfg = {
            "model_name": "test/model",
            "tasks": [
                {"name": "gsm8k", "metrics": [{"name": "exact_match"}]},
            ],
        }
        # Validator returns None on success; raises SystemExit on failure.
        self.mod._validate_config(cfg, Path("test.yaml"))

    def test_config_validator_rejects_malformed_inputs(self):
        bad_configs = [
            # Top-level not a mapping
            ([], "list at top level"),
            # Missing model_name
            ({"tasks": [{"name": "gsm8k"}]}, "missing model_name"),
            # Missing tasks
            ({"model_name": "test"}, "missing tasks"),
            # Empty model_name
            ({"model_name": "", "tasks": [{"name": "gsm8k"}]}, "empty model_name"),
            # Empty tasks
            ({"model_name": "test", "tasks": []}, "empty tasks"),
            # Task without name
            ({"model_name": "test", "tasks": [{}]}, "task missing name"),
            # tasks not a list
            ({"model_name": "test", "tasks": {}}, "tasks not a list"),
        ]
        for cfg, label in bad_configs:
            with self.subTest(label=label):
                with self.assertRaises(
                    SystemExit,
                    msg=f"{label!r}: should have raised",
                ):
                    self.mod._validate_config(cfg, Path("bad.yaml"))


if __name__ == "__main__":
    unittest.main()
