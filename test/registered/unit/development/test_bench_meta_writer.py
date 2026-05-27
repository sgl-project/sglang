"""Registered regression for ``development/_bench_meta_writer.py``.

Round 23 introduced a sidecar writer that spliced raw
``/get_server_info`` JSON into a Python heredoc as source code.
JSON ``true`` / ``false`` / ``null`` are NOT valid Python identifiers,
so the writer crashed with ``NameError`` after every successful
benchmark run on real hardware. Round 24 extracted the writer into a
standalone Python helper that reads JSON from an env var and calls
``json.loads``. These tests lock that contract by invoking the helper
as a subprocess with realistic, empty, and malformed
``SERVER_ARGS_JSON`` payloads.
"""

from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
WRITER = REPO_ROOT / "development" / "_bench_meta_writer.py"


def _run_writer(env_overrides: dict) -> dict:
    """Invoke the writer with the given env vars and return the parsed JSON.

    Asserts the writer exits 0 and produces JSON; raising the parser's
    error as a useful assertion message if not.
    """
    env = {
        "COMMIT_SHA": "abc123",
        "MODE": "double_sparsity",
        "CONCURRENCY": "32",
        "SEED": "431",
        "NUM_PROMPTS": "320",
        "ISL_TOTAL_TOKENS": "4096",
        "OSL_TOKENS": "512",
        "TIMESTAMP_UTC": "2026-05-27T12:00:00Z",
        "SERVER_ARGS_JSON": "{}",
    }
    env.update(env_overrides)
    # Inherit PATH so python3 still resolves.
    env["PATH"] = os.environ.get("PATH", "")
    proc = subprocess.run(
        ["python3", str(WRITER)],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"writer exited {proc.returncode}; stderr:\n{proc.stderr}",
        )
    return json.loads(proc.stdout)


class TestBenchMetaWriter(unittest.TestCase):

    def test_writer_exists(self):
        self.assertTrue(WRITER.exists(), f"missing writer: {WRITER}")

    def test_realistic_server_args_with_true_false_null(self):
        """Round 23 regression: JSON true/false/null must not crash the writer."""
        server_args = {
            "disable_radix_cache": True,
            "kv_events": None,
            "chunked_prefill_size": 4096,
            "tp_size": 8,
            "enable_double_sparsity": True,
            "nested": {"key": "value", "n": 0},
        }
        out = _run_writer({"SERVER_ARGS_JSON": json.dumps(server_args)})
        self.assertEqual(out["chunked_prefill_size"], 4096)
        self.assertIs(out["server_args"]["disable_radix_cache"], True)
        self.assertIsNone(out["server_args"]["kv_events"])
        self.assertEqual(out["server_args"]["nested"], {"key": "value", "n": 0})
        self.assertIsNone(out["server_args_error"])

    def test_empty_server_args_falls_back_to_unknown(self):
        out = _run_writer({"SERVER_ARGS_JSON": ""})
        self.assertEqual(out["server_args"], {})
        self.assertEqual(out["chunked_prefill_size"], "unknown")
        self.assertIsNotNone(out["server_args_error"])
        self.assertIn("empty", out["server_args_error"])

    def test_malformed_server_args_records_error(self):
        out = _run_writer({"SERVER_ARGS_JSON": "{not json"})
        self.assertEqual(out["server_args"], {})
        self.assertEqual(out["chunked_prefill_size"], "unknown")
        self.assertIsNotNone(out["server_args_error"])
        self.assertIn("parse_error", out["server_args_error"])

    def test_non_object_server_args_records_error(self):
        out = _run_writer({"SERVER_ARGS_JSON": "[1, 2, 3]"})
        self.assertEqual(out["server_args"], {})
        self.assertIn("not_object", out["server_args_error"])

    def test_trial_id_defaults_to_1(self):
        out = _run_writer({})
        self.assertEqual(out["trial_id"], "1")

    def test_trial_id_honors_env(self):
        out = _run_writer({"TRIAL_ID": "3"})
        self.assertEqual(out["trial_id"], "3")

    def test_ac11_reproducibility_fields_present_even_when_null(self):
        """warmup_requests / measurement_window_seconds default to null."""
        out = _run_writer({})
        self.assertIn("warmup_requests", out)
        self.assertIn("measurement_window_seconds", out)
        self.assertIsNone(out["warmup_requests"])
        self.assertIsNone(out["measurement_window_seconds"])

    def test_ac11_reproducibility_fields_carry_numeric_overrides(self):
        out = _run_writer({
            "WARMUP_REQUESTS": "120",
            "MEASUREMENT_WINDOW_S": "600.0",
            "TRIAL_ID": "2",
        })
        self.assertEqual(out["warmup_requests"], 120)
        self.assertAlmostEqual(out["measurement_window_seconds"], 600.0)
        self.assertEqual(out["trial_id"], "2")

    def test_chunked_prefill_size_from_server_args(self):
        out = _run_writer({
            "SERVER_ARGS_JSON": json.dumps({"chunked_prefill_size": 8192}),
        })
        self.assertEqual(out["chunked_prefill_size"], 8192)

    def test_output_is_valid_pretty_printed_json(self):
        """The sidecar file must be re-parseable; check indented form."""
        env = {
            "COMMIT_SHA": "abc123",
            "MODE": "native_nsa",
            "CONCURRENCY": "16",
            "SEED": "213",
            "NUM_PROMPTS": "320",
            "ISL_TOTAL_TOKENS": "4096",
            "OSL_TOKENS": "512",
            "TIMESTAMP_UTC": "2026-05-27T12:00:00Z",
            "SERVER_ARGS_JSON": json.dumps({"a": True, "b": None}),
            "PATH": os.environ.get("PATH", ""),
        }
        proc = subprocess.run(
            ["python3", str(WRITER)],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(proc.returncode, 0)
        # Multi-line pretty JSON.
        self.assertIn("\n", proc.stdout)
        json.loads(proc.stdout)  # raises if not valid


if __name__ == "__main__":
    unittest.main()
