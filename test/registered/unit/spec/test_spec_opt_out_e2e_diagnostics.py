"""CPU regressions for evidence collection in the manual model smoke test."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

path = (
    Path(__file__).resolve().parents[3]
    / "manual/spec/test_per_request_spec_opt_out_e2e.py"
)
spec = importlib.util.spec_from_file_location("opt_out_e2e_diagnostics", path)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)

analysis_spec = importlib.util.spec_from_file_location(
    "prefix_analysis", path.with_name("analyze_spec_opt_out_prefix.py")
)
analysis = importlib.util.module_from_spec(analysis_spec)
analysis_spec.loader.exec_module(analysis)


class TestPrefixAnalysis(unittest.TestCase):
    def test_same_last_token_does_not_establish_same_prefix(self):
        self.assertTrue(analysis.same_prefix([1, 2], [3, 4], [1, 2, 3]))
        self.assertFalse(analysis.same_prefix([9, 2], [3, 4], [1, 2, 3]))
        self.assertFalse(analysis.same_prefix([1], [2], [1, 2, 3]))

    def test_exact_ties_and_small_positive_margins_remain_distinct(self):
        tied = analysis.score_summary([22.875, 22.875], 0, [22.75, 22.875])
        self.assertEqual(tied["tied_max_ids"], [0, 1])
        self.assertTrue(tied["greedy_argmax_correct"])
        self.assertEqual(tied["top_margin"], 0)
        self.assertFalse(
            analysis.score_summary([1, 1], 1, [1, 1])["greedy_argmax_correct"]
        )
        near = analysis.score_summary([1, 1.000001], 0, [1, 1.000001])
        self.assertFalse(near["greedy_argmax_correct"])
        self.assertGreater(near["top_margin"], 0)

    def test_invalid_logits_are_not_numerical_success(self):
        for values in ([float("nan"), 1], [1, float("inf")], [1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                analysis.score_summary(values, 0, [1, 1])


class TestOptOutE2EDiagnostics(unittest.TestCase):
    def test_stream_prefix_must_be_monotonic(self):
        module_spec = importlib.util.spec_from_file_location(
            "lifecycle", path.with_name("test_eagle3_opt_out_lifecycle.py")
        )
        lifecycle = importlib.util.module_from_spec(module_spec)
        with patch.dict(sys.modules, test_per_request_spec_opt_out_e2e=smoke):
            module_spec.loader.exec_module(lifecycle)
        lifecycle.check_stream([7], [7, 8])
        lifecycle.check_stream([7, 8], [7, 8])
        for current in ([7], [7, 9], []):
            with self.subTest(current=current), self.assertRaises(AssertionError):
                lifecycle.check_stream([7, 8], current)

    def test_failed_response_is_saved_before_validation(self):
        response = {
            "output_ids": [9],
            "meta_info": {"output_token_logprobs": [[-0.1, 7, None]]},
        }
        rows = [response] * 3
        http = SimpleNamespace(
            ok=True, status_code=200, text=json.dumps(rows), json=lambda: rows
        )
        process = Mock(pid=12345)
        process.poll.return_value = None
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(smoke.os.environ, OPT_OUT_RESULTS=directory),
            patch.object(smoke.subprocess, "Popen", return_value=process),
            patch.object(smoke.os, "killpg") as killpg,
            patch.object(smoke.requests, "get", return_value=SimpleNamespace(ok=True)),
            patch.object(smoke.requests, "post", return_value=http),
        ):
            with self.assertRaisesRegex(AssertionError, "disagree"):
                smoke.run()
            saved = json.loads((Path(directory) / "target-wave-1.json").read_text())
            self.assertEqual(json.loads(saved["response_text"]), rows)
            self.assertEqual(saved["request"]["top_logprobs_num"], 5)
            killpg.assert_called_once_with(process.pid, smoke.signal.SIGTERM)

    def test_emitted_tokens_are_checked_against_logprob_ids(self):
        response = {
            "output_ids": [7, 8],
            "meta_info": {"output_token_logprobs": [[-0.1, 7, None], [-0.2, 8, None]]},
        }
        self.assertEqual(smoke.response_token_ids(response, 2), [7, 8])
        # The old harness would report a match to baseline [7, 8] even though
        # the actual generated output differs. Equal lengths alone do not help.
        response["output_ids"] = [7, 9]
        with self.assertRaisesRegex(AssertionError, "disagree"):
            smoke.response_token_ids(response, 2)

    def test_missing_or_short_output_is_not_replaced_by_metadata(self):
        response = {"meta_info": {"output_token_logprobs": [[-0.1, 7, None]]}}
        with self.assertRaises(KeyError):
            smoke.response_token_ids(response, 1)
        response["output_ids"] = [7]
        with self.assertRaises(AssertionError):
            smoke.response_token_ids(response, 2)

    def test_first_divergence_and_wave_are_preserved(self):
        result = smoke.token_mismatch([7, 9, 3], [7, 8, 4], mode="eager", row=1, wave=4)
        self.assertEqual(result["first_difference"], 1)
        self.assertEqual(result["wave"], 4)
        self.assertEqual(result["actual"], [7, 9, 3])
        self.assertEqual(result["expected"], [7, 8, 4])

    def test_length_mismatch_and_equal_sequences(self):
        for actual, expected in (([7], [7, 8]), ([7, 8], [7]), ([], [7])):
            with self.subTest(actual=actual, expected=expected):
                result = smoke.token_mismatch(
                    actual, expected, mode="graph", row=0, wave=1
                )
                self.assertEqual(
                    result["first_difference"], min(len(actual), len(expected))
                )
        self.assertIsNone(smoke.token_mismatch([7], [7], mode="graph", row=0, wave=1))


if __name__ == "__main__":
    unittest.main()
