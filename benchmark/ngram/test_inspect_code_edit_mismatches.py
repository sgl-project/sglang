"""Artifact inspection tests using synthetic saved trials; no GPU required."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from code_edit_report import build_summary, summarize_trial
from inspect_code_edit_mismatches import inspect_results, print_report
from test_code_edits import record


class FakeTokenizer:
    def decode(self, ids, **kwargs):
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return "".join(f"<{token}>" for token in ids)

    def convert_ids_to_tokens(self, token):
        return f"token-{token}"


class TestInspectMismatches(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.trials = {}
        for repetition in (1, 2, 3):
            for mode in ("baseline", "legacy_bfs", "path_probability"):
                rows = [record(0), record(1)]
                if mode == "legacy_bfs":
                    rows[0]["output_ids"][2] = 8
                elif mode == "path_probability":
                    rows[0]["output_ids"][5] = 9
                self.trials[mode, repetition] = rows
        (self.directory / "experiment.json").write_text(
            json.dumps({"source_revision": "fixture-source"})
        )
        (self.directory / "inputs.json").write_text(
            json.dumps(
                {
                    "requests": [
                        {"task_id": task, "input_ids": [128000, 9]} for task in (0, 1)
                    ]
                }
            )
        )
        self.save_trials()

    def save_trials(self):
        summaries = []
        for (mode, repetition), rows in self.trials.items():
            path = self.directory / f"{mode}-{repetition}.requests.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            summaries.append(
                {**summarize_trial(mode, repetition, rows, 4), "requests": rows}
            )
        (self.directory / "summary.json").write_text(
            json.dumps(build_summary(summaries))
        )

    def test_counts_unique_tasks_repeatability_and_pairwise_outputs(self):
        report = inspect_results(self.directory)
        bfs = report["modes"]["legacy_bfs"]
        self.assertEqual(bfs["mismatch_count"], 3)
        self.assertEqual(bfs["request_repetition_pairs"], 6)
        self.assertEqual(bfs["unique_mismatching_task_ids"], [0])
        self.assertTrue(all(not r["mismatches"] for r in bfs["repeatability"]))
        self.assertTrue(all(c["token_index"] == 2 for c in bfs["cases"]))
        pair = next(
            p
            for p in report["cross_mode"]
            if p["reference_mode"] == "legacy_bfs"
            and p["actual_mode"] == "path_probability"
        )
        # Equal counts and affected tasks do not imply equal generated outputs.
        self.assertEqual(pair["mismatch_count"], 3)

    def test_changed_baseline_is_paired_with_its_own_repetition(self):
        self.trials["baseline", 2][1]["output_ids"][7] = 10
        self.save_trials()
        report = inspect_results(self.directory)
        self.assertEqual(report["modes"]["baseline"]["mismatch_count"], 1)
        bfs_cases = report["modes"]["legacy_bfs"]["cases"]
        cases = [c for c in bfs_cases if c["task_id"] == 1]
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["repetition"], 2)
        self.assertEqual(cases[0]["baseline_repetition"], 2)
        self.assertEqual(cases[0]["baseline_token_id"], 10)

    def test_same_first_divergence_does_not_hide_later_repeat_differences(self):
        self.trials["legacy_bfs", 3][0]["output_ids"][20] = 11
        self.save_trials()
        report = inspect_results(self.directory)
        repeats = report["modes"]["legacy_bfs"]["repeatability"]
        self.assertEqual(repeats[1]["mismatches"][0]["token_index"], 20)

    def test_decoded_windows_include_common_prefix_and_token_pieces(self):
        report = inspect_results(self.directory, FakeTokenizer(), context_tokens=1)
        case = report["modes"]["legacy_bfs"]["cases"][0]
        self.assertEqual(case["common_prefix_tail_text"], "<3>")
        self.assertEqual(case["baseline_window_text"], "<3><3><3>")
        self.assertEqual(case["actual_window_text"], "<3><8><3>")
        self.assertEqual(case["actual_token_piece"], "token-8")
        stream = io.StringIO()
        with redirect_stdout(stream):
            print_report(report, mode="legacy_bfs", task_id="0", limit=1)
        self.assertEqual(stream.getvalue().count("first differing output token"), 1)

    def test_raw_artifacts_must_agree_with_summary_and_inputs(self):
        path = self.directory / "legacy_bfs-1.requests.jsonl"
        rows = self.trials["legacy_bfs", 1]
        for field, value, error in (
            ("output_ids", [42] * 256, "summary disagrees"),
            ("prompt_token_ids", [99], "Prompt IDs differ"),
            ("task_id", 99, "Task IDs/order differ"),
        ):
            with self.subTest(field=field):
                changed = [{**rows[0], field: value}, rows[1]]
                path.write_text("".join(json.dumps(row) + "\n" for row in changed))
                with self.assertRaisesRegex(ValueError, error):
                    inspect_results(self.directory)
        self.save_trials()

    def test_cli_writes_report_without_tokenizer_and_never_overwrites(self):
        script = Path(__file__).with_name("inspect_code_edit_mismatches.py")
        output = self.directory / "inspection.json"
        command = [
            sys.executable,
            str(script),
            str(self.directory),
            "--output",
            str(output),
        ]
        first = subprocess.run(command, text=True, capture_output=True, check=True)
        self.assertIn("No model inference performed", first.stdout)
        original = output.read_bytes()
        second = subprocess.run(command, text=True, capture_output=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("Output already exists", second.stderr)
        self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
