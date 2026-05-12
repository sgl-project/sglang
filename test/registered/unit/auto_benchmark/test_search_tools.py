import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from auto_benchmark import AutoBenchmarkTestCase

from sglang.auto_benchmark_lib import (
    append_jsonl,
    build_qps_plan,
    build_server_candidates,
    classify_failure,
    collect_stale_server_pids,
    describe_search_tier,
    estimate_trials_per_candidate,
    expand_dataset_scenarios,
    format_best_progress,
    render_scenario_summary_markdown,
    rendered_launch_command,
    resolve_max_candidates,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=6, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=6, suite="stage-b-test-1-gpu-small-amd")


class TestAutoBenchmarkSearchTools(AutoBenchmarkTestCase):
    def test_build_candidates_by_tier(self):
        base_flags = {"model_path": "/model", "tp_size": 4}
        search_space = {
            "prefill_attention_backend": ["fa3", "flashinfer", "triton"],
            "decode_attention_backend": ["fa3", "flashinfer"],
            "chunked_prefill_size": [4096, 8192],
            "max_running_requests": [64, 128],
            "schedule_policy": ["lpm", "fcfs"],
        }

        tier1 = self._build_candidates_for_capability(
            base_flags,
            search_space,
            tier=1,
            max_candidates=None,
            capability=None,
        )
        tier2 = self._build_candidates_for_capability(
            base_flags,
            search_space,
            tier=2,
            max_candidates=None,
            capability=None,
        )
        tier3 = self._build_candidates_for_capability(
            base_flags,
            search_space,
            tier=3,
            max_candidates=32,
            capability=None,
        )

        self.assertGreater(len(tier1), 1)
        self.assertGreater(len(tier2), len(tier1))
        self.assertGreater(len(tier3), len(tier2))
        self.assertEqual(tier1[0]["model_path"], "/model")

    def test_parallel_search_derives_dp_size(self):
        server_cfg = {
            "env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
            "base_flags": {"model_path": "/model"},
            "parallel": {
                "tp": [4, 2],
                "pp_size": [1],
            },
            "search_space": {},
        }

        candidates = build_server_candidates(server_cfg, tier=2, max_candidates=None)
        tp_dp_pairs = {
            (candidate["tp_size"], candidate["dp_size"]) for candidate in candidates
        }
        self.assertIn((4, 2), tp_dp_pairs)
        self.assertIn((2, 4), tp_dp_pairs)

    def test_build_server_candidates_filters_unsupported_fa3_on_sm100(self):
        server_cfg = {
            "base_flags": {"model_path": "/model", "tp_size": 1},
            "search_space": {
                "prefill_attention_backend": ["fa3", "flashinfer"],
                "decode_attention_backend": ["fa3", "flashinfer"],
                "chunked_prefill_size": [4096, 8192],
            },
        }

        candidates = self._build_server_candidates_for_capability(
            server_cfg,
            tier=2,
            max_candidates=None,
            capability=(10, 0),
        )

        self.assertGreater(len(candidates), 0)
        for candidate in candidates:
            self.assertNotEqual(candidate.get("attention_backend"), "fa3")
            self.assertNotEqual(candidate.get("prefill_attention_backend"), "fa3")
            self.assertNotEqual(candidate.get("decode_attention_backend"), "fa3")

    def test_build_server_candidates_keeps_fa3_on_sm90(self):
        server_cfg = {
            "base_flags": {"model_path": "/model", "tp_size": 1},
            "search_space": {
                "prefill_attention_backend": ["fa3", "flashinfer"],
                "decode_attention_backend": ["fa3", "flashinfer"],
            },
        }

        candidates = self._build_server_candidates_for_capability(
            server_cfg,
            tier=2,
            max_candidates=None,
            capability=(9, 0),
        )

        self.assertTrue(
            any(
                candidate.get("prefill_attention_backend") == "fa3"
                or candidate.get("decode_attention_backend") == "fa3"
                for candidate in candidates
            )
        )

    def test_ep_alias_and_oom_classification(self):
        server_cfg = {
            "base_flags": {"model_path": "/model", "tp_size": 8},
            "search_space": {"ep": [1, 4]},
        }

        candidates = build_server_candidates(server_cfg, tier=2, max_candidates=None)
        ep_sizes = {candidate.get("ep_size", 1) for candidate in candidates}
        self.assertEqual(ep_sizes, {1, 4})

        diagnosis, hint = classify_failure("RuntimeError: CUDA out of memory")
        self.assertEqual(diagnosis, "oom")
        self.assertIn("Increase GPU count", hint)

    def test_expand_random_dataset_scenarios(self):
        scenarios = expand_dataset_scenarios(
            {
                "kind": "random",
                "scenario_names": ["chat", "summarization"],
                "input_len": [1000, 8000],
                "output_len": [1000, 1000],
            }
        )

        self.assertEqual(len(scenarios), 2)
        self.assertEqual(scenarios[0]["name"], "chat")
        self.assertEqual(scenarios[0]["cfg"]["random_input_len"], 1000)
        self.assertEqual(scenarios[1]["cfg"]["random_input_len"], 8000)
        self.assertEqual(scenarios[1]["cfg"]["random_output_len"], 1000)

    def test_estimate_trials_and_tier_descriptions(self):
        benchmark_cfg = {
            "qps": {"lower": 0.25, "upper": 4.0, "tolerance": 0.1},
            "max_concurrency": [None, 8, 16],
        }

        self.assertEqual(estimate_trials_per_candidate(benchmark_cfg), 15)
        self.assertIn("default", describe_search_tier(2))
        self.assertIn("slowest", describe_search_tier(3))

    def test_resolve_max_candidates_defaults_to_eight(self):
        self.assertEqual(resolve_max_candidates({}), 8)
        self.assertIsNone(resolve_max_candidates({"max_candidates": None}))

    def test_resolve_max_candidates_rejects_non_positive_values(self):
        with self.assertRaisesRegex(ValueError, "search.max_candidates"):
            resolve_max_candidates({"max_candidates": 0})

    def test_build_qps_plan_accepts_numeric_request_rate(self):
        mode, values, tolerance, max_rounds = build_qps_plan({"request_rate": 3.5})
        self.assertEqual(mode, "fixed")
        self.assertEqual(values, [3.5])
        self.assertEqual(tolerance, 0.0)
        self.assertEqual(max_rounds, 0)

    def test_build_qps_plan_clamps_binary_rounds(self):
        mode, values, tolerance, max_rounds = build_qps_plan(
            {"qps": {"lower": 1.0, "upper": 16.0, "tolerance": 0.1, "max_rounds": 99}}
        )

        self.assertEqual(mode, "search")
        self.assertEqual(values, [1.0, 16.0])
        self.assertEqual(tolerance, 0.1)
        self.assertEqual(max_rounds, 5)

    def test_format_best_progress(self):
        text = format_best_progress(
            {
                "candidate_id": 3,
                "requested_qps": 3.5,
                "server_flags": {
                    "tp_size": 4,
                    "ep_size": 4,
                    "mem_fraction_static": 0.84,
                    "max_running_requests": 96,
                },
                "metrics": {
                    "output_throughput": 1234.56,
                    "mean_ttft_ms": 250.12,
                    "mean_tpot_ms": 14.78,
                },
            }
        )

        self.assertIn("qps=3.5000", text)
        self.assertIn("tok/s=1234.6", text)
        self.assertIn("ttft=250.1ms", text)
        self.assertIn("tpot=14.8ms", text)
        self.assertIn("tp=4", text)
        self.assertIn("ep=4", text)

    def test_append_jsonl(self):
        path = self.tmpdir_path / "live_results.jsonl"
        append_jsonl(
            str(path),
            [
                {"candidate_id": 1, "requested_qps": 2.0},
                {"candidate_id": 2, "requested_qps": 3.0},
            ],
        )

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["candidate_id"], 1)
        self.assertEqual(json.loads(lines[1])["requested_qps"], 3.0)

    def test_collect_stale_server_pids_dedups(self):
        def fake_run(command, capture_output, text, check):
            stdout = "123\n" if command[0] == "lsof" else "123\n456\n"
            return SimpleNamespace(returncode=0, stdout=stdout)

        with mock.patch(
            "sglang.auto_benchmark_lib.subprocess.run", side_effect=fake_run
        ):
            self.assertEqual(collect_stale_server_pids(30000), [123, 456])

    def test_rendered_launch_command_includes_env(self):
        text = rendered_launch_command(
            {
                "env": {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "HF_TOKEN": "secret-value",
                },
                "extra_args": [],
            },
            {"model_path": "Qwen/Qwen3-32B", "tp_size": 1, "port": 30000},
        )

        self.assertIn("CUDA_VISIBLE_DEVICES=0", text)
        self.assertIn("--model-path Qwen/Qwen3-32B", text)
        self.assertNotIn("HF_TOKEN", text)

    def test_render_scenario_summary_markdown_keeps_rows_in_single_table(self):
        text = render_scenario_summary_markdown(
            [
                {
                    "scenario_name": "chat",
                    "scenario_dir": "/tmp/chat",
                    "status": "ok",
                    "requested_qps": 11.914,
                    "output_throughput": 1867.28,
                    "mean_ttft_ms": 99.58,
                    "mean_tpot_ms": 21.09,
                    "launch_command": "python -m sglang.launch_server --port 30000",
                },
                {
                    "scenario_name": "summarization",
                    "scenario_dir": "/tmp/summarization",
                    "status": "ok",
                    "requested_qps": 11.914,
                    "output_throughput": 537.17,
                    "mean_ttft_ms": 709.99,
                    "mean_tpot_ms": 26.89,
                    "launch_command": "python -m sglang.launch_server --port 30001",
                },
            ]
        )

        header = (
            "| Scenario | Status | QPS | Output tok/s | TTFT ms | TPOT ms | Summary |"
        )
        self.assertEqual(text.count(header), 1)
        self.assertLess(text.index("| chat |"), text.index("## chat"))
        self.assertLess(text.index("| summarization |"), text.index("## chat"))
        self.assertLess(text.index("| summarization |"), text.index("## summarization"))


if __name__ == "__main__":
    unittest.main()
