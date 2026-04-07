import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

sys.modules.setdefault("zmq", types.SimpleNamespace())

from sglang.auto_benchmark_lib import (
    SearchDeadlineExceeded,
    append_jsonl,
    build_candidates,
    build_qps_plan,
    build_server_candidates,
    classify_failure,
    collect_stale_server_pids,
    describe_search_tier,
    estimate_trials_per_candidate,
    expand_dataset_scenarios,
    format_best_progress,
    infer_backend,
    prepare_dataset,
    render_scenario_summary_markdown,
    rendered_launch_command,
    resolve_max_candidates,
    run_candidate,
)
from sglang.benchmark.datasets.autobench import sample_autobench_requests
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small", disabled="Flaky test")


def create_lightweight_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3}
    vocab.update({f"tok_{i}": i + 4 for i in range(4096)})

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    hf_tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )
    return hf_tokenizer


class TestAutoBenchmarkTools(CustomTestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir_path = Path(self.tmpdir.name)
        self.tokenizer = create_lightweight_tokenizer()
        self.tokenizer_dir = self.tmpdir_path / "tok"
        self.tokenizer.save_pretrained(self.tokenizer_dir)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_autobench_jsonl(self) -> str:
        rows = [
            {"prompt": "tok_1 tok_2 tok_3", "output_len": 32},
            {
                "messages": [{"role": "user", "content": "tok_4 tok_5"}],
                "output_len": 24,
                "extra_request_body": {"temperature": 0.0},
            },
            {
                "system": "tok_6",
                "content": ["tok_7 tok_8", "tok_9", "tok_10 tok_11"],
                "output_len": 16,
            },
        ]
        path = self.tmpdir_path / "sample.autobench.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return str(path)

    def _write_sharegpt_json(self) -> str:
        rows = [
            {
                "conversations": [
                    {"value": "tok_1 tok_2 tok_3"},
                    {"value": "tok_4 tok_5"},
                ]
            },
            {
                "conversations": [
                    {"value": "tok_6 tok_7"},
                    {"value": "tok_8 tok_9 tok_10"},
                ]
            },
        ]
        path = self.tmpdir_path / "sharegpt.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f)
        return str(path)

    def test_prepare_custom_autobench_dataset(self):
        dataset_path = self._write_autobench_jsonl()
        output_path = self.tmpdir_path / "prepared.autobench.jsonl"

        prepared_path, rows, summary = prepare_dataset(
            dataset_cfg={
                "kind": "custom",
                "path": dataset_path,
                "num_prompts": 2,
            },
            tokenizer_path=str(self.tokenizer_dir),
            model=None,
            output_path=str(output_path),
        )

        self.assertEqual(prepared_path, str(output_path))
        self.assertEqual(summary["num_requests"], 2)
        self.assertTrue(Path(prepared_path).exists())
        converted_rows = sample_autobench_requests(
            dataset_path=prepared_path,
            num_requests=0,
            tokenizer=self.tokenizer,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(converted_rows), 2)

    def test_invalid_json_like_prompt_falls_back_to_plain_text(self):
        path = self.tmpdir_path / "jsonlike.autobench.jsonl"
        path.write_text(
            json.dumps({"prompt": "[not actually json", "output_len": 8}) + "\n",
            encoding="utf-8",
        )

        rows = sample_autobench_requests(
            dataset_path=str(path),
            num_requests=0,
            tokenizer=self.tokenizer,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].prompt, "[not actually json")

    def test_prepare_sharegpt_dataset(self):
        sharegpt_path = self._write_sharegpt_json()
        output_path = self.tmpdir_path / "sharegpt.autobench.jsonl"

        prepared_path, rows, summary = prepare_dataset(
            dataset_cfg={
                "kind": "sharegpt",
                "path": sharegpt_path,
                "num_prompts": 2,
            },
            tokenizer_path=str(self.tokenizer_dir),
            model=None,
            output_path=str(output_path),
        )

        self.assertEqual(prepared_path, str(output_path))
        self.assertEqual(summary["num_requests"], 2)
        self.assertEqual(len(rows), 2)

    def test_prepare_custom_dataset_requires_path(self):
        with self.assertRaisesRegex(ValueError, "dataset.path is required"):
            prepare_dataset(
                dataset_cfg={"kind": "custom"},
                tokenizer_path=str(self.tokenizer_dir),
                model=None,
                output_path=str(self.tmpdir_path / "missing.autobench.jsonl"),
            )

    def test_infer_backend(self):
        prompt_rows = [SimpleNamespace(prompt="tok_1 tok_2")]
        chat_rows = [SimpleNamespace(prompt=[{"role": "user", "content": "tok_1"}])]
        token_id_rows = [SimpleNamespace(prompt=[1, 2, 3])]

        self.assertEqual(infer_backend("auto", prompt_rows), "sglang-oai")
        self.assertEqual(infer_backend("auto", chat_rows), "sglang-oai-chat")
        self.assertEqual(infer_backend("auto", token_id_rows), "sglang")

    def test_build_candidates_by_tier(self):
        base_flags = {"model_path": "/model", "tp_size": 4}
        search_space = {
            "prefill_attention_backend": ["fa3", "flashinfer", "triton"],
            "decode_attention_backend": ["fa3", "flashinfer"],
            "chunked_prefill_size": [4096, 8192],
            "max_running_requests": [64, 128],
            "schedule_policy": ["lpm", "fcfs"],
        }

        tier1 = build_candidates(base_flags, search_space, tier=1, max_candidates=None)
        tier2 = build_candidates(base_flags, search_space, tier=2, max_candidates=None)
        tier3 = build_candidates(base_flags, search_space, tier=3, max_candidates=32)

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

        with mock.patch(
            "sglang.auto_benchmark_lib.detect_current_cuda_capability",
            return_value=(10, 0),
        ):
            candidates = build_server_candidates(
                server_cfg, tier=2, max_candidates=None
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

        with mock.patch(
            "sglang.auto_benchmark_lib.detect_current_cuda_capability",
            return_value=(9, 0),
        ):
            candidates = build_server_candidates(
                server_cfg, tier=2, max_candidates=None
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

    def test_run_candidate_binary_search_avoids_rounding_loop(self):
        benchmark_cfg = {
            "qps": {"lower": 1.0, "upper": 1.00000001, "tolerance": 1e-12},
            "max_concurrency": [None],
        }
        calls = []

        def fake_run_trial(**kwargs):
            calls.append(kwargs["request_rate"])
            return {
                "stage": "base",
                "candidate_id": kwargs["candidate_id"],
                "requested_qps": kwargs["request_rate"],
                "max_concurrency": kwargs["max_concurrency"],
                "server_flags": kwargs["server_flags"],
                "sla_passed": True,
                "metrics": {
                    "output_throughput": 1.0,
                    "mean_ttft_ms": 1.0,
                    "mean_tpot_ms": 1.0,
                },
            }

        with mock.patch(
            "sglang.auto_benchmark_lib.run_trial", side_effect=fake_run_trial
        ):
            records = run_candidate(
                stage_name="base",
                candidate_id=0,
                server_cfg={"host": "127.0.0.1", "port": 30000},
                benchmark_cfg=benchmark_cfg,
                dataset_summary={"num_requests": 1},
                backend="sglang-oai",
                dataset_path="/tmp/fake.jsonl",
                tokenizer_path=str(self.tokenizer_dir),
                server_flags={"model_path": "/model"},
                output_dir=str(self.tmpdir_path),
            )

        self.assertLess(len(calls), 40)
        self.assertEqual(len(records), len(calls))

    def test_run_candidate_binary_search_respects_max_rounds(self):
        benchmark_cfg = {
            "qps": {"lower": 1.0, "upper": 32.0, "tolerance": 1e-12, "max_rounds": 2},
            "max_concurrency": [None],
        }
        calls = []

        def fake_run_trial(**kwargs):
            calls.append(kwargs["request_rate"])
            return {
                "stage": "base",
                "candidate_id": kwargs["candidate_id"],
                "requested_qps": kwargs["request_rate"],
                "max_concurrency": kwargs["max_concurrency"],
                "server_flags": kwargs["server_flags"],
                "sla_passed": True,
                "metrics": {
                    "output_throughput": 1.0,
                    "mean_ttft_ms": 1.0,
                    "mean_tpot_ms": 1.0,
                },
            }

        with mock.patch(
            "sglang.auto_benchmark_lib.run_trial", side_effect=fake_run_trial
        ):
            records = run_candidate(
                stage_name="base",
                candidate_id=0,
                server_cfg={"host": "127.0.0.1", "port": 30000},
                benchmark_cfg=benchmark_cfg,
                dataset_summary={"num_requests": 1},
                backend="sglang-oai",
                dataset_path="/tmp/fake.jsonl",
                tokenizer_path=str(self.tokenizer_dir),
                server_flags={"model_path": "/model"},
                output_dir=str(self.tmpdir_path),
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(records), 2)

    def test_run_candidate_stops_when_search_budget_is_exhausted(self):
        benchmark_cfg = {
            "qps": {"lower": 1.0, "upper": 2.0, "tolerance": 0.1},
            "max_concurrency": [None],
        }

        with self.assertRaises(SearchDeadlineExceeded):
            run_candidate(
                stage_name="base",
                candidate_id=0,
                server_cfg={"host": "127.0.0.1", "port": 30000},
                benchmark_cfg=benchmark_cfg,
                dataset_summary={"num_requests": 1},
                backend="sglang-oai",
                dataset_path="/tmp/fake.jsonl",
                tokenizer_path=str(self.tokenizer_dir),
                server_flags={"model_path": "/model"},
                output_dir=str(self.tmpdir_path),
                search_deadline=time.time() - 1.0,
                search_budget_hours=0.1,
            )

    def test_run_candidate_resume_skips_existing_fixed_trials(self):
        benchmark_cfg = {
            "qps": [1.0, 2.0],
            "max_concurrency": [None],
        }
        existing_records = [
            {
                "stage": "base",
                "candidate_id": 0,
                "requested_qps": 1.0,
                "max_concurrency": None,
                "server_flags": {"model_path": "/model"},
                "sla_passed": True,
                "metrics": {
                    "output_throughput": 1.0,
                    "mean_ttft_ms": 1.0,
                    "mean_tpot_ms": 1.0,
                },
            }
        ]
        calls = []

        def fake_run_trial(**kwargs):
            calls.append(kwargs["request_rate"])
            return {
                "stage": "base",
                "candidate_id": kwargs["candidate_id"],
                "requested_qps": kwargs["request_rate"],
                "max_concurrency": kwargs["max_concurrency"],
                "server_flags": kwargs["server_flags"],
                "sla_passed": True,
                "metrics": {
                    "output_throughput": 2.0,
                    "mean_ttft_ms": 2.0,
                    "mean_tpot_ms": 2.0,
                },
            }

        with mock.patch(
            "sglang.auto_benchmark_lib.run_trial", side_effect=fake_run_trial
        ):
            records = run_candidate(
                stage_name="base",
                candidate_id=0,
                server_cfg={"host": "127.0.0.1", "port": 30000},
                benchmark_cfg=benchmark_cfg,
                dataset_summary={"num_requests": 1},
                backend="sglang-oai",
                dataset_path="/tmp/fake.jsonl",
                tokenizer_path=str(self.tokenizer_dir),
                server_flags={"model_path": "/model"},
                output_dir=str(self.tmpdir_path),
                existing_records=existing_records,
            )

        self.assertEqual(calls, [2.0])
        self.assertEqual([record["requested_qps"] for record in records], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
