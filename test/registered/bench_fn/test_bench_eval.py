import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


class TestBenchEvalImports(unittest.TestCase):
    def test_subpackage_importable(self):
        from sglang.benchmark import eval_harness  # noqa: F401

    def test_public_symbols(self):
        from sglang.benchmark.eval_harness import (
            BenchServingLM,
            merge_report,
            write_report,
        )
        self.assertTrue(callable(BenchServingLM))
        self.assertTrue(callable(merge_report))
        self.assertTrue(callable(write_report))


from types import SimpleNamespace
from unittest.mock import patch

from test_benchmark_datasets_api import (
    create_lightweight_tokenizer,
)


def _fake_request(prompt, until=None, max_gen_toks=64, **kw):
    """Minimal object matching lm_eval.api.instance.Instance.args[0..1]."""
    gen_kwargs = {"until": until or [], "max_gen_toks": max_gen_toks, **kw}
    return SimpleNamespace(args=(prompt, gen_kwargs))


class TestBenchServingLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = create_lightweight_tokenizer()

    def _make_lm(self, **overrides):
        from sglang.benchmark.eval_harness import BenchServingLM

        defaults = dict(
            base_url="http://mock:0",
            backend="sglang-oai",
            model_id="mock-model",
            tokenizer=self.tokenizer,
            request_rate=float("inf"),
            max_concurrency=4,
            enable_thinking=False,
        )
        defaults.update(overrides)
        return BenchServingLM(**defaults)

    def test_rejects_loglikelihood(self):
        lm = self._make_lm()
        with self.assertRaises(NotImplementedError):
            lm.loglikelihood([_fake_request("x")])
        with self.assertRaises(NotImplementedError):
            lm.loglikelihood_rolling([_fake_request("x")])

    def test_generate_until_calls_bench_serving(self):
        lm = self._make_lm()
        requests = [
            _fake_request("Q1?", until=["\n"], max_gen_toks=32),
            _fake_request("Q2?", until=["\n"], max_gen_toks=32),
        ]

        async def fake_benchmark(**kwargs):
            # Assert we passed proper DatasetRows with our prompts.
            rows = kwargs["input_requests"]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0].prompt, "Q1?")
            self.assertEqual(rows[0].output_len, 32)
            self.assertEqual(kwargs["request_rate"], float("inf"))
            self.assertEqual(kwargs["max_concurrency"], 4)
            return {
                "generated_texts": ["A1", "A2"],
                "mean_ttft_ms": 1.0,
                "mean_itl_ms": 0.5,
                "output_throughput": 10.0,
                "completed": 2,
            }

        with patch("sglang.bench_serving.benchmark", new=fake_benchmark), \
             patch("sglang.bench_serving.set_global_args"):
            outputs = lm.generate_until(requests)

        self.assertEqual(outputs, ["A1", "A2"])
        self.assertEqual(lm.last_perf["completed"], 2)
        self.assertAlmostEqual(lm.last_perf["mean_ttft_ms"], 1.0)

    def test_generate_until_forwards_stop_strings(self):
        lm = self._make_lm()
        requests = [_fake_request("Q?", until=["\n", "Question:"], max_gen_toks=16)]

        captured = {}

        async def fake_benchmark(**kwargs):
            captured["rows"] = kwargs["input_requests"]
            captured["extra"] = kwargs["extra_request_body"]
            return {"generated_texts": ["."], "completed": 1}

        with patch("sglang.bench_serving.benchmark", new=fake_benchmark), \
             patch("sglang.bench_serving.set_global_args"):
            lm.generate_until(requests)

        # Per-request stop strings go on DatasetRow.extra_request_body.
        self.assertEqual(
            captured["rows"][0].extra_request_body.get("stop"),
            ["\n", "Question:"],
        )

    def test_generate_until_populates_all_args_defaults(self):
        """After set_global_args + _apply_arg_defaults, the module-level args
        namespace must have all attrs that bench_serving internals might read."""
        lm = self._make_lm()
        requests = [_fake_request("Q?", max_gen_toks=16)]

        captured_args = {}

        async def fake_benchmark(**kwargs):
            # At this point the module-level args global is populated.
            from sglang import bench_serving as _bs
            captured_args["args"] = _bs.args
            return {"generated_texts": ["ok"], "completed": 1}

        with patch("sglang.bench_serving.benchmark", new=fake_benchmark):
            lm.generate_until(requests)

        args = captured_args["args"]
        # Sample of attrs read by benchmark() / request funcs that are NOT
        # in our own explicit Namespace construction — these must have been
        # filled in by _apply_arg_defaults.
        for attr in (
            "plot_throughput", "disable_stream", "disable_ignore_eos",
            "return_logprob", "top_logprobs_num", "token_ids_logprob",
            "logprob_start_len", "use_trace_timestamps",
            "mooncake_slowdown_factor", "mooncake_num_rounds",
            "served_model_name", "tokenize_prompt", "warmup_requests",
            "max_concurrency", "output_details", "return_routed_experts",
            "profile_start_step", "profile_steps", "extra_request_body",
            "seed",
        ):
            self.assertTrue(hasattr(args, attr), f"missing attr: {attr}")

    def test_apply_chat_template_plain(self):
        lm = self._make_lm()
        msg = [{"role": "user", "content": "hi"}]
        out = lm.apply_chat_template(msg)
        self.assertIn("user:", out)  # lightweight tokenizer template
        self.assertIn("hi", out)

    def test_apply_chat_template_with_thinking(self):
        """When enable_thinking=True, apply_chat_template must forward it."""
        lm = self._make_lm(enable_thinking=True)
        calls = []
        real_apply = lm.tokenizer.apply_chat_template

        def spy(chat, **kwargs):
            calls.append(dict(kwargs))
            # Strip the thinking kwarg so the lightweight tokenizer doesn't choke.
            kwargs.pop("enable_thinking", None)
            return real_apply(chat, **kwargs)

        lm.tokenizer.apply_chat_template = spy
        lm.apply_chat_template([{"role": "user", "content": "hi"}])
        self.assertTrue(calls, "apply_chat_template was not called on tokenizer")
        self.assertTrue(calls[0].get("enable_thinking"))


import json
import tempfile
from pathlib import Path


class TestReport(unittest.TestCase):
    def test_merge_report_shape(self):
        from sglang.benchmark.eval_harness import merge_report

        lm_eval_results = {
            "results": {
                "gsm8k": {
                    "exact_match,strict-match": 0.62,
                    "exact_match_stderr,strict-match": 0.013,
                    "exact_match,flexible-extract": 0.68,
                    "exact_match_stderr,flexible-extract": 0.012,
                    "alias": "gsm8k",
                }
            },
            "n-samples": {"gsm8k": {"original": 1319, "effective": 1319}},
            "config": {"num_fewshot": 5},
        }
        perf = {
            "mean_ttft_ms": 12.0, "mean_itl_ms": 4.5,
            "output_throughput": 1000.0, "request_throughput": 2.0,
            "completed": 1319, "duration": 600.0,
            "total_output_tokens": 2_500_000,
            "mean_e2e_latency_ms": 500.0,
        }
        merged = merge_report(
            task_name="gsm8k", lm_eval_results=lm_eval_results, perf=perf,
            run_config={"backend": "sglang-oai", "request_rate": 4.0},
        )
        self.assertEqual(merged["task"], "gsm8k")
        self.assertIn("exact_match,strict-match", merged["accuracy"])
        self.assertAlmostEqual(merged["accuracy"]["exact_match,strict-match"], 0.62)
        self.assertAlmostEqual(merged["perf"]["mean_ttft_ms"], 12.0)
        self.assertEqual(merged["n_samples"]["effective"], 1319)
        self.assertEqual(merged["run"]["backend"], "sglang-oai")

    def test_write_report_appends(self):
        from sglang.benchmark.eval_harness import merge_report, write_report

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "out.jsonl"
            r1 = merge_report(task_name="t", lm_eval_results={
                "results": {"t": {"acc": 0.5}}, "n-samples": {"t": {"effective": 10}},
                "config": {"num_fewshot": 0},
            }, perf={"mean_ttft_ms": 1.0}, run_config={})
            r2 = merge_report(task_name="t", lm_eval_results={
                "results": {"t": {"acc": 0.6}}, "n-samples": {"t": {"effective": 10}},
                "config": {"num_fewshot": 0},
            }, perf={"mean_ttft_ms": 2.0}, run_config={})
            write_report(str(path), r1)
            write_report(str(path), r2)

            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertAlmostEqual(json.loads(lines[0])["accuracy"]["acc"], 0.5)
            self.assertAlmostEqual(json.loads(lines[1])["accuracy"]["acc"], 0.6)


class TestRunBenchEvalEndToEnd(unittest.TestCase):
    """End-to-end: aiohttp mock mimics SGLang /v1/completions; run_bench_eval
    drives simple_evaluate through BenchServingLM and produces a merged report
    with both accuracy and perf fields populated."""

    def test_end_to_end_with_mock_server(self):
        import asyncio as _asyncio
        from aiohttp import web

        async def completions(request):
            resp = {
                "id": "cmpl", "object": "text_completion",
                "choices": [{"text": " reasoning... #### 42",
                             "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 8, "total_tokens": 12},
            }
            return web.Response(text=json.dumps(resp), content_type="application/json")

        async def models(request):
            return web.json_response({"data": [{"id": "mock-model"}]})

        async def flush(request):
            return web.Response(text="ok")

        async def _run():
            app = web.Application()
            app.router.add_post("/v1/completions", completions)
            app.router.add_get("/v1/models", models)
            app.router.add_get("/flush_cache", flush)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            try:
                from sglang.bench_eval import run_bench_eval

                with tempfile.TemporaryDirectory() as d:
                    out_file = Path(d) / "out.jsonl"
                    # run_bench_eval is synchronous and calls asyncio.run()
                    # internally via BenchServingLM.generate_until. Run it in
                    # a thread so it sees no running event loop.
                    loop = _asyncio.get_running_loop()
                    report = await loop.run_in_executor(
                        None,
                        lambda: run_bench_eval(
                            task="gsm8k",
                            base_url=f"http://127.0.0.1:{port}",
                            backend="sglang-oai",
                            model="mock-model",
                            tokenizer_path="hf-internal-testing/llama-tokenizer",
                            num_fewshot=0,
                            limit=3,
                            max_gen_toks=32,
                            request_rate=float("inf"),
                            max_concurrency=2,
                            apply_chat_template=False,
                            enable_thinking=False,
                            output_file=str(out_file),
                            include_per_doc=False,
                        ),
                    )
                    return report, out_file.read_text()
            finally:
                await runner.cleanup()

        report, jsonl = _asyncio.run(_run())
        self.assertEqual(report["task"], "gsm8k")
        # gsm8k's filter pipeline produces two tagged variants.
        self.assertIn("exact_match,strict-match", report["accuracy"])
        self.assertIn("exact_match,flexible-extract", report["accuracy"])
        self.assertGreater(report["perf"]["mean_ttft_ms"], 0.0)
        self.assertEqual(report["perf"]["completed"], 3)
        self.assertEqual(report["n_samples"].get("effective"), 3)
        self.assertEqual(len(jsonl.strip().splitlines()), 1)


class TestBenchEvalCLI(unittest.TestCase):
    def test_parser_accepts_all_flags(self):
        from sglang.bench_eval import build_parser

        args = build_parser().parse_args([
            "--task", "gsm8k",
            "--base-url", "http://localhost:30000",
            "--backend", "sglang-oai",
            "--model", "mock",
            "--tokenizer", "mock",
            "--num-fewshot", "5",
            "--limit", "10",
            "--max-gen-toks", "512",
            "--request-rate", "8.0",
            "--max-concurrency", "16",
            "--apply-chat-template",
            "--enable-thinking",
            "--fewshot-as-multiturn",
            "--output-file", "out.jsonl",
            "--include-per-doc",
        ])
        self.assertEqual(args.task, "gsm8k")
        self.assertEqual(args.num_fewshot, 5)
        self.assertEqual(args.max_gen_toks, 512)
        self.assertTrue(args.apply_chat_template)
        self.assertTrue(args.enable_thinking)
        self.assertTrue(args.fewshot_as_multiturn)
        self.assertTrue(args.include_per_doc)
        self.assertAlmostEqual(args.request_rate, 8.0)


if __name__ == "__main__":
    unittest.main()
