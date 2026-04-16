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


if __name__ == "__main__":
    unittest.main()
