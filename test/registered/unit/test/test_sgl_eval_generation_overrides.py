import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sgl_eval.registry import EvalSpec
from sgl_eval.types import Example, ExampleResult, GenConfig, RunResult, Sample

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestSglEvalGenerationOverrides(CustomTestCase):
    def test_top_k_preserves_other_default_request_options(self):
        for extra_body in (None, {"ignore_eos": True, "top_k": 7}):
            with (
                self.subTest(extra_body=extra_body),
                tempfile.TemporaryDirectory() as tmp,
            ):
                original_body = None if extra_body is None else dict(extra_body)
                default_gen = GenConfig(extra_body=extra_body)
                result = RunResult(
                    name="fixture",
                    per_example=[
                        ExampleResult(
                            example=Example(id="one", inputs={}, target="42"),
                            samples=[Sample(text="42", finish_reason="stop")],
                            scores=[1.0],
                            extracted=["42"],
                        )
                    ],
                    aggregate={"score": 1.0},
                    latency=1.0,
                    num_examples=1,
                    n_repeats=1,
                )
                run = Mock(return_value=result)
                spec = EvalSpec("fixture", "test", "local fixture", default_gen, 1, run)
                sampler = Mock(model="fixture-model")
                args = SimpleNamespace(
                    eval_name="fixture",
                    top_k=20,
                    sgl_eval_out_dir=tmp,
                    base_url="http://127.0.0.1:1/v1",
                )
                with (
                    patch("sgl_eval.registry.get", return_value=spec),
                    patch(
                        "sgl_eval.sampler.ChatCompletionSampler", return_value=sampler
                    ),
                    patch("sglang.test.test_utils.dump_metric"),
                ):
                    metrics = run_sgl_eval(args)
                expected = {**(extra_body or {}), "top_k": 20}
                self.assertEqual(run.call_args.kwargs["gen"].extra_body, expected)
                self.assertEqual(default_gen.extra_body, original_body)
                recorded = json.loads(
                    Path(metrics["sgl_eval_metrics_path"]).read_text()
                )
                self.assertEqual(recorded["generation"]["top_k"], 20)
                if extra_body:
                    self.assertTrue(recorded["generation"]["ignore_eos"])
                sampler.abort.assert_called_once()


if __name__ == "__main__":
    unittest.main()
