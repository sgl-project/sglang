"""Verify NPU evaluation routing without launching remote servers."""

import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu-intel")


class TestNpuEvalDispatch(CustomTestCase):
    def setUp(self):
        remote = ModuleType("sglang.test.ascend.e2e.test_npu_multi_node_utils")
        remote.SERVICE_PORT = 30000
        remote.check_role = lambda **kwargs: lambda function: function
        for name in (
            "kill_process_group",
            "launch_pd_mix_node",
            "launch_pd_separation_node",
            "launch_router",
            "wait_server_ready",
        ):
            setattr(remote, name, Mock())
        process_utils = ModuleType("sglang.srt.utils")
        process_utils.kill_process_tree = Mock()
        path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/test/ascend/e2e/test_npu_accuracy_utils.py"
        )
        with patch.dict(
            sys.modules,
            {remote.__name__: remote, process_utils.__name__: process_utils},
        ):
            self.evaluate = runpy.run_path(str(path))["run_accuracy_benchmark"]

    def test_supported_benchmarks_bypass_evalscope(self):
        from sgl_eval.registry import list_evals

        names = [spec.name for spec in list_evals()] + ["gpqa_diamond"]
        for name in names:
            with self.subTest(dataset=name):
                evaluate = Mock(return_value={"accuracy": 0.9})
                with patch.dict(self.evaluate.__globals__, run_sgl_eval=evaluate):
                    metrics = self.evaluate(
                        "localhost",
                        30000,
                        "test-model",
                        [name],
                        limit=10,
                        generation_config={
                            "max_tokens": 32768,
                            "temperature": 0.6,
                            "top_k": 20,
                            "extra_body": {
                                "chat_template_kwargs": {"enable_thinking": True}
                            },
                        },
                    )
                self.assertEqual(metrics, {"accuracy": 0.9})
                args = evaluate.call_args.args[0]
                self.assertEqual(
                    args.eval_name, "gpqa" if name == "gpqa_diamond" else name
                )
                self.assertEqual(args.max_tokens, 32768)
                self.assertEqual(args.top_k, 20)
                self.assertEqual(args.chat_template_kwargs, {"enable_thinking": True})

    def test_supported_benchmarks_reject_legacy_prompt_overrides(self):
        with self.assertRaisesRegex(ValueError, "belongs to sgl-eval"):
            self.evaluate(
                "localhost",
                30000,
                "test-model",
                ["aime25"],
                dataset_args={"few_shot_num": 5},
            )

    def test_mixed_benchmarks_require_separate_accuracy_gates(self):
        with self.assertRaisesRegex(ValueError, "separately"):
            self.evaluate("localhost", 30000, "test-model", ["gpqa_diamond", "mmmu"])


if __name__ == "__main__":
    unittest.main()
