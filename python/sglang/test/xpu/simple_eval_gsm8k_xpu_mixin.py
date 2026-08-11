"""simple-evals GSM8K accuracy mixin for Intel XPU nightly tests.

Launches an SGLang server with XPU flags, then calls ``sglang.test.run_eval``
with ``eval_name="gsm8k"`` so the ``simple_eval_gsm8k.GSM8KEval`` evaluator
scores the run.

Subclasses set ``model``, ``tp_size``, ``accuracy``, and may override
``other_args`` / ``env`` / ``num_examples`` / ``num_threads``.
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    write_github_step_summary,
)
from sglang.test.xpu.test_xpu_utils import write_results_to_github_step_summary


class SimpleEvalGSM8KXPUMixin(ABC):
    model: str = ""
    tp_size: int = 1

    timeout_for_server_launch = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    other_args: list[str] = [
        "--device",
        "xpu",
        "--attention-backend",
        "intel_xpu",
        "--dtype",
        "bfloat16",
        "--trust-remote-code",
        "--disable-overlap-schedule",
        "--disable-radix-cache",
    ]
    env: dict | None = None

    server_cmd: str = ""
    # Subset that fits run_suite's per-file timeout; set None for the full set.
    num_examples: int | None = 200
    # Single-stream: intel_xpu attention at TP>=2 wedges the Level Zero driver
    # on concurrent prefill. Subclasses may bump on hardware that handles it.
    num_threads: int = 1
    # Short generations reduce prefill->decode handoffs that trip the same wedge.
    max_tokens: int = 512

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = {**os.environ, **(cls.env or {})}
        args = list(cls.other_args) + ["--tp-size", str(cls.tp_size)]
        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=cls.timeout_for_server_launch,
                other_args=args,
                env=env,
            )
            cls.server_cmd = subprocess.list2cmdline(cls.process.args)
        except Exception as e:
            write_github_step_summary(f"Failed to launch server for {cls.model}: {e}")
            raise AssertionError(f"Test failed for {cls.model}: {e}")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        accuracy_threshold = getattr(self, "accuracy", 0.0)
        output_throughput_threshold = getattr(self, "output_throughput", 0.0)

        model_metrics = {
            "server": self.server_cmd,
            "client": "simple_eval_gsm8k",
            "accuracy_threshold": getattr(self, "accuracy", "N/A"),
            "output_throughput_threshold": getattr(self, "output_throughput", "N/A"),
        }

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                num_examples=self.num_examples,
                num_threads=self.num_threads,
                max_tokens=self.max_tokens,
            )
            metrics = run_eval(args)
            model_metrics["accuracy"] = metrics["score"]
            model_metrics["output_throughput"] = metrics.get("output_throughput")
            model_metrics["latency"] = metrics["latency"]
            self.assertGreaterEqual(
                metrics["score"],
                accuracy_threshold,
                f'Accuracy of {self.model} is {metrics["score"]}, '
                f"is lower than {accuracy_threshold}",
            )
            if "output_throughput" in metrics:
                self.assertGreaterEqual(
                    metrics["output_throughput"],
                    output_throughput_threshold,
                    f"Output throughput of {self.model} is "
                    f'{metrics["output_throughput"]}, is lower than '
                    f"{output_throughput_threshold}",
                )
        except Exception as e:
            model_metrics["error"] = str(e)
            self.fail(f"Test failed for {self.model}: {e}")
        finally:
            write_results_to_github_step_summary({self.model: model_metrics})
