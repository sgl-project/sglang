"""simple-evals GSM8K accuracy mixin for Intel XPU nightly tests.

Mirrors the AMD/NVIDIA nightly flow (``test_gsm8k_eval_amd.py`` /
``test_text_models_gsm8k_eval.py``): launch an SGLang server with XPU
flags, then call ``sglang.test.run_eval`` with ``eval_name="gsm8k"`` so
the same ``simple_eval_gsm8k.GSM8KEval`` evaluator scores every backend.

Replaces the deprecated SGL-frontend ``few_shot_gsm8k`` path used by
``GSM8KXPUMixin`` (which had a data-leakage bug and a regex that dropped
signs and decimals).

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
    # 200 questions matches the limit used by the XPU 70B lm-eval YAML and
    # fits inside run_suite's per-file timeout when num_threads=1 keeps
    # throughput low. Subclasses on cheaper-per-token hardware (TP=1, no
    # Level Zero wedge) can raise this or set None for the full 1319-question
    # GSM8K test set, matching the AMD/NVIDIA nightly defaults.
    num_examples: int | None = 200
    # Single-stream eval: intel_xpu attention at TP>=2 wedges the Level Zero
    # driver in ur_command_list_manager::appendUSMMemcpy on concurrent prefill.
    # Subclasses on hardware that handles parallel prefill cleanly may bump.
    num_threads: int = 1
    # Short generations reduce the rate of prefill->decode->prefill handoffs,
    # which is what trips the same Level Zero wedge on TP>=2 (observed at the
    # default 2048; 512 matches the original few_shot_gsm8k limit and is still
    # enough for GSM8K CoT answers).
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
