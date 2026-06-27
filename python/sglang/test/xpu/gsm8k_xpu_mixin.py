"""GSM8K few-shot accuracy mixin for Intel XPU nightly tests.

Models compose this mixin alongside ``CustomTestCase``; subclasses set
``model``, ``tp_size``, ``accuracy``, and may override ``other_args`` /
``env`` / ``num_questions`` / ``gsm8k_num_shots``.

The mixin launches an SGLang server with the supplied flags, runs
``few_shot_gsm8k.run_eval`` against it, and asserts ``accuracy >=
self.accuracy``.

Pattern mirrors ``python/sglang/test/ascend/gsm8k_ascend_mixin.py`` so
nightly result tables look the same across NPU and XPU.
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    write_github_step_summary,
)
from sglang.test.xpu.test_xpu_utils import write_results_to_github_step_summary


class GSM8KXPUMixin(ABC):
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
    gsm8k_num_shots: int = 8
    num_questions: int = 200
    # Single-stream eval: with intel_xpu attention at TP>=2, concurrent prefill
    # wedges the Level Zero driver inside ur_command_list_manager::appendUSMMemcpy.
    # Subclasses may override on hardware that handles parallel prefill cleanly.
    gsm8k_parallel: int = 1
    max_new_tokens: int = 512

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
            "client": "few_shot_gsm8k",
            "accuracy_threshold": getattr(self, "accuracy", "N/A"),
            "output_throughput_threshold": getattr(self, "output_throughput", "N/A"),
        }

        try:
            args = SimpleNamespace(
                num_shots=self.gsm8k_num_shots,
                data_path=None,
                num_questions=self.num_questions,
                max_new_tokens=self.max_new_tokens,
                parallel=self.gsm8k_parallel,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval(args)
            model_metrics["accuracy"] = metrics["accuracy"]
            model_metrics["output_throughput"] = metrics["output_throughput"]
            model_metrics["latency"] = metrics["latency"]
            self.assertGreaterEqual(
                metrics["accuracy"],
                accuracy_threshold,
                f'Accuracy of {self.model} is {metrics["accuracy"]}, '
                f"is lower than {accuracy_threshold}",
            )
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
