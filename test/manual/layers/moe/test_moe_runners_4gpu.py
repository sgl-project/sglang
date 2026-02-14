import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestMoERunner4GPU(CustomTestCase):
    BASE_URL = DEFAULT_URL_FOR_TEST
    TIMEOUT = 6000
    DEFAULT_EVAL_KWARGS = {
        "eval_name": "mmlu",
        "num_examples": 5,
        "num_threads": 1,
    }

    CONFIGS = {
        "moe_runner_cutlass_w4a8": {
            "model": "tencent/DeepSeek-V3.1-Terminus-W4AFP8",  # FP8 W8A8 MoE model
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "cutlass",
                "--attention-backend",
                "triton",
                "--sampling-backend",
                "pytorch",
                "--tp-size",
                "4",
            ],
        },
        "moe_runner_cutlass_w4a8_deepep_normal": {
            "model": "tencent/DeepSeek-V3.1-Terminus-W4AFP8",  # FP8 W8A8 MoE model
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "cutlass",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "normal",
                "--attention-backend",
                "triton",
                "--sampling-backend",
                "pytorch",
                "--tp-size",
                "4",
            ],
        },
        "moe_runner_cutlass_w4a8_deepep_ll": {
            "model": "tencent/DeepSeek-V3.1-Terminus-W4AFP8",  # FP8 W8A8 MoE model
            "env_overrides": {"SGLANG_DEEPEP_BF16_DISPATCH": "1"},
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "cutlass",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "low_latency",
                "--attention-backend",
                "triton",
                "--sampling-backend",
                "pytorch",
                "--tp-size",
                "4",
            ],
        },
    }

    def _run_config(self, config: dict) -> None:
        model = config["model"]
        other_args = config.get("other_args", [])
        eval_kwargs = self.DEFAULT_EVAL_KWARGS
        env = dict(os.environ)
        env["SGLANG_ENABLE_JIT_DEEPGEMM"] = "1"
        env["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
        env.update(config.get("env_overrides", {}))
        timeout = config.get("timeout", self.TIMEOUT)

        process = popen_launch_server(
            model,
            self.BASE_URL,
            timeout=timeout,
            other_args=other_args,
            env=env,
        )
        try:
            args = SimpleNamespace(
                base_url=self.BASE_URL,
                model=model,
                **eval_kwargs,
            )
            metrics = run_eval(args)
            print(f"{metrics=}")
            self.assertGreaterEqual(metrics["score"], 0.48)
        finally:
            kill_process_tree(process.pid)


for _name, _cfg in TestMoERunner4GPU.CONFIGS.items():
    setattr(
        TestMoERunner4GPU,
        f"test_{_name}",
        (lambda self, cfg=_cfg: self._run_config(cfg)),
    )


if __name__ == "__main__":
    unittest.main()
