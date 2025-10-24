import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestMoERunnerTriton(CustomTestCase):
    BASE_URL = DEFAULT_URL_FOR_TEST
    TIMEOUT = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    DEFAULT_MODEL = DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT
    DEFAULT_EVAL_KWARGS = {
        "eval_name": "mmlu",
        "num_examples": 5,
        "num_threads": 1,
    }

    CONFIGS = {
        "tp2_torch_native": {
            "model": DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "triton",
                "--tp",
                "2",
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
                "--max-total-tokens",
                "2048",
            ],
        },
    }

    def _run_config(self, config: dict) -> None:
        model = config.get("model", self.DEFAULT_MODEL)
        other_args = config.get("other_args", [])
        eval_kwargs = self.DEFAULT_EVAL_KWARGS | config.get("eval_kwargs", {})

        process = popen_launch_server(
            model,
            self.BASE_URL,
            timeout=self.TIMEOUT,
            other_args=other_args,
        )
        try:
            args = SimpleNamespace(
                base_url=self.BASE_URL,
                model=model,
                **eval_kwargs,
            )
            metrics = run_eval(args)
            print(f"{metrics=}")
            self.assertGreaterEqual(metrics["score"], 0.0)
        finally:
            kill_process_tree(process.pid)


def _make_test(config_name: str, config: dict):
    def test(self):
        self._run_config(config)

    test.__name__ = f"test_{config_name}"
    return test


for _name, _config in TestMoERunnerTriton.CONFIGS.items():
    setattr(TestMoERunnerTriton, f"test_{_name}", _make_test(_name, _config))


if __name__ == "__main__":
    unittest.main()
