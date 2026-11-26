"""Manual AMX validation for AutoRound W8A16 and mixed W4/W8 checkpoints."""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import cpu_has_amx_support, kill_process_tree
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

AUTOROUND_CPU_MIXED_MODELS = (
    "INCModel4/Phi-4-multimodal-instruct-W8A16-AutoRound",
    "INCModel4/Phi-4-multimodal-instruct-W4A16-Mixed-AutoRound",
)


@unittest.skipUnless(
    cpu_has_amx_support(),
    "AutoRound W4A8/W8A8 CPU inference requires the Intel AMX CPU backend.",
)
class TestAutoRoundCPUW4W8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_mmlu(self):
        for model in AUTOROUND_CPU_MIXED_MODELS:
            with self.subTest(model=model):
                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    env={"SGLANG_USE_CPU_ENGINE": "1"},
                    other_args=[
                        "--trust-remote-code",
                        "--quantization",
                        "auto-round",
                        "--device",
                        "cpu",
                        "--dtype",
                        "bfloat16",
                        "--disable-overlap-schedule",
                        "--context-length",
                        "4096",
                        "--mem-fraction-static",
                        "0.5",
                    ],
                    device="cpu",
                )
                try:
                    args = SimpleNamespace(
                        base_url=self.base_url,
                        model=model,
                        eval_name="mmlu",
                        num_examples=128,
                        num_threads=32,
                        max_tokens=256,
                        device="cpu",
                    )
                    metrics = run_sgl_eval(args)
                    self.assertGreaterEqual(metrics["score"], 0.5)
                finally:
                    kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
