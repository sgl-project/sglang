from types import SimpleNamespace
from sglang.test.test_utils import CustomTestCase
from sglang.test.few_shot_gsm8k import run_eval

from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.srt.utils import kill_process_tree


class TestMixtralAccuracy(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "/large_models/mistralai_Mixtral-8x7B-Instruct-v0.1"
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--tp",
            "2",
            "--mem-fraction-static",
            "0.9",
            "--context-length",
            "38768",
            "--quantization",
            "int4fp8_moe"
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=45 * 60,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=1400,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.60)
