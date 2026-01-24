from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_amd_ci(est_time=313, suite="stage-b-test-small-1-gpu-amd")


class TestMixtralAccuracy(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--tp",
            "2",
            "--mem-fraction-static",
            "0.9",
            "--context-length",
            "38768",
            "--quantization",
            "quark_int4fp8_moe",
            # The default aiter attention backend raises segmentation faults and other errors - as quark_int4fp8_moe is not related to attention, let's just use triton here.
            "--attention-backend",
            "triton",
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
        self.assertGreater(metrics["accuracy"], 0.56)
