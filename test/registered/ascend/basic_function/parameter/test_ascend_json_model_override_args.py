import unittest

import requests
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (

    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestJsonModelOverrideArgs(CustomTestCase):
    """Testcase：Verify set --json-model-override-args = {"max_position_embeddings": 50},
    send tokens more than 50 the inference request is fail.

       [Test Category] Parameter
       [Test Target] --json-model-override-args
       """
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--json-model-override-args",
            '{"max_position_embeddings": 50}',
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_json_model_override_args(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The ancient Romans made significant contributions to various fields, "
                        "including law, philosophy, science, and literature. They were known "
                        "for their engineering achievements, such as the construction of the Colosseum and the Pantheon. "
                        "Their art and architecture were also highly esteemed, with the Colosseum being a symbol of "
                        "their power and influence. In science, they made important contributions to astronomy "
                        "and mathematics. Literature was also a major part of their culture, ",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("longer than the model's context length (50 tokens)", response.text)


if __name__ == "__main__":
    unittest.main()
