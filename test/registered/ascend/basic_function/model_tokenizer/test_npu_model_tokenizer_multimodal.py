import json
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestNpuModelTokenizerMultimodal(CustomTestCase):
    """The combination of model and tokenizer parameters was tested, and the streaming request inference was successful.

    [Test Category] Functional
    [Test Target] model & tokenizer on NPU
    --enable-multimodal; --revision; --model-impl
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--enable-multimodal",
            "--revision",
            "1.0.0",
            "--model-impl",
            "sglang",
        ]
        cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up after the test class by killing the server process and removing generated directories."""
        kill_process_tree(cls.process.pid)
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")

    def test_model_tokenizer_stream_request(self):
        text1 = "The capital of France is"
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": text1,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 64,
                },
                "stream": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        has_text = False
        # Stream With Reasoning
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data:") and not line.startswith("data: [DONE]"):
                    data = json.loads(line[6:])
                    if "text" in data and len(data["text"]) > 0:
                        has_text = True
        self.assertTrue(
            has_text,
            "The text is a stream response",
        )

        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        # The `--model-impl` configuration specifies assertions for sglang parameters.
        self.assertIn("type=Qwen3VLForConditionalGeneration", content)
        # Assertions for the --enable-multimodal parameter
        self.assertIn("Using sdpa as multimodal attention backend", content)
        self.out_log_file.close()
        self.err_log_file.close()


if __name__ == "__main__":
    unittest.main()
