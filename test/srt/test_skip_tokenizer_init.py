import json
import unittest
from io import BytesIO

import requests
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestSkipTokenizerInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2-VL-2B"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model, use_fast=False)
        cls.processor = AutoProcessor.from_pretrained(cls.model, trust_remote_code=True)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--skip-tokenizer-init"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(
        self,
        prompt_text="The capital of France is",
        max_new_tokens=32,
        return_logprob=False,
        top_logprobs_num=0,
        n=1,
    ):
        input_ids = self.get_input_ids(prompt_text)
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0 if n == 1 else 0.5,
                    "max_new_tokens": max_new_tokens,
                    "n": n,
                    "stop_token_ids": [self.tokenizer.eos_token_id],
                },
                "stream": False,
                "return_logprob": return_logprob,
                "top_logprobs_num": top_logprobs_num,
                "logprob_start_len": 0,
            },
        )
        ret = response.json()
        print(json.dumps(ret, indent=2))

        def assert_one_item(item):
            if item["meta_info"]["finish_reason"]["type"] == "stop":
                self.assertEqual(
                    item["meta_info"]["finish_reason"]["matched"],
                    self.tokenizer.eos_token_id,
                )
            elif item["meta_info"]["finish_reason"]["type"] == "length":
                self.assertEqual(
                    len(item["token_ids"]), item["meta_info"]["completion_tokens"]
                )
                self.assertEqual(len(item["token_ids"]), max_new_tokens)
                self.assertEqual(item["meta_info"]["prompt_tokens"], len(input_ids))

                if return_logprob:
                    self.assertEqual(
                        len(item["meta_info"]["input_token_logprobs"]),
                        len(input_ids),
                        f'{len(item["meta_info"]["input_token_logprobs"])} mismatch with {len(input_ids)}',
                    )
                    self.assertEqual(
                        len(item["meta_info"]["output_token_logprobs"]),
                        max_new_tokens,
                    )

        # Determine whether to assert a single item or multiple items based on n
        if n == 1:
            assert_one_item(ret)
        else:
            self.assertEqual(len(ret), n)
            for i in range(n):
                assert_one_item(ret[i])

        print("=" * 100)

    def test_simple_decode(self):
        self.run_decode()

    def test_parallel_sample(self):
        self.run_decode(n=3)

    def test_logprob(self):
        for top_logprobs_num in [0, 3]:
            self.run_decode(return_logprob=True, top_logprobs_num=top_logprobs_num)

    def test_eos_behavior(self):
        self.run_decode(max_new_tokens=256)

    def get_input_ids(self, prompt_text) -> list[int]:
        input_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"][
            0
        ].tolist()
        return input_ids


class TestSkipTokenizerInitVLM(TestSkipTokenizerInit):
    @classmethod
    def setUpClass(cls):
        cls.image_url = "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
        response = requests.get(cls.image_url)
        cls.image = Image.open(BytesIO(response.content))
        cls.model = "Qwen/Qwen2-VL-2B"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model, use_fast=False)
        cls.processor = AutoProcessor.from_pretrained(cls.model, trust_remote_code=True)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--skip-tokenizer-init"],
        )

    def get_input_ids(self, _) -> list[int]:
        chat_template = get_chat_template_by_model_path(self.model)
        text = f"{chat_template.image_token}What is in this picture?"
        inputs = self.processor(
            text=[text],
            images=[self.image],
            return_tensors="pt",
        )

        return inputs.input_ids[0].tolist()


if __name__ == "__main__":
    unittest.main()
