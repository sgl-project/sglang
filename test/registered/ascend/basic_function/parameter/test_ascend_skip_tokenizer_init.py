import json
import unittest
from io import BytesIO

import requests
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_IMAGE_URL,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestSkipTokenizerInit(CustomTestCase):
    """Testcase：Verify that for LLM models with the --skip-tokenizer-init parameter configured,
    the streaming/non-streaming inference, parallel sampling, log probability return functions,
    and EOS Token termination trigger function all work properly.

    [Test Category] Parameter
    [Test Target] --skip-tokenizer-init
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                [
                    "--skip-tokenizer-init",
                    "--stream-output",
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
                ]
            ),
        )
        cls.eos_token_id = [119690]
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model, use_fast=False)

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

        request = self.get_request_json(
            input_ids=input_ids,
            return_logprob=return_logprob,
            top_logprobs_num=top_logprobs_num,
            max_new_tokens=max_new_tokens,
            stream=False,
            n=n,
        )
        response = requests.post(
            self.base_url + "/generate",
            json=request,
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
                    len(item["output_ids"]), item["meta_info"]["completion_tokens"]
                )
                self.assertEqual(len(item["output_ids"]), max_new_tokens)
                self.assertEqual(item["meta_info"]["prompt_tokens"], len(input_ids))

                if return_logprob:
                    num_input_logprobs = len(input_ids) - request["logprob_start_len"]
                    if num_input_logprobs > len(input_ids):
                        num_input_logprobs -= len(input_ids)
                    self.assertEqual(
                        len(item["meta_info"]["input_token_logprobs"]),
                        num_input_logprobs,
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

    def run_decode_stream(self, return_logprob=False, top_logprobs_num=0, n=1):
        max_new_tokens = 32
        input_ids = self.get_input_ids("The capital of France is")
        requests.post(self.base_url + "/flush_cache")
        response = requests.post(
            self.base_url + "/generate",
            json=self.get_request_json(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                return_logprob=return_logprob,
                top_logprobs_num=top_logprobs_num,
                stream=False,
                n=n,
            ),
        )
        ret = response.json()
        print(json.dumps(ret))
        output_ids = ret["output_ids"]
        print("output from non-streaming request:")
        print(output_ids)
        print(self.tokenizer.decode(output_ids, skip_special_tokens=True))

        requests.post(self.base_url + "/flush_cache")
        response_stream = requests.post(
            self.base_url + "/generate",
            json=self.get_request_json(
                input_ids=input_ids,
                return_logprob=return_logprob,
                top_logprobs_num=top_logprobs_num,
                stream=True,
                n=n,
            ),
        )

        response_stream_json = []
        for line in response_stream.iter_lines():
            print(line)
            if line.startswith(b"data: ") and line[6:] != b"[DONE]":
                response_stream_json.append(json.loads(line[6:]))
        out_stream_ids = []
        for x in response_stream_json:
            out_stream_ids += x["output_ids"]
        print("output from streaming request:")
        print(out_stream_ids)
        print(self.tokenizer.decode(out_stream_ids, skip_special_tokens=True))

        assert output_ids == out_stream_ids

    def test_simple_decode(self):
        # Verify successful text generation for non-streaming inference with default parameters
        self.run_decode()

    def test_parallel_sample(self):
        # Verify successful text generation for non-streaming inference with default parameters
        self.run_decode(n=3)

    def test_logprob(self):
        # Verify logprob return content matches the configuration
        for top_logprobs_num in [0, 3]:
            self.run_decode(return_logprob=True, top_logprobs_num=top_logprobs_num)

    def test_eos_behavior(self):
        # Verify that the EOS Token function is triggered when the number of generated tokens reaches max_new_tokens
        self.run_decode(max_new_tokens=256)

    def test_simple_decode_stream(self):
        # Verify that the results of streaming inference are consistent with those of non-streaming inference.
        self.run_decode_stream()

    def get_input_ids(self, prompt_text) -> list[int]:
        input_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"][
            0
        ].tolist()
        return input_ids

    def get_request_json(
        self,
        input_ids,
        max_new_tokens=32,
        return_logprob=False,
        top_logprobs_num=0,
        stream=False,
        n=1,
    ):
        return {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0 if n == 1 else 0.5,
                "max_new_tokens": max_new_tokens,
                "n": n,
                "stop_token_ids": self.eos_token_id,
            },
            "stream": stream,
            "return_logprob": return_logprob,
            "top_logprobs_num": top_logprobs_num,
            "logprob_start_len": 0,
        }


class TestSkipTokenizerInitVLM(TestSkipTokenizerInit):
    """Testcase：Verify that for LLM models with the --skip-tokenizer-init parameter configured,
    the streaming/non-streaming inference, parallel sampling, log probability return functions,
    and EOS Token termination trigger function all work properly.
    """

    model = QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        image_path = DEFAULT_IMAGE_URL
        cls.image_url = "https://gh.llkk.cc/" + image_path
        response = requests.get(cls.image_url)
        cls.image = Image.open(BytesIO(response.content))
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model, use_fast=False)
        cls.processor = AutoProcessor.from_pretrained(cls.model, trust_remote_code=True)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--skip-tokenizer-init",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
        )
        cls.eos_token_id = [cls.tokenizer.eos_token_id]

    def get_input_ids(self, _prompt_text) -> list[int]:
        chat_template = get_chat_template_by_model_path(self.model)
        text = f"{chat_template.image_token}What is in this picture?"
        inputs = self.processor(
            text=[text],
            images=[self.image],
            return_tensors="pt",
        )

        return inputs.input_ids[0].tolist()

    def get_request_json(self, *args, **kwargs):
        ret = super().get_request_json(*args, **kwargs)
        ret["image_data"] = [self.image_url]
        # Do not try to calculate logprobs of image embeddings.
        ret["logprob_start_len"] = -1
        return ret


if __name__ == "__main__":
    unittest.main()
