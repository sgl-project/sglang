import multiprocessing
import random
import time
import unittest

import requests
from transformers import AutoConfig, AutoTokenizer

import sglang as sgl
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestEAGLEEngine(unittest.TestCase):

    def test_eagle_accuracy(self):
        prompt = "Today is a sunny day and I like"
        target_model_path = "meta-llama/Llama-2-7b-chat-hf"
        speculative_draft_model_path = "lmzheng/sglang-EAGLE-llama2-chat-7B"

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=target_model_path,
            speculative_draft_model_path=speculative_draft_model_path,
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=4,
            speculative_num_draft_tokens=16,
        )
        out1 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        engine = sgl.Engine(model_path=target_model_path)
        out2 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        print("==== Answer 1 ====")
        print(out1)

        print("==== Answer 2 ====")
        print(out2)
        self.assertEqual(out1, out2)

    def test_eagle_end_check(self):
        prompt = "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nToday is a sunny day and I like [/INST]"
        target_model_path = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(target_model_path)
        speculative_draft_model_path = "lmzheng/sglang-EAGLE-llama2-chat-7B"

        sampling_params = {
            "temperature": 0,
            "max_new_tokens": 1024,
            "skip_special_tokens": False,
        }

        engine = sgl.Engine(
            model_path=target_model_path,
            speculative_draft_model_path=speculative_draft_model_path,
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=4,
            speculative_num_draft_tokens=16,
        )
        out1 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()
        print("==== Answer 1 ====")
        print(repr(out1))
        tokens = tokenizer.encode(out1, truncation=False)
        assert tokenizer.eos_token_id not in tokens


prompts = [
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nToday is a sunny day and I like[/INST]"
    '[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nWhat are the mental triggers in Jeff Walker\'s Product Launch Formula and "Launch" book?[/INST]',
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nSummarize Russell Brunson's Perfect Webinar Script...[/INST]",
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nwho are you?[/INST]",
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nwhere are you from?[/INST]",
]


def process(server_url: str):
    time.sleep(random.uniform(0, 2))
    for prompt in prompts:
        url = server_url
        data = {
            "model": "base",
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1024,
            },
        }
        response = requests.post(url, json=data)
        assert response.status_code == 200


def abort_process(server_url: str):
    for prompt in prompts:
        try:
            time.sleep(1)
            url = server_url
            data = {
                "model": "base",
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1024,
                },
            }
            # set timeout = 1s,mock disconnected
            requests.post(url, json=data, timeout=1)
        except:
            pass


class TestEAGLELaunchServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        speculative_draft_model_path = "lmzheng/sglang-EAGLE-llama2-chat-7B"
        cls.model = "meta-llama/Llama-2-7b-chat-hf"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                speculative_draft_model_path,
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "4",
                "--speculative-num-draft-tokens",
                "16",
                "--served-model-name",
                "base",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_eagle_server_concurrency(self):
        concurrency = 4
        processes = [
            multiprocessing.Process(
                target=process,
                kwargs={"server_url": self.base_url + "/generate"},
            )
            for _ in range(concurrency)
        ]
        for worker in processes:
            worker.start()
        for p in processes:
            p.join()

    def test_eagle_server_request_abort(self):
        concurrency = 4
        processes = [
            multiprocessing.Process(
                target=process,
                kwargs={"server_url": self.base_url + "/generate"},
            )
            for _ in range(concurrency)
        ] + [
            multiprocessing.Process(
                target=abort_process,
                kwargs={"server_url": self.base_url + "/generate"},
            )
            for _ in range(concurrency)
        ]
        for worker in processes:
            worker.start()
        for p in processes:
            p.join()


if __name__ == "__main__":
    unittest.main()
