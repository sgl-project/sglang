import random
import threading
import time
import unittest

import requests

import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestEAGLEEngine(unittest.TestCase):

    def test_eagle_accuracy(self):
        prompt = "Today is a sunny day and I like"
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        # Get the reference output
        ref_engine = sgl.Engine(model_path=DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST)
        ref_output = ref_engine.generate(prompt, sampling_params)["text"]
        ref_engine.shutdown()

        # Launch EAGLE engine
        engine = sgl.Engine(
            model_path=DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            speculative_draft_model_path=DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
            speculative_algorithm="EAGLE",
            speculative_num_steps=5,
            speculative_eagle_topk=8,
            speculative_num_draft_tokens=64,
        )

        # Test the output of EAGLE engine is the same as normal engine
        out1 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()
        print(f"{out1=}, {ref_output=}")
        self.assertEqual(out1, ref_output)

        # Test the output of EAGLE engine does not contain unexpected EOS
        prompt = "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nToday is a sunny day and I like [/INST]"
        sampling_params = {
            "temperature": 0,
            "max_new_tokens": 1024,
            "skip_special_tokens": False,
        }

        tokenizer = get_tokenizer(DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST)
        out2 = engine.generate(prompt, sampling_params)["text"]
        tokens = tokenizer.encode(out2, truncation=False)
        assert tokenizer.eos_token_id not in tokens

        # Shutdown the engine
        engine.shutdown()


prompts = [
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nToday is a sunny day and I like[/INST]"
    '[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nWhat are the mental triggers in Jeff Walker\'s Product Launch Formula and "Launch" book?[/INST]',
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nSummarize Russell Brunson's Perfect Webinar Script...[/INST]",
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nwho are you?[/INST]",
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nwhere are you from?[/INST]",
]


class TestEAGLEServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
                "--speculative-num-steps",
                "5",
                "--speculative-eagle-topk",
                "8",
                "--speculative-num-draft-tokens",
                "64",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def send_request(self):
        time.sleep(random.uniform(0, 2))
        for prompt in prompts:
            url = (self.base_url,)
            data = {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1024,
                },
            }
            response = requests.post(url, json=data)
            assert response.status_code == 200

    def send_requests_abort(self):
        for prompt in prompts:
            try:
                time.sleep(1)
                url = (self.base_url,)
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
            except Exception as e:
                print(e)
                pass

    def test_eagle_server_request_abort(self):
        concurrency = 4
        threads = [
            threading.Thread(
                target=self.send_request,
                kwargs={"server_url": self.base_url + "/generate"},
            )
            for _ in range(concurrency)
        ] + [
            threading.Thread(
                target=self.send_requests_abort,
                kwargs={"server_url": self.base_url + "/generate"},
            )
            for _ in range(concurrency)
        ]
        for worker in threads:
            worker.start()
        for p in threads:
            p.join()


if __name__ == "__main__":
    unittest.main()
