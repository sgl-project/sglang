import json
import unittest
from multiprocessing import Process

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestBatchPenalizerE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--random-seed",
                "0",
            ),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(
        self,
        return_logprob=True,
        top_logprobs_num=5,
        return_text=True,
        n=1,
        **sampling_params,
    ):
        response = requests.post(
            self.base_url + "/generate",
            json={
                # prompt that is supposed to generate < 32 tokens
                "text": "<|start_header_id|>user<|end_header_id|>\n\nWhat is the answer for 1 + 1 = ?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "sampling_params": {
                    "max_new_tokens": 32,
                    "n": n,
                    **sampling_params,
                },
                "stream": False,
                "return_logprob": return_logprob,
                "top_logprobs_num": top_logprobs_num,
                "return_text_in_logprobs": return_text,
                "logprob_start_len": 0,
            },
        )
        print(json.dumps(response.json()))
        print("=" * 100)

    def test_default_values(self):
        self.run_decode()

    def test_mixed(self):
        """
        Sends two requests with one with penalizers disabled, and the other with penalizers enabled.
        This will cause two different {ScheduleBatch} to be initialized and eventually gets merged.

        Merging batch with penalizers enabled with enabled, or disabled is trivial. However disabled + enabled is not.
        This is because the penalizer will not be prepared if it is not required, then it will be prepared during the merge.

        This test triggers the merge of disabled + enabled.
        """

        processes = []

        p = Process(
            target=self.run_decode,
        )
        processes.append(p)
        p.start()

        p = Process(
            target=self.run_decode,
            kwargs={
                "frequency_penalty": 2,
                "min_new_tokens": 16,
                "presence_penalty": 2,
                "repetition_penalty": 2,
            },
        )
        processes.append(p)
        p.start()

        for p in processes:
            p.join()

    def test_frequency_penalty(self):
        self.run_decode(frequency_penalty=2)

    def test_min_new_tokens(self):
        self.run_decode(min_new_tokens=16)

    def test_presence_penalty(self):
        self.run_decode(presence_penalty=2)

    def test_repetition_penalty(self):
        self.run_decode(repetition_penalty=2)


if __name__ == "__main__":
    unittest.main()
