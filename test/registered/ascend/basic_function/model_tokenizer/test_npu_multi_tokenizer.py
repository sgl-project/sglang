import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    META_LLAMA_3_1_8B_INSTRUCT,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestMultiTokenizer(CustomTestCase):
    """Test multi-tokenizer worker performance on NPU.

    [Test Category] Performance
    [Test Target] --tokenizer-worker-num; TTFT latency
    """

    @classmethod
    def setUpClass(cls):
        cls.model = META_LLAMA_3_1_8B_INSTRUCT
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tokenizer-worker-num",
                8,
                "--mem-fraction-static",
                0.8,
                "--attention-backend",
                "ascend",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        gsm8k_num_shots = 8
        num_questions = 200
        args = SimpleNamespace(
            max_tokens=1024,
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            num_examples=num_questions,
            num_threads=128,
            gsm8k_data_path=None,
            num_shots=gsm8k_num_shots,
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.845)

    def test_batch_input_ids_routing(self):
        # Regression guard for sgl-project/sglang#29878 (introduced by #29214).
        #
        # A batch of pre-tokenized `input_ids` (no text / multimodal) is the one
        # case that takes the batch-tokenization path (_send_batch_request ->
        # BatchTokenizedGenerateReqInput). In multi-tokenizer mode this batch
        # must stamp each sub-request's `http_worker_ipc` so the scheduler can
        # route every reply back to its owning tokenizer worker. If it is missing,
        # the requests hang forever.
        #
        # The existing ttft test only sends *text*, so it never exercises this
        # path — this case does, and uses a short timeout so a routing hang
        # fails fast instead of stalling until the server launch timeout.
        batch_input_ids = [
            [1, 2, 3, 4, 5],
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
            [30, 31, 32, 33, 34],
        ]
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": batch_input_ids,
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)
        results = response.json()
        # Every batched request must get its reply routed back — not hang.
        self.assertEqual(len(results), len(batch_input_ids))
        for result in results:
            self.assertIn("text", result)


if __name__ == "__main__":
    unittest.main()
