import json
import threading
import requests
import unittest
import time
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

responses = []
def send_requests(url, **kwargs):
    response = requests.post(DEFAULT_URL_FOR_TEST + url, json=kwargs)
    responses.append(response)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestAscendApi(CustomTestCase):
    """Testcase: Verify the functionality of /abort_request API to terminate a running /generate request on Ascend backend.

    [Test Category] Interface
    [Test Target] /abort_request
    """
    @classmethod
    def setUpClass(cls):
        cls.model = "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
        other_args = (
            [
                "--attention-backend",
                "ascend",
            ]
        )
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_api_abort_request(self):
        # Create thread 1: Send a long-running /generate request with rid=10086
        thread1 = threading.Thread(target=send_requests, args=('/generate',), kwargs={'rid': '10086', 'text': 'who are you?', 'sampling_params': {'temperature': 0.0, 'max_new_tokens': 1024}})
        # Create thread 2: Send an /abort_request to terminate the request with rid=10086
        thread2 = threading.Thread(target=send_requests, args=('/abort_request',), kwargs={'rid': "10086"})
        thread1.start()
        time.sleep(0.5)
        thread2.start()
        thread1.join()
        thread2.join()
        print(responses[1].text)


if __name__ == "__main__":

    unittest.main()
