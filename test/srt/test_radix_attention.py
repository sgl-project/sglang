import unittest

import requests

from sglang.srt.environ import envs
from sglang.test.kits.radix_cache_server_kit import gen_radix_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    kill_process_tree,
    popen_launch_server,
)


class TestRadixCacheFCFS(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--chunked-prefill-size",
                "128",
                "--max-total-tokens",
                "20000",
                "--schedule-policy",
                "fcfs",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_radix_attention(self):
        nodes = gen_radix_tree()
        data = {
            "input_ids": [node["input_ids"] for node in nodes],
            "sampling_params": [
                {"max_new_tokens": node["decode_len"], "temperature": 0}
                for node in nodes
            ],
        }

        res = requests.post(self.base_url + "/generate", json=data)
        assert res.status_code == 200


@unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
class TestRadixCacheLPM(TestRadixCacheFCFS):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--chunked-prefill-size",
                "128",
                "--max-total-tokens",
                "20000",
                "--schedule-policy",
                "lpm",
            ],
        )


class TestRadixCacheNonOverlapLPM(TestRadixCacheFCFS):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-overlap-schedule",
                "--chunked-prefill-size",
                "128",
                "--max-total-tokens",
                "20000",
                "--schedule-policy",
                "lpm",
            ],
        )


if __name__ == "__main__":
    envs.SGLANG_TEST_RETRACT.set(True)
    unittest.main()
