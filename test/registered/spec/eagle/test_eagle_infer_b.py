import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.eagle_fixture import TestEAGLEServer
from sglang.test.test_utils import (
    DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_cuda_ci(est_time=473, suite="stage-b-test-small-1-gpu")


class TestEAGLERetract(TestEAGLEServer):
    @classmethod
    def setUpClass(cls):
        # These config helps find a leak.
        os.environ["SGLANG_CI_SMALL_KV_SIZE"] = "4500"
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
                5,
                "--speculative-eagle-topk",
                8,
                "--speculative-num-draft-tokens",
                64,
                "--mem-fraction-static",
                0.7,
                "--chunked-prefill-size",
                128,
                "--max-running-requests",
                64,
            ],
        )


class TestEAGLEServerTriton(TestEAGLEServer):
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
                5,
                "--speculative-eagle-topk",
                8,
                "--speculative-num-draft-tokens",
                64,
                "--mem-fraction-static",
                0.7,
                "--attention-backend",
                "triton",
                "--max-running-requests",
                8,
            ],
        )


class TestEAGLEServerPageSize(TestEAGLEServer):
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
                5,
                "--speculative-eagle-topk",
                1,
                "--speculative-num-draft-tokens",
                6,
                "--mem-fraction-static",
                0.7,
                "--chunked-prefill-size",
                128,
                "--max-running-requests",
                8,
                "--page-size",
                4,
                "--attention-backend",
                "flashinfer",
            ],
        )


class TestEAGLEServerPageSizeTopk(TestEAGLEServer):
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
                5,
                "--speculative-eagle-topk",
                8,
                "--speculative-num-draft-tokens",
                64,
                "--mem-fraction-static",
                0.7,
                "--chunked-prefill-size",
                128,
                "--max-running-requests",
                8,
                "--page-size",
                4,
                "--attention-backend",
                "flashinfer",
            ],
        )


if __name__ == "__main__":
    unittest.main()
