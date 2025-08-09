import unittest
from types import SimpleNamespace

from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import CustomTestCase


class TestElasticEpGsm8k(CustomTestCase):

    # TODO Start the service manually
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1200,
            parallel=1200,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=TEST_PORT,
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        # TODO  Now some requests are lost
        self.assertGreater(metrics["accuracy"], 0.87)


import argparse
import sys

def parse_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port number for the server")
    args, remaining = parser.parse_known_args()
    global TEST_PORT
    TEST_PORT = args.port
    sys.argv = [sys.argv[0]] + remaining

parse_test_args()

if __name__ == "__main__":
    unittest.main()
