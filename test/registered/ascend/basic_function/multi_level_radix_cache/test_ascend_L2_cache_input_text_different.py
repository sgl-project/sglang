import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestAscendL2CacheInputTextDifferent(CustomTestCase):
    """Testcase: Test enable L2 cache(--enable-hierarchical-cache), inputting different text inference requests
            will not reuse the same text.
   [Test Category] Parameter
   [Test Target] --enable-hierarchical-cache
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_32B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            0.8,
            "--tp-size",
            2,
            "--enable-hierarchical-cache",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_L2_cache_text_different(self):
        texts = [
            "Marie ordered one chicken meal that costs $12, 5 packs of milk that costs $3 each, 4 apples that cost "
            "$1.50 each, and some boxes of pizza. Marie paid a total of $50. How many boxes of pizza did Marie order"
            " if each box costs $8.50?Marie ordered one chicken meal that costs $12, 5 packs of milk that costs "
            "$3 each, 4 apples that cost $1.50 each, and some boxes of pizza. Marie paid a total of $50. "
            "How many boxes of pizza did Marie order if each box costs $8.50?Marie ordered one chicken meal that "
            "costs $12, 5 packs of milk that costs $3 each, 4 apples that cost $1.50 each, and some boxes of pizza. "
            "Marie paid a total of $50. How many boxes of pizza did Marie order if each box costs $8.50?Marie "
            "ordered one chicken meal that costs $12, 5 packs of milk that costs $3 each, 4 apples that cost $1.50 "
            "each, and some boxes of pizza. Marie paid a total of $50. How many boxes of pizza did Marie order if "
            "each box costs $8.50?Marie ordered one chicken meal that costs $12, 5 packs of milk that costs $3 each,"
            " 4 apples that cost $1.50 each, and some boxes of pizza. Marie paid a total of $50. How many boxes of "
            "pizza did Marie order if each box costs $8.50?Marie ordered one chicken meal that costs $12, 5 packs "
            "of milk that costs $3 each, 4 apples that cost $1.50 each, and some boxes of pizza. Marie paid a total "
            "of $50. How many boxes of pizza did Marie order if each box costs $8.50?Marie ordered one chicken meal "
            "that costs $12, 5 packs of milk that costs $3 each, 4 apples that cost $1.50 each, and some boxes of "
            "pizza. Marie paid a total of $50. How many boxes of pizza did Marie order if each box costs $8.50?"
            "Marie ordered one chicken meal that costs $12, 5 packs of milk that costs $3 each, 4 apples that cost "
            "$1.50 each, and some boxes of pizza. Marie paid a total of $50. How many boxes of pizza did Marie order"
            " if each box costs $8.50?",
            "Mishka bought 3 pairs of shorts, 3 pairs of pants, and 3 pairs of shoes. One pair of shorts costs "
            "$16.50. One pair of pants costs $22.50 and one pair of shoes costs $42. How many dollars did Mishka "
            "spend on all the clothing items?Mishka bought 3 pairs of shorts, 3 pairs of pants, and 3 pairs of "
            "shoes. One pair of shorts costs $16.50. One pair of pants costs $22.50 and one pair of shoes costs $42."
            " How many dollars did Mishka spend on all the clothing items?Mishka bought 3 pairs of shorts, 3 pairs "
            "of pants, and 3 pairs of shoes. One pair of shorts costs $16.50. One pair of pants costs $22.50 and one"
            " pair of shoes costs $42. How many dollars did Mishka spend on all the clothing items?Mishka bought 3 "
            "pairs of shorts, 3 pairs of pants, and 3 pairs of shoes. One pair of shorts costs $16.50. One pair of "
            "pants costs $22.50 and one pair of shoes costs $42. How many dollars did Mishka spend on all the "
            "clothing items?Mishka bought 3 pairs of shorts, 3 pairs of pants, and 3 pairs of shoes. One pair of "
            "shorts costs $16.50. One pair of pants costs $22.50 and one pair of shoes costs $42. How many dollars "
            "did Mishka spend on all the clothing items?Mishka bought 3 pairs of shorts, 3 pairs of pants, and 3 "
            "pairs of shoes. One pair of shorts costs $16.50. One pair of pants costs $22.50 and one pair of shoes "
            "costs $42. How many dollars did Mishka spend on all the clothing items?Mishka bought 3 pairs of shorts,"
            " 3 pairs of pants, and 3 pairs of shoes. One pair of shorts costs $16.50. One pair of pants costs "
            "$22.50 and one pair of shoes costs $42. How many dollars did Mishka spend on all the clothing items?",
        ]
        for text in texts:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": text,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 10,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            # cached_tokens: Number of tokens cached in KV Cache
            self.assertTrue(int(response.json()["meta_info"]["cached_tokens"]) == 0)


if __name__ == "__main__":
    unittest.main()
