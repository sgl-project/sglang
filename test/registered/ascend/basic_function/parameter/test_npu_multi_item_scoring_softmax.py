import logging
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

# Qwen3 <|endoftext|> token ID, used as multi-item scoring delimiter.
_DELIMITER_TOKEN_ID = 151643
# Token IDs for "Yes" (9454) and "No" (2753), used as label_token_ids.
_LABEL_TOKEN_IDS = [9454, 2753]

_QUERY = "Is this the correct result of 1 plus 2? "
_ITEMS = ["It is 3", "It is 4", "It is 5"]


def send_score_request(
    base_url,
    query,
    items,
    label_token_ids,
    apply_softmax=False,
    item_first=False,
    timeout=180,
):
    """Send a POST request to /v1/score and return the raw Response."""
    return requests.post(
        url=f"{base_url}/v1/score",
        json={
            "query": query,
            "items": items,
            "label_token_ids": label_token_ids,
            "apply_softmax": apply_softmax,
            "item_first": item_first,
        },
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )


class TestMultiItemScoringBasic(CustomTestCase):
    """Test multi-item scoring: basic functionality, softmax normalization,
    tokenized input, and item_first parameter.

    [Test Category] Feature
    [Test Target] --multi-item-scoring-delimiter; /v1/score
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            model=QWEN3_32B_WEIGHTS_PATH,
            base_url=cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--tp-size",
                "4",
                "--disable-radix-cache",
                "--chunked-prefill-size",
                "-1",
                "--multi-item-scoring-delimiter",
                str(_DELIMITER_TOKEN_ID),
            ],
        )
        cls.tokenizer = get_tokenizer(QWEN3_32B_WEIGHTS_PATH)
        logger.info("Server and tokenizer ready.")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        logger.info("Server terminated.")

    def test_softmax_true_text_input(self):
        """apply_softmax=True with text input: verify structure, normalization, and semantic ordering.

        Checks:
          - HTTP 200, 'scores' field present, shape == (len(items), len(label_token_ids)).
          - Each sub-list sums to 1.0 (softmax guarantee).
          - Correct item ("It is 3"): Yes-prob > No-prob.
          - Wrong items ("It is 4", "It is 5"): Yes-prob < No-prob.
        """
        response = send_score_request(
            base_url=self.base_url,
            query=_QUERY,
            items=_ITEMS,
            label_token_ids=_LABEL_TOKEN_IDS,
            apply_softmax=True,
        )
        self.assertEqual(response.status_code, 200)
        scores = response.json()["scores"]

        self.assertEqual(len(scores), len(_ITEMS))
        for idx, score_list in enumerate(scores):
            self.assertEqual(len(score_list), len(_LABEL_TOKEN_IDS))
            self.assertTrue(all(isinstance(v, float) for v in score_list))
            self.assertAlmostEqual(
                sum(score_list),
                1.0,
                places=5,
                msg=f"scores[{idx}] must sum to 1.0 with apply_softmax=True.",
            )

        self.assertGreater(
            scores[0][0],
            scores[0][1],
            "Correct item 'It is 3': Yes-prob should exceed No-prob.",
        )
        self.assertLess(
            scores[1][0],
            scores[1][1],
            "Wrong item 'It is 4': Yes-prob should be less than No-prob.",
        )
        self.assertLess(
            scores[2][0],
            scores[2][1],
            "Wrong item 'It is 5': Yes-prob should be less than No-prob.",
        )
        logger.info("Softmax=True verified: normalization + semantic ordering correct.")

    def test_softmax_false_tokenized_input(self):
        """apply_softmax=False with pre-tokenized input: values in [0, 1].

        Also cross-checks that softmax=True and softmax=False produce different values
        for the same tokenized input, confirming the normalization path differs.
        """
        query_tokens = self.tokenizer.encode(_QUERY, add_special_tokens=False)
        items_tokens = [
            self.tokenizer.encode(item, add_special_tokens=False) for item in _ITEMS
        ]

        response_false = send_score_request(
            base_url=self.base_url,
            query=query_tokens,
            items=items_tokens,
            label_token_ids=_LABEL_TOKEN_IDS,
            apply_softmax=False,
        )
        self.assertEqual(response_false.status_code, 200)
        scores_false = response_false.json()["scores"]
        logger.info("scores (apply_softmax=False): %s", scores_false)

        for idx, score_list in enumerate(scores_false):
            for j, val in enumerate(score_list):
                self.assertIsInstance(val, float)
                self.assertGreaterEqual(
                    val, 0.0, f"scores_false[{idx}][{j}] must be >= 0.0."
                )
                self.assertLessEqual(
                    val, 1.0, f"scores_false[{idx}][{j}] must be <= 1.0."
                )

        response_true = send_score_request(
            base_url=self.base_url,
            query=query_tokens,
            items=items_tokens,
            label_token_ids=_LABEL_TOKEN_IDS,
            apply_softmax=True,
        )
        self.assertEqual(response_true.status_code, 200)
        scores_true = response_true.json()["scores"]
        logger.info("scores (apply_softmax=True): %s", scores_true)

        self.assertNotEqual(
            scores_false,
            scores_true,
            "apply_softmax=True and apply_softmax=False must produce different values.",
        )
        logger.info("Softmax=False vs Softmax=True distinction verified.")

    def test_item_first_flag(self):
        """item_first is ignored when --multi-item-scoring-delimiter is active.

        Verifies that item_first=True produces identical scores to item_first=False,
        confirming the parameter has no effect in multi-item scoring mode.
        """
        common_kwargs = dict(
            base_url=self.base_url,
            query=_QUERY,
            items=_ITEMS,
            label_token_ids=_LABEL_TOKEN_IDS,
        )

        resp_false = send_score_request(**common_kwargs, item_first=False)
        self.assertEqual(resp_false.status_code, 200)
        scores_false = resp_false.json()["scores"]

        resp_true = send_score_request(**common_kwargs, item_first=True)
        self.assertEqual(resp_true.status_code, 200)
        scores_true = resp_true.json()["scores"]

        self.assertEqual(
            scores_false,
            scores_true,
            "item_first should be ignored in multi-item scoring mode: scores must be identical.",
        )
        logger.info("item_first=True verified: scores identical to item_first=False.")


if __name__ == "__main__":
    unittest.main()
