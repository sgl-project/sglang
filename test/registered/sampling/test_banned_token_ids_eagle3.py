"""--sampling-banned-token-ids with EAGLE3 speculative decoding: the target and
draft are both masked, so no banned id appears in the output or the returned
top logprobs."""

import unittest

import test_banned_token_ids

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
)

register_cuda_ci(est_time=200, stage="extra-a", runner_config="1-gpu-small")


class TestBannedTokenIdsEagle3(test_banned_token_ids.TestBannedTokenIds):
    model = DEFAULT_TARGET_MODEL_EAGLE3
    extra_args = [
        "--dtype",
        "float16",
        "--attention-backend",
        "triton",
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        DEFAULT_DRAFT_MODEL_EAGLE3,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mem-fraction-static",
        "0.7",
    ]


if __name__ == "__main__":
    unittest.main()
