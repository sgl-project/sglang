"""speculative decoding (EAGLE) + chunked prefill.

Manual fixture for the chunked-prefill refactor accuracy net. Spec verify
forms the canonical last-chunk-in-flight scenario alongside the target
model's chunked prefill — spec accept/reject decisions can race the
chunked admission for the same request.

Server arg template borrowed from
``python/sglang/test/server_fixtures/eagle_fixture.py`` and
``test/registered/spec/eagle/test_eagle_infer_b.py``.

GPU requirement: 1 GPU (large; Llama-2-7b-chat-hf target + EAGLE draft).

Not registered with CI. Run by hand from
``test/manual/chunked_prefill/``.
"""

import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedRefactorTestBase
from sglang.test.test_utils import DEFAULT_DRAFT_MODEL_EAGLE, DEFAULT_TARGET_MODEL_EAGLE


class TestChunkedFeatureSpec(ChunkedRefactorTestBase):
    model = DEFAULT_TARGET_MODEL_EAGLE
    feature_args = [
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-draft-model-path",
        DEFAULT_DRAFT_MODEL_EAGLE,
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "8",
        "--speculative-num-draft-tokens",
        "64",
        "--mem-fraction-static",
        "0.7",
    ]


if __name__ == "__main__":
    unittest.main()
