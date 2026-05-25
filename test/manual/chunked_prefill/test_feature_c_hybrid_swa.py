"""Feature (c): hybrid SWA + chunked prefill.

Sliding-window attention is only exercised when the prompt exceeds the
window size; below that, SWA degenerates to full attention and we measure
nothing useful. ``LONG_PROMPT_NUM_SHOTS=24`` brings gsm8k prompts to
~3000-4000 tokens which exceeds the default SWA window on the chosen
model. ``--chunked-prefill-size 256`` ensures the long prompt is chunked
*within* the window.

Server arg template borrowed from
``test/registered/sessions/test_streaming_session_swa.py::TestStreamingSessionSWA``.

GPU requirement: 1 large GPU (>= 40 GB; gpt-oss-20b uses ~25 GB at
mem-fraction 0.70).

Not registered with CI. Run by hand from
``test/manual/chunked_prefill/``.
"""

import unittest
from test.manual.chunked_prefill.common import (
    LONG_PROMPT_NUM_SHOTS,
    ChunkedRefactorTestBase,
)


class TestChunkedFeatureC_HybridSWA(ChunkedRefactorTestBase):
    model = "openai/gpt-oss-20b"
    num_shots = LONG_PROMPT_NUM_SHOTS
    feature_args = [
        "--mem-fraction-static",
        "0.70",
        "--disable-piecewise-cuda-graph",
    ]


if __name__ == "__main__":
    unittest.main()
