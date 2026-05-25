"""Feature (h): piecewise CUDA graph + chunked prefill.

Piecewise CUDA graph is enabled by default in current sglang. This
fixture just verifies that a standard server launch with chunked prefill
forced small *does not* disable piecewise CG. The point is to keep
piecewise CG on the well-trodden chunked-prefill path during the
refactor, not to test a specific failure mode.

The fixture deliberately does not pass ``--disable-piecewise-cuda-graph``.

GPU requirement: 1 small GPU.

Not registered with CI. Run by hand from
``test/manual/chunked_prefill/``.
"""

import unittest

from test.manual.chunked_prefill.common import ChunkedRefactorTestBase

from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestChunkedFeatureH_PiecewiseCudaGraph(ChunkedRefactorTestBase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    # Empty feature_args is intentional: piecewise CG is the default, so the
    # whole point of this fixture is "default flags + small chunk_size".
    feature_args = []


if __name__ == "__main__":
    unittest.main()
