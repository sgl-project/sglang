"""piecewise CUDA graph + chunked prefill.

Piecewise CUDA graph is enabled by default in current sglang. This
fixture just verifies that a standard server launch with chunked prefill
forced small *does not* disable piecewise CG. The point is to keep
piecewise CG on the well-trodden chunked-prefill path during the
refactor, not to test a specific failure mode.

The fixture deliberately does not pass ``--disable-piecewise-cuda-graph``
and otherwise uses the bare ``ChunkedRefactorTestBase`` defaults.

Reference config sources:
  - Existing piecewise CG tests for comparison (none of which exercise
    explicit ``--chunked-prefill-size`` — that's the gap we close):
    ``test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py``
    ``test/registered/piecewise_cuda_graph/test_pcg_glm5_fp4.py``
  - Default-on flag wiring in ``server_args.py``
    (``disable_piecewise_cuda_graph: bool = False``)

GPU requirement: 1 small GPU.

Not registered with CI. Run by hand from
``test/manual/chunked_prefill/``.
"""

import unittest
from test.manual.chunked_prefill.common import ChunkedRefactorTestBase

from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestChunkedFeaturePiecewiseCudaGraph(ChunkedRefactorTestBase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    # Empty feature_args is intentional: piecewise CG is the default, so the
    # whole point of this fixture is "default flags + small chunk_size".
    feature_args = []


if __name__ == "__main__":
    unittest.main()
