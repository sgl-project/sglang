"""DeepSeek-V4.1-Flash + DSPARK accuracy on four GB300 GPUs: bounded replay BCG.

TP4 + EP4 with decoder SWA bounded replay inside breakable prefill graphs,
using the default attention backend.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.dsv41_flash_common import TP4_EP4_ARGS, DSV41FlashAccuracyMixin
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=600, stage="base-c", runner_config="4-gpu-gb300")


class TestDSV41FlashTP4EP4DSparkBoundedReplayBCG(
    DSV41FlashAccuracyMixin, CustomTestCase
):
    """Exercise bounded late-layer replay inside the breakable prefill graph."""

    parallel_args = TP4_EP4_ARGS
    # Leave memory for the prefill graph pool. With 32 requests, the tail
    # capacity is 4096 rows, below the largest 8192-token capture shape.
    mem_fraction_static = "0.7"
    max_running_requests = "32"
    graph_args = [
        "--enable-decoder-swa-bounded-replay",
        "--cuda-graph-backend-prefill",
        "breakable",
        "--cuda-graph-max-bs-prefill",
        "8192",
        "--chunked-prefill-size",
        "8192",
    ]


if __name__ == "__main__":
    unittest.main()
