"""DeepSeek-V4.1-Flash + DSPARK accuracy on four GB300 GPUs: TRT-LLM bounded BCG.

TP4 + EP4 with TRT-LLM varlen attention and decoder SWA bounded replay inside
breakable prefill graphs.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.dsv41_flash_common import DSV41FlashAccuracyMixin
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=600, stage="base-c", runner_config="4-gpu-gb300")


class TestDSV41FlashTrtllmBoundedBCG(DSV41FlashAccuracyMixin, CustomTestCase):
    """TRT-LLM varlen attention through the full and bounded-tail graph breaks."""

    parallel_args = [
        "--attn-dp-size",
        "1",
        "--ep-size",
        "4",
        "--dsv4-attn-backend",
        "trtllm",
    ]
    max_running_requests = "64"
    graph_args = [
        "--cuda-graph-backend-prefill",
        "breakable",
        "--cuda-graph-max-bs-prefill",
        "4096",
        "--cuda-graph-max-seq-len-prefill",
        "16384",
        "--chunked-prefill-size",
        "4096",
    ]


if __name__ == "__main__":
    unittest.main()
