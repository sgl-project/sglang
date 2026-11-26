"""Run the ordinary MLA C8 model on eight A5 NPUs, without MTP or DeepEP.

From the repository root, with the NPU runtime environment loaded:
    python test/manual/ascend/llm_models/test_npu_deepseek_v3_modelslim_mla_fp8.py -v

DS_V3_MODELSLIM_MODEL_PATH can override the local ModelSlim C8 checkpoint.
"""

import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.test_utils import CustomTestCase


class TestDeepSeekV3ModelSlimMLAFP8(GSM8KAscendMixin, CustomTestCase):
    model = os.environ.get(
        "DS_V3_MODELSLIM_MODEL_PATH",
        "/mnt/share/weight/DeepSeek-V3.1-Terminus-w4a4c8-mxfp-0902",
    )
    accuracy = 0.95
    timeout_for_server_launch = 3600
    env = {
        **os.environ,
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "ASCEND_USE_FIA": "1",
        "SGLANG_USE_FIA_NZ": "1",
        "SGLANG_NPU_USE_MLAPO": "1",
    }
    other_args = [
        "--trust-remote-code",
        "--device",
        "npu",
        "--attention-backend",
        "ascend",
        "--dtype",
        "bfloat16",
        "--quantization",
        "modelslim",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--tp-size",
        "8",
        "--dp-size",
        "1",
        "--ep-size",
        "1",
        "--moe-a2a-backend",
        "none",
        "--page-size",
        "128",
        "--max-running-requests",
        "16",
        "--cuda-graph-max-bs-decode",
        "16",
        "--mem-fraction-static",
        "0.85",
        "--chunked-prefill-size",
        "1024",
    ]


if __name__ == "__main__":
    unittest.main()
