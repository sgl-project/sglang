import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import GLM_5_1_W4A8_MODEL_PATH
from sglang.test.test_utils import CustomTestCase


class TestGLM51W4A8GraphWithDraftPrefetch(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify the inference accuracy of GLM-5.1-W4A8 on GSM8K with DraftPrefetch feature.

    [Test Category] Model
    [Test Target] GLM-5.1-W4A8
    [Test Config] Prefill+Decode, npu graph enabled, NextN speculative decoding, w4a8 quantization, dp attention, draft prefetch, pin memory
    """

    model = GLM_5_1_W4A8_MODEL_PATH
    accuracy = 0.8
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.87",
        "--attention-backend",
        "ascend",
        "--quantization",
        "modelslim",
        "--tp-size",
        "16",
        "--dp-size",
        "4",
        "--enable-dp-attention",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "4",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "5",
        "--speculative-draft-model-quantization",
        "unquant",
    ]

    env = {
        # copy from GSM8KAscendMixin
        **os.environ,
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "24",
        "USE_VLLM_CUSTOM_ALLREDUCE": "1",
        "HCCL_EXEC_TIMEOUT": "200",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_ENBLE_TORCH_COMILE": "1",
        "AUTO_USE_UC_MEMORY": "0",
        "P2P_HCCL_BUFFSIZE": "20",
        # feature test
        "HCCL_BUFFSIZE": "1000",
        "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS": "0",
        "SGLANG_NPU_ENABLE_PIN_MEMORY": "1",
    }


if __name__ == "__main__":
    unittest.main()
