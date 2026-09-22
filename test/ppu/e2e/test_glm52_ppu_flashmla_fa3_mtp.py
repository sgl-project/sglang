import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# This test reproduces the crash:
#   ValueError: q.shape[0] (380) does not match qo_indptr[-1] (378)
# which occurs when:
#   - PPU platform with fa3 prefill + flashmla decode hybrid backends
#   - --speculative-attention-mode decode --enable-dp-attention
#   - 0.5.16 _create_flashmla_prefill_backend returns FlashMLABackend
#     instead of None, overriding draft_runner.attn_backend and routing
#     EXTEND (prefill) through FlashInferMLA's ragged prefill which
#     does not tolerate DP-attention padding.
#
# The fix gates _create_flashmla_prefill_backend to return None on PPU,
# preserving HybridAttnBackend (fa3 for EXTEND, flashmla for DRAFT_EXTEND_V2).

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "modelscope.cn/organization/T-HEAD/GLM-5.2-W8A8-INT8"
)
SERVED_MODEL_NAME = "GLM-5.2"
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 24000  # match --dist-timeout

# Local GSM8K dataset path — set GSM8K_DATA_PATH env var to avoid
# re-downloading the dataset on every run.
GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


class TestGlm52PpuFlashmlaFa3DpAttnMtp(GSM8KMixin, CustomTestCase):
    """E2E: GLM-5.2 EAGLE MTP on PPU with fa3 prefill + flashmla decode + DP-attn.

    Verifies the server boots without the qo_indptr crash and speculative
    decoding produces correct GSM8K accuracy.
    """

    # --- GSM8KMixin config ---
    model = SERVED_MODEL_NAME
    gsm8k_score_threshold = 0.50
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 1.0
    gsm8k_data_path = GSM8K_DATA_PATH

    # --- server launch config ---
    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "16",
            "--mem-fraction-static",
            "0.9",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "2",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "3",
            "--deepep-mode",
            "auto",
            "--enable-dp-attention",
            "--dp-size",
            "8",
            "--quantization",
            "w8a8_int8",
            "--trust-remote-code",
            "--max-running-requests",
            "40",
            "--num-continuous-decode-steps",
            "1",
            "--dist-timeout",
            "24000",
            "--enable-dp-lm-head",
            "--watchdog-timeout",
            "24000",
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--reasoning-parser",
            "glm45",
            "--tool-call-parser",
            "glm47",
            "--enable-metrics",
            "--decode-attention-backend",
            "flashmla",
            "--prefill-attention-backend",
            "fa3",
            "--disable-custom-all-reduce",
            "--enforce-disable-flashinfer-allreduce-fusion",
            "--speculative-attention-mode",
            "decode",
            "--moe-a2a-backend",
            "deepep",
            "--moe-dense-tp-size",
            "1",
            "--disable-shared-experts-fusion",
            "--cuda-graph-max-bs-decode",
            "40",
            "--disable-prefill-cuda-graph",
        ]
        cls.process = popen_launch_server(
            MODEL_PATH,
            BASE_URL,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env={
                **os.environ,
                "SGLANG_DSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8": "0",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    @property
    def base_url(self):
        return BASE_URL


if __name__ == "__main__":
    unittest.main()
