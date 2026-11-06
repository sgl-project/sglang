"""Qwen3.6 native block-FP8 or MXFP8 TP2 + NEXTN accuracy on Ascend 950DT.

Set QWEN36_FP8_MODEL_PATH and GSM8K_DATA_PATH to local files for offline runs.
ASCEND_RT_VISIBLE_DEVICES selects the two devices; the test never pins card IDs.
SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8=0/1 selects native/MXFP8 (default: 0).
The legacy run_eval backend is intentional: preserve the deployment's chat,
5-shot scoring protocol. With a 1319-row test.jsonl, its first five rows are
demonstrations and the remaining 1314 rows are scored.
"""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=3600, suite="nightly-acc-4-npu-a5", nightly=True)

SERVER_ENVS = {
    "SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8": (
        "1" if envs.SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8.get() else "0"
    ),
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM": "1",
    "ASCEND_USE_FIA": "1",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
}

SERVER_ARGS = [
    "--tp-size",
    "2",
    "--nnodes",
    "1",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    "-1",
    "--max-total-tokens",
    "600000",
    "--max-prefill-tokens",
    "65536",
    "--disable-radix-cache",
    "--trust-remote-code",
    "--enable-prefill-delayer",
    "--max-running-requests",
    "10",
    "--max-mamba-cache-size",
    "20",
    "--mem-fraction-static",
    "0.65",
    "--cuda-graph-bs-decode",
    "2",
    "4",
    "8",
    "12",
    "14",
    "16",
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-draft-model-quantization",
    "fp8",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen3_coder",
]


class TestNpuQwen36BlockFp8Gsm8k(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get("QWEN36_FP8_MODEL_PATH", "Qwen/Qwen3.6-35B-A3B-FP8")
        cls.base_url = os.environ.get("QWEN36_BASE_URL", "http://127.0.0.1:30000")
        cls.data_path = os.environ.get("GSM8K_DATA_PATH")
        if cls.data_path is not None and not Path(cls.data_path).is_file():
            raise FileNotFoundError(f"GSM8K_DATA_PATH does not exist: {cls.data_path}")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=SERVER_ARGS,
            env=SERVER_ENVS,
            device="npu",
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def test_gsm8k(self):
        result = run_eval(
            SimpleNamespace(
                base_url=self.base_url,
                eval_name="gsm8k",
                api="chat",
                gsm8k_data_path=self.data_path,
                num_examples=1319,
                num_shots=5,
                num_threads=16,
                max_tokens=4096,
                temperature=0,
                chat_template_kwargs={"enable_thinking": False},
            )
        )
        # Initial acceptance floor; calibrate with this exact scoring protocol.
        self.assertGreaterEqual(result["score"], 0.90)


if __name__ == "__main__":
    unittest.main()
