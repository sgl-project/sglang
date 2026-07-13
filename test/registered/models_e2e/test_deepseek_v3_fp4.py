import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

# Per-commit: SymmetricMemory variant only.
# - TestDeepseekV3FP4 (TRTLLM) archived to test/manual/quant/test_deepseek_v3_fp4_4gpu_trtllm.py
# - TestDeepseekV3FP4CutlassMoE moved to test_deepseek_v3_fp4_4gpu_extra.py
register_cuda_ci(est_time=330, stage="base-c", runner_config="4-gpu-b200")

FULL_DEEPSEEK_V3_FP4_MODEL_PATH = "nvidia/DeepSeek-V3-0324-FP4"


class TestDeepseekV3FP4SymmetricMemory(GSM8KMixin, DefaultServerBase):
    model = FULL_DEEPSEEK_V3_FP4_MODEL_PATH
    timeout = 1200
    other_args = [
        "--tp",
        "4",
        "--attention-backend",
        "trtllm_mla",
        "--moe-runner-backend",
        "flashinfer_trtllm",
        "--quantization",
        "modelopt_fp4",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true,"num_threads": 64}',
        "--enable-symm-mem",
    ]

    gsm8k_accuracy_thres = 0.93
    gsm8k_num_questions = 1319
    gsm8k_num_threads = 1319
    gsm8k_num_shots = 8


if __name__ == "__main__":
    unittest.main()
