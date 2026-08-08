import unittest
import warnings

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.multi_model_eval_kit import run_multi_model_accuracy_eval
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, ModelLaunchSettings

register_cuda_ci(est_time=7200, stage="nightly", runner_config="2-gpu-large")

MODELS = [
    ModelLaunchSettings("deepseek-ai/deepseek-vl2-small"),
    ModelLaunchSettings("deepseek-ai/Janus-Pro-7B"),
    ModelLaunchSettings("Efficient-Large-Model/NVILA-8B-hf"),
    ModelLaunchSettings("Efficient-Large-Model/NVILA-Lite-2B-hf"),
    ModelLaunchSettings("google/gemma-4-E4B-it"),
    ModelLaunchSettings("google/gemma-4-26B-A4B-it", extra_args=["--tp=2"]),
    ModelLaunchSettings("google/gemma-4-31B-it", extra_args=["--tp=2"]),
    ModelLaunchSettings("mistral-community/pixtral-12b"),
    ModelLaunchSettings("moonshotai/Kimi-VL-A3B-Instruct"),
    ModelLaunchSettings("OpenGVLab/InternVL2_5-2B"),
    ModelLaunchSettings("Qwen/Qwen2-VL-7B-Instruct"),
    ModelLaunchSettings("Qwen/Qwen2.5-VL-7B-Instruct"),
    ModelLaunchSettings("Qwen/Qwen3-VL-30B-A3B-Instruct", extra_args=["--tp=2"]),
    ModelLaunchSettings("unsloth/Mistral-Small-3.1-24B-Instruct-2503"),
    ModelLaunchSettings("XiaomiMiMo/MiMo-VL-7B-RL"),
    ModelLaunchSettings("zai-org/GLM-4.1V-9B-Thinking"),
    ModelLaunchSettings("zai-org/GLM-4.5V-FP8", extra_args=["--tp=2"]),
]

# Conservative floors on 100 MMMU samples. CoT (max_tokens=1024, see #27327)
# lifted most of these well clear of their floor; pixtral is the one it did not
# help, sitting at a steady 0.3399.
MODEL_SCORE_THRESHOLDS = {
    "deepseek-ai/deepseek-vl2-small": 0.320,
    "deepseek-ai/Janus-Pro-7B": 0.285,
    "Efficient-Large-Model/NVILA-8B-hf": 0.270,
    "Efficient-Large-Model/NVILA-Lite-2B-hf": 0.270,
    "google/gemma-4-E4B-it": 0.26,
    "google/gemma-4-26B-A4B-it": 0.27,
    "google/gemma-4-31B-it": 0.28,
    "mistral-community/pixtral-12b": 0.330,
    "moonshotai/Kimi-VL-A3B-Instruct": 0.330,
    "OpenGVLab/InternVL2_5-2B": 0.300,
    "Qwen/Qwen2-VL-7B-Instruct": 0.310,
    "Qwen/Qwen2.5-VL-7B-Instruct": 0.330,
    "Qwen/Qwen3-VL-30B-A3B-Instruct": 0.29,
    "unsloth/Mistral-Small-3.1-24B-Instruct-2503": 0.30,
    "XiaomiMiMo/MiMo-VL-7B-RL": 0.28,
    "zai-org/GLM-4.1V-9B-Thinking": 0.280,
    "zai-org/GLM-4.5V-FP8": 0.26,
}


class TestNightlyVLMMmmuEval(unittest.TestCase):
    def test_mmmu_vlm_models(self):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        run_multi_model_accuracy_eval(
            self,
            MODELS,
            eval_args=dict(
                eval_name="mmmu",
                num_examples=100,
                num_threads=64,
                max_tokens=1024,
            ),
            accuracy_thresholds=MODEL_SCORE_THRESHOLDS,
            base_url=DEFAULT_URL_FOR_TEST,
        )


if __name__ == "__main__":
    unittest.main()
