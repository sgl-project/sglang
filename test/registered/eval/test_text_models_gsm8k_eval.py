import unittest
import warnings

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.multi_model_eval_kit import run_multi_model_accuracy_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2,
    DEFAULT_URL_FOR_TEST,
    ModelLaunchSettings,
    parse_models,
)

register_cuda_ci(est_time=2880, stage="nightly", runner_config="2-gpu-large")

MODEL_SCORE_THRESHOLDS = {
    # sgl-eval (zero-shot chat, \boxed{}, math_verify grading). Thresholds are
    # measured_score - 0.05, baselined on H100 2-GPU over the full 1319 split.
    "meta-llama/Llama-3.1-8B-Instruct": 0.77,  # 81.05% measured - 5%
    "Qwen/Qwen3-8B": 0.76,  # 81.43% measured - 5%
    "Qwen/Qwen3-4B": 0.77,  # 82.41% measured - 5%
    "meta-llama/Llama-3.1-70B-Instruct": 0.90,  # 94.77% measured - 5%
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.39,  # 43.52% measured - 5%
    "Qwen/Qwen2-57B-A14B-Instruct": 0.46,  # 50.87% measured - 5%
    "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8": 0.77,  # 82.34% measured - 5%
    "neuralmagic/Mistral-7B-Instruct-v0.3-FP8": 0.23,  # 27.82% measured - 5%
    "neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8": 0.80,  # 84.91% measured - 5%
    "zai-org/GLM-4.5-Air-FP8": 0.73,  # 77.48% measured - 5%
    "neuralmagic/gemma-2-2b-it-FP8": 0.02,  # 6.52% measured - 5%
    "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8": 0.89,  # 94.01% measured - 5%
    "neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8": 0.35,  # 40.33% measured - 5%
    "neuralmagic/Qwen2-72B-Instruct-FP8": 0.83,  # 87.64% measured - 5%
    "neuralmagic/Qwen2-57B-A14B-Instruct-FP8": 0.40,  # 44.66% measured - 5%
}

# 70B on 2 GPUs leaves little headroom for the KV pool at the default fraction.
_PER_MODEL_ARGS = {
    "meta-llama/Llama-3.1-70B-Instruct": ["--mem-fraction-static", "0.9"],
}


def _build_models():
    by_tp = {
        1: (
            DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1,
            DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1,
        ),
        2: (
            DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2,
            DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2,
        ),
    }
    return [
        ModelLaunchSettings(
            model_path, tp_size=tp_size, extra_args=_PER_MODEL_ARGS.get(model_path)
        )
        for tp_size, name_lists in by_tp.items()
        for names in name_lists
        for model_path in parse_models(names)
    ]


# Do not use `CustomTestCase`: this sweep does not want retry.
class TestNightlyGsm8KEval(unittest.TestCase):
    def test_gsm8k_all_models(self):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        run_multi_model_accuracy_eval(
            self,
            _build_models(),
            eval_args=dict(
                eval_name="gsm8k",
                api="sgl_eval",
                num_examples=None,
                num_threads=1024,
            ),
            accuracy_thresholds=MODEL_SCORE_THRESHOLDS,
            base_url=DEFAULT_URL_FOR_TEST,
        )


if __name__ == "__main__":
    unittest.main()
