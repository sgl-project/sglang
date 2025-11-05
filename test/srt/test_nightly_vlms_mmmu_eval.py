import json
import unittest
import warnings
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    ModelEvalMetrics,
    ModelLaunchSettings,
    check_evaluation_test_results,
    popen_launch_server,
    write_results_to_json,
)

MODEL_THRESHOLDS = {
    # Conservative thresholds on 100 MMMU samples, especially for latency thresholds
    ModelLaunchSettings("deepseek-ai/deepseek-vl2-small"): ModelEvalMetrics(
        0.330, 56.1
    ),
    ModelLaunchSettings("deepseek-ai/Janus-Pro-7B"): ModelEvalMetrics(0.285, 40.3),
    ModelLaunchSettings("Efficient-Large-Model/NVILA-8B-hf"): ModelEvalMetrics(
        0.270, 56.7
    ),
    ModelLaunchSettings("Efficient-Large-Model/NVILA-Lite-2B-hf"): ModelEvalMetrics(
        0.270, 23.8
    ),
    ModelLaunchSettings("google/gemma-3-4b-it"): ModelEvalMetrics(0.360, 10.9),
    ModelLaunchSettings("google/gemma-3n-E4B-it"): ModelEvalMetrics(0.360, 17.7),
    ModelLaunchSettings("mistral-community/pixtral-12b"): ModelEvalMetrics(0.360, 16.6),
    ModelLaunchSettings("moonshotai/Kimi-VL-A3B-Instruct"): ModelEvalMetrics(
        0.330, 22.3
    ),
    ModelLaunchSettings("openbmb/MiniCPM-o-2_6"): ModelEvalMetrics(0.330, 29.3),
    ModelLaunchSettings("openbmb/MiniCPM-v-2_6"): ModelEvalMetrics(0.259, 36.3),
    ModelLaunchSettings("OpenGVLab/InternVL2_5-2B"): ModelEvalMetrics(0.300, 17.0),
    ModelLaunchSettings("Qwen/Qwen2-VL-7B-Instruct"): ModelEvalMetrics(0.310, 83.3),
    ModelLaunchSettings("Qwen/Qwen2.5-VL-7B-Instruct"): ModelEvalMetrics(0.340, 31.9),
    ModelLaunchSettings(
        "Qwen/Qwen3-VL-30B-A3B-Instruct", extra_args=["--tp=2"]
    ): ModelEvalMetrics(0.29, 37.0),
    ModelLaunchSettings(
        "unsloth/Mistral-Small-3.1-24B-Instruct-2503"
    ): ModelEvalMetrics(0.310, 16.7),
    ModelLaunchSettings("XiaomiMiMo/MiMo-VL-7B-RL"): ModelEvalMetrics(0.28, 32.0),
    ModelLaunchSettings("zai-org/GLM-4.1V-9B-Thinking"): ModelEvalMetrics(0.280, 30.4),
}


class TestNightlyVLMMmmuEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = list(MODEL_THRESHOLDS.keys())
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_mmmu_vlm_models(self):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        is_first = True
        all_results = []

        for model in self.models:
            model_path = model.model_path
            with self.subTest(model=model_path):
                process = popen_launch_server(
                    model=model_path,
                    base_url=self.base_url,
                    other_args=model.extra_args,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                )
                try:
                    args = SimpleNamespace(
                        base_url=self.base_url,
                        model=model_path,
                        eval_name="mmmu",
                        num_examples=100,
                        num_threads=64,
                        max_tokens=30,
                    )

                    args.return_latency = True

                    metrics, latency = run_eval(args)

                    metrics["score"] = round(metrics["score"], 4)
                    metrics["latency"] = round(latency, 4)
                    print(
                        f"{'=' * 42}\n{model_path} - metrics={metrics} score={metrics['score']}\n{'=' * 42}\n"
                    )

                    write_results_to_json(model_path, metrics, "w" if is_first else "a")
                    is_first = False

                    all_results.append(
                        (model_path, metrics["score"], metrics["latency"])
                    )
                finally:
                    kill_process_tree(process.pid)

        try:
            with open("results.json", "r") as f:
                print("\nFinal Results from results.json:")
                print(json.dumps(json.load(f), indent=2))
        except Exception as e:
            print(f"Error reading results: {e}")

        model_accuracy_thresholds = {
            model.model_path: threshold.accuracy
            for model, threshold in MODEL_THRESHOLDS.items()
        }
        model_latency_thresholds = {
            model.model_path: threshold.eval_time
            for model, threshold in MODEL_THRESHOLDS.items()
        }
        check_evaluation_test_results(
            all_results,
            self.__class__.__name__,
            model_accuracy_thresholds=model_accuracy_thresholds,
            model_latency_thresholds=model_latency_thresholds,
        )


if __name__ == "__main__":
    unittest.main()
