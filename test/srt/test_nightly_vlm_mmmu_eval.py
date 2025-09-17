import json
import unittest
import warnings
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    ModelDeploySetup,
    ModelEvalMetrics,
    check_model_scores,
    popen_launch_server_wrapper,
    write_results_to_json,
)

MODEL_THRESHOLDS = {
    # Conservative thresholds on 100 MMMU samples
    # ModelDeploySetup("deepseek-ai/deepseek-vl2-small"): ModelEvalMetrics(0.340, 39.6),
    # ModelDeploySetup("deepseek-ai/Janus-Pro-7B"): ModelEvalMetrics(0.295, 37.2),
    # ModelDeploySetup("Efficient-Large-Model/NVILA-Lite-2B-hf-0626"): ModelEvalMetrics(0.32, 10.9),
    # ModelDeploySetup("google/gemma-3-4b-it"): ModelEvalMetrics(0.360, 8.7),
    # ModelDeploySetup("google/gemma-3n-E4B-it"): ModelEvalMetrics(0.360, 11.0),
    # ModelDeploySetup("mistral-community/pixtral-12b"): ModelEvalMetrics(0.360),
    # ModelDeploySetup("moonshotai/Kimi-VL-A3B-Instruct"): ModelEvalMetrics(0.350),
    # ModelDeploySetup("openbmb/MiniCPM-o-2_6"): ModelEvalMetrics(0.350, 19.8),
    # ModelDeploySetup("openbmb/MiniCPM-v-2_6"): ModelEvalMetrics(0.350, 19,8),
    # ModelDeploySetup("OpenGVLab/InternVL2_5-2B"): ModelEvalMetrics(0.300, 8.8),
    # ModelDeploySetup("Qwen/Qwen2-VL-7B-Instruct"): ModelEvalMetrics(0.330, 20.5),
    ModelDeploySetup("Qwen/Qwen2.5-VL-7B-Instruct"): ModelEvalMetrics(0.340, 22.5),
    # ModelDeploySetup("unsloth/Mistral-Small-3.1-24B-Instruct-2503"): ModelEvalMetrics(0.330, 13.8),
    # ModelDeploySetup("XiaomiMiMo/MiMo-VL-7B-RL"): ModelEvalMetrics(0.28, 18.0),
    # ModelDeploySetup("zai-org/GLM-4.1V-9B-Thinking"): ModelEvalMetrics(0.310, 22.4),
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
                process = popen_launch_server_wrapper(
                    self.base_url, model_path, extra_args=model.extra_args
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
        check_model_scores(
            all_results,
            self.__class__.__name__,
            model_accuracy_thresholds,
            model_latency_thresholds,
        )


if __name__ == "__main__":
    unittest.main()
