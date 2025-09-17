import json
import unittest
import warnings
from types import SimpleNamespace
from typing import List

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    check_model_scores,
    popen_launch_server_wrapper,
    write_results_to_json,
)


class ModelDeploySetup:
    def __init__(self, model_path: str, extra_args: List[str] = []):
        self.model_path = model_path
        if "--enable-multimodal" not in extra_args:
            extra_args.append("--enable-multimodal")
        self.extra_args = extra_args


MODEL_SCORE_THRESHOLDS = {
    # Conservative thresholds on 100 MMMU samples
    ModelDeploySetup("Qwen/Qwen2-VL-7B-Instruct"): 0.330,
    ModelDeploySetup("Qwen/Qwen2.5-VL-7B-Instruct"): 0.340,
    ModelDeploySetup("openbmb/MiniCPM-o-2_6"): 0.350,
    ModelDeploySetup("XiaomiMiMo/MiMo-VL-7B-RL"): 0.28,
    ModelDeploySetup("Efficient-Large-Model/NVILA-Lite-2B-hf-0626"): 0.32,
    ModelDeploySetup("mistral-community/pixtral-12b"): 0.360,
    ModelDeploySetup("deepseek-ai/deepseek-vl2-small"): 0.340,
    ModelDeploySetup("unsloth/Mistral-Small-3.1-24B-Instruct-2503"): 0.330,
    ModelDeploySetup("deepseek-ai/Janus-Pro-7B"): 0.295,
    ModelDeploySetup("google/gemma-3-4b-it"): 0.360,
    ModelDeploySetup("google/gemma-3n-E4B-it"): 0.360,
    ModelDeploySetup("moonshotai/Kimi-VL-A3B-Instruct"): 0.350,
    ModelDeploySetup("zai-org/GLM-4.1V-9B-Thinking"): 0.310,
    ModelDeploySetup("OpenGVLab/InternVL2_5-2B"): 0.300,
}


class TestNightlyVLMMmmuEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = list(MODEL_SCORE_THRESHOLDS.keys())
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

                    metrics = run_eval(args)

                    metrics["score"] = round(metrics["score"], 4)
                    print(
                        f"{'=' * 42}\n{model_path} - metrics={metrics} score={metrics['score']}\n{'=' * 42}\n"
                    )

                    write_results_to_json(model_path, metrics, "w" if is_first else "a")
                    is_first = False

                    all_results.append((model_path, metrics["score"]))
                finally:
                    kill_process_tree(process.pid)

        try:
            with open("results.json", "r") as f:
                print("\nFinal Results from results.json:")
                print(json.dumps(json.load(f), indent=2))
        except Exception as e:
            print(f"Error reading results: {e}")

        check_model_scores(all_results, MODEL_SCORE_THRESHOLDS, self.__class__.__name__)


if __name__ == "__main__":
    unittest.main()
