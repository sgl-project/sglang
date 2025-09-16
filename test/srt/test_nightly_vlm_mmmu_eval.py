import json
import os
import unittest
import warnings
from datetime import datetime
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    check_model_scores,
    parse_models,
    popen_launch_server_wrapper,
    write_results_to_json,
)

MODEL_SCORE_THRESHOLDS = {
    # Conservative thresholds on 100 MMMU samples
    "Qwen/Qwen2.5-VL-7B-Instruct": 0.340,
    "OpenGVLab/InternVL2_5-2B": 0.30,
    "google/gemma-3-4b-it": 0.35,
}

DEFAULT_VLM_MODELS = ",".join(
    [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "OpenGVLab/InternVL2_5-2B",
        "google/gemma-3-4b-it",
    ]
)


class TestNightlyVLMMmmuEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Allow overriding models via env, else use defaults
        cls.models = parse_models(
            os.environ.get("NIGHTLY_VLM_MODELS", DEFAULT_VLM_MODELS)
        )
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_mmmu_vlm_models(self):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        is_first = True
        all_results = []

        for model in self.models:
            with self.subTest(model=model):
                process = popen_launch_server_wrapper(self.base_url, model)

                try:
                    args = SimpleNamespace(
                        base_url=self.base_url,
                        model=model,
                        eval_name="mmmu",
                        num_examples=100,
                        num_threads=512,
                        max_tokens=30,
                    )

                    metrics = run_eval(args)

                    metrics["score"] = round(metrics["score"], 3)
                    print(
                        f"{'=' * 42}\n{model} - metrics={metrics} score={metrics['score']}\n{'=' * 42}\n"
                    )

                    write_results_to_json(model, metrics, "w" if is_first else "a")
                    is_first = False

                    all_results.append((model, metrics["score"]))
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
