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
        0.320, 56.1
    ),
    ModelLaunchSettings("deepseek-ai/Janus-Pro-7B"): ModelEvalMetrics(0.285, 40.3),
    ModelLaunchSettings("Efficient-Large-Model/NVILA-8B-hf"): ModelEvalMetrics(
        0.270, 56.7
    ),
    ModelLaunchSettings("Efficient-Large-Model/NVILA-Lite-2B-hf"): ModelEvalMetrics(
        0.270, 23.8
    ),
    ModelLaunchSettings("google/gemma-3-4b-it"): ModelEvalMetrics(0.360, 10.9),
    ModelLaunchSettings("google/gemma-3n-E4B-it"): ModelEvalMetrics(0.270, 17.7),
    ModelLaunchSettings("mistral-community/pixtral-12b"): ModelEvalMetrics(0.360, 16.6),
    ModelLaunchSettings("moonshotai/Kimi-VL-A3B-Instruct"): ModelEvalMetrics(
        0.330, 22.3
    ),
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
    ModelLaunchSettings(
        "zai-org/GLM-4.5V-FP8", extra_args=["--tp=2"]
    ): ModelEvalMetrics(0.26, 32.0),
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
            error_message = None
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
                        num_examples=800,
                        num_threads=256,
                        max_tokens=4096,
                    )
                    
                    # patch for GLM models
                    if "zai-org/GLM-" in model_path:
                        args.response_answer_regex = r"<\|begin_of_box\|>(.*)<\|end_of_box\|>"

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
                        (
                            model_path,
                            metrics["score"],
                            metrics["latency"],
                            error_message,
                        )
                    )
                except Exception as e:
                    # Capture error message for the summary table
                    error_message = str(e)
                    # Still append result with error info (use None for N/A metrics to match else clause)
                    all_results.append((model_path, None, None, error_message))
                    print(f"Error evaluating {model_path}: {error_message}")
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

"""
 | model | status | score | score_threshold | latency | latency_threshold | error |
| ----- | ------ | ----- | --------------- | ------- | ----------------- | ----- |
| Efficient-Large-Model/NVILA-8B-hf | ❌ | 0.425 | 0.27 | 167.1675 | 56.7 | - |
| Efficient-Large-Model/NVILA-Lite-2B-hf | ❌ | 0.2713 | 0.27 | 30.1851 | 23.8 | - |
| OpenGVLab/InternVL2_5-2B | ❌ | 0.3137 | 0.3 | 55.8175 | 17.0 | - |
| Qwen/Qwen2-VL-7B-Instruct | ✅ | 0.33 | 0.31 | 82.5729 | 83.3 | - |
| Qwen/Qwen2.5-VL-7B-Instruct | ❌ | 0.4226 | 0.34 | 71.5153 | 31.9 | - |
| Qwen/Qwen3-VL-30B-A3B-Instruct | ❌ | 0.53 | 0.29 | 190.0179 | 37.0 | - |
| XiaomiMiMo/MiMo-VL-7B-RL | ❌ | 0.2612 | 0.28 | 366.2807 | 32.0 | - |
| deepseek-ai/Janus-Pro-7B | ❌ | 0.3725 | 0.285 | 209.8493 | 40.3 | - |
| deepseek-ai/deepseek-vl2-small | ❌ | N/A | 0.32 | N/A | 56.1 | Model not evaluated |
| google/gemma-3-4b-it | ❌ | 0.3113 | 0.36 | 114.1445 | 10.9 | - |
| google/gemma-3n-E4B-it | ❌ | 0.3713 | 0.27 | 97.559 | 17.7 | - |
| mistral-community/pixtral-12b | ❌ | 0.3962 | 0.36 | 75.4589 | 16.6 | - |
| moonshotai/Kimi-VL-A3B-Instruct | ❌ | N/A | 0.33 | N/A | 22.3 | Model not evaluated |
| openbmb/MiniCPM-v-2_6 | ❌ | 0.3113 | 0.259 | 53.2697 | 36.3 | - |
| unsloth/Mistral-Small-3.1-24B-Instruct-2503 | ❌ | 0.4838 | 0.31 | 93.8087 | 16.7 | - |
| zai-org/GLM-4.1V-9B-Thinking | ❌ | 0.5626 | 0.28 | 140.657 | 30.4 | - |
| zai-org/GLM-4.5V-FP8 | ❌ | 0.5313 | 0.26 | 257.0079 | 32.0 | - |

Some models failed the evaluation.
F
======================================================================
ERROR: test_mmmu_vlm_models (__main__.TestNightlyVLMMmmuEval.test_mmmu_vlm_models) (model='deepseek-ai/deepseek-vl2-small')
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/root/xinyuan/sglang/test/nightly/test_vlms_mmmu_eval.py", line 71, in test_mmmu_vlm_models
    process = popen_launch_server(
              ^^^^^^^^^^^^^^^^^^^^
  File "/root/xinyuan/sglang/python/sglang/test/test_utils.py", line 658, in popen_launch_server
    raise Exception(
Exception: Server process exited with code -9. Check server logs for errors.

======================================================================
ERROR: test_mmmu_vlm_models (__main__.TestNightlyVLMMmmuEval.test_mmmu_vlm_models) (model='moonshotai/Kimi-VL-A3B-Instruct')
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/root/xinyuan/sglang/test/nightly/test_vlms_mmmu_eval.py", line 71, in test_mmmu_vlm_models
    process = popen_launch_server(
              ^^^^^^^^^^^^^^^^^^^^
  File "/root/xinyuan/sglang/python/sglang/test/test_utils.py", line 658, in popen_launch_server
    raise Exception(
Exception: Server process exited with code -9. Check server logs for errors.

======================================================================
FAIL: test_mmmu_vlm_models (__main__.TestNightlyVLMMmmuEval.test_mmmu_vlm_models)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/root/xinyuan/sglang/test/nightly/test_vlms_mmmu_eval.py", line 136, in test_mmmu_vlm_models
    check_evaluation_test_results(
  File "/root/xinyuan/sglang/python/sglang/test/test_utils.py", line 1857, in check_evaluation_test_results
    raise AssertionError("\n".join(failed_models))
AssertionError:
Latency Check Failed: Efficient-Large-Model/NVILA-8B-hf
Model Efficient-Large-Model/NVILA-8B-hf latency (167.1675) is above threshold (56.7000)

Latency Check Failed: Efficient-Large-Model/NVILA-Lite-2B-hf
Model Efficient-Large-Model/NVILA-Lite-2B-hf latency (30.1851) is above threshold (23.8000)

Latency Check Failed: OpenGVLab/InternVL2_5-2B
Model OpenGVLab/InternVL2_5-2B latency (55.8175) is above threshold (17.0000)

Latency Check Failed: Qwen/Qwen2.5-VL-7B-Instruct
Model Qwen/Qwen2.5-VL-7B-Instruct latency (71.5153) is above threshold (31.9000)

Latency Check Failed: Qwen/Qwen3-VL-30B-A3B-Instruct
Model Qwen/Qwen3-VL-30B-A3B-Instruct latency (190.0179) is above threshold (37.0000)

Score Check Failed: XiaomiMiMo/MiMo-VL-7B-RL
Model XiaomiMiMo/MiMo-VL-7B-RL score (0.2612) is below threshold (0.2800)

Latency Check Failed: XiaomiMiMo/MiMo-VL-7B-RL
Model XiaomiMiMo/MiMo-VL-7B-RL latency (366.2807) is above threshold (32.0000)

Latency Check Failed: deepseek-ai/Janus-Pro-7B
Model deepseek-ai/Janus-Pro-7B latency (209.8493) is above threshold (40.3000)
Model failed to launch or be evaluated: deepseek-ai/deepseek-vl2-small

Score Check Failed: google/gemma-3-4b-it
Model google/gemma-3-4b-it score (0.3113) is below threshold (0.3600)

Latency Check Failed: google/gemma-3-4b-it
Model google/gemma-3-4b-it latency (114.1445) is above threshold (10.9000)

Latency Check Failed: google/gemma-3n-E4B-it
Model google/gemma-3n-E4B-it latency (97.5590) is above threshold (17.7000)

Latency Check Failed: mistral-community/pixtral-12b
Model mistral-community/pixtral-12b latency (75.4589) is above threshold (16.6000)
Model failed to launch or be evaluated: moonshotai/Kimi-VL-A3B-Instruct

Latency Check Failed: openbmb/MiniCPM-v-2_6
Model openbmb/MiniCPM-v-2_6 latency (53.2697) is above threshold (36.3000)

Latency Check Failed: unsloth/Mistral-Small-3.1-24B-Instruct-2503
Model unsloth/Mistral-Small-3.1-24B-Instruct-2503 latency (93.8087) is above threshold (16.7000)

Latency Check Failed: zai-org/GLM-4.1V-9B-Thinking
Model zai-org/GLM-4.1V-9B-Thinking latency (140.6570) is above threshold (30.4000)

Latency Check Failed: zai-org/GLM-4.5V-FP8
Model zai-org/GLM-4.5V-FP8 latency (257.0079) is above threshold (32.0000)

----------------------------------------------------------------------
Ran 1 test in 4469.198s

FAILED (failures=1, errors=2)

"""