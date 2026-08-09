"""Accuracy sweep over a list of models, one server at a time.

The scheduled eval suites each walk a model list, launch a server per model, run
one eval, and assert every score at the end -- so a single bad model reports as
one failure rather than aborting the sweep. This holds that loop; a test file
supplies the model list, the eval arguments and the score floors.

Latency deliberately has no place here. These evals let the model generate
freely, so wall time reflects how much the model chose to say (a thinking model
emits a whole chain of thought) rather than engine speed. Performance is
measured by the `perf/` suites, which drive bench_one_batch_server with a fixed
input/output length and ignore_eos.
"""

import json
from types import SimpleNamespace
from typing import Dict, List, Optional

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    ModelLaunchSettings,
    check_evaluation_test_results,
    popen_launch_server,
    write_results_to_json,
)

# Scheduled evals run models large enough to need a download on cache miss.
DEFAULT_EVAL_SERVER_TIMEOUT = 1800


def run_multi_model_accuracy_eval(
    test_case,
    models: List[ModelLaunchSettings],
    eval_args: Dict,
    accuracy_thresholds: Dict[str, float],
    *,
    base_url: str,
    server_timeout: int = DEFAULT_EVAL_SERVER_TIMEOUT,
    result_key: Optional[str] = None,
):
    """Evaluate every model in `models` and assert all scores at the end.

    `eval_args` is merged into the namespace handed to run_eval, so a caller
    passes only what its eval needs (eval_name, num_examples, num_threads, ...).
    A model that raises is recorded with its error and the sweep continues.
    """
    all_results = []
    is_first = True

    for model_setup in models:
        with test_case.subTest(model=model_setup.model_path):
            process = None
            try:
                process = popen_launch_server(
                    model=model_setup.model_path,
                    base_url=base_url,
                    other_args=list(model_setup.extra_args),
                    timeout=server_timeout,
                )

                args = SimpleNamespace(
                    base_url=base_url,
                    model=model_setup.model_path,
                    **eval_args,
                )
                metrics = run_eval(args)
                metrics["score"] = round(metrics["score"], 4)
                print(
                    f"{'=' * 42}\n{model_setup.model_path} - "
                    f"metrics={metrics} score={metrics['score']}\n{'=' * 42}\n"
                )

                write_results_to_json(
                    model_setup.model_path, metrics, "w" if is_first else "a"
                )
                is_first = False
                all_results.append(
                    (model_setup.model_path, metrics["score"], 0.0, None)
                )
            except Exception as e:
                error_message = str(e)
                all_results.append((model_setup.model_path, None, None, error_message))
                print(f"Error evaluating {model_setup.model_path}: {error_message}")
            finally:
                if process is not None:
                    kill_process_tree(process.pid)

    try:
        with open("results.json", "r") as f:
            print("\nFinal Results from results.json:")
            print(json.dumps(json.load(f), indent=2))
    except Exception as e:
        print(f"Error reading results.json: {e}")

    check_evaluation_test_results(
        all_results,
        result_key or type(test_case).__name__,
        model_accuracy_thresholds=accuracy_thresholds,
        model_count=len(models),
    )
