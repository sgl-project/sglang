import os
import shutil
import signal
import subprocess
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    get_available_url,
    is_in_ci,
    launch_server_in_nightly_test,
    parse_models,
)


def parse_humaneval_results(output_text):
    results = {}
    for line in output_text.split("\n"):
        if "pass@1" in line:
            test_type = (
                "base"
                if "base tests"
                in output_text.split("\n")[output_text.split("\n").index(line) - 1]
                else "extra"
            )
            score = float(line.split("\t")[1])
            results[test_type] = score
    return results


class TestEvalAccuracyLarge(unittest.TestCase):

    def run_evalplus(self, model, is_tp2, logging_details=False):
        print("Delete evalplus results")
        shutil.rmtree("evalplus_results", ignore_errors=True)
        cmd = [
            "evalplus.evaluate",
            "--model",
            model,
            "--dataset",
            "humaneval",
            "--backend",
            "openai",
            "--base-url",
            f"{self.base_url}/v1",
            "--greedy",
        ]

        stdout_stderr = None if logging_details else (subprocess.PIPE, subprocess.PIPE)

        try:
            self.eval_process = subprocess.Popen(
                cmd,
                text=True,
                stdout=stdout_stderr[0] if stdout_stderr else None,
                stderr=stdout_stderr[1] if stdout_stderr else None,
                preexec_fn=os.setsid,
            )

            stdout, stderr = self.eval_process.communicate(
                timeout=1200 if is_tp2 else 600
            )

            assert (
                self.eval_process.returncode == 0
            ), f"Failed to human eval model={model}, err={stderr}"

            print("=" * 42)
            print(stdout)
            print("=" * 42)
            return stdout
        except subprocess.TimeoutExpired:
            if self.eval_process:
                os.killpg(os.getpgid(self.eval_process.pid), signal.SIGTERM)
            print(f"Timeout during evaluation for model={model}")
            raise AssertionError(f"Evaluation timeout for model={model}")
        except Exception as e:
            print(f"Error running evalplus for model={model}: {str(e)}")
            if self.eval_process:
                os.killpg(os.getpgid(self.eval_process.pid), signal.SIGTERM)
            raise AssertionError(f"Evaluation failed for model={model}: {str(e)}")

    def test_human_eval_all_models(self):
        model_groups = [
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1), False, False),
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2), False, True),
            (
                parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1),
                True,
                False,
            ),
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2), True, True),
        ]
        self.base_url = get_available_url(DEFAULT_URL_FOR_TEST)
        print(f"Using port {self.base_url}")
        self.model_results = {}
        errors = []
        for model_group, is_fp8, is_tp2 in model_groups:
            for model in model_group:
                if "Llama" not in model:
                    continue
                self.server_process = None
                self.eval_process = None
                logging_details = False
                try:
                    self.server_process = launch_server_in_nightly_test(
                        self.base_url,
                        model,
                        is_fp8,
                        is_tp2,
                        logging_details=logging_details,
                    )
                    self.model_results[model] = parse_humaneval_results(
                        self.run_evalplus(
                            model, is_tp2=is_tp2, logging_details=logging_details
                        )
                    )

                except Exception as e:
                    errors.append(f"Error in model {model}: {str(e)}")
                    self.model_results[model] = e

                finally:
                    for process in [self.server_process, self.eval_process]:
                        if process:
                            try:
                                kill_process_tree(process.pid)
                            except Exception as cleanup_error:
                                errors.append(f"Cleanup error: {cleanup_error}")
        print(self.model_results)
        if is_in_ci():
            from sglang.test.test_utils import write_github_step_summary

            write_github_step_summary(self.model_results)
        if errors:
            raise Exception("\n".join(errors))


if __name__ == "__main__":
    unittest.main()
