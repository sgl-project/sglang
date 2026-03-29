from types import SimpleNamespace
from typing import Optional

import requests

from sglang.test.few_shot_gsm8k import run_eval as run_eval_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import is_in_amd_ci, is_in_ci, write_github_step_summary

_THRESHOLD_NOT_SET = float("nan")


def _check_accept_length(test_case, base_url, threshold):
    """Check speculative decoding accept length from server info."""
    server_info = requests.get(base_url + "/get_server_info").json()
    avg_spec_accept_length = server_info["internal_states"][0]["avg_spec_accept_length"]
    print(f"{avg_spec_accept_length=}")
    test_case.assertGreater(avg_spec_accept_length, threshold)


class GSM8KMixin:
    """Mixin for few-shot GSM8K evaluation.

    Required attributes on the test class:
        base_url: str
        gsm8k_accuracy_thres: float
    """

    gsm8k_accuracy_thres: float = _THRESHOLD_NOT_SET
    gsm8k_accept_length_thres: Optional[float] = None
    gsm8k_num_questions: int = 200
    gsm8k_parallel: int = 128

    def test_gsm8k(self):
        assert (
            self.gsm8k_accuracy_thres == self.gsm8k_accuracy_thres
        ), f"{type(self).__name__} must set gsm8k_accuracy_thres"

        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=self.gsm8k_num_questions,
            max_new_tokens=512,
            parallel=self.gsm8k_parallel,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_gsm8k(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["accuracy"], self.gsm8k_accuracy_thres)

        if self.gsm8k_accept_length_thres is not None:
            _check_accept_length(self, self.base_url, self.gsm8k_accept_length_thres)


class MMLUMixin:
    """Mixin for MMLU evaluation.

    Required attributes on the test class:
        base_url: str
        model: str
        mmlu_score_threshold: float
    """

    mmlu_score_threshold: float = _THRESHOLD_NOT_SET
    mmlu_accept_length_thres: Optional[float] = None
    mmlu_num_examples: int = 5000
    mmlu_num_threads: int = 1024

    def test_mmlu(self):
        assert (
            self.mmlu_score_threshold == self.mmlu_score_threshold
        ), f"{type(self).__name__} must set mmlu_score_threshold"

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=self.mmlu_num_examples,
            num_threads=self.mmlu_num_threads,
        )

        metrics = run_eval(args)

        if is_in_ci():
            write_github_step_summary(f"### test_mmlu\n{metrics['score']=:.4f}\n")

        self.assertGreaterEqual(metrics["score"], self.mmlu_score_threshold)

        if self.mmlu_accept_length_thres is not None:
            _check_accept_length(self, self.base_url, self.mmlu_accept_length_thres)


class HumanEvalMixin:
    """Mixin for HumanEval evaluation.

    Required attributes on the test class:
        base_url: str
        model: str
        humaneval_score_threshold: float
    """

    humaneval_score_threshold: float = _THRESHOLD_NOT_SET
    humaneval_score_threshold_amd: Optional[float] = None
    humaneval_num_threads: int = 1024

    def test_human_eval(self):
        assert (
            self.humaneval_score_threshold == self.humaneval_score_threshold
        ), f"{type(self).__name__} must set humaneval_score_threshold"

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="humaneval",
            num_examples=None,
            num_threads=self.humaneval_num_threads,
        )

        metrics = run_eval(args)

        if is_in_ci():
            write_github_step_summary(f"### test_human_eval\n{metrics['score']=:.4f}\n")

        threshold = self.humaneval_score_threshold
        if is_in_amd_ci() and self.humaneval_score_threshold_amd is not None:
            threshold = self.humaneval_score_threshold_amd

        self.assertGreaterEqual(metrics["score"], threshold)


class MGSMEnMixin:
    """Mixin for MGSM English evaluation.

    Required attributes on the test class:
        base_url: str
        model: str
        mgsm_en_score_threshold: float
    """

    mgsm_en_score_threshold: float = _THRESHOLD_NOT_SET
    mgsm_en_num_examples: Optional[int] = None
    mgsm_en_num_threads: int = 1024

    def test_mgsm_en(self):
        assert (
            self.mgsm_en_score_threshold == self.mgsm_en_score_threshold
        ), f"{type(self).__name__} must set mgsm_en_score_threshold"

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=self.mgsm_en_num_examples,
            num_threads=self.mgsm_en_num_threads,
        )

        metrics = run_eval(args)

        if is_in_ci():
            write_github_step_summary(f"### test_mgsm_en\n{metrics['score']=:.4f}\n")

        self.assertGreaterEqual(metrics["score"], self.mgsm_en_score_threshold)
