from types import SimpleNamespace
from typing import Optional

import requests

from sglang.test.run_eval import run_eval
from sglang.test.test_utils import is_in_amd_ci, is_in_ci, write_github_step_summary

_THRESHOLD_NOT_SET = float("nan")


def _check_accept_length(test_case, base_url, threshold=None):
    """Print accept length; optionally assert it exceeds threshold."""
    try:
        server_info = requests.get(base_url + "/server_info").json()
        val = server_info["internal_states"][0]["avg_spec_accept_length"]
    except (KeyError, IndexError, requests.RequestException):
        return
    print(f"avg_spec_accept_length={val:.4f}")
    if threshold is not None:
        test_case.assertGreater(val, threshold)


class GSM8KMixin:
    """Mixin for GSM8K evaluation via OpenAI Chat API.

    Required attributes on the test class:
        base_url: str
        gsm8k_accuracy_thres: float

    Optional attributes:
        model: str (if not set, auto-detected from server)
    """

    gsm8k_accuracy_thres: float = _THRESHOLD_NOT_SET
    gsm8k_accept_length_thres: Optional[float] = None
    gsm8k_num_questions: int = 200
    gsm8k_num_threads: int = 128
    gsm8k_num_shots: int = 5

    def test_gsm8k(self):
        assert (
            self.gsm8k_accuracy_thres == self.gsm8k_accuracy_thres
        ), f"{type(self).__name__} must set gsm8k_accuracy_thres"

        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=self.gsm8k_num_questions,
            num_threads=self.gsm8k_num_threads,
            num_shots=self.gsm8k_num_shots,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(f"### test_gsm8k\n{metrics['score']=:.4f}\n")

        self.assertGreaterEqual(metrics["score"], self.gsm8k_accuracy_thres)

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

        _check_accept_length(self, self.base_url, self.mmlu_accept_length_thres)


class GPQAMixin:
    """Mixin for GPQA-Diamond evaluation (graduate-level multiple choice).

    Required attributes on the test class:
        base_url: str
        model: str
        gpqa_score_threshold: float

    Optional sampling knobs (default to run_eval's defaults when unset). Set
    these for reasoning models that need a large generation budget and the
    model's recommended sampling -- e.g. DeepSeek-V4 thinking mode wants
    gpqa_max_tokens=400000, gpqa_temperature=1.0, gpqa_top_p=1.0.
    """

    gpqa_score_threshold: float = _THRESHOLD_NOT_SET
    gpqa_accept_length_thres: Optional[float] = None
    gpqa_num_examples: Optional[int] = None
    gpqa_num_threads: int = 1024
    gpqa_max_tokens: Optional[int] = None
    gpqa_temperature: Optional[float] = None
    gpqa_top_p: Optional[float] = None

    def test_gpqa(self):
        assert (
            self.gpqa_score_threshold == self.gpqa_score_threshold
        ), f"{type(self).__name__} must set gpqa_score_threshold"

        kwargs = dict(
            base_url=self.base_url,
            model=self.model,
            eval_name="gpqa",
            num_examples=self.gpqa_num_examples,
            num_threads=self.gpqa_num_threads,
        )
        # Only override run_eval's defaults when explicitly set, so the common
        # case stays identical to the other mixins.
        if self.gpqa_max_tokens is not None:
            kwargs["max_tokens"] = self.gpqa_max_tokens
        if self.gpqa_temperature is not None:
            kwargs["temperature"] = self.gpqa_temperature
        if self.gpqa_top_p is not None:
            kwargs["top_p"] = self.gpqa_top_p

        metrics = run_eval(SimpleNamespace(**kwargs))

        if is_in_ci():
            write_github_step_summary(f"### test_gpqa\n{metrics['score']=:.4f}\n")

        self.assertGreaterEqual(metrics["score"], self.gpqa_score_threshold)

        _check_accept_length(self, self.base_url, self.gpqa_accept_length_thres)


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

        _check_accept_length(self, self.base_url)


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

        _check_accept_length(self, self.base_url)
