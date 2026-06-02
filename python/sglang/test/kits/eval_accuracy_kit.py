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


def _run_accuracy_eval(
    test_case,
    *,
    eval_name: str,
    score_threshold: float,
    num_examples: Optional[int],
    num_threads: int,
    accept_length_thres: Optional[float] = None,
    summary_label: Optional[str] = None,
    **eval_overrides,
):
    """Shared driver for the accuracy mixins below.

    Runs ``run_eval`` for ``eval_name`` against the test class's server
    (``base_url`` / ``model``), records a CI step summary, asserts the score
    meets ``score_threshold``, and checks the speculative accept length.

    ``eval_overrides`` (e.g. ``api``, ``max_tokens``, ``temperature``,
    ``top_p``, ``num_shots``) are forwarded to ``run_eval`` only when not
    ``None``, so the common case stays identical to ``run_eval``'s defaults.
    Returns the metrics dict.
    """
    # NaN sentinel (NaN != NaN) means the subclass forgot to set the threshold.
    assert (
        score_threshold == score_threshold
    ), f"{type(test_case).__name__} must set the {eval_name} score threshold"

    kwargs = dict(
        base_url=test_case.base_url,
        model=getattr(test_case, "model", None),
        eval_name=eval_name,
        num_examples=num_examples,
        num_threads=num_threads,
    )
    kwargs.update({k: v for k, v in eval_overrides.items() if v is not None})

    metrics = run_eval(SimpleNamespace(**kwargs))
    print(f"{eval_name} {metrics=}")

    if is_in_ci():
        label = summary_label or f"test_{eval_name}"
        write_github_step_summary(f"### {label}\n{metrics['score']=:.4f}\n")

    test_case.assertGreaterEqual(metrics["score"], score_threshold)
    _check_accept_length(test_case, test_case.base_url, accept_length_thres)
    return metrics


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
        requests.get(self.base_url + "/flush_cache")
        _run_accuracy_eval(
            self,
            eval_name="gsm8k",
            score_threshold=self.gsm8k_accuracy_thres,
            num_examples=self.gsm8k_num_questions,
            num_threads=self.gsm8k_num_threads,
            accept_length_thres=self.gsm8k_accept_length_thres,
            api="completion",
            max_tokens=512,
            num_shots=self.gsm8k_num_shots,
        )


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
        _run_accuracy_eval(
            self,
            eval_name="mmlu",
            score_threshold=self.mmlu_score_threshold,
            num_examples=self.mmlu_num_examples,
            num_threads=self.mmlu_num_threads,
            accept_length_thres=self.mmlu_accept_length_thres,
        )


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
        _run_accuracy_eval(
            self,
            eval_name="gpqa",
            score_threshold=self.gpqa_score_threshold,
            num_examples=self.gpqa_num_examples,
            num_threads=self.gpqa_num_threads,
            accept_length_thres=self.gpqa_accept_length_thres,
            max_tokens=self.gpqa_max_tokens,
            temperature=self.gpqa_temperature,
            top_p=self.gpqa_top_p,
        )


class AIME25Mixin:
    """Mixin for AIME 2025 evaluation (competition math, integer answers).

    Required attributes on the test class:
        base_url: str
        model: str
        aime25_score_threshold: float

    Optional sampling knobs (default to run_eval's defaults when unset). Set
    these for reasoning models -- e.g. DeepSeek-V4 thinking mode wants
    aime25_max_tokens=400000, aime25_temperature=1.0, aime25_top_p=1.0. AIME25
    has only 30 problems, so it is high variance; average over several runs
    (e.g. via the dataset's repeat support) for a stable number.
    """

    aime25_score_threshold: float = _THRESHOLD_NOT_SET
    aime25_accept_length_thres: Optional[float] = None
    aime25_num_examples: Optional[int] = None
    aime25_num_threads: int = 1024
    aime25_max_tokens: Optional[int] = None
    aime25_temperature: Optional[float] = None
    aime25_top_p: Optional[float] = None

    def test_aime25(self):
        _run_accuracy_eval(
            self,
            eval_name="aime25",
            score_threshold=self.aime25_score_threshold,
            num_examples=self.aime25_num_examples,
            num_threads=self.aime25_num_threads,
            accept_length_thres=self.aime25_accept_length_thres,
            max_tokens=self.aime25_max_tokens,
            temperature=self.aime25_temperature,
            top_p=self.aime25_top_p,
        )


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
        threshold = self.humaneval_score_threshold
        if is_in_amd_ci() and self.humaneval_score_threshold_amd is not None:
            threshold = self.humaneval_score_threshold_amd

        _run_accuracy_eval(
            self,
            eval_name="humaneval",
            score_threshold=threshold,
            num_examples=None,
            num_threads=self.humaneval_num_threads,
            summary_label="test_human_eval",
        )


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
        _run_accuracy_eval(
            self,
            eval_name="mgsm_en",
            score_threshold=self.mgsm_en_score_threshold,
            num_examples=self.mgsm_en_num_examples,
            num_threads=self.mgsm_en_num_threads,
        )
