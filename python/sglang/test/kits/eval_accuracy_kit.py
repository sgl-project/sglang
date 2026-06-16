from types import SimpleNamespace
from typing import Optional

import requests

from sglang.test.run_eval import run_eval
from sglang.test.test_utils import is_in_amd_ci, is_in_ci, write_github_step_summary

_THRESHOLD_NOT_SET = float("nan")


def _check_accept_length(test_case, base_url, threshold=None):
    """Print speculative accept length; optionally assert it exceeds threshold."""
    try:
        server_info = requests.get(base_url + "/server_info").json()
        val = server_info["internal_states"][0]["avg_spec_accept_length"]
    except (KeyError, IndexError, requests.RequestException):
        return
    print(f"avg_spec_accept_length={val:.4f}")
    if threshold is not None:
        test_case.assertGreater(val, threshold)


def _finalize_eval(
    test_case,
    *,
    eval_name: str,
    score: float,
    score_threshold: float,
    accept_length_thres: Optional[float] = None,
    summary_label: Optional[str] = None,
):
    """Shared driver tail: CI step summary, accept-length check, threshold assert."""
    if is_in_ci():
        label = summary_label or f"test_{eval_name}"
        write_github_step_summary(f"### {label}\n{eval_name}_score={score:.4f}\n")
    _check_accept_length(test_case, test_case.base_url, accept_length_thres)
    test_case.assertGreaterEqual(score, score_threshold)


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
    _finalize_eval(
        test_case,
        eval_name=eval_name,
        score=metrics["score"],
        score_threshold=score_threshold,
        accept_length_thres=accept_length_thres,
        summary_label=summary_label,
    )
    return metrics


def _run_sgl_eval(
    test_case,
    *,
    eval_name: str,
    score_threshold: float,
    metric: str = "score",
    n_repeats: int = 1,
    num_examples: Optional[int] = None,
    num_threads: int = 512,
    thinking: bool = True,
    reasoning_effort: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    accept_length_thres: Optional[float] = None,
    summary_label: Optional[str] = None,
):
    """Shared sgl-eval driver for the reasoning mixins and the ``sgl_eval`` backend.

    Runs ``eval_name`` via the sgl-eval Python API (``registry.get`` ->
    ``EvalSpec.run``) against the test class's server, records a CI step summary,
    asserts the score meets ``score_threshold``, and checks the speculative accept
    length. ``thinking=True`` sends per-request ``chat_template_kwargs={"thinking":
    True}`` so the server separates reasoning from the final answer. Skips the test
    if sgl-eval (git-only) is not installed. Returns the RunResult.
    """
    assert (
        score_threshold == score_threshold
    ), f"{type(test_case).__name__} must set the {eval_name} score threshold"

    try:
        from sgl_eval.registry import get as get_eval_spec
        from sgl_eval.sampler import ChatCompletionSampler
        from sgl_eval.types import GenConfig
    except ImportError:
        test_case.skipTest(
            "sgl-eval not installed; pip install "
            "'sgl-eval @ git+https://github.com/sgl-project/sgl-eval'"
        )

    base_url = test_case.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    sampler = ChatCompletionSampler(
        base_url=base_url, model=getattr(test_case, "model", None), api_key="EMPTY"
    )

    gen_kwargs = dict(
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        chat_template_kwargs={"thinking": True} if thinking else None,
    )
    if temperature is not None:
        gen_kwargs["temperature"] = temperature
    if top_p is not None:
        gen_kwargs["top_p"] = top_p

    result = get_eval_spec(eval_name).run(
        sampler=sampler,
        gen=GenConfig(**gen_kwargs),
        n_repeats=n_repeats,
        num_examples=num_examples,
        num_threads=num_threads,
        predictions_writer=None,
        load_examples=None,
    )
    score = result.aggregate[metric]
    print(f"{eval_name} sgl-eval {metric}={score:.4f}")
    _finalize_eval(
        test_case,
        eval_name=eval_name,
        score=score,
        score_threshold=score_threshold,
        accept_length_thres=accept_length_thres,
        summary_label=summary_label,
    )
    return result


class GSM8KMixin:
    """Mixin for GSM8K evaluation.

    Backend is selectable via ``gsm8k_backend`` (default ``"run_eval"``: OpenAI
    completion API, 5-shot; or ``"sgl_eval"``: sgl-eval chat + boxed/sympy grader,
    skipped if sgl-eval is not installed). The canonical threshold/count knobs are
    ``gsm8k_score_threshold`` / ``gsm8k_num_examples``; the legacy
    ``gsm8k_accuracy_thres`` / ``gsm8k_num_questions`` are still honored.

    Required attributes on the test class:
        base_url: str
        gsm8k_score_threshold: float

    Optional attributes:
        model: str (if not set, auto-detected from server)
    """

    gsm8k_score_threshold: float = _THRESHOLD_NOT_SET
    gsm8k_accuracy_thres: float = _THRESHOLD_NOT_SET  # legacy alias
    gsm8k_num_examples: Optional[int] = None
    gsm8k_num_questions: int = 200  # legacy alias
    gsm8k_accept_length_thres: Optional[float] = None
    gsm8k_num_threads: int = 128
    gsm8k_num_shots: int = 5  # run_eval backend only
    gsm8k_backend: str = "run_eval"  # "run_eval" | "sgl_eval"
    gsm8k_thinking: bool = False  # sgl_eval backend
    gsm8k_n_repeats: int = 1  # sgl_eval backend

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")
        threshold = self.gsm8k_score_threshold
        if threshold != threshold:  # canonical unset (NaN) -> legacy alias
            threshold = self.gsm8k_accuracy_thres
        num_examples = (
            self.gsm8k_num_examples
            if self.gsm8k_num_examples is not None
            else self.gsm8k_num_questions
        )
        if self.gsm8k_backend == "sgl_eval":
            _run_sgl_eval(
                self,
                eval_name="gsm8k",
                score_threshold=threshold,
                n_repeats=self.gsm8k_n_repeats,
                num_examples=num_examples,
                num_threads=self.gsm8k_num_threads,
                thinking=self.gsm8k_thinking,
                accept_length_thres=self.gsm8k_accept_length_thres,
            )
        else:
            _run_accuracy_eval(
                self,
                eval_name="gsm8k",
                score_threshold=threshold,
                num_examples=num_examples,
                num_threads=self.gsm8k_num_threads,
                accept_length_thres=self.gsm8k_accept_length_thres,
                api="completion",
                max_tokens=512,
                num_shots=self.gsm8k_num_shots,
            )


class MMLUMixin:
    """Mixin for MMLU evaluation.

    Backend is selectable via ``mmlu_backend`` (default ``"run_eval"``; or
    ``"sgl_eval"``: sgl-eval multichoice grader, skipped if sgl-eval is not
    installed).

    Required attributes on the test class:
        base_url: str
        model: str
        mmlu_score_threshold: float
    """

    mmlu_score_threshold: float = _THRESHOLD_NOT_SET
    mmlu_accept_length_thres: Optional[float] = None
    mmlu_num_examples: int = 5000
    mmlu_num_threads: int = 1024
    mmlu_backend: str = "run_eval"  # "run_eval" | "sgl_eval"
    mmlu_thinking: bool = False  # sgl_eval backend
    mmlu_n_repeats: int = 1  # sgl_eval backend

    def test_mmlu(self):
        if self.mmlu_backend == "sgl_eval":
            _run_sgl_eval(
                self,
                eval_name="mmlu",
                score_threshold=self.mmlu_score_threshold,
                n_repeats=self.mmlu_n_repeats,
                num_examples=self.mmlu_num_examples,
                num_threads=self.mmlu_num_threads,
                thinking=self.mmlu_thinking,
                accept_length_thres=self.mmlu_accept_length_thres,
            )
        else:
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

    Runs via the sgl-eval Python API (the test is skipped if sgl-eval is not
    installed). ``gpqa_thinking`` defaults to True, which
    enables per-request thinking so the server separates reasoning from the final
    answer.

    Required attributes on the test class:
        base_url: str
        model: str
        gpqa_score_threshold: float

    Optional sampling knobs (default to sgl-eval's defaults when unset). Set these
    for reasoning models -- e.g. DeepSeek-V4 Think-Max wants
    gpqa_reasoning_effort="max", gpqa_max_tokens=200000, gpqa_temperature=1.0,
    gpqa_top_p=1.0. GPQA-Diamond is 198 questions; raise gpqa_n_repeats (e.g. 16)
    for a stable number.
    """

    gpqa_score_threshold: float = _THRESHOLD_NOT_SET
    gpqa_accept_length_thres: Optional[float] = None
    gpqa_num_examples: Optional[int] = None
    gpqa_num_threads: int = 1024
    gpqa_n_repeats: int = 1
    gpqa_thinking: bool = True
    gpqa_reasoning_effort: Optional[str] = None
    gpqa_max_tokens: Optional[int] = None
    gpqa_temperature: Optional[float] = None
    gpqa_top_p: Optional[float] = None

    def test_gpqa(self):
        _run_sgl_eval(
            self,
            eval_name="gpqa",
            score_threshold=self.gpqa_score_threshold,
            n_repeats=self.gpqa_n_repeats,
            num_examples=self.gpqa_num_examples,
            num_threads=self.gpqa_num_threads,
            thinking=self.gpqa_thinking,
            reasoning_effort=self.gpqa_reasoning_effort,
            max_tokens=self.gpqa_max_tokens,
            temperature=self.gpqa_temperature,
            top_p=self.gpqa_top_p,
            accept_length_thres=self.gpqa_accept_length_thres,
        )


class AIME25Mixin:
    """Mixin for AIME 2025 evaluation (competition math, integer answers).

    Runs via the sgl-eval Python API (the test is skipped if sgl-eval is not
    installed). ``aime25_thinking`` defaults to True, which
    enables per-request thinking so the server separates reasoning from the final
    answer.

    Required attributes on the test class:
        base_url: str
        model: str
        aime25_score_threshold: float

    Optional sampling knobs (default to sgl-eval's defaults when unset). Set these
    for reasoning models -- e.g. DeepSeek-V4 Think-Max wants
    aime25_reasoning_effort="max", aime25_max_tokens=200000, aime25_temperature=1.0,
    aime25_top_p=1.0. AIME25 has only 30 problems, so it is high variance; raise
    aime25_n_repeats (e.g. 16) for a stable number.
    """

    aime25_score_threshold: float = _THRESHOLD_NOT_SET
    aime25_accept_length_thres: Optional[float] = None
    aime25_num_examples: Optional[int] = None
    aime25_num_threads: int = 1024
    aime25_n_repeats: int = 1
    aime25_thinking: bool = True
    aime25_reasoning_effort: Optional[str] = None
    aime25_max_tokens: Optional[int] = None
    aime25_temperature: Optional[float] = None
    aime25_top_p: Optional[float] = None

    def test_aime25(self):
        _run_sgl_eval(
            self,
            eval_name="aime25",
            score_threshold=self.aime25_score_threshold,
            n_repeats=self.aime25_n_repeats,
            num_examples=self.aime25_num_examples,
            num_threads=self.aime25_num_threads,
            thinking=self.aime25_thinking,
            reasoning_effort=self.aime25_reasoning_effort,
            max_tokens=self.aime25_max_tokens,
            temperature=self.aime25_temperature,
            top_p=self.aime25_top_p,
            accept_length_thres=self.aime25_accept_length_thres,
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
