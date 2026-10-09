from types import SimpleNamespace
from typing import Optional

import requests

from sglang.test.run_eval import run_eval
from sglang.test.sgl_eval_utils import SGL_EVAL_BENCHMARKS, run_sgl_eval
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
    num_threads: Optional[int],
    accept_length_thres: Optional[float] = None,
    summary_label: Optional[str] = None,
    **eval_overrides,
):
    """Shared driver for the accuracy mixins below.

    ``eval_overrides`` are forwarded only when not ``None``, so unset knobs keep
    the evaluator's defaults.
    """
    assert score_threshold == score_threshold, (
        f"{type(test_case).__name__} must set the {eval_name} score threshold"
    )

    model = eval_overrides.pop("model", getattr(test_case, "model", None))
    kwargs = dict(
        base_url=test_case.base_url,
        model=model,
        eval_name=eval_name,
        num_examples=num_examples,
        num_threads=num_threads,
    )
    kwargs.update({k: v for k, v in eval_overrides.items() if v is not None})

    evaluate = run_sgl_eval if eval_name in SGL_EVAL_BENCHMARKS else run_eval
    metrics = evaluate(SimpleNamespace(**kwargs))
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
    num_examples: Optional[int] = None,
    num_threads: int = 512,
    thinking: bool = True,
    chat_template_kwargs: Optional[dict] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    accept_length_thres: Optional[float] = None,
):
    """Shared sgl-eval driver for the MMLU sanity gate and the GSM8K ``sgl_eval`` backend.

    ``thinking=True`` sends per-request ``chat_template_kwargs={"thinking": True}``
    so the server separates reasoning from the final answer.
    """
    assert score_threshold == score_threshold, (
        f"{type(test_case).__name__} must set the {eval_name} score threshold"
    )

    try:
        import sgl_eval  # noqa: F401
    except ImportError:
        test_case.skipTest("sgl-eval not installed; pip install 'sglang[test]'")

    args = SimpleNamespace(
        eval_name=eval_name,
        base_url=test_case.base_url,
        model=getattr(test_case, "model", None),
        num_examples=num_examples,
        num_threads=num_threads,
        max_tokens=max_tokens,
        temperature=temperature,
        chat_template_kwargs=(
            chat_template_kwargs
            if chat_template_kwargs is not None
            else ({"thinking": True} if thinking else None)
        ),
    )
    metrics = run_sgl_eval(args)
    score = metrics["score"]
    print(f"{eval_name} sgl-eval score={score:.4f}")
    _finalize_eval(
        test_case,
        eval_name=eval_name,
        score=score,
        score_threshold=score_threshold,
        accept_length_thres=accept_length_thres,
    )
    return metrics


class MMLUSanityMixin:
    """Short MMLU accuracy gate shared by ordinary and speculative sanity tests."""

    mmlu_score_threshold: float = 0.60
    mmlu_accept_length_thres: Optional[float] = None

    def test_accuracy_floor(self):
        _run_sgl_eval(
            self,
            eval_name="mmlu",
            score_threshold=self.mmlu_score_threshold,
            num_examples=200,
            num_threads=64,
            thinking=False,
            chat_template_kwargs={"enable_thinking": False},
            max_tokens=1024,
            temperature=0,
            accept_length_thres=self.mmlu_accept_length_thres,
        )


class GSM8KMixin:
    """Mixin for GSM8K evaluation.

    ``"run_eval"`` backend: OpenAI completion API, 5-shot; ``"sgl_eval"``: sgl-eval
    chat + boxed/sympy grader. The legacy ``gsm8k_accuracy_thres`` /
    ``gsm8k_num_questions`` are honored when the canonical knobs are unset.
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
    gsm8k_max_tokens: Optional[int] = None  # sgl_eval backend
    # None keeps run_eval's greedy default; set both to route the run through
    # the sampling path.
    gsm8k_temperature: Optional[float] = None
    gsm8k_top_p: Optional[float] = None

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
                num_examples=num_examples,
                num_threads=self.gsm8k_num_threads,
                thinking=self.gsm8k_thinking,
                max_tokens=self.gsm8k_max_tokens,
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
                temperature=self.gsm8k_temperature,
                top_p=self.gsm8k_top_p,
            )


class MMLUMixin:
    """Mixin for MMLU evaluation via sgl-eval (2048-token cap, no thinking)."""

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


class MMMUProMixin:
    """Mixin for the standard 10-option MMMU-Pro evaluation via sgl-eval.

    The model preset supplies the endpoint model and all generation settings;
    reasoning models' token budget and sampling differ from run_eval defaults.
    """

    mmmu_pro_score_threshold: float = _THRESHOLD_NOT_SET
    mmmu_pro_accept_length_thres: Optional[float] = None
    mmmu_pro_num_examples: Optional[int] = 300
    mmmu_pro_num_threads: Optional[int] = None
    mmmu_pro_load_preset_from_model_id: Optional[str] = None

    def test_mmmu_pro(self):
        assert self.mmmu_pro_load_preset_from_model_id, (
            f"{type(self).__name__} must set mmmu_pro_load_preset_from_model_id"
        )
        _run_accuracy_eval(
            self,
            eval_name="mmmu_pro",
            score_threshold=self.mmmu_pro_score_threshold,
            num_examples=self.mmmu_pro_num_examples,
            num_threads=self.mmmu_pro_num_threads,
            accept_length_thres=self.mmmu_pro_accept_length_thres,
            model=None,
            load_preset_from_model_id=self.mmmu_pro_load_preset_from_model_id,
        )


class HumanEvalMixin:
    """Mixin for HumanEval evaluation."""

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
    """Mixin for MGSM English evaluation."""

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
