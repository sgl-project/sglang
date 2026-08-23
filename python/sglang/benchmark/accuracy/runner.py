"""Accuracy evaluation runner for benchmark.serving (#1096)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

ACCURACY_DATASET_NAMES = frozenset(
    {
        "gsm8k",
        "gpqa",
        "aime",
        "aime24",
        "aime25",
        "aime26",
        "mmmu",
    }
)

_DATASET_TO_EVAL_NAME = {
    "gsm8k": "gsm8k",
    "gpqa": "gpqa",
    "aime": "aime25",
    "aime24": "aime24",
    "aime25": "aime25",
    "aime26": "aime26",
    "mmmu": "mmmu",
}


@dataclass
class AccuracyResult:
    dataset: str
    accuracy: float
    num_samples: int
    model: Optional[str] = None
    eval_name: Optional[str] = None
    latency: Optional[float] = None
    output_throughput: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "dataset": self.dataset,
            "accuracy": self.accuracy,
            "num_samples": self.num_samples,
            "model": self.model,
            "eval_name": self.eval_name,
            "latency": self.latency,
            "output_throughput": self.output_throughput,
        }
        payload.update(self.extra)
        return payload


def resolve_eval_name(dataset_name: str, aime_year: Optional[int] = None) -> str:
    if dataset_name == "aime":
        year = aime_year or 25
        return f"aime{year}"
    return _DATASET_TO_EVAL_NAME[dataset_name]


def _build_run_eval_args(args, base_url: str, eval_name: str) -> SimpleNamespace:
    num_threads = getattr(args, "accuracy_num_threads", None)
    if num_threads is None:
        num_threads = args.max_concurrency or 64

    num_examples = args.num_prompts
    if num_examples is None or num_examples <= 0:
        num_examples = None

    api = "chat"
    if eval_name == "gsm8k":
        api = "completion"

    return SimpleNamespace(
        base_url=base_url,
        host=args.host,
        port=args.port,
        model=args.model,
        eval_name=eval_name,
        api=api,
        num_examples=num_examples,
        num_threads=num_threads,
        max_tokens=getattr(args, "accuracy_max_tokens", 2048),
        temperature=getattr(args, "temperature", 0.0),
        top_p=getattr(args, "top_p", 1.0),
        repeat=1,
        num_shots=getattr(args, "gsm8k_num_shots", 5),
        gsm8k_data_path=getattr(args, "gsm8k_data_path", None),
        response_answer_regex=getattr(args, "response_answer_regex", None),
        return_latency=False,
    )


def run_accuracy_eval(args, base_url: str) -> AccuracyResult:
    """
    Run dataset accuracy via sglang.test.run_eval (reuses simple_eval graders).
    """
    import os

    hf_endpoint = getattr(args, "hf_endpoint", None) or os.environ.get("HF_ENDPOINT")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    from sglang.test.run_eval import run_eval

    dataset_name = args.dataset_name
    if dataset_name not in ACCURACY_DATASET_NAMES:
        raise ValueError(
            f"dataset '{dataset_name}' is not an accuracy dataset; "
            f"choose one of {sorted(ACCURACY_DATASET_NAMES)}"
        )

    eval_name = resolve_eval_name(
        dataset_name, getattr(args, "aime_year", None)
    )
    eval_args = _build_run_eval_args(args, base_url, eval_name)
    metrics = run_eval(eval_args)

    score = metrics.get("score", metrics.get("mean_score"))
    if score is None:
        raise KeyError("run_eval metrics missing score")

    num_samples = metrics.get("num_examples")
    if num_samples is None:
        num_samples = eval_args.num_examples or metrics.get("n", 0)
    if not num_samples:
        num_samples = args.num_prompts

    return AccuracyResult(
        dataset=dataset_name,
        accuracy=float(score),
        num_samples=int(num_samples) if num_samples else 0,
        model=args.model,
        eval_name=eval_name,
        latency=metrics.get("latency"),
        output_throughput=metrics.get("output_throughput"),
        extra={
            k: v
            for k, v in metrics.items()
            if k not in {"score", "latency", "output_throughput", "mean_score"}
        },
    )


def write_accuracy_result(
    result: AccuracyResult, output_path: Optional[str] = None
) -> str:
    path = output_path or f"accuracy_{result.dataset}.json"
    if not path:
        path = f"accuracy_{result.dataset}.json"

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)
        fh.write("\n")
    return str(out)
