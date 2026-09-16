#!/usr/bin/env python3
"""
Phase III of the Reasoning-Aware Compression (RAC) recipe: serve one or more
checkpoints with SGLang and score them on MATH-500.

The point of this script is the *pair* of numbers it reports. The RAC paper
(https://arxiv.org/abs/2509.12464, ICLR 2026) shows that a badly calibrated
pruned reasoning model is not just less accurate -- it also rambles, emitting
far more chain-of-thought tokens for a worse answer, so it is slower than the
dense model it was supposed to speed up (paper Fig. 1, and the runtime columns
of Tables 1-2). Accuracy alone hides that. So every row below reports accuracy
next to mean completion length and wall clock.

Pass several checkpoints to compare calibration strategies head to head:

    python rac_serve_and_eval.py \
        --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
                     ./pruned_prompt_only \
                     ./pruned_rac \
        --num-problems 100

Note on grading: the boxed-answer matching here is intentionally simple, good
enough to rank checkpoints during development. For numbers you would put in a
paper, use the lighteval harness the RAC and open-r1 repos use.
"""

import argparse
import time
from typing import List, Optional

import msgspec

import sglang as sgl
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed "
    "responses. You first think about the reasoning process as an internal "
    "monologue and then provide the user with the answer. Respond in the "
    "following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
)


class EvalResult(msgspec.Struct, frozen=True):
    model_path: str
    num_problems: int
    num_correct: int
    mean_completion_tokens: float
    elapsed_seconds: float

    @property
    def accuracy(self) -> float:
        return self.num_correct / self.num_problems if self.num_problems else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score pruned reasoning checkpoints on MATH-500 with SGLang.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        nargs="+",
        required=True,
        help="One or more checkpoints. Several are evaluated in sequence and "
        "reported side by side.",
    )
    parser.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument(
        "--num-problems",
        type=int,
        default=500,
        help="Problems to score. The full MATH-500 set is 500.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Generation budget. The paper evaluates with 32768.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def extract_boxed(text: str) -> Optional[str]:
    """Return the content of the last \\boxed{...} in text, brace-matched."""
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    depth = 0
    for index in range(start + len(marker) - 1, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(marker) : index]

    return None


def normalize_answer(answer: str) -> str:
    """Strip the LaTeX noise that makes identical answers compare unequal."""
    normalized = answer.strip().rstrip(".").replace(" ", "")
    for token in ("\\left", "\\right", "\\!", "\\,", "$", "\\dfrac", "\\tfrac"):
        replacement = "\\frac" if token in ("\\dfrac", "\\tfrac") else ""
        normalized = normalized.replace(token, replacement)

    if normalized.startswith("\\text{") and normalized.endswith("}"):
        normalized = normalized[len("\\text{") : -1]

    return normalized


def is_correct(*, completion: str, reference: str) -> bool:
    predicted = extract_boxed(completion)
    if predicted is None:
        return False
    return normalize_answer(predicted) == normalize_answer(reference)


def load_problems(*, dataset: str, split: str, limit: int) -> tuple:
    from datasets import load_dataset

    rows = load_dataset(dataset, split=split)
    rows = rows.select(range(min(limit, len(rows))))
    return [row["problem"] for row in rows], [row["answer"] for row in rows]


def build_prompts(*, tokenizer, problems: List[str], system_prompt: str) -> List[str]:
    prompts = []
    for problem in problems:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": problem})
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        )
    return prompts


def evaluate(
    *,
    model_path: str,
    problems: List[str],
    references: List[str],
    sampling_params: dict,
    system_prompt: str,
    engine_kwargs: dict,
) -> EvalResult:
    tokenizer = get_tokenizer(model_path)
    prompts = build_prompts(
        tokenizer=tokenizer,
        problems=problems,
        system_prompt=system_prompt,
    )

    llm = sgl.Engine(model_path=model_path, **engine_kwargs)
    try:
        started_at = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed_seconds = time.perf_counter() - started_at
    finally:
        llm.shutdown()

    num_correct = sum(
        is_correct(completion=output["text"], reference=reference)
        for output, reference in zip(outputs, references)
    )
    total_completion_tokens = sum(
        output["meta_info"]["completion_tokens"] for output in outputs
    )

    return EvalResult(
        model_path=model_path,
        num_problems=len(problems),
        num_correct=num_correct,
        mean_completion_tokens=total_completion_tokens / len(problems),
        elapsed_seconds=elapsed_seconds,
    )


def report(results: List[EvalResult]) -> None:
    width = max(len(result.model_path) for result in results)

    print("\n=== MATH-500 ===")
    header = (
        f"{'model':<{width}}  {'acc@1':>7}  {'mean CoT tokens':>16}  {'wall clock':>12}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.model_path:<{width}}  "
            f"{result.accuracy:>7.3f}  "
            f"{result.mean_completion_tokens:>16.0f}  "
            f"{result.elapsed_seconds / 60:>10.1f}m"
        )

    if len(results) > 1:
        print(
            "\nA pruned model that scores worse *and* emits more CoT tokens is "
            "the failure mode RAC targets: calibration drift makes it ramble, "
            "so it is both less accurate and slower."
        )


def main() -> None:
    args = parse_args()

    problems, references = load_problems(
        dataset=args.dataset,
        split=args.dataset_split,
        limit=args.num_problems,
    )
    print(f"[rac] scoring {len(problems)} problems from {args.dataset}")

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    engine_kwargs = {"tp_size": args.tp_size, "random_seed": args.seed}
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static

    results = [
        evaluate(
            model_path=model_path,
            problems=problems,
            references=references,
            sampling_params=sampling_params,
            system_prompt=args.system_prompt,
            engine_kwargs=engine_kwargs,
        )
        for model_path in args.model_path
    ]

    report(results)


# sgl.Engine spawns subprocesses, so the entry point must be guarded.
if __name__ == "__main__":
    main()
