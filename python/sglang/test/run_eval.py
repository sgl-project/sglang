"""
Usage:
python3 -m sglang.test.run_eval --port 30000 --eval-name mmlu --num-examples 10
"""

import argparse
import json
import os
import time

from sglang.test.simple_eval_common import (
    ChatCompletionSampler,
    Eval,
    make_report,
    set_ulimit,
)


def get_thinking_kwargs(args):
    thinking_mode = getattr(args, "thinking_mode", None)
    if thinking_mode in THINKING_MODE_CHOICES:
        if thinking_mode == "deepseek-v3":
            thinking_param = "thinking"
        else:
            thinking_param = "enable_thinking"
        return {
            "chat_template_kwargs": {thinking_param: True},
        }
    return {}


def run_eval_once(args, base_url: str, eval_obj: Eval) -> dict:
    # Get thinking kwargs based on user's choice
    thinking_kwargs = get_thinking_kwargs(args)

    sampler = ChatCompletionSampler(
        model=args.model,
        max_tokens=getattr(args, "max_tokens", 2048),
        base_url=base_url,
        temperature=getattr(args, "temperature", 0.0),
        reasoning_effort=getattr(args, "reasoning_effort", None),
        extra_body=thinking_kwargs,
    )

    # Run eval
    tic = time.perf_counter()
    result = eval_obj(sampler)
    latency = time.perf_counter() - tic

    return result, latency, sampler


def run_eval(args):
    set_ulimit()

    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "EMPTY"

    base_url = (
        f"{args.base_url}/v1" if args.base_url else f"http://{args.host}:{args.port}/v1"
    )

    if args.eval_name == "mmlu":
        from sglang.test.simple_eval_mmlu import MMLUEval

        filename = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
        eval_obj = MMLUEval(filename, args.num_examples, args.num_threads)
    elif args.eval_name == "math":
        from sglang.test.simple_eval_math import MathEval

        equality_checker = ChatCompletionSampler(model="gpt-4-turbo")

        filename = (
            "https://openaipublic.blob.core.windows.net/simple-evals/math_test.csv"
        )
        eval_obj = MathEval(
            filename, equality_checker, args.num_examples, args.num_threads
        )
    elif args.eval_name == "mgsm":
        from sglang.test.simple_eval_mgsm import MGSMEval

        eval_obj = MGSMEval(args.num_examples, args.num_threads)
    elif args.eval_name == "mgsm_en":
        from sglang.test.simple_eval_mgsm import MGSMEval

        eval_obj = MGSMEval(args.num_examples, args.num_threads, languages=["en"])
    elif args.eval_name == "gpqa":
        from sglang.test.simple_eval_gpqa import GPQAEval

        filename = (
            "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
        )
        eval_obj = GPQAEval(filename, args.num_examples, args.num_threads)
    elif args.eval_name == "humaneval":
        from sglang.test.simple_eval_humaneval import HumanEval

        eval_obj = HumanEval(args.num_examples, args.num_threads)
    elif args.eval_name == "longbench_v2":
        from sglang.test.simple_eval_longbench_v2 import LongBenchV2Eval

        # Default to HuggingFace dataset, can be overridden with --dataset-path
        data_source = args.dataset_path
        categories = args.categories.split(",") if args.categories else None

        eval_obj = LongBenchV2Eval(
            model=args.model,
            data_source=data_source,
            num_examples=args.num_examples,
            num_threads=args.num_threads,
            categories=categories,
            max_context_length=getattr(args, "max_context_length", None),
            min_context_length=getattr(args, "min_context_length", None),
        )
    elif args.eval_name == "mmmu":
        # VLM MMMU evaluation with fixed 100 examples by default
        from sglang.test.simple_eval_mmmu_vlm import MMMUVLMEval

        eval_obj = MMMUVLMEval(args.num_examples, args.num_threads)
    else:
        raise ValueError(f"Invalid eval name: {args.eval_name}")

    if getattr(args, "repeat", 1) == 1:
        result, latency, sampler = run_eval_once(args, base_url, eval_obj)
    else:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=args.repeat)

        futures = [
            executor.submit(run_eval_once, args, base_url, eval_obj)
            for _ in range(args.repeat)
        ]

        scores_repeat = []

        for f in futures:
            result, latency, sampler = f.result()
            scores_repeat.append(result.score)

        mean_score = sum(scores_repeat) / len(scores_repeat)
        scores_repeat = [f"{s:.3f}" for s in scores_repeat]
        print("=" * 20)
        print(f"Repeat: {args.repeat}, mean: {mean_score:.3f}")
        print(f"Scores: {scores_repeat}")
        print("=" * 20)

        executor.shutdown()

    # Dump reports
    metrics = result.metrics | {"score": result.score}
    file_stem = f"{args.eval_name}_{sampler.model.replace('/', '_')}"
    report_filename = f"/tmp/{file_stem}.html"
    print(f"Writing report to {report_filename}")
    with open(report_filename, "w") as fh:
        fh.write(make_report(result))
    metrics = result.metrics | {"score": result.score}
    print(metrics)
    result_filename = f"/tmp/{file_stem}.json"
    with open(result_filename, "w") as f:
        f.write(json.dumps(metrics, indent=2))
    print(f"Writing results to {result_filename}")

    # Print results
    print(f"Total latency: {latency:.3f} s")
    print(f"Score: {metrics['score']:.3f}")

    if getattr(args, "return_latency", False):
        return metrics, latency
    return metrics


THINKING_MODE_CHOICES = ["deepseek-r1", "deepseek-v3", "qwen3"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Default host is 0.0.0.0."
    )
    parser.add_argument(
        "--port",
        type=int,
        help="If not set, the default port is configured according to its default value for different LLM Inference Engines.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name or path of the model. If not set, the default model will request /v1/models for conf.",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="repeat the evaluation n times"
    )
    parser.add_argument("--eval-name", type=str, default="mmlu")
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--num-threads", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", type=str)
    parser.add_argument(
        "--thinking-mode",
        default=None,
        type=str,
        choices=THINKING_MODE_CHOICES,
        help="Enable thinking mode in Deepseek R1, V3.1/3.2, or Qwen3",
    )

    # LongBench-v2 specific arguments
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="THUDM/LongBench-v2",
        help="Path to dataset file or HuggingFace dataset name for LongBench-v2",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated list of categories to evaluate for LongBench-v2",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        help="Maximum context length in characters for LongBench-v2",
    )
    parser.add_argument(
        "--min-context-length",
        type=int,
        help="Minimum context length in characters for LongBench-v2",
    )

    args = parser.parse_args()

    run_eval(args)
