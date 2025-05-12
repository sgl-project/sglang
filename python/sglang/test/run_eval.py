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
    make_report,
    set_ulimit,
)


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
    else:
        raise ValueError(f"Invalid eval name: {args.eval_name}")

    sampler = ChatCompletionSampler(
        model=args.model,
        max_tokens=2048,
        base_url=base_url,
        temperature=getattr(args, "temperature", 0.0),
    )

    # Run eval
    tic = time.perf_counter()
    result = eval_obj(sampler)
    latency = time.perf_counter() - tic

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

    return metrics


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
    parser.add_argument("--eval-name", type=str, default="mmlu")
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--num-threads", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    run_eval(args)
