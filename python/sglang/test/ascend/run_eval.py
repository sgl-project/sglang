import json
import os

from sglang.test.run_eval import run_eval_once
from sglang.test.simple_eval_common import (
    make_report,
    set_ulimit,
)


def run_eval(args):
    # Lazy import to avoid circular dependency with test_utils
    from sglang.test.test_utils import dump_metric

    set_ulimit()

    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "EMPTY"

    base_url = (
        f"{args.base_url}/v1" if args.base_url else f"http://{args.host}:{args.port}/v1"
    )

    if args.eval_name == "mmlu":
        from sglang.test.ascend.simple_eval_mmlu import MMLUEval

        filename = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
        eval_obj = MMLUEval(
            filename, args.num_examples, args.num_threads, getattr(args, "num_shots", 0)
        )
    else:
        raise ValueError(f"Invalid eval name: {args.eval_name}")

    if getattr(args, "repeat", 1) == 1:
        result, latency, sampler = run_eval_once(args, base_url, eval_obj)
        metrics = result.metrics | {"score": result.score}
        metrics["latency"] = latency
        print(f"Total latency: {latency:.3f} s")
        print(f"Score: {metrics['score']:.3f}")

        # Compute output throughput from accumulated completion tokens
        total_completion_tokens = sum(sampler._completion_tokens)
        if total_completion_tokens > 0 and latency > 0:
            metrics["output_throughput"] = total_completion_tokens / latency
            print(f"Output throughput: {metrics['output_throughput']:.3f} token/s")

        # Report metrics to unified collection framework
        dump_metric(
            f"{args.eval_name}_score",
            metrics["score"],
            labels={"model": sampler.model, "eval": args.eval_name},
        )
        dump_metric(
            f"{args.eval_name}_latency",
            latency,
            labels={"model": sampler.model, "eval": args.eval_name},
        )
    else:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=args.repeat)

        futures = [
            executor.submit(run_eval_once, args, base_url, eval_obj)
            for _ in range(args.repeat)
        ]

        scores_repeat = []
        latencies = []
        total_completion_tokens = 0

        for f in futures:
            result, latency, sampler = f.result()
            scores_repeat.append(result.score)
            latencies.append(latency)
            total_completion_tokens += sum(sampler._completion_tokens)

        mean_score = sum(scores_repeat) / len(scores_repeat)
        mean_latency = sum(latencies) / len(latencies)
        total_latency = sum(latencies)
        scores_repeat = [f"{s:.3f}" for s in scores_repeat]
        print("=" * 20)
        print(f"Repeat: {args.repeat}, mean: {mean_score:.3f}")
        print(f"Scores: {scores_repeat}")
        print(f"Mean latency: {mean_latency:.3f} s")
        print("=" * 20)
        metrics = result.metrics | {"scores": scores_repeat}
        metrics = metrics | {"mean_score": mean_score}
        metrics["latency"] = mean_latency

        if total_completion_tokens > 0 and total_latency > 0:
            metrics["output_throughput"] = total_completion_tokens / total_latency
            print(f"Output throughput: {metrics['output_throughput']:.3f} token/s")

        # Report metrics to unified collection framework
        dump_metric(
            f"{args.eval_name}_mean_score",
            mean_score,
            labels={
                "model": sampler.model,
                "eval": args.eval_name,
                "repeat": args.repeat,
            },
        )

        executor.shutdown()

    # Dump reports
    file_stem = f"{args.eval_name}_{sampler.model.replace('/', '_')}"
    report_filename = f"/tmp/{file_stem}.html"
    print(f"Writing report to {report_filename}")
    with open(report_filename, "w") as fh:
        fh.write(make_report(result))
    print(metrics)
    result_filename = f"/tmp/{file_stem}.json"
    with open(result_filename, "w") as f:
        f.write(json.dumps(metrics, indent=2))
    print(f"Writing results to {result_filename}")

    if getattr(args, "return_latency", False):
        return metrics, latency
    return metrics
