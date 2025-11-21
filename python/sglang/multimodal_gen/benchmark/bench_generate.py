import argparse

from sglang.multimodal_gen import DiffGenerator
from sglang.multimodal_gen.runtime.utils.performance_logger import (
    OnDemandFileHandler,
    _initialize_perf_logger,
    perf_logger,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark serving with DiffGenerator")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="Path to the model or HuggingFace model ID",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="Path to the file containing prompts, e.g., https://github.com/facebookresearch/MovieGenBench/blob/main/benchmark/MovieGenVideoBench.txt",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of prompts to use for benchmarking (default: all)",
    )
    return parser.parse_args()


def load_prompts_from_file(file_path, num_prompts=None):
    with open(file_path, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    if num_prompts is not None:
        prompts = prompts[:num_prompts]

    return prompts


def parse_performance_log(log_file_path):
    """Parse the performance log file and extract key metrics."""
    import json

    metrics = {"total_inference_times": [], "step_durations": [], "num_steps": 0}

    try:
        with open(log_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    if entry.get("tag") == "total_inference_time":
                        total_duration = entry.get("total_duration_ms")
                        if total_duration is not None:
                            metrics["total_inference_times"].append(total_duration)

                        steps = entry.get("steps", [])
                        if steps:
                            step_durations = [
                                step.get("duration_ms")
                                for step in steps
                                if step.get("duration_ms") is not None
                            ]
                            metrics["step_durations"].extend(step_durations)
                            metrics["num_steps"] = len(step_durations)
                except json.JSONDecodeError:
                    continue

        return metrics
    except FileNotFoundError:
        print(f"Warning: Performance log file {log_file_path} not found.")
        return metrics
    except Exception as e:
        print(f"Error parsing performance log: {e}")
        return metrics


def print_performance_metrics(metrics):
    if not metrics["total_inference_times"]:
        print("No performance data available.")
        return

    import statistics

    total_times = metrics["total_inference_times"]
    avg_total_time = statistics.mean(total_times)
    min_total_time = min(total_times)
    max_total_time = max(total_times)

    print("\n" + "=" * 50)
    print("Performance Metrics")
    print("=" * 50)
    print(f"Number of inferences: {len(total_times)}")
    print(
        f"Average total inference time: {avg_total_time:.2f} ms ({avg_total_time/1000:.2f} s)"
    )
    print(
        f"Min total inference time: {min_total_time:.2f} ms ({min_total_time/1000:.2f} s)"
    )
    print(
        f"Max total inference time: {max_total_time:.2f} ms ({max_total_time/1000:.2f} s)"
    )

    if len(total_times) > 1:
        # Calculate 95th and 99th percentiles
        sorted_times = sorted(total_times)
        n = len(sorted_times)

        # 95th percentile
        index_95 = int(0.95 * (n - 1))
        percentile_95 = sorted_times[index_95]
        print(
            f"95th percentile of total inference time: {percentile_95:.2f} ms ({percentile_95/1000:.2f} s)"
        )

        # 99th percentile
        index_99 = int(0.99 * (n - 1))
        percentile_99 = sorted_times[index_99]
        print(
            f"99th percentile of total inference time: {percentile_99:.2f} ms ({percentile_99/1000:.2f} s)"
        )

    if metrics["step_durations"]:
        step_durations = metrics["step_durations"]
        avg_step_duration = statistics.mean(step_durations)
        min_step_duration = min(step_durations)
        max_step_duration = max(step_durations)

        print(f"\nNumber of denoising steps per inference: {metrics['num_steps']}")
        print(
            f"Average time per denoising step: {avg_step_duration:.2f} ms ({avg_step_duration/1000:.2f} s)"
        )
        print(
            f"Min time for a denoising step: {min_step_duration:.2f} ms ({min_step_duration/1000:.2f} s)"
        )
        print(
            f"Max time for a denoising step: {max_step_duration:.2f} ms ({max_step_duration/1000:.2f} s)"
        )

        if len(step_durations) > 1:
            # Calculate 95th and 99th percentiles for step durations
            sorted_steps = sorted(step_durations)
            n = len(sorted_steps)

            # 95th percentile
            index_95 = int(0.95 * (n - 1))
            percentile_95 = sorted_steps[index_95]
            print(
                f"95th percentile of denoising step time: {percentile_95:.2f} ms ({percentile_95/1000:.2f} s)"
            )

            # 99th percentile
            index_99 = int(0.99 * (n - 1))
            percentile_99 = sorted_steps[index_99]
            print(
                f"99th percentile of denoising step time: {percentile_99:.2f} ms ({percentile_99/1000:.2f} s)"
            )

    print("=" * 50)


def main():
    args = parse_args()
    _initialize_perf_logger()

    performance_log = None
    for handler in perf_logger.handlers:
        if isinstance(handler, OnDemandFileHandler):
            performance_log = handler.baseFilename
    assert performance_log is not None, f"{performance_log=} is None"
    print(f"{performance_log=}")

    generator = DiffGenerator.from_pretrained(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
    )

    prompts = load_prompts_from_file(args.prompts_file, args.num_prompts)
    video = generator.generate(
        prompts,
        return_frames=False,
        save_output=False,
    )

    # Parse and print performance metrics
    metrics = parse_performance_log(performance_log)
    print_performance_metrics(metrics)


if __name__ == "__main__":
    main()
