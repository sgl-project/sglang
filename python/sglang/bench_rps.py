import argparse
import json
import re
import subprocess
from datetime import datetime


def run_benchmark(request_rate, backend="sglang", num_prompts=1000):
    command = [
        "python3",
        "-m",
        "sglang.bench_serving",
        "--backend",
        backend,
        "--request-rate",
        str(request_rate),
        "--num-prompts",
        str(num_prompts),
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    ttft_pattern = r"Median TTFT.*?(\d+\.\d+)"
    ttft_match = re.search(ttft_pattern, result.stdout)

    itl_pattern = r"Median ITL.*?(\d+\.\d+)"
    itl_match = re.search(itl_pattern, result.stdout)

    output_token_throughput_pattern = r"Output token throughput.*?(\d+\.\d+)"
    output_token_throughput_match = re.search(
        output_token_throughput_pattern, result.stdout
    )

    if ttft_match and itl_match and output_token_throughput_match:
        return (
            float(ttft_match.group(1)),
            float(itl_match.group(1)),
            float(output_token_throughput_match.group(1)),
        )
    else:
        return None, None, None


def parse_request_rate_range(request_rate_range):
    start, stop, step = map(int, request_rate_range.split(","))
    return list(range(start, stop, step))


def main():
    parser = argparse.ArgumentParser(description="Run benchmark tests.")
    parser.add_argument(
        "--backend",
        type=str,
        default="sglang",
        help="The backend to use for benchmarking",
    )
    parser.add_argument(
        "--request-rate-range",
        type=str,
        default="2,34,2",
        help="Range of request rates in the format start,stop,step",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to use in the benchmark",
    )
    parser.add_argument("--output-file", type=str, help="Output JSONL file name")

    args = parser.parse_args()

    request_rates = parse_request_rate_range(args.request_rate_range)

    results = []

    for rate in request_rates:
        ttft, itl, output_token_throughput = run_benchmark(
            rate, args.backend, args.num_prompts
        )
        if ttft is not None and itl is not None and output_token_throughput is not None:
            result = {
                "request_rate": rate,
                "ttft": ttft,
                "itl": itl,
                "output_token_throughput": output_token_throughput,
            }
            results.append(result)
            print(f"Request Rate: {rate}")
            print(f"Median TTFT: {ttft}")
            print(f"Median ITL: {itl}")
            print(f"Output token throughput: {output_token_throughput}")
            print("-" * 30)
        else:
            print(f"Error running benchmark for request rate: {rate}")
            print("-" * 30)

    # Determine output file name
    if args.output_file:
        output_file_name = args.output_file
    else:
        now = datetime.now().strftime("%m%d%H")
        output_file_name = f"{args.backend}_{now}.jsonl"

    # Write results to a JSONL file
    with open(output_file_name, "w") as file:
        for result in results:
            file.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
