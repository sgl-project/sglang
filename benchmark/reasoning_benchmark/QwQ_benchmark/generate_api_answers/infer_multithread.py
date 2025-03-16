import argparse
import collections
import concurrent.futures
import copy
import json
import os
import threading

from tqdm import tqdm
from utils_vllm import get_content

file_lock = threading.Lock()


def count_completed_samples(output_file):
    prompt_counts = collections.defaultdict(int)
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    prompt = item["prompt"]
                    gen_count = len(item.get("gen", []))
                    prompt_counts[prompt] += gen_count
                except json.JSONDecodeError:
                    continue
    return prompt_counts


def process_item(item, output_file, base_url, model_name):
    result = copy.deepcopy(item)

    response = get_content(item["prompt"], base_url, model_name)

    if "gen" not in result:
        result["gen"] = []

    result["gen"].append(response)
    with file_lock:
        with open(output_file, "a", encoding="utf-8") as g:
            g.write(json.dumps(result, ensure_ascii=False) + "\n")
            g.flush()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on model with prompts from a jsonl file"
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Input jsonl file path"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Output file path"
    )
    parser.add_argument(
        "--n_samples", type=int, default=64, help="Number of samples per prompt"
    )
    parser.add_argument(
        "--max_workers", type=int, default=128, help="Maximum number of worker threads"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://10.77.249.36:8030/v1",
        help="base url of vllm server",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/QwQ-32B",
        help="model name of vllm server",
    )
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f]

    if os.path.exists(args.output_file):
        completed_counts = count_completed_samples(args.output_file)
        total_completed = sum(completed_counts.values())
        print(f"Found {total_completed} completed samples from previous run")
    else:
        with open(args.output_file, "w", encoding="utf-8") as g:
            completed_counts = dict()

    expanded_data = []
    for item in data:
        prompt = item["prompt"]
        completed = completed_counts.get(prompt, 0)
        remaining = args.n_samples - completed
        for _ in range(remaining):
            expanded_data.append(copy.deepcopy(item))

    total_tasks = len(expanded_data)
    print(f"Total remaining samples to process: {total_tasks}")

    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        future_to_item = {
            executor.submit(
                process_item, item, args.output_file, args.base_url, args.model_name
            ): i
            for i, item in enumerate(expanded_data)
        }

        with tqdm(total=len(expanded_data), desc="Processing samples") as pbar:
            for future in concurrent.futures.as_completed(future_to_item):
                idx = future_to_item[future]
                try:
                    future.result()
                    completed_count += 1
                except Exception as exc:
                    print(f"Error processing sample {idx}: {exc}")
                pbar.update(1)

    print(f"Completed {completed_count}/{len(expanded_data)} samples")
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
