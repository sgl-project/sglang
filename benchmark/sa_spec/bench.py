#!/usr/bin/env python3
"""Compare ordinary decoding, NGram, and suffix automaton on one GPU.

Run inside the TensorRT-LLM release:1.3.0rc25 container, from /work:

    /tmp/sa-download-env/bin/python bench.py prepare
    python bench.py run 10 1 smoke
    python bench.py run 100 3 results

The preparation environment needs huggingface_hub, datasets, and pyyaml.
Authenticate with `hf auth login` before downloading the gated Llama model.
Run benchmarks with the container's Python, not the preparation environment.

This reconstructs the code-edit workload from TensorRT-LLM PR #11434. Its
exact request sample and full configuration were not published. This script
uses a newer runtime and an explicit four-token draft budget.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import median

MODEL = "meta-llama/Llama-3.1-8B"
DATASET = "glaiveai/code_edits_sample"
NUM_REQUESTS = 100
OUTPUT_TOKENS = 256
MODES = ("baseline", "ngram", "sa3", "sa_longest")


def prepare():
    """Download pinned inputs and write the shared dataset and configurations."""
    import yaml
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    manifest_path = Path("manifest.json")

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "model": MODEL,
            "model_revision": api.model_info(MODEL).sha,
            "dataset": DATASET,
            "dataset_revision": api.dataset_info(DATASET).sha,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

    snapshot_download(
        MODEL,
        revision=manifest["model_revision"],
        local_dir="model",
        allow_patterns=["*.json", "*.safetensors", "*.model"],
        ignore_patterns=["original/*"],
    )

    rows = load_dataset(
        DATASET,
        revision=manifest["dataset_revision"],
        split="train",
        streaming=True,
    )

    requests = []
    for index, row in enumerate(rows.take(NUM_REQUESTS)):
        prompt = (
            "\nApply the following edit to the following code:\n\n"
            f"<edit>\n{row['edit']}\n</edit>\n\n\n"
            f"<code>\n{row['code']}\n</code>\n\n\n"
            "Result:\n<edited_code>\n"
        )
        requests.append(
            {
                "task_id": index,
                "prompt": prompt,
                "output_tokens": OUTPUT_TOKENS,
            }
        )

    if len(requests) != NUM_REQUESTS:
        raise RuntimeError(f"Expected {NUM_REQUESTS} dataset rows, got {len(requests)}")

    data = "".join(json.dumps(request) + "\n" for request in requests)
    Path("dataset.jsonl").write_text(data, encoding="utf-8")
    manifest["row_indices"] = list(range(NUM_REQUESTS))
    manifest["dataset_sha256"] = hashlib.sha256(data.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2))

    common = {
        "dtype": "bfloat16",
        "disable_overlap_scheduler": False,
        "cuda_graph_config": {"batch_sizes": [1]},
        "kv_cache_config": {"enable_block_reuse": False},
    }
    specs = {
        "baseline": None,
        "ngram": {
            "decoding_type": "NGram",
            "max_draft_len": 4,
            "max_matching_ngram_size": 3,
            "is_keep_all": False,
            "is_public_pool": False,
        },
        "sa3": {
            "decoding_type": "SA",
            "max_draft_len": 4,
            "max_matching_ngram_size": 3,
            "enable_global_pool": False,
        },
        "sa_longest": {
            "decoding_type": "SA",
            "max_draft_len": 4,
            "max_matching_ngram_size": -1,
            "enable_global_pool": False,
        },
    }

    Path("configs").mkdir(exist_ok=True)
    for mode, spec in specs.items():
        config = dict(common)
        if spec is not None:
            config["speculative_config"] = spec
        Path(f"configs/{mode}.yaml").write_text(yaml.safe_dump(config))

    sampler = {"temperature": 0, "top_k": 1, "ignore_eos": True}
    Path("sampler.yaml").write_text(yaml.safe_dump(sampler))
    print("Downloads and configuration complete.", flush=True)


def print_summary(reports, directory):
    """Print median throughput, latency, and decoded-text consistency."""
    throughput = {
        mode: median(
            report["performance"]["system_output_throughput_tok_s"]
            for report in reports[mode]
        )
        for mode in MODES
    }
    reference = json.loads((directory / "baseline-1.outputs.json").read_text())

    print("\nMedian results:")
    print(
        f"{'Mode':<13} {'tok/s':>10} {'TPOT ms':>10} "
        f"{'vs baseline':>13} {'vs NGram':>11} {'Same text':>10}"
    )

    for mode in MODES:
        tpot = median(
            report["streaming_metrics"]["avg_tpot_ms"] for report in reports[mode]
        )
        same_text = all(
            json.loads(path.read_text()) == reference
            for path in directory.glob(f"{mode}-*.outputs.json")
        )
        baseline_gain = 100 * (throughput[mode] / throughput["baseline"] - 1)
        ngram_gain = 100 * (throughput[mode] / throughput["ngram"] - 1)
        print(
            f"{mode:<13} {throughput[mode]:>10.2f} {tpot:>10.3f} "
            f"{baseline_gain:>+12.1f}% {ngram_gain:>+10.1f}% "
            f"{str(same_text):>10}"
        )


def run(num_requests, repeats, directory):
    """Run all modes on one GPU, rotating their order between repetitions."""
    if not 1 <= num_requests <= NUM_REQUESTS or repeats < 1:
        raise ValueError(f"Use 1–{NUM_REQUESTS} requests and at least one repetition")

    directory = Path(directory)
    directory.mkdir()  # A new directory keeps each experiment's reports separate.
    reports = {mode: [] for mode in MODES}

    for repetition in range(1, repeats + 1):
        offset = (repetition - 1) % len(MODES)
        order = MODES[offset:] + MODES[:offset]

        for mode in order:
            prefix = directory / f"{mode}-{repetition}"
            print(f"\nRunning {mode}, repetition {repetition}", flush=True)

            command = [
                "trtllm-bench",
                "--model",
                MODEL,
                "--model_path",
                str(Path("model").resolve()),
                "throughput",
                "--backend",
                "pytorch",
                "--dataset",
                str(Path("dataset.jsonl").resolve()),
                "--tp",
                "1",
                "--max_batch_size",
                "1",
                "--max_num_tokens",
                "8192",
                "--concurrency",
                "1",
                "--num_requests",
                str(num_requests),
                "--warmup",
                "10",
                "--streaming",
                "--disable_chunked_context",
                "--scheduler_policy",
                "guaranteed_no_evict",
                "--config",
                f"configs/{mode}.yaml",
                "--sampler_options",
                "sampler.yaml",
                "--report_json",
                f"{prefix}.report.json",
                "--output_json",
                f"{prefix}.outputs.json",
            ]
            subprocess.run(command, check=True)

            report = json.loads(Path(f"{prefix}.report.json").read_text())
            if report["request_info"]["num_requests"] != num_requests:
                raise RuntimeError(f"{prefix}: unexpected completed request count")
            if report["request_info"]["avg_output_length"] != OUTPUT_TOKENS:
                raise RuntimeError(f"{prefix}: unexpected output length")
            reports[mode].append(report)

    print_summary(reports, directory)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Download inputs and generate configurations")

    run_parser = subparsers.add_parser("run", help="Run all four decoding modes")
    run_parser.add_argument(
        "requests", type=int, help="Number of requests, from 1 to 100"
    )
    run_parser.add_argument("repeats", type=int, help="Number of measured repetitions")
    run_parser.add_argument("directory", help="A new directory for the results")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        run(args.requests, args.repeats, args.directory)


if __name__ == "__main__":
    main()
