"""CPU comparison of baseline and optimized offset-based scheduler padding.

Run from the repository root:
    PYTHONPATH=python python benchmark/io/bench_mm_token_padding.py --baseline-ref BASE_SHA

Inputs are synthetic. Both paths include the scheduler's final ownership copy.
This measures padding CPU time, not model forward time or end-to-end TTFT.
"""

import argparse
import ast
import gc
import json
import platform
import statistics
import subprocess
import time
from array import array
from pathlib import Path

import torch

from sglang.srt.managers import mm_utils
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    duplicate_pad_mm_input_ids,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)


def load_baseline(ref):
    """Load the exact baseline helper classes without replacing installed SGLang."""
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=root, text=True
    ).strip()
    path = "python/sglang/srt/managers/mm_utils.py"
    source = subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=root, text=True
    )
    names = {
        "MultiModalityDataPaddingPattern",
        "MultiModalityDataPaddingPatternMultimodalTokens",
    }
    classes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    if {node.name for node in classes} != names:
        raise ValueError("Baseline does not contain the expected padding helpers")
    namespace = dict(vars(mm_utils))
    # Reuse dependency imports, but compile the original method bodies verbatim.
    exec(
        compile(ast.Module(body=classes, type_ignores=[]), f"{commit}:{path}", "exec"),
        namespace,
    )
    return commit, namespace["MultiModalityDataPaddingPatternMultimodalTokens"]()


def make_case(num_tokens, num_items, span_tokens):
    if num_items and num_tokens < 3 * num_items:
        raise ValueError(
            "Each item needs three tokens for its span and surrounding text"
        )
    input_ids = array("q", range(257, 257 + num_tokens))
    items = []
    stride = num_tokens // max(num_items, 1)
    for index in range(num_items):
        start = index * stride + 1
        end = start + min(span_tokens, stride - 2) - 1
        items.append(
            MultimodalDataItem(
                modality=Modality.IMAGE,
                pad_value=2**40 + index,
                offsets=[(start, end)],
            )
        )
    return input_ids, MultimodalInputs(mm_items=items, im_token_id=1)


def benchmark_case(num_tokens, num_items, args, baseline):
    pattern = MultiModalityDataPaddingPatternMultimodalTokens()
    input_ids, mm_inputs = make_case(num_tokens, num_items, args.span_tokens)
    original = input_ids.tobytes()

    def baseline_padding():
        return array("q", baseline.pad_input_tokens(input_ids, mm_inputs))

    def optimized_padding():
        return duplicate_pad_mm_input_ids(
            input_ids, mm_inputs, pattern.pad_input_tokens
        )

    expected = baseline_padding()
    actual = optimized_padding()
    assert actual == expected, "Token values differ"
    assert input_ids.tobytes() == original, "Original input was mutated"
    assert actual is not input_ids, "Scheduler result aliases the original input"

    functions = {"baseline": baseline_padding, "optimized": optimized_padding}
    timings = {name: [] for name in functions}
    for _ in range(args.warmups):
        for function in functions.values():
            function()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for trial in range(args.trials):
            order = (
                ("baseline", "optimized") if trial % 2 else ("optimized", "baseline")
            )
            for name in order:
                start = time.perf_counter_ns()
                result = functions[name]()
                elapsed = time.perf_counter_ns() - start
                timings[name].append(elapsed / 1e6)
                del result
    finally:
        if gc_was_enabled:
            gc.enable()
    assert input_ids.tobytes() == original
    medians = {name: statistics.median(samples) for name, samples in timings.items()}
    return {
        "tokens": num_tokens,
        "items": num_items,
        "baseline_ms": medians["baseline"],
        "optimized_ms": medians["optimized"],
        "speedup": medians["baseline"] / medians["optimized"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-ref", required=True, help="Local pre-change git ref"
    )
    parser.add_argument(
        "--tokens", nargs="+", type=int, default=[128, 4096, 16384, 131072]
    )
    parser.add_argument("--items", nargs="+", type=int, default=[0, 1, 10])
    parser.add_argument("--span-tokens", type=int, default=32)
    parser.add_argument("--trials", type=int, default=41)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if (
        min(args.tokens + args.items) < 0
        or min(args.span_tokens, args.trials, args.threads) < 1
        or args.warmups < 0
    ):
        parser.error("Lengths/items/warmups must be nonnegative; other counts positive")
    if any(n < 3 * items for n in args.tokens for items in args.items):
        parser.error("Each token length must accommodate at least 3 tokens per item")
    torch.set_num_threads(args.threads)
    baseline_commit, baseline = load_baseline(args.baseline_ref)
    rows = [
        benchmark_case(tokens, items, args, baseline)
        for tokens in args.tokens
        for items in args.items
    ]
    if args.json:
        print(
            json.dumps(
                {
                    "baseline_commit": baseline_commit,
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "architecture": platform.machine(),
                    "threads": args.threads,
                    "trials": args.trials,
                    "warmups": args.warmups,
                    "span_tokens": args.span_tokens,
                    "results": rows,
                },
                indent=2,
            )
        )
        return
    print("Offset-based CPU padding; median milliseconds; synthetic inputs")
    print(
        f"{'Tokens':>8} {'Items':>6} {'Baseline (ms)':>14} {'Optimized (ms)':>14} {'Speedup':>9}"
    )
    for row in rows:
        print(
            f"{row['tokens']:>8} {row['items']:>6} "
            f"{row['baseline_ms']:>14.4f} {row['optimized_ms']:>14.4f} "
            f"{row['speedup']:>8.2f}x"
        )


if __name__ == "__main__":
    main()
