"""
Compare kernel shapes between different inference runs.

This is useful for comparing:
- Prefill phase vs Decode phase
- Different batch sizes
- Different sequence lengths
- Different models

Usage:
    python compare_shapes.py <log1.jsonl> <log2.jsonl> [options]
    
Examples:
    # Compare two runs
    python compare_shapes.py prefill.jsonl decode.jsonl
    
    # Compare with labels
    python compare_shapes.py prefill.jsonl decode.jsonl --labels "Prefill" "Decode"
    
    # Only show differences
    python compare_shapes.py run1.jsonl run2.jsonl --only-diff
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Any, Set


def load_log_file(log_file: str) -> List[Dict[str, Any]]:
    """Load JSONL log file."""
    entries = []
    with open(log_file, "r") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def extract_operation_stats(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract statistics from log entries."""
    op_counts = defaultdict(int)
    op_unique_shapes = defaultdict(set)
    op_examples = {}

    for entry in entries:
        op_name = entry["operation"]
        op_counts[op_name] += 1

        # Track unique output shapes
        output_str = json.dumps(entry.get("outputs"), sort_keys=True)
        op_unique_shapes[op_name].add(output_str)

        # Keep first example of each operation
        if op_name not in op_examples:
            op_examples[op_name] = entry

    return {
        "total_ops": len(entries),
        "op_counts": dict(op_counts),
        "op_unique_shapes": {op: len(shapes) for op, shapes in op_unique_shapes.items()},
        "op_examples": op_examples,
        "unique_ops": set(op_counts.keys()),
    }


def compare_stats(stats1: Dict[str, Any], stats2: Dict[str, Any], label1: str, label2: str):
    """Compare statistics from two runs."""
    print("\n" + "=" * 80)
    print("Overall Comparison")
    print("=" * 80)

    print(f"\n{label1}:")
    print(f"  Total operations: {stats1['total_ops']:,}")
    print(f"  Unique operations: {len(stats1['unique_ops'])}")

    print(f"\n{label2}:")
    print(f"  Total operations: {stats2['total_ops']:,}")
    print(f"  Unique operations: {len(stats2['unique_ops'])}")

    # Set operations
    only_in_1 = stats1['unique_ops'] - stats2['unique_ops']
    only_in_2 = stats2['unique_ops'] - stats1['unique_ops']
    common = stats1['unique_ops'] & stats2['unique_ops']

    print(f"\n{label1} only: {len(only_in_1)} operations")
    print(f"{label2} only: {len(only_in_2)} operations")
    print(f"Common: {len(common)} operations")

    return only_in_1, only_in_2, common


def compare_operation_counts(stats1: Dict[str, Any], stats2: Dict[str, Any], 
                             label1: str, label2: str, common_ops: Set[str]):
    """Compare operation counts for common operations."""
    print("\n" + "=" * 80)
    print("Operation Count Comparison (Common Operations)")
    print("=" * 80)

    # Calculate differences
    differences = []
    for op in common_ops:
        count1 = stats1['op_counts'].get(op, 0)
        count2 = stats2['op_counts'].get(op, 0)
        diff = count2 - count1
        diff_pct = (diff / count1 * 100) if count1 > 0 else float('inf')
        differences.append((op, count1, count2, diff, diff_pct))

    # Sort by absolute difference
    differences.sort(key=lambda x: abs(x[3]), reverse=True)

    # Print top differences
    print(f"\n{'Operation':<50} | {label1:>10} | {label2:>10} | {'Diff':>10} | {'%':>8}")
    print("-" * 100)

    for op, count1, count2, diff, diff_pct in differences[:20]:
        op_short = op if len(op) <= 50 else op[:47] + "..."
        print(f"{op_short:<50} | {count1:>10,} | {count2:>10,} | {diff:>+10,} | {diff_pct:>+7.1f}%")


def compare_shape_diversity(stats1: Dict[str, Any], stats2: Dict[str, Any],
                            label1: str, label2: str, common_ops: Set[str]):
    """Compare shape diversity for common operations."""
    print("\n" + "=" * 80)
    print("Shape Diversity Comparison")
    print("=" * 80)

    differences = []
    for op in common_ops:
        shapes1 = stats1['op_unique_shapes'].get(op, 0)
        shapes2 = stats2['op_unique_shapes'].get(op, 0)
        diff = shapes2 - shapes1
        differences.append((op, shapes1, shapes2, diff))

    # Sort by absolute difference
    differences.sort(key=lambda x: abs(x[3]), reverse=True)

    print(f"\n{'Operation':<50} | {label1:>10} | {label2:>10} | {'Diff':>8}")
    print("-" * 90)

    for op, shapes1, shapes2, diff in differences[:20]:
        if diff != 0:  # Only show operations with different shape counts
            op_short = op if len(op) <= 50 else op[:47] + "..."
            print(f"{op_short:<50} | {shapes1:>10} | {shapes2:>10} | {diff:>+8}")


def show_unique_operations(only_in_1: Set[str], only_in_2: Set[str], 
                          stats1: Dict[str, Any], stats2: Dict[str, Any],
                          label1: str, label2: str):
    """Show operations that appear in only one log."""
    if only_in_1:
        print("\n" + "=" * 80)
        print(f"Operations only in {label1}")
        print("=" * 80)
        
        sorted_ops = sorted([(op, stats1['op_counts'][op]) for op in only_in_1],
                           key=lambda x: x[1], reverse=True)
        
        print(f"\n{'Count':>8} | {'Operation'}")
        print("-" * 80)
        for op, count in sorted_ops[:20]:
            print(f"{count:>8} | {op}")

    if only_in_2:
        print("\n" + "=" * 80)
        print(f"Operations only in {label2}")
        print("=" * 80)
        
        sorted_ops = sorted([(op, stats2['op_counts'][op]) for op in only_in_2],
                           key=lambda x: x[1], reverse=True)
        
        print(f"\n{'Count':>8} | {'Operation'}")
        print("-" * 80)
        for op, count in sorted_ops[:20]:
            print(f"{count:>8} | {op}")


def show_example_shapes(stats1: Dict[str, Any], stats2: Dict[str, Any],
                       label1: str, label2: str, operations: List[str]):
    """Show example shapes for specific operations."""
    print("\n" + "=" * 80)
    print("Example Shape Comparison")
    print("=" * 80)

    for op in operations:
        print(f"\n{op}")
        print("-" * 80)

        if op in stats1['op_examples']:
            example1 = stats1['op_examples'][op]
            print(f"\n{label1}:")
            print(f"  Inputs:  {json.dumps(example1.get('inputs'), indent=11)}")
            print(f"  Outputs: {json.dumps(example1.get('outputs'), indent=11)}")

        if op in stats2['op_examples']:
            example2 = stats2['op_examples'][op]
            print(f"\n{label2}:")
            print(f"  Inputs:  {json.dumps(example2.get('inputs'), indent=11)}")
            print(f"  Outputs: {json.dumps(example2.get('outputs'), indent=11)}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare kernel shapes between different runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("log1", help="First log file")
    parser.add_argument("log2", help="Second log file")
    parser.add_argument("--labels", nargs=2, default=["Log 1", "Log 2"],
                       help="Labels for the two logs")
    parser.add_argument("--only-diff", action="store_true",
                       help="Only show operations with differences")
    parser.add_argument("--show-examples", nargs="+",
                       help="Show example shapes for specific operations")
    parser.add_argument("--top", type=int, default=20,
                       help="Number of top items to show (default: 20)")

    args = parser.parse_args()

    print("=" * 80)
    print("Shape Log Comparison")
    print("=" * 80)

    # Load logs
    print(f"\nLoading {args.log1}...")
    entries1 = load_log_file(args.log1)
    print(f"  Loaded {len(entries1):,} entries")

    print(f"\nLoading {args.log2}...")
    entries2 = load_log_file(args.log2)
    print(f"  Loaded {len(entries2):,} entries")

    # Extract stats
    print("\nAnalyzing logs...")
    stats1 = extract_operation_stats(entries1)
    stats2 = extract_operation_stats(entries2)

    label1, label2 = args.labels

    # Compare
    only_in_1, only_in_2, common = compare_stats(stats1, stats2, label1, label2)

    if not args.only_diff or (only_in_1 or only_in_2):
        show_unique_operations(only_in_1, only_in_2, stats1, stats2, label1, label2)

    if common:
        compare_operation_counts(stats1, stats2, label1, label2, common)
        compare_shape_diversity(stats1, stats2, label1, label2, common)

    # Show example shapes if requested
    if args.show_examples:
        show_example_shapes(stats1, stats2, label1, label2, args.show_examples)

    print("\n" + "=" * 80)
    print("Comparison Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
