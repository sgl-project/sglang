"""
Analyze kernel shape logs with clear, readable output.

This script reads the shape logs and displays them in a human-friendly format,
showing each kernel followed by its input and output shapes.

Usage:
    # Analyze a shape log file
    python analyze_clear_shapes.py model_tp8_clear_shapes.jsonl

    # Show only top N most frequent kernels
    python analyze_clear_shapes.py model_tp8_clear_shapes.jsonl --top 20

    # Show detailed view of specific kernel
    python analyze_clear_shapes.py model_tp8_clear_shapes.jsonl --kernel "aten.mm.default"

    # Export summary to JSON
    python analyze_clear_shapes.py model_tp8_clear_shapes.jsonl --export summary.json
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Any


def load_shape_log(log_file: str) -> List[Dict[str, Any]]:
    """Load shape log from JSONL file."""
    entries = []
    with open(log_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                entries.append(entry)
            except Exception as e:
                print(f"Warning: Error parsing line {line_num}: {e}")
    return entries


def analyze_kernels(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze kernel usage and shape patterns."""
    kernel_counts = defaultdict(int)
    kernel_examples = defaultdict(list)
    unique_shape_patterns = defaultdict(set)
    
    for entry in entries:
        kernel = entry.get("kernel", "unknown")
        kernel_counts[kernel] += 1
        
        # Store first 3 examples of each kernel
        if len(kernel_examples[kernel]) < 3:
            kernel_examples[kernel].append(entry)
        
        # Track unique shape patterns
        input_shapes_str = json.dumps(entry.get("input_shapes", []), sort_keys=True)
        output_shapes_str = json.dumps(entry.get("output_shapes", []), sort_keys=True)
        pattern = f"{input_shapes_str} -> {output_shapes_str}"
        unique_shape_patterns[kernel].add(pattern)
    
    return {
        "kernel_counts": kernel_counts,
        "kernel_examples": kernel_examples,
        "unique_patterns": {k: len(v) for k, v in unique_shape_patterns.items()},
    }


def print_kernel_details(kernel_name: str, examples: List[Dict[str, Any]], count: int):
    """Print detailed information about a specific kernel."""
    print(f"\n{'='*80}")
    print(f"Kernel: {kernel_name}")
    print(f"{'='*80}")
    print(f"Total occurrences: {count}")
    print(f"\nExample invocations (showing up to 3):")
    
    for i, example in enumerate(examples[:3], 1):
        print(f"\n  Example {i} (call #{example['call_id']}):")
        
        input_shapes = example.get("input_shapes", [])
        output_shapes = example.get("output_shapes", [])
        
        if input_shapes:
            print(f"    Input shapes:")
            for j, shape in enumerate(input_shapes, 1):
                print(f"      [{j}] {shape}")
        else:
            print(f"    Input shapes: (none)")
        
        if output_shapes:
            print(f"    Output shapes:")
            for j, shape in enumerate(output_shapes, 1):
                print(f"      [{j}] {shape}")
        else:
            print(f"    Output shapes: (none)")


def print_summary(analysis: Dict[str, Any], top_n: int = None):
    """Print summary statistics."""
    kernel_counts = analysis["kernel_counts"]
    unique_patterns = analysis["unique_patterns"]
    kernel_examples = analysis["kernel_examples"]
    
    total_ops = sum(kernel_counts.values())
    unique_kernels = len(kernel_counts)
    
    print(f"\n{'='*80}")
    print("SHAPE LOG SUMMARY")
    print(f"{'='*80}")
    print(f"Total operations: {total_ops:,}")
    print(f"Unique kernels: {unique_kernels}")
    
    # Sort kernels by frequency
    sorted_kernels = sorted(kernel_counts.items(), key=lambda x: x[1], reverse=True)
    
    if top_n:
        sorted_kernels = sorted_kernels[:top_n]
        print(f"\nTop {top_n} Most Frequent Kernels:")
    else:
        print(f"\nAll Kernels (sorted by frequency):")
    
    print(f"{'='*80}")
    print(f"{'Count':<12} {'Patterns':<10} {'Kernel'}")
    print(f"{'-'*12} {'-'*10} {'-'*50}")
    
    for kernel, count in sorted_kernels:
        patterns = unique_patterns.get(kernel, 0)
        # Truncate long kernel names
        display_kernel = kernel if len(kernel) <= 50 else kernel[:47] + "..."
        print(f"{count:<12,} {patterns:<10} {display_kernel}")
    
    print(f"{'='*80}")


def print_all_kernels_detailed(analysis: Dict[str, Any], top_n: int = None):
    """Print detailed information for all (or top N) kernels."""
    kernel_counts = analysis["kernel_counts"]
    kernel_examples = analysis["kernel_examples"]
    
    # Sort kernels by frequency
    sorted_kernels = sorted(kernel_counts.items(), key=lambda x: x[1], reverse=True)
    
    if top_n:
        sorted_kernels = sorted_kernels[:top_n]
    
    for kernel, count in sorted_kernels:
        examples = kernel_examples[kernel]
        print_kernel_details(kernel, examples, count)


def export_summary(analysis: Dict[str, Any], output_file: str, entries: List[Dict[str, Any]]):
    """Export summary to JSON file."""
    kernel_counts = analysis["kernel_counts"]
    unique_patterns = analysis["unique_patterns"]
    
    # Get TP size from first entry
    tp_size = entries[0].get("tp_size", 1) if entries else 1
    
    summary = {
        "metadata": {
            "total_operations": sum(kernel_counts.values()),
            "unique_kernels": len(kernel_counts),
            "tp_size": tp_size,
        },
        "kernels": [
            {
                "name": kernel,
                "count": count,
                "unique_patterns": unique_patterns.get(kernel, 0),
            }
            for kernel, count in sorted(kernel_counts.items(), key=lambda x: x[1], reverse=True)
        ],
    }
    
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze kernel shape logs with clear, readable output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "log_file",
        type=str,
        help="Path to the shape log file (JSONL format)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show only top N most frequent kernels",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default=None,
        help="Show detailed information for a specific kernel",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed information for all kernels (or top N with --top)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export summary to JSON file",
    )
    
    args = parser.parse_args()
    
    print(f"Loading shape log from: {args.log_file}")
    entries = load_shape_log(args.log_file)
    print(f"Loaded {len(entries):,} operations")
    
    if not entries:
        print("Error: No valid entries found in log file")
        return
    
    # Analyze the entries
    analysis = analyze_kernels(entries)
    
    # Handle different output modes
    if args.kernel:
        # Show specific kernel
        kernel_examples = analysis["kernel_examples"].get(args.kernel, [])
        kernel_count = analysis["kernel_counts"].get(args.kernel, 0)
        
        if kernel_count == 0:
            print(f"\nError: Kernel '{args.kernel}' not found in log")
            print(f"\nAvailable kernels:")
            for kernel in sorted(analysis["kernel_counts"].keys()):
                print(f"  {kernel}")
        else:
            print_kernel_details(args.kernel, kernel_examples, kernel_count)
    
    elif args.detailed:
        # Show detailed info for all/top kernels
        print_summary(analysis, args.top)
        print_all_kernels_detailed(analysis, args.top)
    
    else:
        # Default: just show summary
        print_summary(analysis, args.top)
    
    # Export if requested
    if args.export:
        export_summary(analysis, args.export, entries)


if __name__ == "__main__":
    main()
