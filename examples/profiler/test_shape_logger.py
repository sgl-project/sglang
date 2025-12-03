"""
Test script for the torch shape logger.

This script runs a simple test to verify the shape logger works correctly
without requiring a large model download.

Usage:
    python test_shape_logger.py
"""

import sys
import torch
import torch.nn as nn
from torch_shape_logger import ShapeLogger, CompactShapeLogger, analyze_shape_log


class SimpleModel(nn.Module):
    """A simple test model."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_basic_logger():
    """Test basic shape logger functionality."""
    print("=" * 80)
    print("Test 1: Basic ShapeLogger")
    print("=" * 80)

    model = SimpleModel()
    x = torch.randn(32, 100)

    output_file = "test_shapes.jsonl"

    with ShapeLogger(output_file, verbose=False) as logger:
        output = model(x)

    summary = logger.get_summary()
    print(f"\n✓ Test passed!")
    print(f"  Total operations: {summary['total_operations']}")
    print(f"  Unique operations: {summary['unique_operations']}")
    print(f"  Output file: {output_file}")

    # Quick check
    assert summary['total_operations'] > 0, "No operations captured!"
    assert summary['unique_operations'] > 0, "No unique operations!"

    return output_file


def test_compact_logger():
    """Test compact shape logger."""
    print("\n" + "=" * 80)
    print("Test 2: CompactShapeLogger")
    print("=" * 80)

    model = SimpleModel()
    x = torch.randn(16, 100)

    output_file = "test_shapes_compact.jsonl"

    with CompactShapeLogger(output_file, verbose=False) as logger:
        output = model(x)

    summary = logger.get_summary()
    print(f"\n✓ Test passed!")
    print(f"  Total operations: {summary['total_operations']}")
    print(f"  Unique operations: {summary['unique_operations']}")
    print(f"  Output file: {output_file}")

    return output_file


def test_verbose_mode():
    """Test verbose mode."""
    print("\n" + "=" * 80)
    print("Test 3: Verbose Mode (showing first few operations)")
    print("=" * 80)

    # Simple operation to avoid too much output
    x = torch.randn(5, 5)

    with ShapeLogger("test_verbose.jsonl", verbose=True) as logger:
        y = torch.mm(x, x)
        z = torch.relu(y)

    print(f"\n✓ Test passed!")


def test_cuda_if_available():
    """Test with CUDA if available."""
    if not torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("Test 4: CUDA Test - SKIPPED (CUDA not available)")
        print("=" * 80)
        return

    print("\n" + "=" * 80)
    print("Test 4: CUDA Operations")
    print("=" * 80)

    model = SimpleModel().cuda()
    x = torch.randn(8, 100).cuda()

    with ShapeLogger("test_cuda.jsonl", verbose=False) as logger:
        output = model(x)

    summary = logger.get_summary()
    print(f"\n✓ Test passed!")
    print(f"  Total operations: {summary['total_operations']}")
    print(f"  Device: CUDA")


def compare_file_sizes():
    """Compare file sizes between regular and compact mode."""
    print("\n" + "=" * 80)
    print("File Size Comparison")
    print("=" * 80)

    import os

    files = [
        ("test_shapes.jsonl", "Regular"),
        ("test_shapes_compact.jsonl", "Compact"),
    ]

    for filename, mode in files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"  {mode:10s}: {size:8,} bytes ({filename})")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Torch Shape Logger Test Suite")
    print("=" * 80)

    try:
        # Run tests
        regular_log = test_basic_logger()
        compact_log = test_compact_logger()
        test_verbose_mode()
        test_cuda_if_available()

        # Compare results
        compare_file_sizes()

        # Test analysis
        print("\n" + "=" * 80)
        print("Test 5: Log Analysis")
        print("=" * 80)
        analyze_shape_log(regular_log)

        print("\n" + "=" * 80)
        print("All Tests Passed! ✓")
        print("=" * 80)
        print("\nYou can now use the shape logger with real models.")
        print("Example:")
        print("  python qwen_shape_logger.py --model Qwen/Qwen2.5-7B-Instruct")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
