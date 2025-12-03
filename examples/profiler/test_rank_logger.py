"""
Test the rank-aware shape logger with a simple example.

Usage:
    # Test on current rank (single GPU)
    python test_rank_logger.py
    
    # Test with simulated rank
    RANK=0 python test_rank_logger.py
    
    # Test logging only on rank 0
    python test_rank_logger.py --only-rank 0
"""

import argparse
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch_shape_logger_rank import RankAwareShapeLogger, CompactRankAwareShapeLogger, get_current_rank, get_world_size


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


def test_basic_rank_logger(only_rank=None):
    """Test basic rank-aware shape logger functionality."""
    print("=" * 80)
    print("Test: Rank-Aware Shape Logger")
    print("=" * 80)
    print(f"Current rank: {get_current_rank()}")
    print(f"World size: {get_world_size()}")
    print(f"Only logging rank: {only_rank if only_rank is not None else 'all'}")
    print("=" * 80)

    model = SimpleModel()
    x = torch.randn(32, 100)

    output_file = f"test_rank_shapes.jsonl"

    with RankAwareShapeLogger(output_file, verbose=False, only_rank=only_rank) as logger:
        output = model(x)

    if logger.should_log:
        summary = logger.get_summary()
        print(f"\n✓ Test passed on rank {get_current_rank()}!")
        print(f"  Total operations: {summary['total_operations']}")
        print(f"  Unique operations: {summary['unique_operations']}")
        print(f"  Output file: {output_file}")
        print(f"  Rank: {summary['rank']}")
        print(f"  World size: {summary['world_size']}")
        
        # Show top 5 operations
        if summary['operation_counts']:
            print(f"\n  Top 5 operations:")
            sorted_ops = sorted(
                summary['operation_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for op_name, count in sorted_ops[:5]:
                print(f"    {count:4d} : {op_name}")
    else:
        print(f"\n✓ Rank {get_current_rank()} correctly skipped logging (only_rank={only_rank})")

    return output_file


def test_compact_rank_logger():
    """Test compact rank-aware shape logger."""
    print("\n" + "=" * 80)
    print("Test: Compact Rank-Aware Shape Logger")
    print("=" * 80)

    model = SimpleModel()
    x = torch.randn(16, 100)

    output_file = f"test_rank_shapes_compact.jsonl"

    with CompactRankAwareShapeLogger(output_file, verbose=False) as logger:
        output = model(x)

    if logger.should_log:
        summary = logger.get_summary()
        print(f"\n✓ Test passed on rank {get_current_rank()}!")
        print(f"  Total operations: {summary['total_operations']}")
        print(f"  Unique operations: {summary['unique_operations']}")
        print(f"  Output file: {output_file}")

    return output_file


def test_cuda_if_available(only_rank=None):
    """Test with CUDA if available."""
    if not torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("Test: CUDA - SKIPPED (CUDA not available)")
        print("=" * 80)
        return

    print("\n" + "=" * 80)
    print("Test: CUDA Operations with Rank Awareness")
    print("=" * 80)

    model = SimpleModel().cuda()
    x = torch.randn(8, 100).cuda()

    with RankAwareShapeLogger("test_cuda_rank.jsonl", verbose=False, only_rank=only_rank) as logger:
        output = model(x)

    if logger.should_log:
        summary = logger.get_summary()
        print(f"\n✓ Test passed on rank {get_current_rank()}!")
        print(f"  Total operations: {summary['total_operations']}")
        print(f"  Device: CUDA")


def test_rank_filtering():
    """Test that rank filtering works correctly."""
    print("\n" + "=" * 80)
    print("Test: Rank Filtering")
    print("=" * 80)
    
    current_rank = get_current_rank()
    print(f"Current rank: {current_rank}")
    
    # Test 1: Log only on rank 0
    print("\nTest 1: only_rank=0")
    model = SimpleModel()
    x = torch.randn(8, 100)
    
    with RankAwareShapeLogger("test_only_rank0.jsonl", verbose=False, only_rank=0) as logger:
        output = model(x)
    
    if current_rank == 0:
        assert logger.should_log, "Rank 0 should log when only_rank=0"
        print("  ✓ Rank 0 correctly logged")
    else:
        assert not logger.should_log, f"Rank {current_rank} should not log when only_rank=0"
        print(f"  ✓ Rank {current_rank} correctly skipped")
    
    # Test 2: Log on all ranks
    print("\nTest 2: only_rank=None (all ranks)")
    with RankAwareShapeLogger("test_all_ranks.jsonl", verbose=False, only_rank=None) as logger:
        output = model(x)
    
    assert logger.should_log, f"All ranks should log when only_rank=None"
    print(f"  ✓ Rank {current_rank} correctly logged")
    
    print("\n✓ All rank filtering tests passed!")


def verify_log_file(filename):
    """Verify the log file contains rank information."""
    print("\n" + "=" * 80)
    print(f"Verifying log file: {filename}")
    print("=" * 80)
    
    if not os.path.exists(filename):
        print(f"  ⚠ File not found: {filename}")
        return
    
    import json
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    if len(lines) == 0:
        print(f"  ⚠ File is empty: {filename}")
        return
    
    print(f"  Total entries: {len(lines)}")
    
    # Check first entry
    first_entry = json.loads(lines[0])
    print(f"\n  First entry:")
    print(f"    Rank: {first_entry.get('rank', 'N/A')}")
    print(f"    Operation: {first_entry.get('operation', 'N/A')}")
    print(f"    Output shape: {first_entry.get('outputs', 'N/A')}")
    
    # Check if all entries have rank field
    has_rank = all('rank' in json.loads(line) for line in lines)
    if has_rank:
        print(f"\n  ✓ All entries have 'rank' field")
    else:
        print(f"\n  ⚠ Some entries missing 'rank' field")
    
    # Get unique ranks
    ranks = set(json.loads(line).get('rank', -1) for line in lines)
    print(f"  Unique ranks in file: {sorted(ranks)}")


def main():
    parser = argparse.ArgumentParser(description="Test rank-aware shape logger")
    parser.add_argument("--only-rank", type=int, help="Only log on this rank")
    parser.add_argument("--skip-cuda", action="store_true", help="Skip CUDA tests")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Rank-Aware Shape Logger Test Suite")
    print("=" * 80)
    print()

    try:
        # Test 1: Basic logger
        log_file1 = test_basic_rank_logger(only_rank=args.only_rank)
        if get_current_rank() == (args.only_rank or 0):
            verify_log_file(log_file1)

        # Test 2: Compact logger
        log_file2 = test_compact_rank_logger()
        if get_current_rank() == 0:
            verify_log_file(log_file2)

        # Test 3: CUDA if available
        if not args.skip_cuda:
            test_cuda_if_available(only_rank=args.only_rank)

        # Test 4: Rank filtering
        test_rank_filtering()

        print("\n" + "=" * 80)
        print("All Tests Passed! ✓")
        print("=" * 80)
        print("\nThe rank-aware shape logger is working correctly.")
        print("\nYou can now use it to profile GPU-specific shapes:")
        print("  python profile_gpu0_shapes.py --model-path <model> --tp-size 8")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
