#!/usr/bin/env python3
"""
Clean up stale HuggingFace cache artifacts from previous failed downloads.

This script removes incomplete marker files, temporary files, and lock files
from the HuggingFace cache directory. These artifacts can accumulate from
interrupted or failed downloads and may interfere with future downloads.
"""

import os
import sys
from pathlib import Path
from typing import List

try:
    from huggingface_hub import constants

    HF_HUB_AVAILABLE = True
except ImportError:
    print("Warning: huggingface_hub not available")
    HF_HUB_AVAILABLE = False


def get_hf_cache_dir() -> str:
    """Get the HuggingFace cache directory."""
    if HF_HUB_AVAILABLE:
        return constants.HF_HUB_CACHE

    # Fallback to environment variable or default
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(hf_home, "hub")


def find_stale_artifacts(cache_dir: str) -> List[Path]:
    """
    Find stale artifact files in the HuggingFace cache.

    Args:
        cache_dir: HuggingFace cache directory

    Returns:
        List of paths to stale artifact files
    """
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        return []

    # Patterns for stale files to clean up
    patterns = [
        "**/*.incomplete",  # Incomplete download markers
        "**/*.tmp",  # Temporary files
        "**/*.lock",  # Lock files from interrupted downloads
    ]

    stale_files = []
    for pattern in patterns:
        stale_files.extend(cache_path.glob(pattern))

    return stale_files


def cleanup_artifacts(artifacts: List[Path]) -> tuple[int, int]:
    """
    Remove stale artifact files.

    Args:
        artifacts: List of file paths to remove

    Returns:
        Tuple of (successful_removals, failed_removals)
    """
    successful = 0
    failed = 0

    for file_path in artifacts:
        try:
            file_path.unlink()
            print(f"  Removed: {file_path}")
            successful += 1
        except Exception as e:
            print(f"  Warning: Could not remove {file_path}: {e}")
            failed += 1

    return successful, failed


def main() -> int:
    """
    Main cleanup logic.

    Returns:
        Always returns 0 (cleanup is best-effort and should not fail CI)
    """
    print("=" * 70)
    print("HuggingFace Cache Cleanup")
    print("=" * 70)

    # Get cache directory
    cache_dir = get_hf_cache_dir()
    print(f"Cache directory: {cache_dir}")

    if not os.path.exists(cache_dir):
        print("Cache directory does not exist - nothing to clean")
        return 0

    print("-" * 70)

    # Find stale artifacts
    print("Scanning for stale artifacts...")
    stale_artifacts = find_stale_artifacts(cache_dir)

    if not stale_artifacts:
        print("✓ No stale cache artifacts found")
        return 0

    # Clean up artifacts
    print(f"Found {len(stale_artifacts)} stale artifact(s) to remove:")
    successful, failed = cleanup_artifacts(stale_artifacts)

    print("-" * 70)

    # Summary
    if failed > 0:
        print(f"⚠ Cleaned up {successful} file(s), {failed} removal(s) failed")
    else:
        print(f"✓ Successfully cleaned up {successful} stale file(s)")

    # Always return 0 - cleanup failures should not fail CI
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Unexpected error during cleanup: {e}")
        import traceback

        traceback.print_exc()
        # Still return 0 - cleanup failures should not fail CI
        sys.exit(0)
