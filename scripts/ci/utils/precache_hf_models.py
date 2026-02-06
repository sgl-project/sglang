#!/usr/bin/env python3
"""
Pre-download missing HuggingFace models before CI tests start.

This script runs once during CI initialization (in prepare_runner.sh) to:
1. Extract model names from DEFAULT_*MODEL* constants in test_utils.py
2. Scan test/registered/ for hardcoded model references (model = "org/repo")
3. Check which models are NOT yet cached (no snapshot with config.json)
4. Download missing models via snapshot_download(max_workers=1)
5. Use fcntl-based locking (same lock paths as ci_download_with_validation_and_retry)
   with LOCK_NB (non-blocking — skip if locked by another process)

This is best-effort: exits 0 on all failures. Pre-caching is an optimization,
not a requirement.
"""

import fcntl
import glob
import hashlib
import json
import multiprocessing
import os
import re
import sys
import time
from pathlib import Path

# Add python directory to path to import sglang modules
REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

# Time limits
TOTAL_TIME_LIMIT_SECONDS = 600  # 10 minutes total
PER_MODEL_TIMEOUT_SECONDS = 300  # 5 minutes per model


def get_all_default_models():
    """
    Get all model names from DEFAULT_*MODEL* constants in test_utils.py.

    Uses _get_default_models() which handles str, comma-separated str,
    and tuple constants.

    Returns:
        Set of model name strings
    """
    from sglang.test.test_utils import _get_default_models

    models_json = _get_default_models()
    return set(json.loads(models_json))


# Matches model references like: model = "org/repo" or model_path = "org/repo"
# Captures org/repo format (at least one slash, no spaces)
_MODEL_PATTERN = re.compile(
    r"""(?:model|model_path)\s*=\s*["']([a-zA-Z][a-zA-Z0-9_-]*/[a-zA-Z0-9._-]+)"""
)


def get_models_from_test_files():
    """
    Scan test/registered/ for hardcoded model references.

    Extracts model names from patterns like:
        model = "org/repo"
        model_path = "org/repo"

    Strips LoRA adapter suffixes (e.g., "org/repo:adapter" -> "org/repo").

    Returns:
        Set of model name strings
    """
    models = set()
    test_dir = REPO_ROOT / "test" / "registered"

    if not test_dir.is_dir():
        print(f"  Warning: {test_dir} not found, skipping test file scan")
        return models

    for py_file in test_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for match in _MODEL_PATTERN.finditer(content):
            model_name = match.group(1)
            # Strip LoRA adapter suffix (e.g., "meta-llama/Llama-3.1-8B:sql-expert")
            model_name = model_name.split(":")[0]
            models.add(model_name)

    return models


def is_model_cached(model_name):
    """
    Check if a model is already cached (has a snapshot with config.json).

    Args:
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")

    Returns:
        True if model appears to be cached, False otherwise
    """
    try:
        from huggingface_hub import constants

        cache_dir = constants.HF_HUB_CACHE
    except ImportError:
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        cache_dir = os.path.join(hf_home, "hub")

    # models--org--repo format
    repo_folder_name = "models--" + model_name.replace("/", "--")
    snapshots_pattern = os.path.join(
        cache_dir, repo_folder_name, "snapshots", "*", "config.json"
    )
    return len(glob.glob(snapshots_pattern)) > 0


def _get_lock_file_path(model_name):
    """
    Generate a unique lock file path for download coordination.

    Duplicated from ci_weight_validation.py for compatibility — uses the same
    /dev/shm lock paths as ci_download_with_validation_and_retry.

    Args:
        model_name: Model identifier

    Returns:
        Path to the lock file
    """
    key_hash = hashlib.sha256(model_name.encode()).hexdigest()[:16]
    if os.path.isdir("/dev/shm"):
        return f"/dev/shm/sglang_download_lock_{key_hash}"
    return f"/tmp/sglang_download_lock_{key_hash}"


def _download_worker(model_name):
    """
    Worker function for downloading a model in a child process.

    Args:
        model_name: HuggingFace model name to download
    """
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_name, max_workers=1)
    except Exception as e:
        print(f"  Download error: {type(e).__name__}: {e}")
        sys.exit(1)


def precache_model(model_name, timeout):
    """
    Attempt to download a model with non-blocking locking and timeout.

    Uses fcntl.flock with LOCK_NB to skip if another process holds the lock.
    Runs snapshot_download in a child process with a timeout.

    Args:
        model_name: HuggingFace model name to download
        timeout: Maximum seconds to wait for the download

    Returns:
        True if download succeeded, False otherwise
    """
    lock_file_path = _get_lock_file_path(model_name)

    try:
        lock_file = open(lock_file_path, "w")
    except OSError as e:
        print(f"  Cannot open lock file {lock_file_path}: {e}")
        return False

    try:
        # Non-blocking lock — skip if another process is downloading this model
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            print("  Skipping: another process holds the download lock")
            return False

        try:
            # Run download in a child process so we can enforce timeout
            proc = multiprocessing.Process(target=_download_worker, args=(model_name,))
            proc.start()
            proc.join(timeout=timeout)

            if proc.is_alive():
                print(f"  Timeout after {timeout}s, terminating download")
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=5)
                return False

            return proc.exitcode == 0

        except Exception as e:
            print(f"  Error running download process: {e}")
            return False

    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock_file.close()


def main():
    start_time = time.time()

    print("=" * 70)
    print("CI: Pre-caching missing HuggingFace models")
    print("=" * 70)
    print(f"Total time limit: {TOTAL_TIME_LIMIT_SECONDS}s")
    print(f"Per-model timeout: {PER_MODEL_TIMEOUT_SECONDS}s")
    print()

    # Get model names from DEFAULT_*MODEL* constants
    all_models = set()
    print("Extracting model names from test_utils.py constants...")
    try:
        constant_models = get_all_default_models()
        all_models.update(constant_models)
        print(f"  Found {len(constant_models)} from constants")
    except Exception as e:
        print(f"  Failed: {e}")

    # Get model names from test/registered/ files
    print("Scanning test/registered/ for hardcoded model references...")
    try:
        test_file_models = get_models_from_test_files()
        new_models = test_file_models - all_models
        all_models.update(test_file_models)
        print(
            f"  Found {len(test_file_models)} from test files ({len(new_models)} new)"
        )
    except Exception as e:
        print(f"  Failed: {e}")

    if not all_models:
        print("No models found, skipping pre-cache step")
        return

    print(f"Total unique models: {len(all_models)}")
    print()

    # Classify models as cached or missing
    cached_models = []
    missing_models = []

    for model_name in sorted(all_models):
        if is_model_cached(model_name):
            cached_models.append(model_name)
        else:
            missing_models.append(model_name)

    print(f"Already cached: {len(cached_models)}")
    print(f"Missing (need download): {len(missing_models)}")
    print()

    if not missing_models:
        print("All models are cached, nothing to do")
        print("=" * 70)
        return

    # Download missing models
    downloaded = 0
    failed = 0
    skipped = 0

    for i, model_name in enumerate(missing_models):
        elapsed = time.time() - start_time
        remaining = TOTAL_TIME_LIMIT_SECONDS - elapsed

        if remaining <= 0:
            skipped += len(missing_models) - i
            print(
                f"\nTotal time limit reached ({elapsed:.1f}s), "
                f"skipping {len(missing_models) - i} remaining model(s)"
            )
            break

        per_model_timeout = min(PER_MODEL_TIMEOUT_SECONDS, remaining)

        print(f"[{i + 1}/{len(missing_models)}] Downloading: {model_name}")
        model_start = time.time()

        success = precache_model(model_name, per_model_timeout)
        model_elapsed = time.time() - model_start

        if success:
            print(f"  OK ({model_elapsed:.1f}s)")
            downloaded += 1
        else:
            print(f"  FAILED ({model_elapsed:.1f}s)")
            failed += 1

    elapsed_total = time.time() - start_time

    print()
    print("=" * 70)
    print(f"Pre-cache summary (completed in {elapsed_total:.1f}s):")
    print(f"  Already cached:     {len(cached_models)}")
    print(f"  Downloaded:         {downloaded}")
    print(f"  Failed:             {failed}")
    print(f"  Skipped (timeout):  {skipped}")
    print(f"  Total models:       {len(all_models)}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"ERROR: Unexpected error during pre-caching: {e}")
        import traceback

        traceback.print_exc()
    # Always exit 0 — pre-caching is best-effort
    sys.exit(0)
