"""
AMD CI utilities for ROCm GPU memory management and HF cache cleanup.

These utilities help prevent CI eviction due to:
- GPU memory not being released between sequential tests (ROCm is slower than CUDA)
- Disk exhaustion when NFS bandwidth is low and models accumulate
"""

import gc
import os
import shutil
import time

from sglang.multimodal_gen.runtime.utils.common import is_hip
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def cleanup_hf_cache_if_not_persistent() -> None:
    """Clean up HF cache if it's not on a persistent volume.

    When running in CI without persistent cache, downloaded models accumulate
    and can cause disk/memory exhaustion. This cleans up the model after each
    test if the cache is not persistent.

    AMD CI specific: Only clean up when NFS bandwidth was low (SGLANG_NFS_HF_CACHE=0)
    or when the .persistent_cache marker doesn't exist.
    """
    if not is_hip():
        return

    hf_home = os.environ.get("HF_HOME", "")
    if not hf_home:
        return

    hf_hub_cache = os.path.join(hf_home, "hub")

    # Check if HF cache is on a persistent volume by looking for a marker file
    # or checking the SGLANG_NFS_HF_CACHE env var
    persistent_marker = os.path.join(hf_home, ".persistent_cache")
    use_nfs_hf_cache = os.environ.get("SGLANG_NFS_HF_CACHE", "1")

    if os.path.exists(persistent_marker) or use_nfs_hf_cache == "1":
        logger.info("HF cache is persistent, skipping cleanup")
        return

    # Check if the cache directory is empty or was just created
    if not os.path.exists(hf_hub_cache):
        return

    try:
        # Get model cache directories
        model_dirs = [
            d
            for d in os.listdir(hf_hub_cache)
            if d.startswith("models--") and os.path.isdir(os.path.join(hf_hub_cache, d))
        ]

        # If there are cached models but no persistent marker, clean up
        # to prevent disk exhaustion in CI
        if model_dirs:
            logger.info(
                "HF cache appears non-persistent (no .persistent_cache marker), "
                "cleaning up %d model(s) to prevent disk exhaustion",
                len(model_dirs),
            )
            for model_dir in model_dirs:
                model_path = os.path.join(hf_hub_cache, model_dir)
                try:
                    shutil.rmtree(model_path)
                    logger.info("Cleaned up model cache: %s", model_dir)
                except Exception as e:
                    logger.warning("Failed to clean up %s: %s", model_dir, e)
    except Exception as e:
        logger.warning("Error during HF cache cleanup: %s", e)


def cleanup_rocm_gpu_memory(process=None) -> None:
    """ROCm-specific cleanup to ensure GPU memory is fully released.

    Args:
        process: Optional subprocess.Popen object to wait for termination.
    """
    if not is_hip():
        return

    # Wait for process to fully terminate
    if process is not None:
        try:
            process.wait(timeout=30)
        except Exception:
            pass

    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()

    # Clear HIP memory on all GPUs
    try:
        import torch

        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    except Exception:
        pass

    # Wait for GPU memory to be released (ROCm can be much slower than CUDA)
    # The GPU driver needs time to reclaim memory from killed processes
    time.sleep(15)


def wait_for_rocm_gpu_memory_clear(max_wait: float = 60.0) -> None:
    """ROCm-specific: Wait for GPU memory to be mostly free before starting.

    ROCm GPU memory release from killed processes can be significantly slower
    than CUDA, so we need to wait longer and be more patient.

    Args:
        max_wait: Maximum time to wait in seconds (default: 60s).
    """
    if not is_hip():
        return

    try:
        import torch

        if not torch.cuda.is_available():
            return

        start_time = time.time()
        last_total_used = float("inf")

        while time.time() - start_time < max_wait:
            # Check GPU memory usage
            total_used = 0
            for i in range(torch.cuda.device_count()):
                mem_info = torch.cuda.mem_get_info(i)
                free, total = mem_info
                used = total - free
                total_used += used

            # If less than 5GB is used across all GPUs, we're good
            if total_used < 5 * 1024 * 1024 * 1024:  # 5GB
                logger.info(
                    "[server-test] ROCm GPU memory is clear (used: %.2f GB)",
                    total_used / (1024**3),
                )
                return

            # Log progress
            elapsed = int(time.time() - start_time)
            if total_used < last_total_used:
                logger.info(
                    "[server-test] ROCm: GPU memory clearing (used: %.2f GB, elapsed: %ds)",
                    total_used / (1024**3),
                    elapsed,
                )
            else:
                logger.info(
                    "[server-test] ROCm: Waiting for GPU memory (used: %.2f GB, elapsed: %ds)",
                    total_used / (1024**3),
                    elapsed,
                )
            last_total_used = total_used
            time.sleep(3)

        # Final warning with detailed GPU info
        logger.warning(
            "[server-test] ROCm GPU memory not fully cleared after %.0fs (used: %.2f GB). "
            "Proceeding anyway - this may cause OOM.",
            max_wait,
            total_used / (1024**3),
        )
    except Exception as e:
        logger.debug("[server-test] Could not check ROCm GPU memory: %s", e)
