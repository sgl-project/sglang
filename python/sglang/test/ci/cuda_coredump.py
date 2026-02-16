"""CUDA coredump support for CI test runs.

When enabled via SGLANG_CI_CUDA_COREDUMP=1, injects CUDA coredump
environment variables into test subprocesses so that GPU exceptions
(e.g. illegal memory access) produce lightweight coredump files for
post-mortem analysis with cuda-gdb.
"""

import glob
import logging
import os
import shutil

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

CUDA_COREDUMP_DIR = "/tmp/sglang_cuda_coredumps"

CUDA_COREDUMP_ENV = {
    "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
    "CUDA_COREDUMP_SHOW_PROGRESS": "1",
    "CUDA_COREDUMP_GENERATION_FLAGS": (
        "skip_nonrelocated_elf_images,skip_global_memory,"
        "skip_shared_memory,skip_local_memory,skip_constbank_memory"
    ),
    "CUDA_COREDUMP_FILE": f"{CUDA_COREDUMP_DIR}/cuda_coredump_%h.%p.%t",
}


def is_enabled() -> bool:
    return envs.SGLANG_CI_CUDA_COREDUMP.get()


def setup_dir():
    """Clean and recreate the CUDA coredump directory."""
    if os.path.exists(CUDA_COREDUMP_DIR):
        shutil.rmtree(CUDA_COREDUMP_DIR, ignore_errors=True)
    os.makedirs(CUDA_COREDUMP_DIR, exist_ok=True)


def get_env() -> dict:
    """Return env dict with CUDA coredump variables injected."""
    env = os.environ.copy()
    env.update(CUDA_COREDUMP_ENV)
    return env


def report():
    """Log any CUDA coredump files found after a test failure."""
    coredump_files = glob.glob(os.path.join(CUDA_COREDUMP_DIR, "cuda_coredump_*"))
    if not coredump_files:
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"CUDA coredump(s) detected ({len(coredump_files)} file(s)):")
    for f in coredump_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        logger.info(f"  {f} ({size_mb:.1f} MB)")
    logger.info("Use cuda-gdb to analyze: cuda-gdb -c <coredump_file>")
    logger.info(f"{'='*60}\n")
