"""CUDA coredump helpers.

CUDA coredump env vars are auto-injected by environ.py at import time
when SGLANG_CUDA_COREDUMP=1.  This module provides directory management
(setup / cleanup) and coredump file reporting for the CI test runner.
"""

import glob
import logging
import os
import shutil

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def is_enabled() -> bool:
    return envs.SGLANG_CUDA_COREDUMP.get()


def get_dump_dir() -> str:
    return envs.SGLANG_CUDA_COREDUMP_DIR.get()


def setup_dir():
    """Clean and recreate the coredump directory for a fresh test run."""
    dump_dir = get_dump_dir()
    if os.path.exists(dump_dir):
        shutil.rmtree(dump_dir, ignore_errors=True)
    os.makedirs(dump_dir, exist_ok=True)
    logger.info(f"[CUDA Coredump] Enabled. Dump dir: {dump_dir}")


def report():
    """Log any CUDA coredump files found after a test failure."""
    dump_dir = get_dump_dir()
    coredump_files = glob.glob(os.path.join(dump_dir, "cuda_coredump_*"))
    if not coredump_files:
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"CUDA coredump(s) detected ({len(coredump_files)} file(s)):")
    for f in coredump_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        logger.info(f"  {f} ({size_mb:.1f} MB)")
    logger.info("Use cuda-gdb to analyze: cuda-gdb -c <coredump_file>")

    run_id = os.environ.get("GITHUB_RUN_ID")
    if run_id:
        repo = os.environ.get("GITHUB_REPOSITORY", "sgl-project/sglang")
        logger.info(f"Download from CI: gh run download {run_id} --repo {repo}")

    logger.info(f"{'='*60}\n")
