import os
import subprocess
import sys
from typing import Callable

import pytest


def multiprocess_test(file: str, nproc: int, timeout: int = 90) -> None:
    """Launch this script as a torchrun worker and assert success."""
    env = os.environ.copy()
    # Torch 2.12 bundles NCCL 2.29, which hard-fails NVLS multicast bind
    # errors that NCCL 2.28 used to handle by disabling NVLS and continuing.
    env.setdefault("NCCL_NVLS_ENABLE", "0")
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        file,
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"torchrun (nproc={nproc}) timed out after {timeout} seconds\n"
            f"{e.stdout}"
        ) from e

    assert result.returncode == 0, (
        f"torchrun (nproc={nproc}) failed with rc={result.returncode}\n"
        f"{result.stdout}"
    )


def multiprocess_main(file: str, main: Callable[[], None]) -> None:
    """Helper to run a function in a multiprocess torchrun context."""
    if "LOCAL_RANK" in os.environ:
        main()
    else:
        sys.exit(pytest.main([file, "-v", "-s"]))
