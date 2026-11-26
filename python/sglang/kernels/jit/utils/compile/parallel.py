"""``parallel_compile``: build several JIT modules at once."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, List, Optional, TypeVar

T = TypeVar("T")


def parallel_compile(
    builders: Iterable[Callable[[], T]],
    max_workers: Optional[int] = None,
) -> List[T]:
    """Run module builders concurrently and return their results in order.

    A builder is a call that ends in :func:`load_jit`, typically a ``cache_once``
    module factory with its arguments bound. Each build is its own ninja/nvcc
    subprocess under its own build lock, so the threads only wait on them, and a
    cached module returns at once. Use it to build every variant a dispatch may
    reach up front (e.g. before CUDA graph capture) instead of one at a time on
    first use. Every builder runs to completion; the first exception is raised
    after that.
    """
    builders = list(builders)
    if len(builders) <= 1:
        return [build() for build in builders]
    workers = max_workers or min(len(builders), (os.cpu_count() or 1) + 4)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="jit-build") as ex:
        futures = [ex.submit(build) for build in builders]
    return [future.result() for future in futures]
