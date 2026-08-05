"""Deprecated import path for ``sglang.benchmark.one_batch``.

``python -m sglang.bench_one_batch`` and ``from sglang.bench_one_batch import ...``
still work, but the implementation now lives in ``sglang.benchmark.one_batch``.
Update references to the new path.
"""

import warnings

from sglang.benchmark.one_batch import *  # noqa: F401,F403
from sglang.benchmark.one_batch import cli_main

warnings.warn(
    "`sglang.bench_one_batch` is deprecated and will be removed in a future "
    "release; use `sglang.benchmark.one_batch` instead "
    "(e.g. `python -m sglang.benchmark.one_batch`).",
    FutureWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    cli_main()
