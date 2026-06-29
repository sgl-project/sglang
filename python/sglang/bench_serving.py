"""Deprecated import path for ``sglang.benchmark.serving``.

``python -m sglang.bench_serving`` and ``from sglang.bench_serving import ...``
still work, but the implementation now lives in ``sglang.benchmark.serving``.
Update references to the new path.
"""

import warnings

from sglang.benchmark.serving import *  # noqa: F401,F403
from sglang.benchmark.serving import cli_main

warnings.warn(
    "`sglang.bench_serving` is deprecated and will be removed in a future "
    "release; use `sglang.benchmark.serving` instead "
    "(e.g. `python -m sglang.benchmark.serving`).",
    FutureWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    cli_main()
