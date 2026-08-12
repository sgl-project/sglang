"""Deprecated import path for ``sglang.benchmark.one_batch_server``.

``python -m sglang.bench_one_batch_server`` and
``from sglang.bench_one_batch_server import ...`` still work, but the
implementation now lives in ``sglang.benchmark.one_batch_server``.
Update references to the new path.
"""

import warnings

from sglang.benchmark.one_batch_server import *  # noqa: F401,F403
from sglang.benchmark.one_batch_server import cli_main

warnings.warn(
    "`sglang.bench_one_batch_server` is deprecated and will be removed in a "
    "future release; use `sglang.benchmark.one_batch_server` instead "
    "(e.g. `python -m sglang.benchmark.one_batch_server`).",
    FutureWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    cli_main()
