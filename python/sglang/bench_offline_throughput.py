"""Deprecated import path for ``sglang.benchmark.offline_throughput``.

``python -m sglang.bench_offline_throughput`` and
``from sglang.bench_offline_throughput import ...`` still work, but the
implementation now lives in ``sglang.benchmark.offline_throughput``.
Update references to the new path.
"""

import warnings

from sglang.benchmark.offline_throughput import *  # noqa: F401,F403
from sglang.benchmark.offline_throughput import cli_main

warnings.warn(
    "`sglang.bench_offline_throughput` is deprecated and will be removed in a "
    "future release; use `sglang.benchmark.offline_throughput` instead "
    "(e.g. `python -m sglang.benchmark.offline_throughput`).",
    FutureWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    cli_main()
