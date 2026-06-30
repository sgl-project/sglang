"""Back-compat shim. The implementation now lives in
``sglang.benchmark.one_batch_server``; this module preserves the
``python -m sglang.bench_one_batch_server`` entry point and the
``from sglang.bench_one_batch_server import ...`` imports.
"""

import warnings

from sglang.benchmark.one_batch_server import *  # noqa: F401,F403
from sglang.benchmark.one_batch_server import main

warnings.warn(
    "`sglang.bench_one_batch_server` is deprecated and will be removed in a "
    "future release; use `sglang.benchmark.one_batch_server` instead "
    "(e.g. `python -m sglang.benchmark.one_batch_server`).",
    FutureWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    main()
