"""Back-compat shim. The implementation now lives in
``sglang.benchmark.one_batch_server``; this module preserves the
``python -m sglang.bench_one_batch_server`` entry point and the
``from sglang.bench_one_batch_server import ...`` imports.
"""

from sglang.benchmark.one_batch_server import *  # noqa: F401,F403
from sglang.benchmark.one_batch_server import main

if __name__ == "__main__":
    main()
