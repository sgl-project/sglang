# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim for the relocated serving benchmark entrypoint."""

import warnings

from sglang.benchmark.serving import *  # noqa: F403
from sglang.benchmark.serving import (  # noqa: F401
    _create_bench_client_session,
    cli_main,
)

warnings.warn(
    "sglang.bench_serving is deprecated; use sglang.benchmark.serving instead.",
    FutureWarning,
    stacklevel=1,
)


if __name__ == "__main__":
    cli_main()
