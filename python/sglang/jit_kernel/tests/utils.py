import sys
from typing import Callable, List, Optional, Sequence

import pytest

from sglang.jit_kernel.mp import multigpu_launch


def multigpu_pytest_main(
    name: str,
    file: str,
    num_gpus: Sequence[int],
    *,
    pre_launch_fn: Optional[Callable[[List[int]], None]] = None,
    timeout: Optional[int] = 600,
) -> None:
    """cudalib-style multi-GPU pytest entry point.

    Drop this at the bottom of a test file::

        multigpu_pytest_main(__name__, __file__, num_gpus=range(2, 9))

    When the file is run with ``python <file>``, it relaunches itself under
    ``torchrun --nproc_per_node=N <file>`` for each N in ``num_gpus``. Inside
    each worker, ``pytest.main([file, ...forwarded_args])`` runs the collected
    tests. Pass ``--num-gpu 2,4`` on the command line to override ``num_gpus``.

    ``pre_launch_fn`` (kw-only) runs once in the outer process before any
    torchrun child starts, receiving the runnable world sizes. Use it for
    parallel JIT precompilation so torchrun children hit a warm disk cache
    instead of compiling kernels on first call.

    ``timeout`` (kw-only, seconds) bounds each per-world-size torchrun
    invocation. The default budget covers the cold-cache first invocation
    (the worker pays the full triton + cutlass JIT compile cost, 60-180s
    observed on H200) plus the nightly full sweep, which runs every size x
    dtype x algo x graph-mode parametrisation rather than the reduced in-CI
    range. A worker that exceeds the budget is killed and the run fails. Pass
    ``None`` to wait indefinitely.
    """

    def inner() -> int:
        # CI's run_unittest_files invokes `python3 <file> -f` (legacy
        # unittest failfast). Translate to pytest's `-x` so it survives.
        pytest_args = ["-x" if a == "-f" else a for a in sys.argv[1:]]
        return pytest.main([file] + pytest_args)

    return multigpu_launch(
        name,
        file,
        num_gpus,
        env_key="_IS_TEST_MULTIGPU_SGLANG_JIT_KERNEL",
        inner=inner,
        kind="test",
        pre_launch_fn=pre_launch_fn,
        timeout=timeout,
    )
