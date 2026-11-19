"""Which FlashInfer top-k ops each leg of the V4.1 CUDA-graph replay test captures.

Run on the devbox from the sglang tree root with the branch on PYTHONPATH:
    python3 /hai-workspace/dsv41-fi-topk/tools/verify_graph_backends.py
Wraps flashinfer.topk.get_topk_module so every kernel op call is logged with
whether it ran under stream capture, then runs the test's graph method once.
"""

import collections
import contextlib
import sys
import time
import unittest
from unittest import mock

import flashinfer.topk as fi_topk
import torch

sys.path.insert(0, "test/registered/kernels/ops/attention")
import test_dsv41_topk_selector as selector_test  # noqa: E402

LOG = []
_real_module = fi_topk.get_topk_module
_real_make_selector = selector_test.make_selector


class _Proxy:
    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        attr = getattr(self._module, name)
        kernel = name.startswith(("radix_", "cub_", "fast_topk")) and not name.endswith(
            "_size"
        )
        if not kernel:
            return attr

        def logged(*args, **kwargs):
            shape = tuple(args[0].shape) if isinstance(args[0], torch.Tensor) else None
            dtype = str(args[0].dtype) if isinstance(args[0], torch.Tensor) else None
            LOG.append(("op", name, shape, dtype, torch.cuda.is_current_stream_capturing()))
            return attr(*args, **kwargs)

        return logged


@contextlib.contextmanager
def _logged_make_selector(*variant):
    LOG.append(("leg", variant))
    with _real_make_selector(*variant) as selector:
        yield selector


def main() -> int:
    with (
        mock.patch.object(fi_topk, "get_topk_module", lambda: _Proxy(_real_module())),
        mock.patch.object(selector_test, "make_selector", _logged_make_selector),
    ):
        suite = unittest.defaultTestLoader.loadTestsFromName(
            "TestDSV41TopKSelector.test_flashinfer_paths_replay_under_cuda_graphs",
            selector_test,
        )
        result = unittest.TestResult()
        start = time.perf_counter()
        suite.run(result)
        elapsed = time.perf_counter() - start
    print(f"run={result.testsRun} failures={len(result.failures)} errors={len(result.errors)} "
          f"elapsed={elapsed:.2f}s")
    for _, trace in result.failures + result.errors:
        print(trace)
    legs = []
    for entry in LOG:
        if entry[0] == "leg":
            legs.append((entry[1], collections.Counter()))
        elif legs and entry[4]:
            legs[-1][1][(entry[1], entry[2], entry[3])] += 1
    for variant, captured in legs:
        _, tie_break, deterministic = variant
        print(f"tie_break={tie_break} deterministic={deterministic}")
        for (op, shape, dtype), count in sorted(captured.items()):
            print(f"    captured {op} input={shape} {dtype} x{count}")
    return 1 if result.failures or result.errors else 0


if __name__ == "__main__":
    sys.exit(main())
