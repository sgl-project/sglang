"""Unit tests for forward-hook CUDA graph skip warning — no server, no GPU.

hook_manager is loaded straight from its file: importing sglang as a package
pulls in the engine's runtime dependencies, which would make a pure-logic test
require a full install.
"""

import importlib.util
import os
import unittest

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # standalone run without an sglang install; CI parses the
    # registration below from the AST, so the stub changes nothing for CI.
    def register_cpu_ci(**kwargs):
        pass


register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_HOOK_MANAGER = os.path.normpath(
    os.path.join(_HERE, "../../../../python/sglang/srt/model_executor/hook_manager.py")
)
_spec = importlib.util.spec_from_file_location(
    "sglang.srt.model_executor.hook_manager", _HOOK_MANAGER
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FORWARD_HOOKS_CUDA_GRAPH_WARNING = _mod.FORWARD_HOOKS_CUDA_GRAPH_WARNING
warn_if_cuda_graphs_skip_forward_hooks = _mod.warn_if_cuda_graphs_skip_forward_hooks


class TestForwardHookCudaGraphWarning(unittest.TestCase):
    def test_warns_when_hooks_and_cuda_graphs_are_on(self):
        with self.assertLogs(
            "sglang.srt.model_executor.hook_manager", level="WARNING"
        ) as captured:
            warn_if_cuda_graphs_skip_forward_hooks(
                [{"name": "probe"}],
                cuda_graphs_enabled=True,
            )
        self.assertTrue(
            any(
                FORWARD_HOOKS_CUDA_GRAPH_WARNING in message
                for message in captured.output
            )
        )

    def test_silent_when_cuda_graphs_are_disabled(self):
        with self.assertNoLogs(
            "sglang.srt.model_executor.hook_manager", level="WARNING"
        ):
            warn_if_cuda_graphs_skip_forward_hooks(
                [{"name": "probe"}],
                cuda_graphs_enabled=False,
            )

    def test_silent_when_no_hooks_are_configured(self):
        with self.assertNoLogs(
            "sglang.srt.model_executor.hook_manager", level="WARNING"
        ):
            warn_if_cuda_graphs_skip_forward_hooks(None, cuda_graphs_enabled=True)
            warn_if_cuda_graphs_skip_forward_hooks([], cuda_graphs_enabled=True)


if __name__ == "__main__":
    unittest.main()
