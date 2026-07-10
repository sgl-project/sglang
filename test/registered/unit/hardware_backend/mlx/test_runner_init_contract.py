"""Guard the MLX initialize override against ModelRunner contract drift.

MlxModelRunnerStub.initialize must stay callable exactly as ModelRunner invokes
it. The check is signature-only and MLX-gated because importing the stub pulls in
mlx.core.
"""

from __future__ import annotations

import importlib.util
import inspect
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

if _HAS_MLX:
    from sglang.srt.hardware_backend.mlx.model_runner_stub import MlxModelRunnerStub
    from sglang.srt.model_executor.model_runner import ModelRunner


def _required_params_beyond_self(func) -> list[str]:
    """Names of parameters (after ``self``) a caller MUST supply.

    Excludes ``self``, anything carrying a default, and ``*args`` / ``**kwargs``.
    """
    params = list(inspect.signature(func).parameters.values())[1:]
    return [
        p.name
        for p in params
        if p.default is inspect.Parameter.empty
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxRunnerInitContract(unittest.TestCase):
    """``MlxModelRunnerStub.initialize`` must match how the base calls it."""

    def test_base_initialize_takes_no_extra_args(self):
        # The assumption this guard rests on: base ModelRunner.initialize is
        # parameterless and is invoked as ``self.initialize()`` (model_runner.py).
        # If the base re-introduces a required parameter, the override contract
        # below must be revisited -- fail loudly here so that change is noticed.
        required = _required_params_beyond_self(ModelRunner.initialize)
        self.assertEqual(
            required,
            [],
            msg=(
                "Base ModelRunner.initialize gained required parameter(s) "
                f"{required}. Since #23862 it is parameterless and called as "
                "self.initialize(); if that changed, re-check the MLX override "
                "(MlxModelRunnerStub.initialize, #28660)."
            ),
        )

    def test_stub_initialize_binds_like_base_call(self):
        # Core regression guard for #28660: the base calls self.initialize() with
        # zero extra args, so the override must bind with the instance alone. The
        # pre-#28660 signature (self, pre_model_load_memory) raises here.
        sig = inspect.signature(MlxModelRunnerStub.initialize)
        try:
            sig.bind(object())  # stands in for ``self``; mirrors self.initialize()
        except TypeError as exc:
            self.fail(
                "MlxModelRunnerStub.initialize is not call-compatible with the "
                "base ModelRunner call site self.initialize() (no extra args): "
                f"{exc}. Base initialize(self) has been parameterless since "
                "#23862; the override must not require an argument the base no "
                "longer passes (regression of #28660)."
            )


if __name__ == "__main__":
    unittest.main()
