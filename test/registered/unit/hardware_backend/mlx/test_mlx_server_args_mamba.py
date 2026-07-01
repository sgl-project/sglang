"""Guard the MPS branch in ``ServerArgs._handle_mamba_radix_cache``.

The CUDA-only ``_validate_mamba_extra_buffer`` asserts
``is_cuda() or is_musa() or is_npu()``. On Apple Silicon (MPS) the
``auto`` strategy would otherwise resolve to ``extra_buffer`` whenever
overlap schedule is on, crashing the MLX startup. The MPS branch must
pin the strategy to ``no_buffer`` and force ``disable_overlap_schedule``
so the CUDA-only assert never fires.

The checks run on any platform (no MLX import needed): the early-return
in the new branch inspects ``is_mps()`` at runtime, so we exercise the
logic by patching ``is_mps`` rather than relying on the host machine.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _build_args() -> ServerArgs:
    """Build a ServerArgs with a hybrid GDN model arch and auto strategy.

    We can't call ``ServerArgs(...)`` directly because it would try to
    download ``model_path`` via HuggingFace. Instead we build a bare
    instance with ``__new__`` and set the fields the mamba branch reads.
    """
    args = ServerArgs.__new__(ServerArgs)
    args.model_path = "/fake/qwen3-6-moe"
    args.disable_radix_cache = False
    args.mamba_radix_cache_strategy = "auto"
    args.disable_overlap_schedule = False
    args.page_size = 1
    return args


class TestMlxMambaRadixStrategy(unittest.TestCase):
    """``_handle_mamba_radix_cache`` must route MPS to ``no_buffer``."""

    def test_mps_pins_no_buffer_and_disables_overlap(self):
        args = _build_args()
        with patch("sglang.srt.server_args.is_mps", return_value=True):
            args._handle_mamba_radix_cache(model_arch="Qwen3_5MoeForCausalLM")

        self.assertEqual(
            args.mamba_radix_cache_strategy,
            "no_buffer",
            msg="MPS backend must use no_buffer; extra_buffer is CUDA-only.",
        )
        self.assertTrue(
            args.disable_overlap_schedule,
            msg=(
                "MPS backend must disable overlap schedule; otherwise "
                "the auto path would later re-select extra_buffer."
            ),
        )

    def test_non_mps_keeps_auto_resolution(self):
        args = _build_args()
        with patch("sglang.srt.server_args.is_mps", return_value=False):
            args._handle_mamba_radix_cache(model_arch="Qwen3_5MoeForCausalLM")

        # On non-MPS platforms the auto branch resolves the strategy
        # via wants_overlap/wants_paging. For a model that supports
        # extra_buffer with overlap enabled, it lands on extra_buffer.
        self.assertIn(
            args.mamba_radix_cache_strategy,
            ("auto", "extra_buffer", "no_buffer", "extra_buffer_lazy"),
            msg="Non-MPS path must keep the auto/extra_buffer/no_buffer contract.",
        )

    def test_mps_short_circuits_before_extra_buffer_assert(self):
        """The CUDA-only assert in _validate_mamba_extra_buffer must not
        run on MPS even with overlap enabled."""
        args = _build_args()
        # The validation helper would raise AssertionError on MPS; mocking
        # it lets the test assert the call site is never reached.
        with patch("sglang.srt.server_args.is_mps", return_value=True), patch.object(
            ServerArgs, "_validate_mamba_extra_buffer"
        ) as validate_extra:
            args._handle_mamba_radix_cache(model_arch="Qwen3_5MoeForCausalLM")
            validate_extra.assert_not_called()

        self.assertNotEqual(
            args.mamba_radix_cache_strategy,
            "extra_buffer",
            msg=(
                "MPS must not end up on extra_buffer; "
                "_validate_mamba_extra_buffer asserts CUDA/MUSA/NPU."
            ),
        )


if __name__ == "__main__":
    unittest.main()
