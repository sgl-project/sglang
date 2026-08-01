"""Backend selection in initialize_linear_attn_config.

The SM100 GDN default reaches the module state as an argument rather than as a
ServerArgs mutation, so the precedence between an explicit flag, that default,
and the shared base backend is pinned here.
"""

import unittest

from sglang.srt.layers.attention.linear import utils as linear_utils
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    initialize_linear_attn_config,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLinearAttnConfig(CustomTestCase):
    def setUp(self):
        saved = (
            linear_utils.LINEAR_ATTN_DECODE_BACKEND,
            linear_utils.LINEAR_ATTN_PREFILL_BACKEND,
        )

        def restore():
            (
                linear_utils.LINEAR_ATTN_DECODE_BACKEND,
                linear_utils.LINEAR_ATTN_PREFILL_BACKEND,
            ) = saved

        self.addCleanup(restore)

    def _init(self, prefill_default=None, **fields):
        args = ServerArgs(model_path="dummy")
        for key, value in fields.items():
            setattr(args, key, value)
        initialize_linear_attn_config(args, prefill_default)
        return (
            linear_utils.LINEAR_ATTN_PREFILL_BACKEND,
            linear_utils.LINEAR_ATTN_DECODE_BACKEND,
        )

    def test_default_applies_when_the_flag_is_unset(self):
        prefill, _ = self._init(
            prefill_default="flashinfer", linear_attn_backend="triton"
        )
        self.assertEqual(prefill, LinearAttnKernelBackend.FLASHINFER)

    def test_explicit_flag_wins_over_the_default(self):
        prefill, _ = self._init(
            prefill_default="flashinfer",
            linear_attn_backend="triton",
            linear_attn_prefill_backend="cutedsl",
        )
        self.assertEqual(prefill, LinearAttnKernelBackend.CUTEDSL)

    def test_base_backend_applies_without_a_default(self):
        prefill, decode = self._init(linear_attn_backend="triton")
        self.assertEqual(prefill, LinearAttnKernelBackend.TRITON)
        self.assertEqual(decode, LinearAttnKernelBackend.TRITON)

    def test_the_default_does_not_reach_the_decode_backend(self):
        _, decode = self._init(
            prefill_default="flashinfer", linear_attn_backend="triton"
        )
        self.assertEqual(decode, LinearAttnKernelBackend.TRITON)


if __name__ == "__main__":
    unittest.main()
