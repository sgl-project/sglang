"""A mamba-only extra_buffer strategy on a non-mamba arch must fail fast at
resolution, not arm the ping-pong track buffers and crash on a None buffer at
decode. auto/no_buffer stay inert on such an arch.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.model_hook import handle_mamba_radix_cache
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_NON_MAMBA_ARCH = "Qwen2ForCausalLM"


class TestMambaExtraBufferArchGuard(CustomTestCase):
    def _server_args(self, strategy):
        sa = ServerArgs(model_path="dummy", mamba_radix_cache_strategy=strategy)
        sa._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[_NON_MAMBA_ARCH])
        )
        return sa

    def test_extra_buffer_on_non_mamba_arch_raises(self):
        for strategy in ("extra_buffer", "extra_buffer_lazy"):
            with self.subTest(strategy=strategy):
                with self.assertRaisesRegex(
                    ValueError, "needs a model with mamba state"
                ):
                    handle_mamba_radix_cache(
                        self._server_args(strategy), _NON_MAMBA_ARCH
                    )

    def test_inert_strategy_on_non_mamba_arch_is_accepted(self):
        for strategy in ("auto", "no_buffer"):
            with self.subTest(strategy=strategy):
                handle_mamba_radix_cache(self._server_args(strategy), _NON_MAMBA_ARCH)


if __name__ == "__main__":
    unittest.main()
