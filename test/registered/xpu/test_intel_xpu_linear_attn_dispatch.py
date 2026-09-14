"""Intel XPU linear-attn (GDN) backend dispatch: intel_xpu is opt-in only
(default stays triton, like every other platform) and fails fast rather than
silently degrading when misconfigured. Pure dispatch-logic tests -- no XPU
device required -- kept under test/registered/xpu to separate Intel-XPU-only
behavior from the platform-agnostic linear-attn dispatch tests.
"""

import unittest
from unittest.mock import patch

from sglang.srt.layers.attention.linear import gdn_backend
from sglang.srt.layers.attention.linear.gdn_backend import GDNKernelDispatcher
from sglang.srt.layers.attention.linear.kda_backend import KDAKernelDispatcher
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.server_args import LINEAR_ATTN_KERNEL_BACKEND_CHOICES
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=5, suite="stage-b-test-1-gpu-xpu")


class TestIntelXpuGDNDispatch(CustomTestCase):
    def test_intel_xpu_is_a_registered_backend_choice(self):
        self.assertIn("intel_xpu", LINEAR_ATTN_KERNEL_BACKEND_CHOICES)

    def test_intel_xpu_requires_xpu_hardware(self):
        with patch.object(gdn_backend, "is_xpu", return_value=False):
            with self.assertRaisesRegex(ValueError, "requires Intel XPU"):
                GDNKernelDispatcher(
                    LinearAttnKernelBackend.INTEL_XPU,
                    LinearAttnKernelBackend.TRITON,
                )
            with self.assertRaisesRegex(ValueError, "requires Intel XPU"):
                GDNKernelDispatcher(
                    LinearAttnKernelBackend.TRITON,
                    LinearAttnKernelBackend.INTEL_XPU,
                )

    def test_intel_xpu_uses_triton_as_the_dispatcher_fallback_kernel(self):
        # The fused SYCL kernel is dispatched outside GDNKernelDispatcher (via
        # XpuGDNAttnBackend.forward_fused_gdn); the dispatcher itself only
        # needs a valid fallback kernel for requests that hook declines
        # (e.g. verify), which is Triton.
        with patch.object(gdn_backend, "is_xpu", return_value=True):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.INTEL_XPU,
                LinearAttnKernelBackend.INTEL_XPU,
            )

        self.assertIsInstance(dispatcher.decode_kernel, TritonGDNKernel)
        self.assertIsInstance(dispatcher.extend_kernel, TritonGDNKernel)
        self.assertIsInstance(dispatcher.verify_kernel, TritonGDNKernel)


class TestIntelXpuKDADispatch(CustomTestCase):
    def test_intel_xpu_is_not_a_supported_kda_backend(self):
        # Unlike GDN, KDA has no Intel XPU SYCL kernel: intel_xpu must not be
        # silently treated as Triton, it should fail fast like any other
        # backend KDA does not implement.
        with self.assertRaisesRegex(ValueError, "Unsupported KDA decode backend"):
            KDAKernelDispatcher(
                decode_backend=LinearAttnKernelBackend.INTEL_XPU,
                prefill_backend=LinearAttnKernelBackend.TRITON,
                verify_backend=LinearAttnKernelBackend.TRITON,
            )
        with self.assertRaisesRegex(ValueError, "Unsupported KDA prefill backend"):
            KDAKernelDispatcher(
                decode_backend=LinearAttnKernelBackend.TRITON,
                prefill_backend=LinearAttnKernelBackend.INTEL_XPU,
                verify_backend=LinearAttnKernelBackend.TRITON,
            )


if __name__ == "__main__":
    unittest.main()
