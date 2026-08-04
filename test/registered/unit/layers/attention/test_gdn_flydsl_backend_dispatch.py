import unittest
from unittest.mock import patch

from sglang.srt.layers.attention.linear.gdn_backend import GDNKernelDispatcher
from sglang.srt.layers.attention.linear.kernels.gdn_flydsl import FlyDSLGDNKernel
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.server_args import LINEAR_ATTN_KERNEL_BACKEND_CHOICES
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FLYDSL = LinearAttnKernelBackend.FLYDSL
TRITON = LinearAttnKernelBackend.TRITON


class TestFlyDSLGDNBackendDispatch(unittest.TestCase):
    """The flydsl GDN backend must be selectable, and must degrade to Triton
    rather than raise when the aiter FlyDSL kernels are not importable (the
    common case: any non-ROCm host, or a ROCm host without aiter)."""

    def dispatch(self, *, extend_ok, decode_ok):
        with (
            patch.object(FlyDSLGDNKernel, "supports_flydsl_extend", extend_ok),
            patch.object(FlyDSLGDNKernel, "supports_flydsl_decode", decode_ok),
        ):
            return GDNKernelDispatcher(decode_backend=FLYDSL, prefill_backend=FLYDSL)

    def test_backend_name_is_a_server_arg_choice(self):
        self.assertIn("flydsl", LINEAR_ATTN_KERNEL_BACKEND_CHOICES)
        self.assertIs(LinearAttnKernelBackend("flydsl"), FLYDSL)
        self.assertTrue(FLYDSL.is_flydsl())
        self.assertFalse(TRITON.is_flydsl())

    def test_selects_flydsl_for_both_phases_when_available(self):
        d = self.dispatch(extend_ok=True, decode_ok=True)
        self.assertIsInstance(d.decode_kernel, FlyDSLGDNKernel)
        self.assertIsInstance(d.extend_kernel, FlyDSLGDNKernel)
        # one shared instance, not two
        self.assertIs(d.decode_kernel, d.extend_kernel)

    def test_falls_back_to_triton_when_aiter_is_unavailable(self):
        d = self.dispatch(extend_ok=False, decode_ok=False)
        for kernel in (d.decode_kernel, d.extend_kernel):
            self.assertIsInstance(kernel, TritonGDNKernel)
            self.assertNotIsInstance(kernel, FlyDSLGDNKernel)

    def test_falls_back_per_phase(self):
        """A build with only the prefill kernel still gets the prefill win."""
        d = self.dispatch(extend_ok=True, decode_ok=False)
        self.assertIsInstance(d.extend_kernel, FlyDSLGDNKernel)
        self.assertNotIsInstance(d.decode_kernel, FlyDSLGDNKernel)

    def test_tree_verify_always_stays_on_triton(self):
        d = self.dispatch(extend_ok=True, decode_ok=True)
        self.assertIsInstance(d.tree_verify_kernel, TritonGDNKernel)
        self.assertNotIsInstance(d.tree_verify_kernel, FlyDSLGDNKernel)

    def test_routes_around_the_packed_decode_fast_path(self):
        """flydsl_gdr_decode takes split q/k/v, so the packed path must be off."""
        self.assertFalse(FlyDSLGDNKernel.supports_packed_decode)
        self.assertTrue(TritonGDNKernel.supports_packed_decode)


if __name__ == "__main__":
    unittest.main()
