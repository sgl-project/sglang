import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.linear.kda_backend import KDAKernelDispatcher
from sglang.srt.layers.attention.linear.kernels.kda_flashinfer import CakeKDAKernel
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _decode_kwargs(lower_bound):
    t = torch.zeros(1)
    return dict(
        A_log=t,
        dt_bias=t,
        ssm_states=t,
        cache_indices=t,
        query_start_loc=t,
        lower_bound=lower_bound,
    )


class TestKDADispatcherBoundedGateDecode(unittest.TestCase):
    """Kimi-K3 decodes with a bounded ("safe") gate (gate_lower_bound=-5.0).

    The dispatcher used to accept a bounded gate only for TritonKDAKernel, so
    ``--linear-attn-decode-backend cake`` (whose packed decode *is* the K3
    bounded-gate contract) died at the first decode. The guard is now the
    kernel's declared capability.
    """

    def _dispatcher(self, kernel):
        dispatcher = KDAKernelDispatcher.__new__(KDAKernelDispatcher)
        dispatcher.decode_kernel = kernel
        return dispatcher

    def test_kernels_declare_bounded_gate_decode(self):
        self.assertTrue(TritonKDAKernel.supports_bounded_gate_decode)
        self.assertTrue(CakeKDAKernel.supports_bounded_gate_decode)

    def test_bounded_gate_forwarded_to_capable_kernel(self):
        kernel = MagicMock(supports_bounded_gate_decode=True)
        t = torch.zeros(1)
        self._dispatcher(kernel).decode(t, t, t, t, t, **_decode_kwargs(-5.0))
        kernel.decode.assert_called_once()
        self.assertEqual(kernel.decode.call_args.kwargs["lower_bound"], -5.0)

    def test_bounded_gate_refused_without_capability(self):
        kernel = MagicMock(supports_bounded_gate_decode=False)
        t = torch.zeros(1)
        with self.assertRaises(NotImplementedError):
            self._dispatcher(kernel).decode(t, t, t, t, t, **_decode_kwargs(-5.0))
        kernel.decode.assert_not_called()

    def test_unbounded_gate_is_not_gated(self):
        kernel = MagicMock(supports_bounded_gate_decode=False)
        t = torch.zeros(1)
        self._dispatcher(kernel).decode(t, t, t, t, t, **_decode_kwargs(None))
        kernel.decode.assert_called_once()


if __name__ == "__main__":
    unittest.main()
