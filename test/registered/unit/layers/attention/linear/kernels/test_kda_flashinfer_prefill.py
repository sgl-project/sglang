"""Unit tests for the FlashInfer KDA prefill kernel wrapper and its dispatch.

Both cases are regressions from serving Kimi-K3 with the radix cache on
SM103. FlashInfer's ``recurrent_kda`` runs one of two kernels for a packed
prefill: the flash prefill kernel when some sequence extends more than one
token, and the grouped recurrent kernel otherwise. The grouped kernel binds
``dt_bias`` with a unit leading stride and reads the state pool as bf16.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kda_backend import KDAKernelDispatcher
from sglang.srt.layers.attention.linear.kernels.kda_flashinfer_prefill import (
    FlashInferKDAPrefillKernel,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

H, K, V = 4, 128, 128


def _make_inputs(num_tokens, seq_lens):
    q = torch.zeros(1, num_tokens, H, K, dtype=torch.bfloat16)
    g = torch.zeros(1, num_tokens, H, K, dtype=torch.bfloat16)
    beta = torch.zeros(1, num_tokens, H, dtype=torch.bfloat16)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0)))
    return dict(
        q=q,
        k=q.clone(),
        v=q.clone(),
        g=g,
        beta=beta,
        ssm_states=torch.zeros(8, H, V, K, dtype=torch.float32),
        cache_indices=torch.arange(1, len(seq_lens) + 1),
        query_start_loc=cu,
    )


class _RecurrentKDAContract:
    """Stand-in for ``flashinfer.kda.recurrent_kda`` that enforces the parts
    of the grouped kernel's tensor contract the wrapper is responsible for.

    FlashInfer compiles the grouped kernel with
    ``mark_layout_dynamic(leading_dim=0)`` on ``dt_bias``; CuTe then requires
    ``dt_bias.stride(0) == 1`` and rejects a ``[H, K]`` view ("Expected
    strides[leading_dim] == 1, but got 128").
    """

    def __init__(self):
        self.dt_bias = None

    def __call__(self, q, k, v, g, beta, A_log, dt_bias, **kwargs):
        if dt_bias.stride(0) != 1:
            raise RuntimeError(
                f"Expected strides[leading_dim] == 1, but got {dt_bias.stride(0)}."
            )
        if dt_bias.dtype != torch.float32 or dt_bias.numel() != q.shape[2] * K:
            raise RuntimeError("dt_bias must be float32 with H*K elements")
        self.dt_bias = dt_bias
        return torch.zeros_like(v), None


class _RecordingKernel:
    supports_safe_gate = True
    supports_track_state_snapshot = True

    def __init__(self):
        self.calls = 0

    def extend(self, *args, **kwargs):
        self.calls += 1
        return args[2], None


class TestFlashInferKDAPrefillDtBias(CustomTestCase):
    def test_dt_bias_is_passed_flat_with_unit_stride(self):
        """A ``[H, K]`` ``dt_bias`` (how the layer stores it) must reach FlashInfer
        flattened to ``[H*K]``: the grouped kernel's layout binding rejects a
        two-dimensional view, which surfaced as a compile error on the first
        extend that reused a cached prefix."""
        kernel = FlashInferKDAPrefillKernel.__new__(FlashInferKDAPrefillKernel)
        kernel._recurrent_kda = fake = _RecurrentKDAContract()
        dt_bias = torch.arange(H * K, dtype=torch.float32).view(H, K)
        inputs = _make_inputs(num_tokens=6, seq_lens=[6])

        out = kernel.extend(
            inputs.pop("q"),
            inputs.pop("k"),
            inputs.pop("v"),
            inputs.pop("g"),
            inputs.pop("beta"),
            A_log=torch.zeros(H),
            dt_bias=dt_bias,
            lower_bound=-5.0,
            **inputs,
        )

        self.assertEqual(out.shape, (1, 6, H, V))
        self.assertEqual(fake.dt_bias.shape, (H * K,))
        # Same values in the same order as the layer's parameter.
        self.assertTrue(torch.equal(fake.dt_bias, dt_bias.reshape(-1)))


class TestFlashInferKDAPrefillDispatch(CustomTestCase):
    def _make_dispatcher(self):
        flashinfer_kernel = _RecordingKernel()
        with (
            patch(
                "sglang.srt.layers.attention.linear.kda_backend.is_cuda",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.attention.linear.kernels.kda_flashinfer_prefill"
                ".FlashInferKDAPrefillKernel",
                return_value=flashinfer_kernel,
            ),
        ):
            dispatcher = KDAKernelDispatcher(
                decode_backend=LinearAttnKernelBackend.TRITON,
                prefill_backend=LinearAttnKernelBackend.FLASHINFER,
                verify_backend=LinearAttnKernelBackend.TRITON,
            )
        self.assertIsInstance(dispatcher.triton_kernel, TritonKDAKernel)
        dispatcher.triton_kernel = triton_kernel = _RecordingKernel()
        return dispatcher, flashinfer_kernel, triton_kernel

    def _extend(self, dispatcher, num_tokens, seq_lens):
        inputs = _make_inputs(num_tokens, seq_lens)
        return dispatcher.extend(
            inputs.pop("q"),
            inputs.pop("k"),
            inputs.pop("v"),
            inputs.pop("g"),
            inputs.pop("beta"),
            A_log=torch.zeros(H),
            dt_bias=torch.zeros(H * K),
            lower_bound=-5.0,
            **inputs,
        )

    def test_one_token_per_sequence_batch_goes_to_triton(self):
        """A packed batch where every sequence extends exactly one token (e.g.
        retracted requests re-entering with a fully cached prefix) would make
        FlashInfer pick its grouped recurrent kernel, which reads the state pool
        as bf16; the KDA pool is fp32, so the launch fails to compile
        ("autovec_copy only supports equal ... bit widths"). Such batches must
        run on Triton, not just the single-sequence ``T == 1`` case."""
        dispatcher, flashinfer_kernel, triton_kernel = self._make_dispatcher()

        self._extend(dispatcher, num_tokens=4, seq_lens=[1, 1, 1, 1])

        self.assertEqual(triton_kernel.calls, 1)
        self.assertEqual(flashinfer_kernel.calls, 0)

    def test_multi_token_batch_stays_on_flashinfer(self):
        """Once any sequence extends more than one token FlashInfer takes the
        flash prefill kernel, which accepts the fp32 pool; the reroute must not
        widen into a blanket Triton fallback for ordinary prefills."""
        dispatcher, flashinfer_kernel, triton_kernel = self._make_dispatcher()

        self._extend(dispatcher, num_tokens=5, seq_lens=[2, 1, 1, 1])

        self.assertEqual(flashinfer_kernel.calls, 1)
        self.assertEqual(triton_kernel.calls, 0)


if __name__ == "__main__":
    unittest.main()
