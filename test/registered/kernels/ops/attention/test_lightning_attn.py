"""Minimax prefill must decay carried state before producing each token (#53616)."""

import unittest

import torch

from sglang.kernels.ops.attention.linear.lightning_attn import (
    lightning_attention,
    linear_decode_forward_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestLightningAttention(CustomTestCase):
    def _inputs(self, length, dim):
        torch.manual_seed(53616)
        shape = (2, 4, length, dim)
        q, k, v = [
            torch.randn(shape, device="cuda", dtype=torch.float32) for _ in range(3)
        ]
        q = q * dim**-0.5
        slopes = torch.tensor([0.00390625, 0.03, 0.3, 0.9], device="cuda")
        return q, k, v, slopes

    @staticmethod
    def _reference(q, k, v, slopes, history=None):
        # Independent sequential recurrence, including the current token's KV.
        q, k, v = (x.double() for x in (q, k, v))
        state = (
            torch.zeros(
                (*q.shape[:2], q.shape[-1], v.shape[-1]),
                device=q.device,
                dtype=torch.float64,
            )
            if history is None
            else history.double().clone()
        )
        decay = (-slopes.double()).exp()[None, :, None, None]
        output = []
        for t in range(q.shape[2]):
            state = decay * state + k[:, :, t, :, None] * v[:, :, t, None, :]
            output.append((q[:, :, t, :, None] * state).sum(-2))
        return torch.stack(output, dim=2), state

    def _assert_output_close(self, actual, expected):
        # Normalize by each head's RMS token norm, avoiding relative errors at
        # individual near-zero elements while still checking every token.
        error = (actual.double() - expected).norm(dim=-1)
        scale = expected.square().sum(-1).mean(-1, keepdim=True).sqrt()
        max_error = (error / scale.clamp_min(1e-12)).max().item()
        self.assertLess(max_error, 0.01)

    def _assert_state_close(self, actual, expected):
        error = (actual.double() - expected).flatten(-2).norm(dim=-1)
        scale = expected.flatten(-2).norm(dim=-1)
        self.assertLess((error / scale.clamp_min(1e-12)).max().item(), 0.005)

    def test_fresh_prefill_block_boundaries(self):
        """Fresh prefill must match recurrence on both sides of token 256."""
        for dim in (64, 128):
            for length in (1, 255, 256, 257, 577):
                with self.subTest(dim=dim, length=length):
                    q, k, v, slopes = self._inputs(length, dim)
                    expected, state = self._reference(q, k, v, slopes)
                    actual, states = lightning_attention(q, k, v, slopes)
                    self._assert_state_close(states[:, :, -1], state)
                    self._assert_output_close(actual[:, :, :256], expected[:, :, :256])
                    if length > 256:
                        self._assert_output_close(
                            actual[:, :, 256:], expected[:, :, 256:]
                        )

    def test_continuation_from_prefill_state(self):
        """A continuation must decay its incoming state even at its first token."""
        for dim in (64, 128):
            for length in (1, 65, 300):
                with self.subTest(dim=dim, length=length):
                    q, k, v, slopes = self._inputs(256 + length, dim)
                    _, prefix_states = lightning_attention(
                        q[:, :, :256], k[:, :, :256], v[:, :, :256], slopes
                    )
                    history = prefix_states[:, :, -1].contiguous()
                    saved_history = history.clone()
                    _, prefix_ref = self._reference(
                        q[:, :, :256], k[:, :, :256], v[:, :, :256], slopes
                    )
                    expected, state = self._reference(
                        q[:, :, 256:], k[:, :, 256:], v[:, :, 256:], slopes, prefix_ref
                    )
                    actual, states = lightning_attention(
                        q[:, :, 256:],
                        k[:, :, 256:],
                        v[:, :, 256:],
                        slopes,
                        kv_history=history,
                    )
                    torch.testing.assert_close(history, saved_history, rtol=0, atol=0)
                    self._assert_state_close(states[:, :, -1], state)
                    self._assert_output_close(actual, expected)

    def test_prefill_matches_decode_at_block_boundary(self):
        """Token 256 must agree whether processed by prefill or by decode."""
        for dim in (64, 128):
            with self.subTest(dim=dim):
                q, k, v, slopes = self._inputs(257, dim)
                actual, states = lightning_attention(q, k, v, slopes)
                _, prefix_states = lightning_attention(
                    q[:, :, :256], k[:, :, :256], v[:, :, :256], slopes
                )
                cache = prefix_states[:, :, -1].contiguous()
                decoded = linear_decode_forward_triton(
                    q[:, :, 256:].contiguous(),
                    k[:, :, 256:].contiguous(),
                    v[:, :, 256:].contiguous(),
                    cache,
                    slopes,
                    torch.arange(q.shape[0], device="cuda", dtype=torch.int32),
                ).reshape(q.shape[0], q.shape[1], 1, q.shape[-1])
                self._assert_state_close(states[:, :, -1], cache.double())
                self._assert_output_close(actual[:, :, 256:], decoded.double())


if __name__ == "__main__":
    unittest.main()
