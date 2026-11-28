"""KDA and causal-conv Triton launches on contiguous storage under torch.compile.

The kernels take strides read at trace time. A compiled graph may copy a view
operand into a contiguous buffer, and the kernel then addresses that buffer with
the view's strides. Under capture the kernels launch on contiguous operands, or
on the storage the view's strides describe.
"""

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.attention.fla import kda as kda_module
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    _conv_state_storage,
    _token_major_conv_operands,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestKDATritonOperandLayout(CustomTestCase):
    def setUp(self):
        torch._dynamo.reset()

    def test_kda_gate_launches_on_contiguous_beta_under_capture(self):
        beta = torch.randn(1, 5, 224)[..., :96]
        for compiling, beta_stride_t in ((False, 224), (True, 96)):
            with (
                self.subTest(compiling=compiling),
                mock.patch.object(
                    kda_module, "kda_gate_chunk_cumsum_vector_kernel"
                ) as kernel,
                mock.patch.object(
                    torch.compiler, "is_compiling", return_value=compiling
                ),
            ):
                kda_module.kda_gate_chunk_cumsum(
                    torch.randn(1, 5, 96, 4),
                    torch.randn(96),
                    chunk_size=64,
                    beta=beta,
                )
                launch = kernel.__getitem__.return_value.call_args.kwargs
                self.assertEqual(launch["beta_stride_t"], beta_stride_t)
                self.assertEqual(launch["beta"].stride(1), beta_stride_t)
                torch.testing.assert_close(launch["beta"], beta)

    def test_conv_operands_are_contiguous_token_major_buffers(self):
        """``empty_like`` of the channel-last input fell back to a
        ``Max(1, tokens)`` stride for a symbolic token count."""

        def prepare(mixed_qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x_tokens, out_tokens = _token_major_conv_operands(mixed_qkv.t())
            for buffer in (x_tokens, out_tokens):
                torch._check(buffer.stride(0) == 12)
                torch._check(buffer.stride(1) == 1)
            return x_tokens, out_tokens

        mixed_qkv = torch.arange(5 * 16, dtype=torch.float32).view(5, 16)[:, :12]
        torch._dynamo.decorators.mark_unbacked(mixed_qkv, 0, min=1, max=64)
        x_tokens, out_tokens = torch.compile(
            prepare, backend="eager", dynamic=True, fullgraph=True
        )(mixed_qkv)
        torch.testing.assert_close(x_tokens, mixed_qkv)
        self.assertEqual(x_tokens.stride(), (12, 1))
        self.assertEqual(out_tokens.shape, (5, 12))
        self.assertEqual(out_tokens.stride(), (12, 1))

    def test_conv_kernels_launch_on_the_conv_state_pool_storage(self):
        # (slots, state_len, dim), passed to the kernels as its transpose.
        pool = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
        conv_state = pool.transpose(-1, -2)
        self.assertIs(_conv_state_storage(conv_state), conv_state)

        compiled = torch.compile(_conv_state_storage, backend="eager", fullgraph=True)
        storage = compiled(conv_state)
        self.assertEqual(storage.shape, pool.shape)
        self.assertTrue(storage.is_contiguous())
        torch.testing.assert_close(storage, pool)
        self.assertEqual(storage.transpose(-1, -2).stride(), conv_state.stride())
        torch.testing.assert_close(compiled(pool), pool)


if __name__ == "__main__":
    unittest.main()
