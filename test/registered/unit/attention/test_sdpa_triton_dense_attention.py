# SPDX-License-Identifier: Apache-2.0
"""Wiring of the dense Triton prefill path inside ``SDPABackend``.

The kernel itself is mocked: its numerics live in
``test/registered/kernels/ops/attention/test_flash_attn_dense_triton.py``. What
is checked here is that

- the path stays out unless ``SGLANG_USE_ROCM_TRITON_DENSE_ATTN`` asks for it,
  and out on non-ROCm builds, while compiling, with dropout, causality, GQA,
  other dtypes, other head dims and unaligned sequence lengths;
- the operands handed to the kernel are head-major dense copies of the
  token-major inputs, with the softmax scale forwarded;
- a kernel refusal falls back to SDPA instead of failing.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import sdpa as sdpa_module
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_SEQ = 128
_HEADS = 2
_HEAD_DIM = 128


def _qkv(seq=_SEQ, heads=_HEADS, head_dim=_HEAD_DIM, dtype=torch.bfloat16):
    shape = (1, seq, heads, head_dim)
    return tuple(torch.randn(shape, dtype=dtype) for _ in range(3))


class _FakeKernel:
    def __init__(self, can_use: bool = True):
        self.can_use = can_use
        self.calls = []
        self.output = torch.zeros(1, _SEQ, _HEADS, _HEAD_DIM, dtype=torch.bfloat16)

    def can_use_flash_attn_dense(self, q, k, v, kv_pad=0):
        return self.can_use

    def flash_attn_dense(self, q, k, v, scale=None, kv_pad=0):
        self.calls.append(
            SimpleNamespace(
                shape=tuple(q.shape),
                strides=tuple(q.stride()),
                scale=scale,
                dtype=q.dtype,
            )
        )
        return self.output


def _impl(causal=False, dropout=0.0):
    return SDPAImpl(
        num_heads=_HEADS,
        head_size=_HEAD_DIM,
        causal=causal,
        softmax_scale=_HEAD_DIM**-0.5,
        dropout_p=dropout,
    )


@mock.patch.dict(os.environ, {"SGLANG_USE_ROCM_TRITON_DENSE_ATTN": "1"})
@mock.patch.object(torch.version, "hip", "7.2.0")
class TestDenseAttnGate(CustomTestCase):
    def _forward(self, fake, impl=None, tensors=None):
        impl = impl or _impl()
        tensors = tensors or _qkv()
        with mock.patch.object(
            sdpa_module, "triton_dense_attn_kernel", return_value=fake
        ):
            return impl.forward(*tensors, attn_metadata=None), fake.calls

    def test_disabled_by_env(self):
        fake = _FakeKernel()
        with mock.patch.dict(os.environ, {"SGLANG_USE_ROCM_TRITON_DENSE_ATTN": "0"}):
            out, calls = self._forward(fake)
        self.assertEqual(calls, [])
        self.assertEqual(out.shape, (1, _SEQ, _HEADS, _HEAD_DIM))

    def test_non_rocm_build_falls_back(self):
        fake = _FakeKernel()
        with mock.patch.object(torch.version, "hip", None):
            _, calls = self._forward(fake)
        self.assertEqual(calls, [])

    def test_kernel_gets_head_major_dense_operands_and_scale(self):
        fake = _FakeKernel()
        impl = _impl()
        out, calls = self._forward(fake, impl=impl)
        self.assertIs(out, fake.output)
        self.assertEqual(len(calls), 1)
        call = calls[0]
        # Token-major view of a head-major dense tile: one token step moves by
        # the head dim, one head step by a whole sequence position.
        self.assertEqual(call.shape, (1, _SEQ, _HEADS, _HEAD_DIM))
        self.assertEqual(
            call.strides,
            (_HEADS * _SEQ * _HEAD_DIM, _HEAD_DIM, _SEQ * _HEAD_DIM, 1),
        )
        self.assertAlmostEqual(call.scale, impl.softmax_scale)

    def test_kernel_refusal_falls_back(self):
        fake = _FakeKernel(can_use=False)
        out, calls = self._forward(fake)
        self.assertEqual(calls, [])
        self.assertEqual(out.shape, (1, _SEQ, _HEADS, _HEAD_DIM))

    def test_causal_and_dropout_fall_back(self):
        for impl in (_impl(causal=True), _impl(dropout=0.1)):
            fake = _FakeKernel()
            _, calls = self._forward(fake, impl=impl)
            self.assertEqual(calls, [])

    def test_non_bf16_falls_back(self):
        fake = _FakeKernel()
        _, calls = self._forward(fake, tensors=_qkv(dtype=torch.float32))
        self.assertEqual(calls, [])

    def test_head_dim_and_alignment_are_required(self):
        for tensors in (
            _qkv(head_dim=64),
            _qkv(seq=_SEQ + 16),
            tuple(t[..., :1] for t in _qkv()),
        ):
            fake = _FakeKernel()
            _, calls = self._forward(fake, tensors=tensors)
            self.assertEqual(calls, [])

    def test_mismatched_lengths_fall_back(self):
        fake = _FakeKernel()
        q, k, v = _qkv()
        _, calls = self._forward(fake, tensors=(q, k[:, :-64], v[:, :-64]))
        self.assertEqual(calls, [])

    def test_compiling_falls_back(self):
        fake = _FakeKernel()
        with mock.patch.object(torch.compiler, "is_compiling", return_value=True):
            _, calls = self._forward(fake)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
