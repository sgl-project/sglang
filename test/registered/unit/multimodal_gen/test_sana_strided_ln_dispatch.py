# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for SANA LayerNorm dispatch validation."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

import sglang.multimodal_gen.runtime.models.dits.sana as sana
from sglang.kernels.ops.diffusion import BitExactFusionGate
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTensor:
    def __init__(self, shape, stride, *, dtype=torch.bfloat16):
        self.shape = tuple(shape)
        self._stride = tuple(stride)
        self.dtype = dtype
        self.device = torch.device("cuda")
        self.is_cuda = True

    def dim(self):
        return len(self.shape)

    def numel(self):
        return torch.Size(self.shape).numel()

    def stride(self, dim=None):
        return self._stride if dim is None else self._stride[dim]

    def is_contiguous(self):
        _, seq, hidden = self.shape
        return self._stride == (seq * hidden, hidden, 1)

    def contiguous(self):
        _, seq, hidden = self.shape
        return _FakeTensor(self.shape, (seq * hidden, hidden, 1), dtype=self.dtype)

    def __getitem__(self, _index):
        return self


class TestSanaLayerNormDispatch(CustomTestCase):
    def _patch_runtime(self, stack, *, capability=(9, 0), capturing=False):
        stack.enter_context(
            patch.object(torch.compiler, "is_compiling", return_value=False)
        )
        stack.enter_context(
            patch.multiple(
                torch.cuda,
                is_current_stream_capturing=lambda: capturing,
                current_stream=lambda: "side",
                default_stream=lambda: "default",
            )
        )
        arch = stack.enter_context(
            patch.object(
                sana,
                "current_arch",
                return_value=SimpleNamespace(
                    cuda=SimpleNamespace(capability=capability)
                ),
            )
        )
        stack.enter_context(
            patch.object(sana, "can_use_fused_layernorm_modulate", return_value=True)
        )
        return arch

    def _patch_gate(self, stack):
        gate = BitExactFusionGate("SANA dispatch test", per_signature=True)
        stack.enter_context(
            patch.multiple(
                sana,
                _SANA_LN_MOD=gate,
                _SANA_LN_MOD_SIGS=gate.verified_sigs,
                _SANA_LN_MOD_DISABLED=False,
            )
        )
        return gate

    @staticmethod
    def _inputs(x_stride=(136, 1, 17), scale_stride=(8, 8, 1)):
        return (
            _FakeTensor((2, 17, 8), x_stride),
            _FakeTensor((2, 1, 8), scale_stride),
            _FakeTensor((2, 1, 8), (8, 8, 1)),
        )

    @staticmethod
    def _patch_outputs(stack, result):
        eager = stack.enter_context(
            patch.object(sana, "_eager_ln_modulate", return_value=result)
        )
        strided = stack.enter_context(
            patch.object(
                sana, "fused_layernorm_modulate_strided_raw", return_value=result
            )
        )
        contiguous = stack.enter_context(
            patch.object(sana, "fused_layernorm_modulate_raw", return_value=result)
        )
        return eager, strided, contiguous

    def test_automatic_dispatch_and_arch_scoped_signatures(self):
        x, scale, shift = self._inputs()
        contiguous_x, _, _ = self._inputs(x_stride=(136, 8, 1))
        norm = nn.LayerNorm(8, elementwise_affine=False)
        expected = torch.tensor([1.0])

        with ExitStack() as stack:
            arch = self._patch_runtime(stack)
            gate = self._patch_gate(stack)
            _, strided, contiguous = self._patch_outputs(stack, expected)

            sana.sana_ln_modulate(norm, x, scale, shift)
            arch.return_value = SimpleNamespace(
                cuda=SimpleNamespace(capability=(10, 0))
            )
            sana.sana_ln_modulate(norm, x, scale, shift)
            sana.sana_ln_modulate(norm, contiguous_x, scale, shift)
            arch.return_value = SimpleNamespace(
                cuda=SimpleNamespace(capability=(12, 0))
            )
            sana.sana_ln_modulate(norm, contiguous_x, scale, shift)

            self.assertEqual(strided.call_count, 2)
            self.assertEqual(contiguous.call_count, 2)
            self.assertEqual(
                {(sig[0], sig[1]) for sig in gate.verified_sigs},
                {
                    ("strided", (9, 0)),
                    ("strided", (10, 0)),
                    ("contiguous", (10, 0)),
                    ("contiguous", (12, 0)),
                },
            )

    def test_cached_signature_cannot_bypass_plain_layernorm_requirement(self):
        x, scale, shift = self._inputs(x_stride=(136, 8, 1))
        plain_norm = nn.LayerNorm(8, eps=1e-6, elementwise_affine=False)
        affine_norm = nn.LayerNorm(8, eps=1e-6, elementwise_affine=True)
        plain_result, affine_result = torch.tensor([1.0]), torch.tensor([2.0])

        with ExitStack() as stack:
            self._patch_runtime(stack)
            self._patch_gate(stack)
            eager = stack.enter_context(
                patch.object(
                    sana,
                    "_eager_ln_modulate",
                    side_effect=[plain_result, affine_result],
                )
            )
            contiguous = stack.enter_context(
                patch.object(
                    sana, "fused_layernorm_modulate_raw", return_value=plain_result
                )
            )

            sana.sana_ln_modulate(plain_norm, x, scale, shift)
            actual = sana.sana_ln_modulate(affine_norm, x, scale, shift)

            self.assertTrue(torch.equal(actual, affine_result))
            contiguous.assert_called_once()
            self.assertEqual(eager.call_count, 2)

    def test_unsupported_layouts_fall_back_appropriately(self):
        expected = torch.tensor([1.0])
        cases = (
            ((135, 1, 17), (8, 8, 1), True),
            ((136, 1, 17), (8, 8, 2), False),
        )
        for x_stride, scale_stride, uses_contiguous in cases:
            with self.subTest(x_stride=x_stride, scale_stride=scale_stride):
                x, scale, shift = self._inputs(x_stride, scale_stride)
                with ExitStack() as stack:
                    self._patch_runtime(stack)
                    gate = self._patch_gate(stack)
                    _, strided, contiguous = self._patch_outputs(stack, expected)

                    actual = sana.sana_ln_modulate(
                        nn.LayerNorm(8, elementwise_affine=False), x, scale, shift
                    )

                    self.assertTrue(torch.equal(actual, expected))
                    strided.assert_not_called()
                    if uses_contiguous:
                        contiguous.assert_called_once()
                    else:
                        contiguous.assert_not_called()
                        self.assertFalse(gate.verified_sigs)

    def test_unverified_signature_uses_eager_during_capture(self):
        x, scale, shift = self._inputs()
        expected = torch.tensor([1.0])
        with ExitStack() as stack:
            self._patch_runtime(stack, capturing=True)
            gate = self._patch_gate(stack)
            _, strided, contiguous = self._patch_outputs(stack, expected)

            sana.sana_ln_modulate(
                nn.LayerNorm(8, elementwise_affine=False), x, scale, shift
            )

            strided.assert_not_called()
            contiguous.assert_not_called()
            self.assertFalse(gate.verified_sigs)


if __name__ == "__main__":
    unittest.main()
