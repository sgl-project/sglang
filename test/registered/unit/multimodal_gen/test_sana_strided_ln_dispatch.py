# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for SANA's automatic strided LayerNorm dispatch."""

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


class _FakeCudaTensor:
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
        return _FakeCudaTensor(
            self.shape, (seq * hidden, hidden, 1), dtype=self.dtype
        )


class _FakeModulation:
    def __init__(self, batch, hidden, *, stride=None):
        self.shape = (batch, 1, hidden)
        self._stride = stride or (hidden, hidden, 1)
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda")

    def dim(self):
        return 3

    def stride(self, dim=None):
        return self._stride if dim is None else self._stride[dim]

    def __getitem__(self, _index):
        return self


class TestSanaStridedLayerNormDispatch(CustomTestCase):
    def _patch_runtime(self, stack, *, capability=(9, 0), capturing=False):
        current_stream = object()
        default_stream = object()
        stack.enter_context(
            patch.object(torch.compiler, "is_compiling", return_value=False)
        )
        stack.enter_context(
            patch.object(
                torch.cuda,
                "is_current_stream_capturing",
                return_value=capturing,
            )
        )
        stack.enter_context(
            patch.object(torch.cuda, "current_stream", return_value=current_stream)
        )
        stack.enter_context(
            patch.object(torch.cuda, "default_stream", return_value=default_stream)
        )
        arch_patch = stack.enter_context(
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
        return arch_patch

    def _patch_gate(self, stack):
        gate = BitExactFusionGate("SANA dispatch test", per_signature=True)
        stack.enter_context(patch.object(sana, "_SANA_LN_MOD", gate))
        stack.enter_context(patch.object(sana, "_SANA_LN_MOD_SIGS", gate.verified_sigs))
        stack.enter_context(patch.object(sana, "_SANA_LN_MOD_DISABLED", False))
        return gate

    @staticmethod
    def _inputs(x_stride=(136, 1, 17), scale_stride=None):
        batch, seq, hidden = 2, 17, 8
        return (
            _FakeCudaTensor((batch, seq, hidden), x_stride),
            _FakeModulation(batch, hidden, stride=scale_stride),
            _FakeModulation(batch, hidden),
            nn.LayerNorm(hidden, elementwise_affine=False),
        )

    def test_eligible_layout_dispatches_automatically_and_verifies_per_arch(self):
        x, scale, shift, norm = self._inputs()
        expected = torch.tensor([1.0])
        with ExitStack() as stack:
            arch_patch = self._patch_runtime(stack)
            gate = self._patch_gate(stack)
            stack.enter_context(
                patch.object(sana, "_eager_ln_modulate", return_value=expected)
            )
            strided = stack.enter_context(
                patch.object(
                    sana,
                    "fused_layernorm_modulate_strided_raw",
                    return_value=expected,
                )
            )
            contiguous = stack.enter_context(
                patch.object(
                    sana, "fused_layernorm_modulate_raw", return_value=expected
                )
            )

            actual = sana.sana_ln_modulate(norm, x, scale, shift)
            self.assertTrue(torch.equal(actual, expected))
            arch_patch.return_value = SimpleNamespace(
                cuda=SimpleNamespace(capability=(10, 0))
            )
            actual = sana.sana_ln_modulate(norm, x, scale, shift)
            self.assertTrue(torch.equal(actual, expected))

            self.assertEqual(strided.call_count, 2)
            contiguous.assert_not_called()
            self.assertEqual(
                {sig[1] for sig in gate.verified_sigs}, {(9, 0), (10, 0)}
            )

    def test_unsupported_stride_uses_the_contiguous_path(self):
        x, scale, shift, norm = self._inputs(x_stride=(135, 1, 17))
        expected = torch.tensor([1.0])
        with ExitStack() as stack:
            self._patch_runtime(stack)
            self._patch_gate(stack)
            stack.enter_context(
                patch.object(sana, "_eager_ln_modulate", return_value=expected)
            )
            strided = stack.enter_context(
                patch.object(sana, "fused_layernorm_modulate_strided_raw")
            )
            contiguous = stack.enter_context(
                patch.object(
                    sana, "fused_layernorm_modulate_raw", return_value=expected
                )
            )

            sana.sana_ln_modulate(norm, x, scale, shift)

            strided.assert_not_called()
            contiguous.assert_called_once()

    def test_incompatible_modulation_falls_back_to_eager(self):
        x, scale, shift, norm = self._inputs(scale_stride=(8, 8, 2))
        expected = torch.tensor([1.0])
        with ExitStack() as stack:
            self._patch_runtime(stack)
            gate = self._patch_gate(stack)
            stack.enter_context(
                patch.object(sana, "_eager_ln_modulate", return_value=expected)
            )
            strided = stack.enter_context(
                patch.object(sana, "fused_layernorm_modulate_strided_raw")
            )
            contiguous = stack.enter_context(
                patch.object(
                    sana,
                    "fused_layernorm_modulate_raw",
                    side_effect=RuntimeError("incompatible modulation strides"),
                )
            )

            actual = sana.sana_ln_modulate(norm, x, scale, shift)

            self.assertTrue(torch.equal(actual, expected))
            strided.assert_not_called()
            contiguous.assert_called_once()
            self.assertTrue(gate.disabled)

    def test_numerical_mismatch_disables_candidate_and_returns_eager(self):
        x, scale, shift, norm = self._inputs()
        expected = torch.tensor([1.0])
        with ExitStack() as stack:
            self._patch_runtime(stack)
            gate = self._patch_gate(stack)
            stack.enter_context(
                patch.object(sana, "_eager_ln_modulate", return_value=expected)
            )
            stack.enter_context(
                patch.object(
                    sana,
                    "fused_layernorm_modulate_strided_raw",
                    return_value=torch.tensor([0.0]),
                )
            )

            actual = sana.sana_ln_modulate(norm, x, scale, shift)

            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(gate.disabled)
            self.assertFalse(gate.verified_sigs)

    def test_unverified_signature_falls_back_during_capture(self):
        x, scale, shift, norm = self._inputs()
        expected = torch.tensor([1.0])
        with ExitStack() as stack:
            self._patch_runtime(stack, capturing=True)
            gate = self._patch_gate(stack)
            stack.enter_context(
                patch.object(sana, "_eager_ln_modulate", return_value=expected)
            )
            strided = stack.enter_context(
                patch.object(sana, "fused_layernorm_modulate_strided_raw")
            )
            contiguous = stack.enter_context(
                patch.object(sana, "fused_layernorm_modulate_raw")
            )

            sana.sana_ln_modulate(norm, x, scale, shift)

            strided.assert_not_called()
            contiguous.assert_not_called()
            self.assertFalse(gate.verified_sigs)


if __name__ == "__main__":
    unittest.main()
