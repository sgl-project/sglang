# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for SANA's opt-in strided LayerNorm dispatch."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

import sglang.multimodal_gen.runtime.models.dits.sana as sana
import sglang.multimodal_gen.runtime.models.dits.sana_video as sana_video
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
        b, s, d = self.shape
        return self._stride == (s * d, d, 1)

    def contiguous(self):
        b, s, d = self.shape
        return _FakeCudaTensor(self.shape, (s * d, d, 1), dtype=self.dtype)


class _FakeModulation:
    def __init__(self, batch, hidden):
        self.shape = (batch, 1, hidden)
        self._stride = (hidden, hidden, 1)
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda")

    def dim(self):
        return 3

    def stride(self, dim=None):
        return self._stride if dim is None else self._stride[dim]

    def __getitem__(self, _index):
        return self


class TestSanaStridedLayerNormDispatch(CustomTestCase):
    def _patch_runtime(self, stack, *, sm90=True, capturing=False):
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
        stack.enter_context(
            patch.object(
                sana,
                "current_arch",
                return_value=SimpleNamespace(
                    is_sm=lambda major, minor: sm90 and (major, minor) == (9, 0)
                ),
            )
        )
        stack.enter_context(
            patch.object(sana, "can_use_fused_layernorm_modulate", return_value=True)
        )

    def _patch_gate(self, stack):
        gate = BitExactFusionGate("SANA dispatch test", per_signature=True)
        stack.enter_context(patch.object(sana, "_SANA_LN_MOD", gate))
        stack.enter_context(patch.object(sana, "_SANA_LN_MOD_SIGS", gate.verified_sigs))
        stack.enter_context(patch.object(sana, "_SANA_LN_MOD_DISABLED", False))
        return gate

    @staticmethod
    def _inputs(stride=(136, 1, 17)):
        batch, seq, hidden = 2, 17, 8
        x = _FakeCudaTensor((batch, seq, hidden), stride)
        return (
            x,
            _FakeModulation(batch, hidden),
            _FakeModulation(batch, hidden),
            nn.LayerNorm(hidden, elementwise_affine=False),
        )

    def test_strided_dispatch_is_opt_in_and_uses_a_distinct_gate_signature(self):
        x, scale, shift, norm = self._inputs()
        expected = torch.tensor([1.0])
        with ExitStack() as stack:
            self._patch_runtime(stack)
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

            sana.sana_ln_modulate(norm, x, scale, shift, prefer_strided=True)
            sana.sana_ln_modulate(norm, x, scale, shift)

            self.assertEqual(strided.call_count, 1)
            self.assertEqual(contiguous.call_count, 1)
            self.assertEqual(
                {sig[0] for sig in gate.verified_sigs}, {"strided", "contiguous"}
            )

    def test_unsupported_layout_and_non_sm90_keep_the_contiguous_path(self):
        expected = torch.tensor([1.0])
        for x_stride, sm90 in (((135, 1, 17), True), ((136, 1, 17), False)):
            x, scale, shift, norm = self._inputs(x_stride)
            with ExitStack() as stack:
                self._patch_runtime(stack, sm90=sm90)
                self._patch_gate(stack)
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

                sana.sana_ln_modulate(norm, x, scale, shift, prefer_strided=True)

                strided.assert_not_called()
                contiguous.assert_called_once()

    def test_unverified_strided_signature_falls_back_during_capture(self):
        x, scale, shift, norm = self._inputs()
        expected = torch.tensor([1.0])
        with ExitStack() as stack:
            self._patch_runtime(stack, capturing=True)
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

            sana.sana_ln_modulate(norm, x, scale, shift, prefer_strided=True)

            strided.assert_not_called()
            contiguous.assert_not_called()
            self.assertFalse(gate.verified_sigs)

    def test_sana_video_opts_in_and_image_block_keeps_the_default(self):
        batch, hidden = 1, 8
        hidden_states = torch.zeros((batch, 1, hidden), dtype=torch.bfloat16)
        timestep = torch.zeros((batch, 1, 6 * hidden))
        rotary_emb = (torch.empty(0), torch.empty(0))

        video = object.__new__(sana_video.SanaVideoTransformerBlock)
        nn.Module.__init__(video)
        video.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        video.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        video.scale_shift_table = nn.Parameter(torch.zeros(6, hidden))
        video.attn1 = lambda x, _rope: torch.zeros_like(x)
        video.attn2 = lambda x, _encoder, _mask: torch.zeros_like(x)
        video.ff = lambda x: torch.zeros_like(x)

        image = object.__new__(sana.SanaTransformerBlock)
        nn.Module.__init__(image)
        image.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        image.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        image.scale_shift_table = nn.Parameter(torch.zeros(6, hidden))
        image.attn1 = lambda x: torch.zeros_like(x)
        image.attn2 = lambda x, _encoder, _mask: torch.zeros_like(x)
        image.ff = lambda x: torch.zeros_like(x)

        with (
            patch.object(
                sana_video,
                "sana_ln_modulate",
                side_effect=lambda _norm, x, _scale, _shift, **_kwargs: x,
            ) as video_ln,
            patch.object(
                sana,
                "sana_ln_modulate",
                side_effect=lambda _norm, x, _scale, _shift, **_kwargs: x,
            ) as image_ln,
            patch.object(
                sana_video,
                "sana_residual_gate_add",
                side_effect=lambda residual, _update, _gate: residual,
            ),
            patch.object(
                sana,
                "sana_residual_gate_add",
                side_effect=lambda residual, _update, _gate: residual,
            ),
        ):
            video(
                hidden_states,
                None,
                None,
                timestep,
                frames=1,
                height=1,
                width=1,
                rotary_emb=rotary_emb,
            )
            image(hidden_states, None, timestep[:, 0], height=1, width=1)

        self.assertEqual(video_ln.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs.get("prefer_strided") is True
                for call in video_ln.call_args_list
            )
        )
        self.assertEqual(image_ln.call_count, 2)
        self.assertTrue(all(not call.kwargs for call in image_ln.call_args_list))


if __name__ == "__main__":
    unittest.main()
