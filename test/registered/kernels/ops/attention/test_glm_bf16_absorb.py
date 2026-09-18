"""Correctness tests for GLM's gfx950 BF16 MLA absorb matmuls."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.fp8_utils import input_to_float8
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_rocm,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=5, suite="jit-kernel-unit-test-amd")

_LOCAL_HEADS = 8
_KV_LORA_RANK = 512
_QK_NOPE_HEAD_DIM = 192
_V_HEAD_DIM = 256


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(),
    "GLM BF16 absorb operators require HIP gfx950",
)
class TestGlmBf16AbsorbOperators(CustomTestCase):
    def test_gfx950_bf16_absorb_bmms_match_fp32_reference(self):
        torch.manual_seed(0)
        device = torch.device("cuda")
        source_w_kc = (
            torch.randn(
                _LOCAL_HEADS,
                _QK_NOPE_HEAD_DIM,
                _KV_LORA_RANK,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.02
        )
        source_w_vc = (
            torch.randn(
                _LOCAL_HEADS,
                _V_HEAD_DIM,
                _KV_LORA_RANK,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.02
        )
        decode_weight, decode_scale = input_to_float8(
            torch.cat(
                [
                    source_w_kc,
                    source_w_vc,
                ],
                dim=1,
            ).flatten(0, 1),
            dtype=torch.float8_e4m3fn,
        )
        decode_weight = decode_weight.unflatten(
            0,
            (_LOCAL_HEADS, _QK_NOPE_HEAD_DIM + _V_HEAD_DIM),
        )
        decode_w_kc, decode_w_vc = decode_weight.split(
            [_QK_NOPE_HEAD_DIM, _V_HEAD_DIM],
            dim=1,
        )
        self_attn = SimpleNamespace(
            num_local_heads=_LOCAL_HEADS,
            o_proj=SimpleNamespace(
                weight=torch.empty(1, device=device, dtype=torch.bfloat16)
            ),
            w_kc=source_w_kc,
            w_vc=source_w_vc.transpose(1, 2),
            w_scale=1.0,
            w_kc_decode=decode_w_kc,
            w_vc_decode=decode_w_vc.transpose(1, 2),
            w_scale_decode=decode_scale,
            use_deep_gemm_bmm=False,
        )

        with patch.object(forward_mla_rocm, "_use_aiter_gfx95", True):
            for num_tokens in (1, 17):
                with self.subTest(num_tokens=num_tokens):
                    q_nope = (
                        torch.randn(
                            num_tokens,
                            _LOCAL_HEADS,
                            _QK_NOPE_HEAD_DIM,
                            device=device,
                            dtype=torch.bfloat16,
                        )
                        * 0.05
                    )
                    actual_q = forward_mla_rocm.rocm_absorb_q_bmm(
                        self_attn,
                        q_nope,
                        is_capture_mode=False,
                    )
                    reference_q = torch.einsum(
                        "thq,hqr->htr",  # codespell:ignore thq hqr htr
                        q_nope.float(),
                        source_w_kc.float(),
                    ).to(torch.bfloat16)

                    attn_output = (
                        torch.randn(
                            num_tokens,
                            _LOCAL_HEADS,
                            _KV_LORA_RANK,
                            device=device,
                            dtype=torch.bfloat16,
                        )
                        * 0.05
                    )
                    actual_v = forward_mla_rocm.rocm_absorb_v_bmm(
                        self_attn,
                        attn_output,
                    )
                    reference_v = (
                        torch.einsum(
                            "thr,hvr->thv",  # codespell:ignore thr hvr thv
                            attn_output.float(),
                            source_w_vc.float(),
                        )
                        .flatten(1, 2)
                        .to(torch.bfloat16)
                    )

                    self.assertEqual(actual_q.dtype, torch.bfloat16)
                    self.assertEqual(actual_v.dtype, torch.bfloat16)
                    torch.testing.assert_close(
                        actual_q,
                        reference_q,
                        atol=2e-2,
                        rtol=2e-2,
                    )
                    torch.testing.assert_close(
                        actual_v,
                        reference_v,
                        atol=2e-2,
                        rtol=2e-2,
                    )

                    actual_decode_q = forward_mla_rocm.rocm_absorb_q_bmm(
                        self_attn,
                        q_nope,
                        is_capture_mode=True,
                        use_decode_weights=True,
                    )
                    reference_decode_q = torch.einsum(
                        "thq,hqr->htr",  # codespell:ignore thq hqr htr
                        q_nope.float(),
                        decode_w_kc.float() * decode_scale,
                    ).to(torch.bfloat16)

                    actual_decode_v = forward_mla_rocm.rocm_absorb_v_bmm(
                        self_attn,
                        attn_output,
                        use_decode_weights=True,
                    )
                    reference_decode_v = (
                        torch.einsum(
                            "thr,hvr->thv",  # codespell:ignore thr hvr thv
                            attn_output.float(),
                            decode_w_vc.float() * decode_scale,
                        )
                        .flatten(1, 2)
                        .to(torch.bfloat16)
                    )

                    torch.testing.assert_close(
                        actual_decode_q,
                        reference_decode_q,
                        atol=2e-1,
                        rtol=2e-1,
                    )
                    torch.testing.assert_close(
                        actual_decode_v,
                        reference_decode_v,
                        atol=2e-1,
                        rtol=2e-1,
                    )


if __name__ == "__main__":
    unittest.main()
