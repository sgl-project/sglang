from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm
from sglang.srt.layers.quantization.modelopt_quant import (
    _maybe_override_nvfp4_activation_scale_for_per_token,
)

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


def test_quantize_hidden_states_fp4_static_path():
    hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)
    input_scale_quant = torch.tensor(0.25, dtype=torch.float32)
    hs_fp4_bytes = torch.randint(0, 255, (2, 8), dtype=torch.uint8)
    hs_sf_bytes = torch.randint(0, 255, (2, 1), dtype=torch.uint8)

    with patch.object(
        flashinfer_trtllm, "fp4_quantize", return_value=(hs_fp4_bytes, hs_sf_bytes)
    ) as mock_fp4_quantize:
        hs_fp4, hs_sf, per_token_scale = flashinfer_trtllm.quantize_hidden_states_fp4(
            hidden_states,
            input_scale_quant,
            use_per_token_nvfp4=False,
        )

    mock_fp4_quantize.assert_called_once_with(
        hidden_states,
        input_scale_quant,
        16,
        False,
        False,
    )
    assert per_token_scale is None
    assert torch.equal(hs_fp4, hs_fp4_bytes)
    assert torch.equal(hs_sf.view(torch.uint8), hs_sf_bytes)


def test_quantize_hidden_states_fp4_per_token_path():
    hidden_states = torch.randn(3, 32, dtype=torch.bfloat16)
    input_scale_quant = torch.tensor(0.5, dtype=torch.float32)
    hs_fp4_bytes = torch.randint(0, 255, (3, 16), dtype=torch.uint8)
    hs_sf_bytes = torch.randint(0, 255, (3, 2), dtype=torch.uint8)
    per_token_scale_ref = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

    with (
        patch.object(
            flashinfer_trtllm,
            "nvfp4_quant_and_per_token_scale",
            return_value=(hs_fp4_bytes, hs_sf_bytes, per_token_scale_ref),
        ) as mock_per_token_quant,
        patch.object(
            flashinfer_trtllm,
            "fp4_quantize",
            side_effect=AssertionError("static fp4_quantize should not be used"),
        ),
    ):
        hs_fp4, hs_sf, per_token_scale = flashinfer_trtllm.quantize_hidden_states_fp4(
            hidden_states,
            input_scale_quant,
            use_per_token_nvfp4=True,
        )

    mock_per_token_quant.assert_called_once_with(
        hidden_states,
        flashinfer_trtllm._NVFP4_PER_TOKEN_GLOBAL_SCALE_INV,
    )
    assert torch.equal(hs_fp4, hs_fp4_bytes)
    assert torch.equal(hs_sf.view(torch.uint8), hs_sf_bytes)
    assert torch.equal(per_token_scale, per_token_scale_ref)


def test_quantize_hidden_states_fp4_per_token_requires_flashinfer_op():
    hidden_states = torch.randn(1, 16, dtype=torch.bfloat16)
    input_scale_quant = torch.tensor(1.0, dtype=torch.float32)

    with patch.object(flashinfer_trtllm, "nvfp4_quant_and_per_token_scale", None):
        with pytest.raises(
            RuntimeError,
            match="SGLANG_FLASHINFER_PER_TOKEN_NVFP4_MOE requires flashinfer.nvfp4_quant_and_per_token_scale",
        ):
            flashinfer_trtllm.quantize_hidden_states_fp4(
                hidden_states,
                input_scale_quant,
                use_per_token_nvfp4=True,
            )


def test_override_nvfp4_activation_scale_for_per_token():
    w2_input_scale = torch.tensor([0.25, 0.5], dtype=torch.float32)

    with envs.SGLANG_FLASHINFER_PER_TOKEN_NVFP4_MOE.override(False):
        out = _maybe_override_nvfp4_activation_scale_for_per_token(
            w2_input_scale,
            enable_flashinfer_trtllm_moe=True,
        )
        assert torch.equal(out, w2_input_scale)

    with envs.SGLANG_FLASHINFER_PER_TOKEN_NVFP4_MOE.override(True):
        out = _maybe_override_nvfp4_activation_scale_for_per_token(
            w2_input_scale,
            enable_flashinfer_trtllm_moe=False,
        )
        assert torch.equal(out, w2_input_scale)

    with envs.SGLANG_FLASHINFER_PER_TOKEN_NVFP4_MOE.override(True):
        out = _maybe_override_nvfp4_activation_scale_for_per_token(
            w2_input_scale,
            enable_flashinfer_trtllm_moe=True,
        )
        assert torch.equal(out, torch.ones_like(w2_input_scale))
