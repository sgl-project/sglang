# SPDX-License-Identifier: Apache-2.0
"""ComfyUI `diffusion_model.` FLUX.2 LoRAs split the fused double-block QKV."""

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.lora.format_adapter import (
    LoRAFormat,
    detect_lora_format_from_state_dict,
    normalize_lora_state_dict,
)

RANK, DIM = 4, 8


def _lora(prefix: str) -> dict[str, torch.Tensor]:
    sd = {}
    for m, out_dim in (
        ("double_blocks.0.img_attn.qkv", 3 * DIM),
        ("double_blocks.0.txt_attn.qkv", 3 * DIM),
        ("double_blocks.0.img_attn.proj", DIM),
        ("single_blocks.0.linear2", DIM),
    ):
        sd[f"{prefix}{m}.lora_A.weight"] = torch.randn(RANK, DIM)
        sd[f"{prefix}{m}.lora_B.weight"] = torch.randn(out_dim, RANK)
    return sd


@pytest.mark.parametrize("prefix", ["", "base_model.model.", "diffusion_model."])
def test_flux2_bfl_lora_splits_double_block_qkv(prefix):
    sd = _lora(prefix)
    assert detect_lora_format_from_state_dict(sd) == LoRAFormat.AI_TOOLKIT_FLUX

    out = normalize_lora_state_dict(sd)

    b = sd[f"{prefix}double_blocks.0.img_attn.qkv.lora_B.weight"]
    for i, name in enumerate(("to_q", "to_k", "to_v")):
        torch.testing.assert_close(
            out[f"transformer_blocks.0.attn.{name}.lora_B.weight"],
            b[i * DIM : (i + 1) * DIM],
        )
    for name in ("add_q_proj", "add_k_proj", "add_v_proj"):
        assert f"transformer_blocks.0.attn.{name}.lora_A.weight" in out
    assert "transformer_blocks.0.attn.to_out.0.lora_A.weight" in out
    assert "single_transformer_blocks.0.attn.to_out.lora_A.weight" in out
