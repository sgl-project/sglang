# SPDX-License-Identifier: Apache-2.0
"""ComfyUI `diffusion_model.` FLUX.2 LoRAs split the fused double-block QKV."""

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.lora.format_adapter import (
    LoRAFormat,
    _bfl_flux_down_up_as_a_b,
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


@pytest.mark.parametrize("suffix", [".weight", ""])
def test_flux2_bfl_lora_with_down_up_suffixes_splits_double_block_qkv(suffix):
    sd = {
        k.replace(".lora_A.weight", f".lora_down{suffix}").replace(
            ".lora_B.weight", f".lora_up{suffix}"
        ): v
        for k, v in _lora("diffusion_model.").items()
    }
    sd["diffusion_model.double_blocks.0.img_attn.qkv.alpha"] = torch.tensor(2.0)
    out = normalize_lora_state_dict(sd)

    b = sd[f"diffusion_model.double_blocks.0.img_attn.qkv.lora_up{suffix}"]
    for i, name in enumerate(("to_q", "to_k", "to_v")):
        torch.testing.assert_close(
            out[f"transformer_blocks.0.attn.{name}.lora_B.weight"],
            b[i * DIM : (i + 1) * DIM],
        )
        # The module alpha follows each split section.
        assert out[f"transformer_blocks.0.attn.{name}.alpha"].item() == 2
    assert "transformer_blocks.0.attn.add_v_proj.lora_A.weight" in out
    assert not any(".lora_down" in k or ".lora_up" in k for k in out)


def test_hunyuan_double_blocks_are_not_renamed_as_flux():
    sd = {
        "diffusion_model.double_blocks.0.img_attn_qkv.lora_down.weight": torch.randn(
            RANK, DIM
        ),
        "diffusion_model.double_blocks.0.img_attn_qkv.lora_up.weight": torch.randn(
            3 * DIM, RANK
        ),
    }
    assert _bfl_flux_down_up_as_a_b(sd) is sd


@pytest.mark.parametrize("prefix", ["", "diffusion_model."])
def test_flux2_bfl_alpha_pattern_follows_the_renamed_module(prefix):
    config = {"r": RANK, "lora_alpha": 8, "alpha_pattern": {"img_attn.proj": 2}}
    out = normalize_lora_state_dict(_lora(prefix), adapter_config=config)

    assert out["transformer_blocks.0.attn.to_out.0.alpha"].item() == 2
    # Modules the pattern does not name keep the global alpha.
    assert "transformer_blocks.0.attn.to_q.alpha" not in out
