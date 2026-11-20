"""DiffSynth-Studio MiniMax-H3 LoRAs carry per-head-interleaved QKV B rows."""

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.lora.format_adapter import (
    normalize_lora_state_dict,
)

HEADS, HEAD_DIM, RANK, HIDDEN = 4, 128, 2, 64


def _per_head_qkv_b():
    q, k, v = (torch.randn(HEADS, HEAD_DIM, RANK) for _ in range(3))
    interleaved = torch.stack([q, k, v], dim=1).reshape(-1, RANK)
    native = torch.cat([t.reshape(-1, RANK) for t in (q, k, v)])
    return interleaved, native


@pytest.mark.parametrize("prefix", ["", "token_refiner."])
def test_diffsynth_qkv_lora_b_is_regrouped(prefix):
    interleaved, native = _per_head_qkv_b()
    out_b = torch.randn(HIDDEN, RANK)
    module = f"{prefix}blocks.25.attn"
    normalized = normalize_lora_state_dict(
        {
            f"{module}.qkv_proj.lora_A.default.weight": torch.randn(RANK, HIDDEN),
            f"{module}.qkv_proj.lora_B.default.weight": interleaved,
            f"{module}.out_proj.lora_A.default.weight": torch.randn(RANK, HIDDEN),
            f"{module}.out_proj.lora_B.default.weight": out_b,
        }
    )

    torch.testing.assert_close(
        normalized[f"{module}.qkv_proj.lora_B.weight"], native, rtol=0, atol=0
    )
    torch.testing.assert_close(
        normalized[f"{module}.out_proj.lora_B.weight"], out_b, rtol=0, atol=0
    )


def test_native_qkv_lora_b_without_default_slot_is_unchanged():
    _, native = _per_head_qkv_b()
    normalized = normalize_lora_state_dict(
        {
            "blocks.0.attn.qkv_proj.lora_A.weight": torch.randn(RANK, HIDDEN),
            "blocks.0.attn.qkv_proj.lora_B.weight": native,
        }
    )

    torch.testing.assert_close(
        normalized["blocks.0.attn.qkv_proj.lora_B.weight"], native, rtol=0, atol=0
    )
