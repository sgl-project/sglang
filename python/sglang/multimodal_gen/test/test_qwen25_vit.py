"""
Test custom Qwen2_5_VisionTransformer output consistency vs HuggingFace reference.

Builds a mini ViT (depth=2, hidden=64, heads=4) with shared weights between
HuggingFace and our custom implementation, then asserts numerical equivalence
across grid sizes representative of qwen_image_edit workloads.

Grid sizes are derived from qwen_image_edit pipeline constants:
- CONDITION_IMAGE_AREA = 384 * 384 (condition image target area)
- Typical aspect ratios: 1:1, 4:3, 3:4, 16:9, 9:16
"""

import math
import os

os.environ["SGLANG_USE_AITER"] = "0"

import copy

import pytest
import torch

# ---------------------------------------------------------------------------
# qwen_image_edit size helpers
# ---------------------------------------------------------------------------

CONDITION_IMAGE_AREA = 384 * 384


def _calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    """Replicate qwen_image_edit's calculate_dimensions (rounds to 32)."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


def _pixel_to_grid(
    h_pixels: int, w_pixels: int, patch_size: int = 14, merge_size: int = 2
) -> tuple[int, int]:
    """Convert pixel dimensions to ViT grid dimensions (in patches).

    Mimics Qwen2.5-VL preprocessing: round to the nearest multiple of
    (patch_size * merge_size), then divide by patch_size so that the
    resulting grid is always divisible by merge_size.
    """
    unit = patch_size * merge_size
    grid_h = max(round(h_pixels / unit), 1) * merge_size
    grid_w = max(round(w_pixels / unit), 1) * merge_size
    return grid_h, grid_w


ASPECT_RATIOS: dict[str, float] = {
    "1_1": 1.0,
    "4_3": 4.0 / 3.0,
    "3_4": 3.0 / 4.0,
    "16_9": 16.0 / 9.0,
    "9_16": 9.0 / 16.0,
}


def _build_grid_cases() -> list[tuple[str, list[tuple[int, int, int]]]]:
    """Build (test_id, grid_specs) for qwen_image_edit-like scenarios."""
    cases: list[tuple[str, list[tuple[int, int, int]]]] = []

    for ratio_name, ratio in ASPECT_RATIOS.items():
        w, h = _calculate_dimensions(CONDITION_IMAGE_AREA, ratio)
        grid_h, grid_w = _pixel_to_grid(h, w)
        cases.append((f"condition_{ratio_name}", [(1, grid_h, grid_w)]))

    # Multi-image: condition + noisy (typical image-edit scenario)
    w1, h1 = _calculate_dimensions(CONDITION_IMAGE_AREA, 1.0)
    g1h, g1w = _pixel_to_grid(h1, w1)
    w2, h2 = _calculate_dimensions(CONDITION_IMAGE_AREA, 4.0 / 3.0)
    g2h, g2w = _pixel_to_grid(h2, w2)
    cases.append(("edit_multi_same", [(1, g1h, g1w), (1, g1h, g1w)]))
    cases.append(("edit_multi_mixed", [(1, g1h, g1w), (1, g2h, g2w)]))

    return cases


GRID_CASES = _build_grid_cases()

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

_tp_initialized = False


def _ensure_tp_initialized():
    global _tp_initialized
    if _tp_initialized:
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    initialize_model_parallel(
        tensor_parallel_degree=1,
        data_parallel_size=1,
        pipeline_parallel_degree=1,
        sequence_parallel_degree=1,
        classifier_free_guidance_degree=1,
    )
    _tp_initialized = True


def _make_mini_vision_config(attn_implementation="eager"):
    from transformers import Qwen2_5_VLConfig

    full_cfg = Qwen2_5_VLConfig()
    vcfg = full_cfg.vision_config
    vcfg.hidden_size = 64
    vcfg.num_heads = 4
    vcfg.depth = 2
    vcfg.intermediate_size = 128
    vcfg.out_hidden_size = 64
    vcfg.spatial_merge_size = 2
    vcfg.patch_size = 14
    vcfg.temporal_patch_size = 2
    vcfg.in_channels = 3
    vcfg.hidden_act = "silu"
    vcfg.window_size = 112
    vcfg.fullatt_block_indexes = [1]
    vcfg._attn_implementation = attn_implementation
    return vcfg


def _copy_weights_hf_to_custom(hf_vit, custom_vit):
    """Copy weights from HF ViT to custom ViT, handling fused gate_up_proj."""
    custom_vit.patch_embed.load_state_dict(hf_vit.patch_embed.state_dict())
    custom_vit.merger.load_state_dict(hf_vit.merger.state_dict())

    for hf_blk, custom_blk in zip(hf_vit.blocks, custom_vit.blocks):
        custom_blk.norm1.weight.data.copy_(hf_blk.norm1.weight.data)
        custom_blk.norm2.weight.data.copy_(hf_blk.norm2.weight.data)
        custom_blk.attn.qkv.weight.data.copy_(hf_blk.attn.qkv.weight.data)
        custom_blk.attn.qkv.bias.data.copy_(hf_blk.attn.qkv.bias.data)
        custom_blk.attn.proj.weight.data.copy_(hf_blk.attn.proj.weight.data)
        custom_blk.attn.proj.bias.data.copy_(hf_blk.attn.proj.bias.data)

        hf_mlp = hf_blk.mlp
        custom_mlp = custom_blk.mlp
        custom_mlp.gate_up_proj.weight.data.copy_(
            torch.cat([hf_mlp.gate_proj.weight.data, hf_mlp.up_proj.weight.data], dim=0)
        )
        custom_mlp.gate_up_proj.bias.data.copy_(
            torch.cat([hf_mlp.gate_proj.bias.data, hf_mlp.up_proj.bias.data], dim=0)
        )
        custom_mlp.down_proj.weight.data.copy_(hf_mlp.down_proj.weight.data)
        custom_mlp.down_proj.bias.data.copy_(hf_mlp.down_proj.bias.data)


def _make_pixel_values(grid_specs, vcfg):
    """Create random pixel_values and grid_thw from grid specifications."""
    grid_thw = torch.tensor(grid_specs)
    total_patches = sum(t * h * w for t, h, w in grid_specs)
    pixel_values = torch.randn(
        total_patches * vcfg.temporal_patch_size,
        vcfg.in_channels,
        vcfg.patch_size,
        vcfg.patch_size,
    )
    return pixel_values, grid_thw


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVisionTransformerConsistency:
    """Verify custom Qwen2_5_VisionTransformer matches HuggingFace output
    on grid sizes representative of qwen_image_edit workloads."""

    @pytest.fixture(scope="class")
    def models(self):
        _ensure_tp_initialized()

        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
        )

        from sglang.multimodal_gen.runtime.models.encoders.qwen2_5vl import (
            Qwen2_5_VisionTransformer,
        )

        use_cuda = torch.cuda.is_available()
        attn_impl = "flash_attention_2" if use_cuda else "sdpa"
        device = torch.device("cuda" if use_cuda else "cpu")
        dtype = torch.bfloat16 if use_cuda else torch.float32

        vcfg = _make_mini_vision_config(attn_implementation=attn_impl)

        hf_vit = Qwen2_5_VisionTransformerPretrainedModel._from_config(vcfg)
        hf_vit.eval().to(device=device, dtype=dtype)

        custom_vit = Qwen2_5_VisionTransformer(copy.deepcopy(vcfg))
        custom_vit.eval().to(device=device, dtype=dtype)

        _copy_weights_hf_to_custom(hf_vit, custom_vit)

        return hf_vit, custom_vit, vcfg, device, dtype

    @pytest.mark.parametrize(
        "name, grid_specs",
        GRID_CASES,
        ids=[c[0] for c in GRID_CASES],
    )
    def test_output_matches(self, models, name, grid_specs):
        hf_vit, custom_vit, vcfg, device, dtype = models
        pixel_values, grid_thw = _make_pixel_values(grid_specs, vcfg)
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        grid_thw = grid_thw.to(device=device)

        atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5

        with torch.no_grad():
            hf_out = hf_vit(pixel_values, grid_thw=grid_thw)
            custom_out = custom_vit(pixel_values, grid_thw=grid_thw)

        assert (
            hf_out.shape == custom_out.shape
        ), f"Shape mismatch for {name}: HF {hf_out.shape} vs custom {custom_out.shape}"
        torch.testing.assert_close(custom_out, hf_out, atol=atol, rtol=rtol)
