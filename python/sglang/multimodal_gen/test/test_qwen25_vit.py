"""
Test custom Qwen2_5_VisionTransformer output consistency vs HuggingFace reference.

Builds a mini ViT (depth=2, hidden=64, heads=4) with shared weights between
HuggingFace and our custom implementation, then asserts numerical equivalence
across different grid sizes and multi-image batches.
"""

import os

os.environ["SGLANG_USE_AITER"] = "0"

import copy

import pytest
import torch

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


class TestVisionTransformerConsistency:
    """Verify custom Qwen2_5_VisionTransformer matches HuggingFace output."""

    @pytest.fixture
    def models_and_input(self):
        _ensure_tp_initialized()

        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
        )

        from sglang.multimodal_gen.runtime.models.encoders.qwen2_5vl import (
            Qwen2_5_VisionTransformer,
        )

        vcfg = _make_mini_vision_config(attn_implementation="eager")

        hf_vit = Qwen2_5_VisionTransformerPretrainedModel._from_config(vcfg)
        hf_vit.eval()

        custom_vit = Qwen2_5_VisionTransformer(copy.deepcopy(vcfg))
        custom_vit.eval()

        _copy_weights_hf_to_custom(hf_vit, custom_vit)

        t, h, w = 1, 28, 28
        grid_thw = torch.tensor([[t, h // vcfg.patch_size, w // vcfg.patch_size]])
        num_patches = t * (h // vcfg.patch_size) * (w // vcfg.patch_size)
        pixel_values = torch.randn(
            num_patches * vcfg.temporal_patch_size,
            vcfg.in_channels,
            vcfg.patch_size,
            vcfg.patch_size,
        )

        return hf_vit, custom_vit, pixel_values, grid_thw

    def test_output_matches(self, models_and_input):
        hf_vit, custom_vit, pixel_values, grid_thw = models_and_input

        with torch.no_grad():
            hf_out = hf_vit(pixel_values, grid_thw=grid_thw)
            custom_out = custom_vit(pixel_values, grid_thw=grid_thw)

        assert hf_out.shape == custom_out.shape
        torch.testing.assert_close(custom_out, hf_out, atol=1e-5, rtol=1e-5)

    def test_output_matches_larger_grid(self, models_and_input):
        hf_vit, custom_vit, _, _ = models_and_input
        vcfg = hf_vit.config if hasattr(hf_vit, "config") else custom_vit.config

        grid_thw = torch.tensor([[1, 4, 4]])
        num_patches = 1 * 4 * 4
        pixel_values = torch.randn(
            num_patches * vcfg.temporal_patch_size,
            vcfg.in_channels,
            vcfg.patch_size,
            vcfg.patch_size,
        )

        with torch.no_grad():
            hf_out = hf_vit(pixel_values, grid_thw=grid_thw)
            custom_out = custom_vit(pixel_values, grid_thw=grid_thw)

        assert hf_out.shape == custom_out.shape
        torch.testing.assert_close(custom_out, hf_out, atol=1e-5, rtol=1e-5)

    def test_multi_image_batch(self, models_and_input):
        hf_vit, custom_vit, _, _ = models_and_input
        vcfg = hf_vit.config if hasattr(hf_vit, "config") else custom_vit.config

        grid_thw = torch.tensor([[1, 2, 2], [1, 4, 4]])
        total_patches = 2 * 2 + 4 * 4
        pixel_values = torch.randn(
            total_patches * vcfg.temporal_patch_size,
            vcfg.in_channels,
            vcfg.patch_size,
            vcfg.patch_size,
        )

        with torch.no_grad():
            hf_out = hf_vit(pixel_values, grid_thw=grid_thw)
            custom_out = custom_vit(pixel_values, grid_thw=grid_thw)

        assert hf_out.shape == custom_out.shape
        torch.testing.assert_close(custom_out, hf_out, atol=1e-5, rtol=1e-5)
