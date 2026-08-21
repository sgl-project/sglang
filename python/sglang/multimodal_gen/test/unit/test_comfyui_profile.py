# SPDX-License-Identifier: Apache-2.0
"""Tests for the comfyui_mode profile of native pipelines.

The existing ComfyUI tests under apps/ComfyUI_SGLDiffusion/test need a real
multi-GB checkpoint and two GPUs, so nothing here is covered in CI. These build
a tiny ComfyUI-format checkpoint instead, which is enough to exercise module
trimming, single-file loading, weight conversion, and stage construction.
"""

import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.models.dits.flux import FluxConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoint import (
    get_comfyui_checkpoint_spec,
)
from sglang.multimodal_gen.registry import get_pipeline_class
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.server_args.server_args import set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

HEADS, HEAD_DIM, LAYERS, SINGLE_LAYERS = 4, 16, 2, 2


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _shrink(arch) -> None:
    arch.num_attention_heads = HEADS
    arch.attention_head_dim = HEAD_DIM
    arch.num_layers = LAYERS
    arch.num_single_layers = SINGLE_LAYERS
    arch.guidance_embeds = True


def _write_comfyui_flux_checkpoint(path: str, config: FluxConfig) -> dict:
    """Write a checkpoint using ComfyUI's parameter names and fused layouts."""
    arch = config.arch_config
    hidden = arch.num_attention_heads * arch.attention_head_dim
    mlp_hidden = int(hidden * getattr(arch, "mlp_ratio", 4.0))
    qkv = 3 * hidden
    out_channels = arch.out_channels or arch.in_channels
    patch = getattr(arch, "patch_size", 1)

    sd: dict[str, torch.Tensor] = {}

    def put(name: str, *shape: int) -> None:
        sd[name] = torch.randn(*shape, dtype=torch.bfloat16)

    for b in range(arch.num_layers):
        for attn in ("img_attn", "txt_attn"):
            put(f"double_blocks.{b}.{attn}.qkv.weight", qkv, hidden)
            put(f"double_blocks.{b}.{attn}.qkv.bias", qkv)
            put(f"double_blocks.{b}.{attn}.proj.weight", hidden, hidden)
            put(f"double_blocks.{b}.{attn}.proj.bias", hidden)
            put(f"double_blocks.{b}.{attn}.norm.query_norm.scale", arch.attention_head_dim)
            put(f"double_blocks.{b}.{attn}.norm.key_norm.scale", arch.attention_head_dim)
        for mlp in ("img_mlp", "txt_mlp"):
            put(f"double_blocks.{b}.{mlp}.0.weight", mlp_hidden, hidden)
            put(f"double_blocks.{b}.{mlp}.0.bias", mlp_hidden)
            put(f"double_blocks.{b}.{mlp}.2.weight", hidden, mlp_hidden)
            put(f"double_blocks.{b}.{mlp}.2.bias", hidden)
        for mod in ("img_mod", "txt_mod"):
            put(f"double_blocks.{b}.{mod}.lin.weight", 6 * hidden, hidden)
            put(f"double_blocks.{b}.{mod}.lin.bias", 6 * hidden)

    for b in range(arch.num_single_layers):
        put(f"single_blocks.{b}.linear1.weight", qkv + mlp_hidden, hidden)
        put(f"single_blocks.{b}.linear1.bias", qkv + mlp_hidden)
        put(f"single_blocks.{b}.linear2.weight", hidden, hidden + mlp_hidden)
        put(f"single_blocks.{b}.linear2.bias", hidden)
        put(f"single_blocks.{b}.norm.query_norm.scale", arch.attention_head_dim)
        put(f"single_blocks.{b}.norm.key_norm.scale", arch.attention_head_dim)
        put(f"single_blocks.{b}.modulation.lin.weight", 3 * hidden, hidden)
        put(f"single_blocks.{b}.modulation.lin.bias", 3 * hidden)

    for stem, in_dim in (
        ("time_in", 256),
        ("vector_in", arch.pooled_projection_dim),
        ("guidance_in", 256),
    ):
        put(f"{stem}.in_layer.weight", hidden, in_dim)
        put(f"{stem}.in_layer.bias", hidden)
        put(f"{stem}.out_layer.weight", hidden, hidden)
        put(f"{stem}.out_layer.bias", hidden)

    put("txt_in.weight", hidden, arch.joint_attention_dim)
    put("txt_in.bias", hidden)
    put("img_in.weight", hidden, arch.in_channels)
    put("img_in.bias", hidden)
    put("final_layer.linear.weight", patch * patch * out_channels, hidden)
    put("final_layer.linear.bias", patch * patch * out_channels)
    put("final_layer.adaLN_modulation.1.weight", 2 * hidden, hidden)
    put("final_layer.adaLN_modulation.1.bias", 2 * hidden)

    save_file(sd, path)
    return sd


@pytest.fixture(scope="module")
def comfyui_flux_pipeline():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    tmpdir = tempfile.mkdtemp(prefix="comfyui_flux_")
    checkpoint = os.path.join(tmpdir, "flux_comfyui.safetensors")

    # The file has to exist before ServerArgs resolves the single-file path.
    seed_config = FluxConfig()
    _shrink(seed_config.arch_config)
    state_dict = _write_comfyui_flux_checkpoint(checkpoint, seed_config)

    server_args = ServerArgs.from_kwargs(
        model_path=checkpoint,
        pipeline_class_name="FluxPipeline",
        comfyui_mode=True,
        num_gpus=1,
    )
    _shrink(server_args.pipeline_config.dit_config.arch_config)
    set_global_server_args(server_args)
    _ensure_single_process_parallel_runtime()

    pipeline = get_pipeline_class("FluxPipeline")(
        model_path=checkpoint, server_args=server_args
    )
    return pipeline, state_dict


def test_specs_cover_every_comfyui_supported_pipeline():
    for pipeline_name in (
        "FluxPipeline",
        "ZImagePipeline",
        "QwenImagePipeline",
        "QwenImageEditPlusPipeline",
    ):
        spec = get_comfyui_checkpoint_spec(pipeline_name)
        assert spec is not None, f"{pipeline_name} has no ComfyUI checkpoint spec"
        assert get_pipeline_class(pipeline_name) is not None
        assert get_pipeline_class(pipeline_name).pipeline_config_cls is not None


def test_comfyui_mode_trims_pipeline_to_a_dit_forward_service(comfyui_flux_pipeline):
    pipeline, _ = comfyui_flux_pipeline

    assert pipeline.required_config_modules == ["transformer", "scheduler"]
    assert (
        type(pipeline.get_module("scheduler")).__name__ == "ComfyUIPassThroughScheduler"
    )
    assert list(pipeline._stage_name_mapping) == [
        "ComfyUILatentPreparationStage",
        "DenoisingStage",
    ]


def test_comfyui_checkpoint_fully_populates_the_transformer(comfyui_flux_pipeline):
    pipeline, _ = comfyui_flux_pipeline
    transformer = pipeline.get_module("transformer")

    unloaded = [name for name, p in transformer.named_parameters() if p.is_meta]
    assert not unloaded, f"parameters never received checkpoint weights: {unloaded[:5]}"


def test_fused_qkv_is_split_into_separate_projections(comfyui_flux_pipeline):
    pipeline, state_dict = comfyui_flux_pipeline
    params = dict(pipeline.get_module("transformer").named_parameters())
    device = next(iter(params.values())).device
    hidden = HEADS * HEAD_DIM

    fused = state_dict["double_blocks.0.img_attn.qkv.weight"].to(device, torch.bfloat16)
    for offset, proj in enumerate(("to_q", "to_k", "to_v")):
        expected = fused[offset * hidden : (offset + 1) * hidden]
        actual = params[f"transformer_blocks.0.attn.{proj}.weight"]
        assert torch.equal(actual, expected), f"{proj} does not match the fused slice"


def test_adaln_modulation_scale_and_shift_are_swapped(comfyui_flux_pipeline):
    pipeline, state_dict = comfyui_flux_pipeline
    params = dict(pipeline.get_module("transformer").named_parameters())
    actual = params["norm_out.linear.weight"]

    source = state_dict["final_layer.adaLN_modulation.1.weight"].to(
        actual.device, torch.bfloat16
    )
    half = source.shape[0] // 2
    # ComfyUI emits [shift, scale]; AdaLayerNormContinuous expects [scale, shift].
    assert torch.equal(actual, torch.cat([source[half:], source[:half]], dim=0))
