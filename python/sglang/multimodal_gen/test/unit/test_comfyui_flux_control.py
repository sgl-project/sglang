# SPDX-License-Identifier: Apache-2.0
"""Flux ControlNet residuals: DiT hook, worker kwargs, ComfyUI numerical parity."""

import os
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.flux import FluxConfig
from sglang.multimodal_gen.configs.pipeline_configs.flux import FluxPipelineConfig
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints import spec
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context

HIDDEN, HEADS, DOUBLE, SINGLE, POOLED, TEXT_DIM, IN_CHANNELS = 64, 2, 3, 4, 16, 24, 16
AXES = (4, 14, 14)
BATCH, GRID_H, GRID_W, TEXT_LEN = 2, 4, 6, 5
SEQ = GRID_H * GRID_W


@pytest.fixture
def cpu_timestep_embedding(monkeypatch):
    # On a CUDA machine the timestep kernel is chosen and rejects CPU tensors.
    from sglang.multimodal_gen.runtime.layers import visual_embedding

    monkeypatch.setattr(visual_embedding, "_is_cuda", False)


def _tiny_config() -> FluxConfig:
    config = FluxConfig()
    arch = config.arch_config
    arch.in_channels = arch.out_channels = IN_CHANNELS
    arch.num_layers, arch.num_single_layers = DOUBLE, SINGLE
    arch.attention_head_dim, arch.num_attention_heads = HIDDEN // HEADS, HEADS
    arch.joint_attention_dim, arch.pooled_projection_dim = TEXT_DIM, POOLED
    arch.axes_dims_rope, arch.patch_size = AXES, 1
    return config


def _server_args(config, path):
    return SimpleNamespace(
        pipeline_config=SimpleNamespace(dit_config=config, dit_precision="fp32"),
        model_paths={},
        transformer_weights_path=None,
        component_weights_paths={},
        model_path=str(path),
        dit_precision="fp32",
        component_precisions={},
        quantization=None,
        nunchaku_config=None,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        pin_cpu_memory=False,
        should_start_component_on_cpu=lambda _: True,
        should_use_fsdp_for_component=lambda _: False,
        attention_backend=None,
        attention_backend_config=None,
        kv_gather_degree=1,
        comfyui_mode=True,
    )


def _step_inputs():
    ids = torch.zeros(GRID_H, GRID_W, 3)
    ids[..., 1] = torch.arange(GRID_H)[:, None]
    ids[..., 2] = torch.arange(GRID_W)[None, :]
    return dict(
        img=torch.randn(BATCH, SEQ, IN_CHANNELS),
        txt=torch.randn(BATCH, TEXT_LEN, TEXT_DIM),
        y=torch.randn(BATCH, POOLED),
        img_ids=ids.reshape(1, SEQ, 3).expand(BATCH, -1, -1),
        txt_ids=torch.zeros(BATCH, TEXT_LEN, 3),
        t=torch.tensor([0.7, 0.3]),
        g=torch.tensor([3.5, 3.5]),
    )


def _residuals():
    double = [torch.randn(BATCH, SEQ, HIDDEN) * 0.5 for _ in range(DOUBLE)]
    single = [torch.randn(BATCH, SEQ, HIDDEN) * 0.5 for _ in range(SINGLE)]
    double[1] = None
    single[2] = None
    return double, single


def _run_dit(dit, step, **kwargs):
    freqs = dit.rotary_emb(torch.cat([step["txt_ids"][0], step["img_ids"][0]], 0))
    with (
        torch.no_grad(),
        set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=None),
    ):
        return dit(
            hidden_states=step["img"],
            encoder_hidden_states=step["txt"],
            pooled_projections=step["y"],
            timestep=step["t"] * 1000,
            guidance=step["g"] * 1000,
            freqs_cis=freqs,
            **kwargs,
        )


def _load_dit_from_comfyui_state(monkeypatch, tmp_path, state_dict):
    from safetensors.torch import save_file

    def bfl_name(key):
        if key.endswith(("query_norm.weight", "key_norm.weight")):
            return key[: -len("weight")] + "scale"
        return key

    path = tmp_path / "flux.safetensors"
    save_file({bfl_name(k): v.contiguous() for k, v in state_dict.items()}, str(path))
    monkeypatch.setattr(spec, "get_local_torch_device", lambda: torch.device("cpu"))
    pipeline = SimpleNamespace(
        pipeline_name="FluxPipeline", model_path=str(path), get_module=lambda _: None
    )
    args = _server_args(_tiny_config(), path)
    return spec.load_comfyui_transformer(pipeline, args)["transformer"].eval()


def _build_comfyui_flux():
    import comfy.ops
    from comfy.ldm.flux.model import Flux

    model = Flux(
        image_model="flux",
        in_channels=IN_CHANNELS,
        out_channels=IN_CHANNELS,
        vec_in_dim=POOLED,
        context_in_dim=TEXT_DIM,
        hidden_size=HIDDEN,
        mlp_ratio=4.0,
        num_heads=HEADS,
        depth=DOUBLE,
        depth_single_blocks=SINGLE,
        axes_dim=list(AXES),
        theta=10000,
        patch_size=1,
        qkv_bias=True,
        guidance_embed=True,
        txt_ids_dims=[],
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.05)
    return model.eval()


def test_dit_residual_hook_matches_block_output_hooks(
    single_process_model_parallel, cpu_timestep_embedding
):
    """Residuals added by forward hooks on the blocks are the reference."""
    from sglang.multimodal_gen.runtime.models.dits.flux import FluxTransformer2DModel

    torch.manual_seed(0)
    config = _tiny_config()
    config.arch_config.guidance_embeds = True
    dit = FluxTransformer2DModel(config=config, hf_config={}).eval()
    for param in dit.parameters():
        torch.nn.init.normal_(param, std=0.05)
    step = _step_inputs()
    double, single = _residuals()

    def add_to_image_stream(residual):
        def hook(_module, _inputs, output):
            return output[0], output[1] + residual

        return hook

    base = _run_dit(dit, step)
    handles = [
        block.register_forward_hook(add_to_image_stream(res))
        for blocks, residuals in (
            (dit.transformer_blocks, double),
            (dit.single_transformer_blocks, single),
        )
        for block, res in zip(blocks, residuals)
        if res is not None
    ]
    expected = _run_dit(dit, step)
    for handle in handles:
        handle.remove()
    actual = _run_dit(
        dit,
        step,
        controlnet_block_samples=double,
        controlnet_single_block_samples=single,
    )
    assert (expected - base).abs().max() > 0.1
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(_run_dit(dit, step), base)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_dit_applies_control_residuals_on_cuda(single_process_model_parallel):
    """Residuals on the worker device and compute dtype change the output, and
    leaving them out reproduces the plain forward."""
    from sglang.multimodal_gen.runtime.models.dits.flux import FluxTransformer2DModel

    dtype = torch.bfloat16  # the RoPE kernel has no fp32 path
    torch.manual_seed(0)
    config = _tiny_config()
    config.arch_config.guidance_embeds = True
    dit = FluxTransformer2DModel(config=config, hf_config={}).eval()
    for param in dit.parameters():
        torch.nn.init.normal_(param, std=0.05)
    dit = dit.to("cuda", dtype)

    def to_cuda(value):
        if value is None:
            return None
        return value.to("cuda", dtype) if value.is_floating_point() else value.cuda()

    step = {k: to_cuda(v) for k, v in _step_inputs().items()}
    step["img_ids"], step["txt_ids"] = step["img_ids"].float(), step["txt_ids"].float()
    double, single = ([to_cuda(r) for r in rs] for rs in _residuals())
    base = _run_dit(dit, step)
    controlled = _run_dit(
        dit,
        step,
        controlnet_block_samples=double,
        controlnet_single_block_samples=single,
    )
    assert torch.isfinite(controlled).all()
    assert (controlled - base).abs().max() > 0.05
    torch.testing.assert_close(_run_dit(dit, step), base)


def test_dit_matches_comfyui_with_control(
    monkeypatch, tmp_path, single_process_model_parallel, cpu_timestep_embedding
):
    """Needs a ComfyUI checkout: COMFYUI_PATH=/path/to/ComfyUI."""
    torch.manual_seed(0)
    comfyui_path = os.environ.get("COMFYUI_PATH")
    if not comfyui_path:
        pytest.skip("set COMFYUI_PATH to a ComfyUI checkout")
    monkeypatch.syspath_prepend(comfyui_path)
    pytest.importorskip("comfy.ldm.flux.model")
    comfy_model = _build_comfyui_flux()
    dit = _load_dit_from_comfyui_state(monkeypatch, tmp_path, comfy_model.state_dict())
    step = _step_inputs()
    double, single = _residuals()

    def run_comfy(control):
        with torch.no_grad():
            return comfy_model.forward_orig(
                step["img"],
                step["img_ids"],
                step["txt"],
                step["txt_ids"],
                step["t"],
                step["y"],
                step["g"],
                control=control,
            )

    base_comfy, base_sgl = run_comfy(None), _run_dit(dit, step)
    torch.testing.assert_close(base_sgl, base_comfy, rtol=1e-3, atol=1e-3)
    expected = run_comfy({"input": double, "output": single})
    actual = _run_dit(
        dit,
        step,
        controlnet_block_samples=double,
        controlnet_single_block_samples=single,
    )
    assert (expected - base_comfy).abs().max() > 0.1
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def _control_batch(extra, did_sp_shard_latents=False):
    return SimpleNamespace(
        extra=extra,
        did_sp_shard_latents=did_sp_shard_latents,
        enable_sequence_shard=False,
    )


def test_pipeline_config_casts_control_residuals_to_compute_dtype():
    double = [torch.ones(1, 4, 8, dtype=torch.float16), None]
    single = [torch.ones(1, 4, 8, dtype=torch.float16)]
    kwargs = FluxPipelineConfig().prepare_control_residual_kwargs(
        _control_batch({"comfyui_control": {"input": double, "output": single}}),
        torch.device("cpu"),
        torch.bfloat16,
    )
    assert kwargs["controlnet_block_samples"][0].dtype == torch.bfloat16
    assert kwargs["controlnet_block_samples"][1] is None
    assert kwargs["controlnet_single_block_samples"][0].dtype == torch.bfloat16


def test_pipeline_config_emits_no_control_kwargs_without_control():
    config = FluxPipelineConfig()
    for extra in ({}, {"comfyui_control": None}, {"comfyui_control": {"input": []}}):
        assert (
            config.prepare_control_residual_kwargs(
                _control_batch(extra), torch.device("cpu"), torch.float32
            )
            == {}
        )


def test_pipeline_config_shards_control_residuals_like_latents(monkeypatch):
    from sglang.multimodal_gen.configs.pipeline_configs import base

    monkeypatch.setattr(base, "get_sp_world_size", lambda: 2)
    monkeypatch.setattr(base, "get_sp_parallel_rank", lambda: 1)
    residual = torch.arange(5.0).reshape(1, 5, 1).expand(1, 5, 4).contiguous()
    kwargs = FluxPipelineConfig().prepare_control_residual_kwargs(
        _control_batch(
            {"comfyui_control": {"input": [residual]}}, did_sp_shard_latents=True
        ),
        torch.device("cpu"),
        torch.float32,
    )
    local = kwargs["controlnet_block_samples"][0]
    assert local.shape == (1, 3, 4)
    assert local[0, :2, 0].tolist() == [3.0, 4.0]
    assert local[0, 2].abs().sum() == 0


def test_shared_control_residuals_cross_the_request_once():
    """ComfyUI reuses one residual object for several blocks; sending a copy per
    block made the per-step request scale with block count (about 1 s per step
    for Flux + InstantX on an H100)."""
    import pickle

    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux import (
        _pack_control,
    )

    shared = [torch.randn(1, 64, 32) for _ in range(2)]
    control = {"input": [shared[0]] * 19, "output": [shared[1]] * 38}
    packed = _pack_control(control, batch_size=1, seq_len=64)
    one_each = len(pickle.dumps(_pack_control({"input": shared}, 1, 64)))
    sent = pickle.loads(pickle.dumps(packed))
    assert len(pickle.dumps(packed)) < 1.5 * one_each
    kwargs = FluxPipelineConfig().prepare_control_residual_kwargs(
        _control_batch({"comfyui_control": sent}),
        torch.device("cpu"),
        torch.bfloat16,
    )
    double = kwargs["controlnet_block_samples"]
    assert len(double) == 19 and all(r is double[0] for r in double)
    torch.testing.assert_close(double[0], shared[0].to(torch.bfloat16))


def test_shared_control_residuals_stay_shared_after_cfg_row_split():
    """With batch > 1 (CFG, batch_size) the executor slices kwargs per row
    before packing; the shared residual must still be one tensor per row."""
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
        _slice_batch_row,
    )
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux import (
        _pack_control,
    )

    shared = torch.randn(2, 64, 32)
    own = torch.randn(2, 64, 32)
    control = {"input": [shared] * 19, "output": [shared] * 37 + [own]}
    for row in range(2):
        packed = _pack_control(_slice_batch_row(control, row, 2), 1, 64)
        tensors = packed["input"] + packed["output"]
        assert len({id(t) for t in tensors}) == 2
        assert all(t is tensors[0] for t in tensors[:56])
        torch.testing.assert_close(tensors[0], shared[row : row + 1])
        torch.testing.assert_close(tensors[-1], own[row : row + 1])
