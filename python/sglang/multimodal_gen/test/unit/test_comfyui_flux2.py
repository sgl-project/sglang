# SPDX-License-Identifier: Apache-2.0
"""ComfyUI integrated mode for FLUX.2 / FLUX.2 Klein: checkpoint spec and loader."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux2 import (
    Flux2Adapter,
)
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints import spec
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.flux2 import (
    FLUX2_KLEIN_BASE_PIPELINE,
    FLUX2_KLEIN_PIPELINE,
    FLUX2_PIPELINE,
    flux2_pipeline_name,
    infer_flux2_geometry,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args

HIDDEN, LAYERS, SINGLES, LATENT, JOINT = 256, 2, 2, 16, 64
MLP_HIDDEN = 3 * HIDDEN


def _bfl_state_dict(*, guidance: bool, norm_suffix: str = "weight", joint=JOINT):
    """Tensor names and shapes of a BFL-layout FLUX.2 file (ComfyUI loads these)."""
    shapes = {
        "img_in.weight": (HIDDEN, LATENT),
        "txt_in.weight": (HIDDEN, joint),
        "time_in.in_layer.weight": (HIDDEN, 256),
        "time_in.out_layer.weight": (HIDDEN, HIDDEN),
        "double_stream_modulation_img.lin.weight": (6 * HIDDEN, HIDDEN),
        "double_stream_modulation_txt.lin.weight": (6 * HIDDEN, HIDDEN),
        "single_stream_modulation.lin.weight": (3 * HIDDEN, HIDDEN),
        "final_layer.adaLN_modulation.1.weight": (2 * HIDDEN, HIDDEN),
        "final_layer.linear.weight": (LATENT, HIDDEN),
    }
    if guidance:
        shapes["guidance_in.in_layer.weight"] = (HIDDEN, 256)
        shapes["guidance_in.out_layer.weight"] = (HIDDEN, HIDDEN)
    for i in range(LAYERS):
        block = f"double_blocks.{i}."
        for stream in ("img", "txt"):
            shapes[f"{block}{stream}_attn.qkv.weight"] = (3 * HIDDEN, HIDDEN)
            shapes[f"{block}{stream}_attn.proj.weight"] = (HIDDEN, HIDDEN)
            for norm in ("query_norm", "key_norm"):
                shapes[f"{block}{stream}_attn.norm.{norm}.{norm_suffix}"] = (128,)
            shapes[f"{block}{stream}_mlp.0.weight"] = (2 * MLP_HIDDEN, HIDDEN)
            shapes[f"{block}{stream}_mlp.2.weight"] = (HIDDEN, MLP_HIDDEN)
    for i in range(SINGLES):
        block = f"single_blocks.{i}."
        shapes[block + "linear1.weight"] = (3 * HIDDEN + 2 * MLP_HIDDEN, HIDDEN)
        shapes[block + "linear2.weight"] = (HIDDEN, HIDDEN + MLP_HIDDEN)
        for norm in ("query_norm", "key_norm"):
            shapes[f"{block}norm.{norm}.{norm_suffix}"] = (128,)
    return {name: torch.randn(*shape) for name, shape in shapes.items()}


def _server_args(model_path):
    # Only the fields the Flux2 DiT constructor and the loader read.
    return SimpleNamespace(
        pipeline_config=SimpleNamespace(dit_config=None, dit_precision="fp32"),
        model_paths={},
        transformer_weights_path=None,
        component_weights_paths={},
        model_path=str(model_path),
        dit_precision="fp32",
        component_precisions={},
        quantization=None,
        nunchaku_config=None,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        pin_cpu_memory=False,
        attention_backend=None,
        attention_backend_config=None,
        comfyui_mode=True,
        disable_autocast=True,
        boundary_ratio=None,
        kv_gather_degree=1,
        should_start_component_on_cpu=lambda _: False,
        should_use_fsdp_for_component=lambda _: False,
    )


@pytest.fixture
def cpu_loader(single_process_model_parallel, monkeypatch):
    monkeypatch.setattr(spec, "get_local_torch_device", lambda: torch.device("cpu"))


def _load(tmp_path, tensors, pipeline_name):
    path = tmp_path / "flux2.safetensors"
    save_file(tensors, path)
    args = _server_args(path)
    set_global_server_args(args)
    pipeline = SimpleNamespace(
        pipeline_name=pipeline_name, model_path=str(path), get_module=lambda _: None
    )
    return args, spec.load_comfyui_transformer(pipeline, args)["transformer"]


@pytest.mark.parametrize("norm_suffix", ["scale", "weight"])
@pytest.mark.parametrize("guidance", [True, False])
def test_bfl_checkpoint_loads_strictly_and_keeps_tensor_values(
    cpu_loader, tmp_path, guidance, norm_suffix
):
    """A mapping gap would fail the strict load or leave a parameter at random init."""
    tensors = _bfl_state_dict(guidance=guidance, norm_suffix=norm_suffix)
    _, model = _load(
        tmp_path,
        tensors,
        FLUX2_KLEIN_BASE_PIPELINE if guidance else FLUX2_KLEIN_PIPELINE,
    )
    params = model.state_dict()

    # Fused BFL qkv is stored [q; k; v]; the model keeps them separate.
    qkv = tensors["double_blocks.0.txt_attn.qkv.weight"].chunk(3, dim=0)
    for part, name in zip(qkv, ("add_q_proj", "add_k_proj", "add_v_proj")):
        assert torch.equal(params[f"transformer_blocks.0.attn.{name}.weight"], part)
    # Single blocks keep the fused linear1 as is.
    assert torch.equal(
        params["single_transformer_blocks.1.attn.to_qkv_mlp_proj.weight"],
        tensors["single_blocks.1.linear1.weight"],
    )
    # BFL stores the final modulation as [scale, shift]; the DiT reads [shift, scale].
    final = tensors["final_layer.adaLN_modulation.1.weight"]
    assert torch.equal(
        params["norm_out.linear.weight"],
        torch.cat([final[HIDDEN:], final[:HIDDEN]]),
    )
    assert ("time_guidance_embed.guidance_embedder.linear_1.weight" in params) == (
        guidance
    )
    assert all(not p.requires_grad for p in model.parameters())


def test_family_is_chosen_from_tensor_shapes():
    def shapes(**kw):
        return {n: tuple(t.shape) for n, t in _bfl_state_dict(**kw).items()}

    dev = infer_flux2_geometry(shapes(guidance=True, joint=15360))
    klein_base = infer_flux2_geometry(shapes(guidance=True, joint=7680))
    klein = infer_flux2_geometry(shapes(guidance=False, joint=12288))
    assert flux2_pipeline_name(dev) == FLUX2_PIPELINE
    assert flux2_pipeline_name(klein_base) == FLUX2_KLEIN_BASE_PIPELINE
    assert flux2_pipeline_name(klein) == FLUX2_KLEIN_PIPELINE
    assert (dev["hidden"], dev["num_layers"], dev["num_single_layers"]) == (
        HIDDEN,
        LAYERS,
        SINGLES,
    )
    assert dev["mlp_ratio"] == 3.0


def test_unsupported_checkpoints_are_rejected_before_loading():
    base = {n: tuple(t.shape) for n, t in _bfl_state_dict(guidance=True).items()}
    with pytest.raises(ValueError, match="Quantized"):
        infer_flux2_geometry({**base, "double_blocks.0.img_attn.qkv.comfy_quant": (1,)})
    with pytest.raises(ValueError, match="BFL-layout"):
        infer_flux2_geometry(
            {"transformer_blocks.0.attn.to_q.weight": (HIDDEN, HIDDEN)}
        )


@pytest.mark.skipif(
    "COMFYUI_PATH" not in os.environ, reason="needs a ComfyUI checkout (COMFYUI_PATH)"
)
@pytest.mark.parametrize("guidance", [True, False])
def test_adapter_and_dit_match_comfyui_with_batch_and_reference_images(
    cpu_loader, tmp_path, guidance
):
    """Same weights, stacked cond/uncond rows and two reference images.

    Covers the id layout, the 0..1 to 0..1000 timestep scaling, guidance and the
    reference-token slicing, none of which a shape check would catch.
    """
    sys.path.insert(0, os.environ["COMFYUI_PATH"])
    import comfy.ops
    from comfy.ldm.flux.model import Flux

    from sglang.multimodal_gen.configs.pipeline_configs.flux import _prepare_text_ids
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )

    torch.manual_seed(0)
    comfy_model = Flux(
        image_model="flux2",
        final_layer=True,
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
        in_channels=LATENT,
        out_channels=LATENT,
        vec_in_dim=None,
        context_in_dim=JOINT,
        hidden_size=HIDDEN,
        mlp_ratio=3.0,
        num_heads=HIDDEN // 128,
        depth=LAYERS,
        depth_single_blocks=SINGLES,
        axes_dim=[32, 32, 32, 32],
        theta=2000,
        patch_size=1,
        qkv_bias=False,
        guidance_embed=guidance,
        txt_ids_dims=[3],
        global_modulation=True,
        mlp_silu_act=True,
        ops_bias=False,
        default_ref_method="index",
        ref_index_scale=10.0,
    ).eval()
    for param in comfy_model.parameters():
        torch.nn.init.normal_(param, std=0.2)
    _, model = _load(
        tmp_path,
        {k: v.contiguous() for k, v in comfy_model.state_dict().items()},
        FLUX2_KLEIN_BASE_PIPELINE if guidance else FLUX2_KLEIN_PIPELINE,
    )

    batch, height, width, text_len = 2, 4, 6, 8
    x = torch.randn(batch, LATENT, height, width)
    context = torch.randn(batch, text_len, JOINT)
    sigma = torch.full((batch,), 0.7)
    guide = torch.tensor([3.0]) if guidance else None
    refs = [
        torch.randn(1, LATENT, 4, 4).expand(batch, -1, -1, -1).contiguous(),
        torch.randn(1, LATENT, 2, 6).expand(batch, -1, -1, -1).contiguous(),
    ]
    with torch.no_grad():
        expected = comfy_model(x, sigma, context, guidance=guide, ref_latents=refs)

    adapter = Flux2Adapter()
    packed = adapter.pack(x, sigma, context, guidance=guide, ref_latents=refs)
    extra = packed.extra_req
    ids = torch.cat(
        [extra["latent_ids"][0], extra["condition_image_latent_ids"][0]], dim=0
    )
    img_cos, img_sin = model.rotary_emb.forward(ids)
    txt_cos, txt_sin = model.rotary_emb.forward(_prepare_text_ids(context)[0])
    freqs = (torch.cat([txt_cos, img_cos]), torch.cat([txt_sin, img_sin]))
    with (
        torch.no_grad(),
        set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=None),
    ):
        out = model(
            torch.cat([packed.latents, extra["image_latent"]], dim=1),
            context,
            packed.timesteps,
            torch.full((batch,), packed.guidance_scale) if guidance else None,
            freqs,
        )
    actual = adapter.unpack(out[:, : packed.latents.shape[1]], packed, x)

    scale = expected.abs().max().item()
    assert (expected - actual).abs().max().item() < 1e-5 * scale + 1e-4
