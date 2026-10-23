# SPDX-License-Identifier: Apache-2.0
"""ComfyUI single-file checkpoint specs: shared hooks and the FLUX.2 spec."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux2 import (
    Flux2Adapter,
)
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints import spec as spec_mod
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.flux2 import (
    FLUX2_KLEIN_BASE_PIPELINE,
    FLUX2_KLEIN_PIPELINE,
    FLUX2_PIPELINE,
    flux2_pipeline_name,
    infer_flux2_geometry,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args

_MARKER = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}


def _load(
    monkeypatch,
    spec,
    h3_markers=None,
    quantization=None,
    local_device="cpu",
    model_path="/models/dit.safetensors",
):
    """Run load_comfyui_transformer with a dummy spec; return the load kwargs."""
    seen = {}

    def maybe_load_fsdp_model(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(parameters=lambda: [])

    def h3_inspection(paths):
        if h3_markers is None:
            raise AssertionError("only the MiniMax-H3 spec uses the H3 reader")
        return None, h3_markers

    monkeypatch.setattr(spec_mod, "get_comfyui_checkpoint_spec", lambda name: spec)
    monkeypatch.setattr(
        spec_mod.ModelRegistry,
        "resolve_model_cls",
        lambda name: (SimpleNamespace(param_names_mapping={}), None),
    )
    monkeypatch.setattr(spec_mod, "resolve_precision", lambda *a, **k: "bf16")
    monkeypatch.setattr(spec_mod, "maybe_load_fsdp_model", maybe_load_fsdp_model)
    monkeypatch.setattr(spec_mod, "inspect_minimax_h3_safetensors", h3_inspection)
    monkeypatch.setattr(
        spec_mod,
        "resolve_minimax_h3_checkpoint_quantization",
        lambda markers: ("quant-config", dict(markers)),
    )
    monkeypatch.setattr(spec_mod, "get_local_torch_device", lambda: local_device)
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils."
        "resolve_transformer_gguf_to_load",
        lambda server_args, component: None,
    )
    server_args = SimpleNamespace(
        model_paths={},
        quantization=quantization,
        nunchaku_config=None,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        pin_cpu_memory=False,
        should_start_component_on_cpu=lambda component: True,
        should_use_fsdp_for_component=lambda component: False,
    )
    pipeline = SimpleNamespace(
        pipeline_name="DummyPipeline",
        model_path=model_path,
        get_module=lambda name: "scheduler",
    )
    spec_mod.load_comfyui_transformer(pipeline, server_args)
    seen["server_args"] = server_args
    return seen


def _spec(**kwargs):
    return spec_mod.ComfyUICheckpointSpec(
        dit_cls_name="DummyDiT",
        build_dit_config=lambda server_args: SimpleNamespace(
            arch_config=SimpleNamespace(param_names_mapping={})
        ),
        **kwargs,
    )


def _is_dit_key(key: str) -> bool:
    return not key.startswith("text_encoders.")


@pytest.mark.parametrize("markers", [{"blocks.0.attn.to_q": _MARKER}, {}])
def test_spec_quant_markers_and_key_filter(monkeypatch, markers):
    """A quantized file gets the quant config and both key filters; an
    unquantized one keeps only the spec's filter."""
    seen = _load(
        monkeypatch,
        _spec(quant_markers=lambda paths: markers, checkpoint_key_filter=_is_dit_key),
    )
    key_filter = seen["checkpoint_key_filter"]
    assert not key_filter("text_encoders.layer.weight")
    if not markers:
        assert "quant_config" not in seen["init_params"]
        assert key_filter is _is_dit_key
        return
    assert seen["init_params"]["quant_config"] == ("quant-config", markers)
    assert key_filter("blocks.0.attn.to_q.weight")
    assert not key_filter("blocks.0.attn.to_q.comfy_quant")


def test_h3_spec_still_rejects_non_int8_markers(monkeypatch):
    h3 = spec_mod.ComfyUICheckpointSpec(
        dit_cls_name="MiniMaxH3DiTModel", build_dit_config=_spec().build_dit_config
    )
    with pytest.raises(ValueError, match="INT8 ConvRot"):
        _load(monkeypatch, h3, h3_markers={"x": {"format": "nvfp4"}})


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
    monkeypatch.setattr(spec_mod, "get_local_torch_device", lambda: torch.device("cpu"))


def _load_flux2(tmp_path, tensors, pipeline_name):
    path = tmp_path / "flux2.safetensors"
    save_file(tensors, path)
    args = _server_args(path)
    set_global_server_args(args)
    pipeline = SimpleNamespace(
        pipeline_name=pipeline_name, model_path=str(path), get_module=lambda _: None
    )
    return args, spec_mod.load_comfyui_transformer(pipeline, args)["transformer"]


@pytest.mark.parametrize("norm_suffix", ["scale", "weight"])
@pytest.mark.parametrize("guidance", [True, False])
def test_bfl_checkpoint_loads_strictly_and_keeps_tensor_values(
    cpu_loader, tmp_path, guidance, norm_suffix
):
    """A mapping gap would fail the strict load or leave a parameter at random init."""
    tensors = _bfl_state_dict(guidance=guidance, norm_suffix=norm_suffix)
    _, model = _load_flux2(
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
    _, model = _load_flux2(
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


def _write_gguf(path, names):
    import gguf
    import numpy as np

    writer = gguf.GGUFWriter(str(path), "flux")
    for i, name in enumerate(names):
        writer.add_tensor(name, np.full((2, 4), i, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_gguf_unet_loads_with_runtime_names(monkeypatch, tmp_path):
    """A GGUF unet loads as the DiT, keyed by runtime layer names for both quant
    lookup and loading: Z-Image's qkv is to_qkv and its w1/w3 fuse into w13."""
    from dataclasses import replace

    from sglang.multimodal_gen.configs.models.dits.zimage import ZImageDitConfig

    gguf_path = tmp_path / "z-image-turbo-Q4_K_M.gguf"
    _write_gguf(
        gguf_path,
        [
            "layers.0.attention.qkv.weight",
            "layers.0.feed_forward.w1.weight",
            "layers.0.feed_forward.w2.weight",
            "layers.0.feed_forward.w3.weight",
        ],
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils."
        "_validate_gguf_runtime_support",
        lambda *args: None,
    )
    zimage = spec_mod.get_comfyui_checkpoint_spec("ZImagePipeline")
    seen = _load(
        monkeypatch,
        replace(zimage, build_dit_config=lambda server_args: ZImageDitConfig()),
        model_path=str(gguf_path),
    )
    meta = seen["init_params"]["quant_config"].tensor_meta
    weights = dict(seen["weights_iterator"])
    assert spec_mod.is_comfyui_single_file(str(gguf_path))
    assert seen["server_args"].model_paths["transformer"] == str(tmp_path)
    assert meta["layers.0.attention.to_qkv.weight"].logical_shape == (2, 4)
    assert meta["layers.0.feed_forward.w13.weight"].logical_shape == (4, 4)
    assert list(weights) == [
        "layers.0.attention.to_qkv.weight",
        "layers.0.feed_forward.w2.weight",
        "layers.0.feed_forward.w13.weight",
    ]
    assert weights["layers.0.feed_forward.w13.weight"][:, 0].tolist() == [1, 1, 3, 3]


def test_gguf_unet_architecture_is_detected_from_its_header(tmp_path):
    """Detection rejected every GGUF, so only MiniMax-H3 with a BF16 companion
    file could load one."""
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.generator import (
        _load_state_dict_for_detection,
    )

    path = tmp_path / "flux1-dev-Q4_K_M.gguf"
    _write_gguf(path, ["img_in.weight"])
    state_dict = _load_state_dict_for_detection(str(path))
    assert {k: tuple(v.shape) for k, v in state_dict.items()} == {
        "img_in.weight": (2, 4)
    }


def test_flux_guidance_probe_reads_gguf_tensor_names(tmp_path):
    from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints import flux

    dev, schnell = tmp_path / "dev.gguf", tmp_path / "schnell.gguf"
    _write_gguf(dev, ["img_in.weight", "guidance_in.in_layer.weight"])
    _write_gguf(schnell, ["img_in.weight"])
    assert flux._checkpoint_has_guidance_embeds(str(dev))
    assert not flux._checkpoint_has_guidance_embeds(str(schnell))


@pytest.mark.parametrize("quantization", ["fp8", "convrot_int8"])
def test_bf16_file_is_quantized_online(monkeypatch, quantization):
    """A BF16 file with --quantization used to load with no quant config, so the
    flag was silently ignored and the DiT stayed in BF16."""
    hook_calls = []
    quant_spec = SimpleNamespace(
        quant_config="online-config",
        param_dtype=None,
        post_load_hooks=[hook_calls.append],
        is_convrot_int8=quantization == "convrot_int8",
        needs_device_weight_postprocess=True,
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils."
        "resolve_transformer_quant_load_spec",
        lambda **kwargs: quant_spec,
    )
    seen = _load(
        monkeypatch,
        _spec(quant_markers=lambda paths: {}),
        quantization=quantization,
        local_device="gpu",
    )
    assert seen["init_params"]["quant_config"] == "online-config"
    assert seen["param_dtype"] is None
    assert len(hook_calls) == 1
    # Online FP8 quantizes on the GPU; only ConvRot INT8 can start from host weights.
    expected = "cpu" if quantization == "convrot_int8" else "gpu"
    assert str(seen["weight_load_plan"].checkpoint_load_device) == expected


def test_bf16_file_rejects_methods_that_need_serialized_weights(monkeypatch):
    with pytest.raises(ValueError, match="pre-quantized"):
        _load(
            monkeypatch,
            _spec(quant_markers=lambda paths: {}),
            quantization="mxfp8",
        )
