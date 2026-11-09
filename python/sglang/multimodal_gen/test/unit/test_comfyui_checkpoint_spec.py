# SPDX-License-Identifier: Apache-2.0
"""ComfyUI single-file checkpoint specs: the quant-marker and key-filter hooks."""

from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints import spec as spec_mod

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
