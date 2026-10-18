# SPDX-License-Identifier: Apache-2.0
"""ComfyUI single-file checkpoint specs: the quant-marker and key-filter hooks."""

from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints import spec as spec_mod

_MARKER = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}


def _load(monkeypatch, spec, h3_markers=None):
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
    monkeypatch.setattr(spec_mod, "get_local_torch_device", lambda: "cpu")
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils."
        "resolve_transformer_gguf_to_load",
        lambda server_args, component: None,
    )
    server_args = SimpleNamespace(
        model_paths={},
        quantization=None,
        nunchaku_config=None,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        pin_cpu_memory=False,
        should_start_component_on_cpu=lambda component: True,
        should_use_fsdp_for_component=lambda component: False,
    )
    pipeline = SimpleNamespace(
        pipeline_name="DummyPipeline",
        model_path="/models/dit.safetensors",
        get_module=lambda name: "scheduler",
    )
    spec_mod.load_comfyui_transformer(pipeline, server_args)
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
    for name in names:
        writer.add_tensor(name, np.zeros((2, 4), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_worker_routes_a_gguf_unet_to_the_comfyui_loader(tmp_path):
    for name in ("dit.safetensors", "dit-Q8_0.gguf", "dit.bin"):
        (tmp_path / name).write_bytes(b"x")
    assert spec_mod.is_comfyui_single_file(str(tmp_path / "dit.safetensors"))
    assert spec_mod.is_comfyui_single_file(str(tmp_path / "dit-Q8_0.gguf"))
    assert not spec_mod.is_comfyui_single_file(str(tmp_path / "dit.bin"))
    assert not spec_mod.is_comfyui_single_file(str(tmp_path))


def test_gguf_model_path_loads_through_the_gguf_branch(monkeypatch, tmp_path):
    gguf_path = str(tmp_path / "minimax_h3_fl2va-Q8_0.gguf")
    seen = {}

    def gguf_load(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(parameters=lambda: [])

    def no_safetensors(*args, **kwargs):
        raise AssertionError("a GGUF model path must not be read as safetensors")

    monkeypatch.setattr(spec_mod, "_load_comfyui_gguf_transformer", gguf_load)
    monkeypatch.setattr(spec_mod, "inspect_minimax_h3_safetensors", no_safetensors)
    monkeypatch.setattr(spec_mod, "safetensors_weights_iterator", no_safetensors)
    monkeypatch.setattr(spec_mod, "get_comfyui_checkpoint_spec", lambda name: _spec())
    monkeypatch.setattr(
        spec_mod.ModelRegistry,
        "resolve_model_cls",
        lambda name: (SimpleNamespace(param_names_mapping={}), None),
    )
    monkeypatch.setattr(spec_mod, "resolve_precision", lambda *a, **k: "bf16")
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils."
        "resolve_transformer_gguf_to_load",
        lambda server_args, component: None,
    )
    server_args = SimpleNamespace(model_paths={})
    pipeline = SimpleNamespace(
        pipeline_name="DummyPipeline",
        model_path=gguf_path,
        get_module=lambda name: "scheduler",
    )
    spec_mod.load_comfyui_transformer(pipeline, server_args)
    assert seen["gguf_file"] == gguf_path
    assert server_args.model_paths["transformer"] == str(tmp_path)


def test_flux_guidance_probe_reads_gguf_tensor_names(tmp_path):
    from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints import flux

    dev, schnell = tmp_path / "dev.gguf", tmp_path / "schnell.gguf"
    _write_gguf(dev, ["img_in.weight", "guidance_in.in_layer.weight"])
    _write_gguf(schnell, ["img_in.weight"])
    assert flux._checkpoint_has_guidance_embeds(str(dev))
    assert not flux._checkpoint_has_guidance_embeds(str(schnell))
