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


def test_quantized_spec_gets_quant_config_and_both_key_filters(monkeypatch):
    seen = _load(
        monkeypatch,
        _spec(
            quant_markers=lambda paths: {"blocks.0.attn.to_q": _MARKER},
            checkpoint_key_filter=_is_dit_key,
        ),
    )
    assert seen["init_params"]["quant_config"] == (
        "quant-config",
        {"blocks.0.attn.to_q": _MARKER},
    )
    key_filter = seen["checkpoint_key_filter"]
    assert key_filter("blocks.0.attn.to_q.weight")
    assert not key_filter("blocks.0.attn.to_q.comfy_quant")
    assert not key_filter("text_encoders.layer.weight")
    assert seen["weight_load_plan"] is not None


def test_unquantized_file_keeps_only_the_spec_key_filter(monkeypatch):
    seen = _load(
        monkeypatch,
        _spec(quant_markers=lambda paths: {}, checkpoint_key_filter=_is_dit_key),
    )
    assert "quant_config" not in seen["init_params"]
    assert seen["checkpoint_key_filter"] is _is_dit_key
    assert seen["weight_load_plan"] is None


def test_spec_without_hooks_loads_as_before(monkeypatch):
    seen = _load(monkeypatch, _spec())
    assert "quant_config" not in seen["init_params"]
    assert seen["checkpoint_key_filter"] is None


def test_h3_spec_still_rejects_non_int8_markers(monkeypatch):
    h3 = spec_mod.ComfyUICheckpointSpec(
        dit_cls_name="MiniMaxH3DiTModel", build_dit_config=_spec().build_dit_config
    )
    with pytest.raises(ValueError, match="INT8 ConvRot"):
        _load(monkeypatch, h3, h3_markers={"x": {"format": "nvfp4"}})
