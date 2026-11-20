from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.weights import hub_file_model
from sglang.multimodal_gen.runtime.weights.hub_file_model import (
    apply_hub_file_model_path,
    find_registered_base_model,
    parse_hub_file_model_path,
)

H3 = "MiniMaxAI/MiniMax-H3"
H3_IMAGE = "MiniMaxAI/MiniMax-H3-Image"
FASTH3 = "FastVideo/FastVideo-FastH3-8-Step-V2"


class FakeApi:
    """Model cards keyed by repo id; ``calls`` records every lookup."""

    def __init__(self, cards, tensors):
        self.cards = cards
        self.tensors = tensors
        self.calls = []

    def parse_safetensors_file_metadata(self, repo_id, filename, revision=None):
        tensors = self.tensors.get(f"{repo_id}/{filename}", ["blocks.0.attn.weight"])
        return SimpleNamespace(tensors=dict.fromkeys(tensors))

    def model_info(self, repo_id, revision=None):
        self.calls.append(repo_id)
        base_model = self.cards.get(repo_id)
        card_data = {"base_model": base_model} if base_model is not None else {}
        return SimpleNamespace(card_data=card_data)


@pytest.fixture
def hub(monkeypatch):
    """Fake Hub: model cards and safetensors headers, no network."""
    state = SimpleNamespace(cards={}, tensors={}, api=None)

    def make_api():
        state.api = FakeApi(state.cards, state.tensors)
        return state.api

    monkeypatch.setattr(hub_file_model, "HfApi", make_api)
    find_registered_base_model.cache_clear()
    yield state
    find_registered_base_model.cache_clear()


@pytest.mark.parametrize(
    "model_path",
    [
        H3,  # a repo, not a file
        "org/repo/transformer",  # a subfolder
        "model.safetensors",  # not owner/repo/file
        "/abs/path/model.safetensors",  # local
        None,
    ],
)
def test_only_hub_weight_files_are_parsed(model_path):
    assert parse_hub_file_model_path(model_path) is None


def test_hub_file_and_url_are_parsed():
    source = parse_hub_file_model_path("org/repo/diffusion_models/x.safetensors")
    assert (source.repo_id, source.filename) == (
        "org/repo",
        "diffusion_models/x.safetensors",
    )
    url = "https://huggingface.co/org/repo/blob/abc123/x-Q4_K.gguf"
    source = parse_hub_file_model_path(url)
    assert (source.repo_id, source.filename, source.revision) == (
        "org/repo",
        "x-Q4_K.gguf",
        "abc123",
    )


def test_base_model_follows_card_links(hub):
    hub.cards.update(
        {
            "user/remix": "Comfy-Org/MiniMax-H3",  # a ComfyUI repack, itself a derivative
            "Comfy-Org/MiniMax-H3": H3,
        }
    )
    assert find_registered_base_model("user/remix") == H3
    assert hub.api.calls == ["user/remix", "Comfy-Org/MiniMax-H3"]
    hub.api = None
    assert find_registered_base_model("user/remix") == H3  # cached
    assert hub.api is None


def test_merge_with_one_supported_base(hub):
    hub.cards["user/merge"] = ["Comfy-Org/MiniMax-H3", H3, "other/lora"]
    assert find_registered_base_model("user/merge") == H3
    assert hub.api.calls == ["user/merge"]


def test_file_name_picks_among_linked_bases(hub):
    hub.cards["user/loras"] = [H3, H3_IMAGE]
    assert (
        find_registered_base_model(
            "user/loras", filename="minimax_h3_style.safetensors"
        )
        == H3
    )


def test_card_base_narrows_to_registered_variant(hub):
    # FastVideo's ComfyUI repack names MiniMax-H3, but holds FastH3 weights
    hub.cards["FastVideo/FastVideo-FastH3-Comfy"] = H3
    assert find_registered_base_model("FastVideo/FastVideo-FastH3-Comfy") == FASTH3


def test_repo_without_card_link_uses_name_detectors(hub):
    assert find_registered_base_model("lightx2v/Minimax-h3-Turbo") == H3


@pytest.mark.parametrize(
    "repo_id, cards, match",
    [
        ("user/mix", {"user/mix": [H3, FASTH3]}, "several supported base models"),
        ("user/thing", {"user/thing": "other/model"}, "Could not find the base model"),
    ],
)
def test_ambiguous_or_unknown_base_is_an_error(hub, repo_id, cards, match):
    hub.cards.update(cards)
    with pytest.raises(ValueError, match=match):
        find_registered_base_model(repo_id)


def test_transformer_file_becomes_weight_override(hub):
    hub.cards["Comfy-Org/MiniMax-H3"] = H3
    weight = "Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors"
    kwargs, explicit = {"model_path": weight}, {"model_path"}

    note = apply_hub_file_model_path(kwargs, explicit)

    assert note == (
        f"Serving {weight} on its base model, as --model-path {H3} "
        f"--component-weights-paths.transformer {weight} --model-variant ref2va"
    )
    assert kwargs == {
        "model_path": H3,
        "component_weights_paths": {"transformer": weight},
        "model_variant": "ref2va",
        "served_model_name": weight,
    }
    assert {"component_weights_paths", "model_variant"} <= explicit


def test_lora_file_becomes_lora_path(hub):
    hub.cards["fal/MiniMax-H3-Realism-People-LoRA"] = H3
    weight = "fal/MiniMax-H3-Realism-People-LoRA/realism.safetensors"
    hub.tensors[weight] = ["diffusion_model.blocks.0.attn.qkv.lora_A.weight"]
    kwargs = {"model_path": weight, "served_model_name": "realism"}

    apply_hub_file_model_path(kwargs, set())

    assert kwargs == {
        "model_path": H3,
        "lora_path": weight,
        "served_model_name": "realism",
    }


def test_gguf_and_explicit_variant_are_kept(hub):
    hub.cards["unsloth/MiniMax-H3-GGUF"] = H3
    weight = "unsloth/MiniMax-H3-GGUF/minimax_h3_fl2va_pruned-Q4_K.gguf"
    kwargs = {"model_path": weight, "model_variant": "hybrid"}

    apply_hub_file_model_path(kwargs, set())

    assert kwargs["component_weights_paths"] == {"transformer": weight}
    assert kwargs["model_variant"] == "hybrid"


def test_single_partition_base_gets_no_variant(hub):
    hub.cards["FastVideo/FastVideo-FastH3-Comfy"] = FASTH3
    weight = "FastVideo/FastVideo-FastH3-Comfy/fasth3_fl2va_bf16.safetensors"
    kwargs = {"model_path": weight}

    apply_hub_file_model_path(kwargs, set())

    assert kwargs["model_path"] == FASTH3
    assert "model_variant" not in kwargs


@pytest.mark.parametrize(
    "extra, tensors",
    [
        ({"component_weights_paths": {"transformer": "a/b/c.safetensors"}}, None),
        ({"lora_path": "a/b"}, ["blocks.0.lora_down.weight"]),
    ],
)
def test_conflicting_overrides_are_rejected(hub, extra, tensors):
    hub.cards["user/ckpt"] = H3
    weight = "user/ckpt/x.safetensors"
    if tensors:
        hub.tensors[weight] = tensors
    with pytest.raises(ValueError, match="pass the base model as --model-path"):
        apply_hub_file_model_path({"model_path": weight, **extra}, set())


@pytest.mark.parametrize(
    "filename, tensors, kind",
    [
        (
            "text_encoders/qwen3vl.safetensors",
            ["model.layers.0.mlp.up_proj.weight", "visual.merger.weight"],
            "text_encoder",
        ),
        (
            "vae/video_vae.safetensors",
            ["encoder.conv_in.weight", "decoder.conv_out.weight"],
            "vae",
        ),
        ("embeddings/bullet_time.safetensors", ["qwen3vl_32b"], "standalone tensors"),
    ],
)
def test_non_model_files_fail_before_download(hub, filename, tensors, kind):
    hub.cards["Comfy-Org/MiniMax-H3"] = H3
    weight = f"Comfy-Org/MiniMax-H3/{filename}"
    hub.tensors[weight] = tensors
    with pytest.raises(ValueError, match=f"holds {kind}"):
        apply_hub_file_model_path({"model_path": weight}, set())


def test_other_model_paths_are_untouched(hub):
    kwargs = {"model_path": H3, "model_variant": "fl2va"}
    assert apply_hub_file_model_path(kwargs, set()) is None
    assert kwargs == {"model_path": H3, "model_variant": "fl2va"}
    assert hub.api is None


def test_cli_routes_hub_file_to_diffusion(hub):
    from sglang.cli.utils import get_is_diffusion_model

    hub.cards["Comfy-Org/MiniMax-H3"] = H3
    assert get_is_diffusion_model(
        "Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_bf16.safetensors"
    )
