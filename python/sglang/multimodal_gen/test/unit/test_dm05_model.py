# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch
from accelerate import init_empty_weights
from safetensors.torch import save_file
from transformers import Gemma3Config, Gemma3TextConfig, SiglipVisionConfig
from transformers.cache_utils import DynamicCache

from sglang.multimodal_gen.runtime.models.vlas.dm05_core import (
    DM05CoreModel,
    linear_fp32,
    time_embedding,
)
from sglang.multimodal_gen.runtime.models.vlas.dm05_policy import (
    DM05PolicyModel,
    load_dm05_weights,
)
from sglang.multimodal_gen.runtime.vla.observation import VLAObservationBatch
from sglang.multimodal_gen.runtime.vla.prefix_cache import (
    PrefixContext,
    VLADensePrefixCache,
)


def tiny_dm05_config():
    language = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        query_pre_attn_scalar=8,
        max_position_embeddings=128,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "linear",
                "rope_theta": 1000000.0,
                "factor": 8.0,
            },
        },
    )
    expert = language.to_dict()
    expert.update(hidden_size=12, intermediate_size=24)
    vision = SiglipVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        image_size=16,
        patch_size=8,
        vision_use_head=False,
    )
    vlm = Gemma3Config(
        text_config=language.to_dict(),
        vision_config=vision.to_dict(),
        mm_tokens_per_image=4,
        image_token_id=31,
        boi_token_id=29,
        eoi_token_id=30,
    )
    return {
        "model_type": "dm05",
        "action_dim": 4,
        "chunk_size": 3,
        "vlm_config": vlm.to_dict(),
        "action_config": expert,
    }


def make_core(payload=None, *, dtype=torch.float32, device="cpu"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        core = DM05CoreModel(payload or tiny_dm05_config())
    return core.to(device=device, dtype=dtype).eval()


def make_policy(core=None, *, device="cpu", graph=False, horizon=3):
    config = SimpleNamespace(
        action_horizon=horizon,
        action_dim=4,
        default_num_inference_steps=2,
        precision_policy="bf16_mixed",
        enable_action_cuda_graph=graph,
        action_cuda_graph_max_entries=2,
        llm_attn_implementation="eager",
        vision_attn_implementation="sdpa",
        action_attn_implementation="eager",
    )
    return DM05PolicyModel(config, core or make_core(), device=device)


def make_observation(*, action_mask=None):
    tokens = torch.tensor([[2, 3, 4, 5]])
    return VLAObservationBatch(
        prompt=["pick"],
        images={},
        image_masks={},
        state=None,
        noise=None,
        tokens=tokens,
        token_masks=torch.ones_like(tokens, dtype=torch.bool),
        batch_size=1,
        metadata={
            "camera_order": (),
            "dm05": {
                "pixel_values": None,
                "token_type_ids": torch.zeros_like(tokens),
                "history_pixel_values": None,
                "history_mask": None,
                "action_mask": torch.ones(1, 1, 4)
                if action_mask is None
                else action_mask,
            },
        },
    )


def write_checkpoint(path, model, payload, *, sharded=False):
    path.mkdir(exist_ok=True)
    (path / "config.json").write_text(json.dumps(payload))
    tensors = {"model." + k: v.detach().clone() for k, v in model.state_dict().items()}
    if not sharded:
        save_file(tensors, path / "model.safetensors")
    else:
        names = list(tensors)
        groups = (names[::2], names[1::2])
        weight_map = {}
        for index, group in enumerate(groups):
            filename = f"model-{index + 1:05d}-of-00002.safetensors"
            save_file({name: tensors[name] for name in group}, path / filename)
            weight_map.update({name: filename for name in group})
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
    return tensors


def test_dm05_rejects_incompatible_prefix_expert_kv_layout():
    payload = tiny_dm05_config()
    payload["action_config"]["num_key_value_heads"] = 2
    with pytest.raises(ValueError, match="num_key_value_heads"):
        DM05CoreModel(payload)


def test_dm05_time_embedding_keeps_fp32_construction():
    time = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)
    default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        embedded = time_embedding(time, 8)
    finally:
        torch.set_default_dtype(default)
    assert embedded.dtype == torch.float32
    torch.testing.assert_close(
        embedded[0], torch.tensor([0.0] * 4 + [1.0] * 4), rtol=0, atol=0
    )


def test_dm05_action_projections_compute_fp32_outside_amp():
    layer = torch.nn.Linear(4, 3).bfloat16()
    values = torch.tensor([[0.13, -0.28, 0.63, 0.41]], dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(
        values.float(), layer.weight.float(), layer.bias.float()
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = linear_fp32(values, layer)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dm05_prefix_reproduces_hf_sliding_cache_and_history_pad_positions():
    core = make_core()
    tokens = torch.tensor([[2, 3, 7, 4, 5, 6, 8, 9, 10, 11, 12, 0]])
    attention = tokens != 0
    context = core.encode_prefix(
        tokens, attention, pixel_values=None, token_type_ids=torch.zeros_like(tokens)
    )
    cache = DynamicCache(config=core.vlm.model.language_model.config)
    mask = attention.masked_fill(tokens == 7, 0)
    embeds = core.vlm.model.get_input_embeddings()(tokens)
    embeds = embeds.masked_fill((tokens == 7).unsqueeze(-1), 0)
    core.vlm.model(
        input_ids=None,
        inputs_embeds=embeds,
        attention_mask=mask,
        position_ids=(mask.cumsum(-1) - 1).clamp_min(0),
        token_type_ids=torch.zeros_like(tokens),
        past_key_values=cache,
        use_cache=True,
    )
    assert context.prefix_len == 12
    assert context.past_key_values[0][0].shape[-2] == 7
    assert context.past_key_values[1][0].shape[-2] == 12
    assert context.prefix_pad_masks.tolist() == [
        [True, True, False, True, True, True, True, True, True, True, True, False]
    ]
    for actual, expected in zip(context.past_key_values, cache.layers, strict=True):
        torch.testing.assert_close(actual[0], expected.keys, rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected.values, rtol=0, atol=0)
    output = core.velocity(context, torch.randn(1, 3, 4), torch.ones(1))
    assert output.shape == (1, 3, 4) and torch.isfinite(output).all()


@pytest.mark.parametrize("sharded", [False, True])
def test_dm05_strict_loader_materializes_meta_parameters_and_tied_weights(
    tmp_path, sharded
):
    payload = tiny_dm05_config()
    reference = make_core(payload)
    tensors = write_checkpoint(tmp_path, reference, payload, sharded=sharded)
    with init_empty_weights(include_buffers=False):
        target = DM05CoreModel(payload)
    target.vlm.tie_weights()
    manifest = load_dm05_weights(
        target, str(tmp_path), device=torch.device("cpu"), dtype=torch.float32
    )
    assert set(manifest.loaded_keys) == set(tensors)
    assert len(manifest.safetensor_files) == (2 if sharded else 1)
    assert not any(p.is_meta for p in target.parameters())
    assert (
        target.vlm.lm_head.weight is target.vlm.model.language_model.embed_tokens.weight
    )
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
    x = torch.tensor([[2, 3, 4, 5]])
    expected = reference.encode_prefix(
        x, torch.ones_like(x), pixel_values=None, token_type_ids=torch.zeros_like(x)
    )
    actual = target.encode_prefix(
        x, torch.ones_like(x), pixel_values=None, token_type_ids=torch.zeros_like(x)
    )
    torch.testing.assert_close(
        actual.past_key_values[1][0], expected.past_key_values[1][0], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "corruption, message",
    [
        ("missing", "Missing DM05"),
        ("unexpected", "Unexpected DM05"),
        ("shape", "shape mismatch"),
        ("tied", "Conflicting tied"),
    ],
)
def test_dm05_loader_rejects_incomplete_or_incompatible_checkpoints(
    tmp_path, corruption, message
):
    payload = tiny_dm05_config()
    reference = make_core(payload)
    tensors = write_checkpoint(tmp_path, reference, payload)
    name = "model.action_in_proj.weight"
    if corruption == "missing":
        tensors.pop(name)
    elif corruption == "unexpected":
        tensors["model.not_a_dm05_parameter"] = torch.ones(1)
    elif corruption == "shape":
        tensors[name] = tensors[name][:-1]
    else:
        tensors["model.vlm.lm_head.weight"] = tensors["model.vlm.lm_head.weight"] + 1
    save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match=message):
        load_dm05_weights(
            make_core(payload),
            str(tmp_path),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_dm05_loader_maps_the_public_siglip_container_without_dropping_bias(tmp_path):
    payload = tiny_dm05_config()
    reference = make_core(payload)
    tensors = write_checkpoint(tmp_path, reference, payload)
    legacy = {
        name.replace(
            "model.vlm.model.vision_tower.",
            "model.vlm.model.vision_tower.vision_model.",
            1,
        ): value
        for name, value in tensors.items()
    }
    save_file(legacy, tmp_path / "model.safetensors")
    target = make_core(payload)
    manifest = load_dm05_weights(
        target, str(tmp_path), device=torch.device("cpu"), dtype=torch.float32
    )
    bias_name = (
        "model.vlm.model.vision_tower.vision_model.embeddings.patch_embedding.bias"
    )
    assert bias_name in manifest.loaded_keys
    torch.testing.assert_close(
        target.vlm.model.vision_tower.embeddings.patch_embedding.bias,
        legacy[bias_name],
        rtol=0,
        atol=0,
    )


def test_dm05_prefix_cache_identity_covers_token_types_and_history_pixels():
    policy = make_policy()
    observation = make_observation()
    observation.metadata["dm05"].update(
        history_pixel_values=torch.zeros(1, 3, 16, 16),
        history_mask=torch.zeros(1, 4, dtype=torch.bool),
    )
    original = policy.build_prefix_cache_key(observation)
    observation.metadata["dm05"]["action_mask"].zero_()
    assert policy.build_prefix_cache_key(observation) == original
    observation.metadata["dm05"]["token_type_ids"][0, 0] = 1
    assert policy.build_prefix_cache_key(observation) != original
    observation.metadata["dm05"]["token_type_ids"].zero_()
    observation.metadata["dm05"]["history_pixel_values"][0, 0, 0, 0] = 1
    assert policy.build_prefix_cache_key(observation) != original
    observation.metadata["dm05"]["history_pixel_values"].zero_()
    observation.metadata["dm05"]["history_mask"][0, 0] = True
    assert policy.build_prefix_cache_key(observation) != original


def test_dm05_euler_mask_dtype_transition_and_final_unmasked_update(monkeypatch):
    core = make_core(dtype=torch.bfloat16)
    policy = make_policy(core)
    observation = make_observation(action_mask=torch.tensor([[[1, 1, 0, 0]]]))
    prefix = PrefixContext(VLADensePrefixCache(), torch.ones(1, 4, dtype=torch.bool), 4)
    seen = []

    def velocity(context, x, time):
        seen.append((x.clone(), time.clone()))
        return torch.ones_like(x, dtype=torch.float32)

    monkeypatch.setattr(core, "velocity", velocity)
    noise = torch.ones(1, 3, 4, dtype=torch.bfloat16)
    actual = policy.sample_actions(
        observation, prefix, noise=noise, num_steps=2, use_cuda_graph=False
    )
    assert seen[0][0].dtype == torch.bfloat16
    assert seen[1][0].dtype == torch.float32
    assert all(time.dtype == torch.bfloat16 for _, time in seen)
    assert all(not x[..., 2:].any() for x, _ in seen)
    torch.testing.assert_close(actual[..., :2], torch.zeros(1, 3, 2), rtol=0, atol=0)
    torch.testing.assert_close(
        actual[..., 2:], torch.full((1, 3, 2), -0.5), rtol=0, atol=0
    )
    assert noise.eq(1).all()


def test_dm05_explicit_noise_repeatability_and_request_local_generator():
    policy = make_policy()
    observation = make_observation()
    prefix = policy.encode_prefix(observation)
    generator = torch.Generator(device="cpu").manual_seed(31)
    initial_global_state = torch.random.get_rng_state().clone()
    noise = policy.sample_noise(generator=generator)
    assert torch.equal(torch.random.get_rng_state(), initial_global_state)
    first = policy.sample_actions(
        observation, prefix, noise=noise, use_cuda_graph=False
    )
    saved = first.clone()
    second = policy.sample_actions(
        observation, prefix, noise=noise, use_cuda_graph=False
    )
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    policy.sample_actions(observation, prefix, noise=noise + 0.5, use_cuda_graph=False)
    torch.testing.assert_close(first, saved, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DM05 graph replay requires CUDA"
)
def test_dm05_graph_replay_refreshes_kv_masks_and_owns_outputs():
    core = make_core(device="cuda", dtype=torch.bfloat16)
    policy = make_policy(core, device="cuda", graph=True)
    observation = make_observation()
    prefix = policy.encode_prefix(observation)
    noise = torch.ones(1, 3, 4, device="cuda", dtype=torch.bfloat16)
    eager = policy.sample_actions(
        observation, prefix, noise=noise, use_cuda_graph=False
    )
    captured = policy.sample_actions(
        observation, prefix, noise=noise, use_cuda_graph=True
    )
    snapshot = captured.clone()
    torch.testing.assert_close(captured, eager, rtol=1e-5, atol=1e-5)
    observation.tokens[0, 1] = 8
    observation.metadata["dm05"]["action_mask"][..., 2:] = 0
    changed_prefix = policy.encode_prefix(observation)
    changed_eager = policy.sample_actions(
        observation, changed_prefix, noise=noise + 0.25, use_cuda_graph=False
    )
    replayed = policy.sample_actions(
        observation, changed_prefix, noise=noise + 0.25, use_cuda_graph=True
    )
    torch.testing.assert_close(replayed, changed_eager, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(captured, snapshot, rtol=0, atol=0)
    info = policy.action_graph_runner.cache_info()
    assert info.captures == 1 and info.hits >= 1 and info.failures == 0
