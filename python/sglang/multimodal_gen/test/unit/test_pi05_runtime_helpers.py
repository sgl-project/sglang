# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import sglang.multimodal_gen.runtime.models.vlas.pi05_policy as pi05_policy_module
from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig
from sglang.multimodal_gen.runtime.models.vlas.pi05_core import (
    Pi05CoreModel,
    Pi05SiglipVisionModel,
    PiGemmaRMSNorm,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)
from sglang.multimodal_gen.runtime.models.vlas.pi05_policy import (
    Pi05CheckpointManifest,
    Pi05PolicyModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.pi05_preprocess import (
    _preprocess_image,
    _resize_with_pad_image_tensor,
)
from sglang.multimodal_gen.runtime.vla.cuda_graph import (
    VLADenoiseGraphRunner,
    VLADenoiseGraphSignature,
    VLAPrefixGraphRunner,
    VLATensorGraphRunner,
    _BoundedCaptureCache,
    _CapturedDenoiseGraph,
)
from sglang.multimodal_gen.runtime.vla.observation import VLAObservationBatch
from sglang.multimodal_gen.runtime.vla.parallel import VLASplitGroup
from sglang.multimodal_gen.runtime.vla.prefix_cache import (
    PrefixContext,
    VLADensePrefixCache,
)
from sglang.multimodal_gen.runtime.vla.prompt_bucketing import (
    bucket_prompt_tokens,
    effective_token_length,
    select_prompt_token_bucket,
)
from sglang.srt.models.siglip import SiglipVisionModel
from sglang.srt.runtime_context import get_context


def _prefix_context(value: float, digest: str | None) -> PrefixContext:
    keys = torch.full((1, 1, 2, 4), value)
    values = torch.full((1, 1, 2, 4), value)
    return PrefixContext(
        past_key_values=VLADensePrefixCache(((keys, values, None),)),
        prefix_pad_masks=torch.ones(1, 2, dtype=torch.bool),
        prefix_len=2,
        cache_key_digest=digest,
    )


def test_vla_split_group_marks_all_action_ranks():
    split = VLASplitGroup(
        group=SimpleNamespace(world_size=2),
        prefix_root=0,
        action_root=1,
        action_ranks=(0, 1),
        rank=0,
    )

    assert split.is_prefix_rank
    assert split.is_action_rank
    assert split.uses_action_sp


def test_denoise_graph_skips_prefix_copy_for_same_digest(monkeypatch):
    runner = VLADenoiseGraphRunner(enabled=True)
    static_context = _prefix_context(1.0, "same")
    captured = _CapturedDenoiseGraph(
        graph=object(),
        static_prefix_context=static_context,
        static_x_t=torch.empty(1, 2, 4),
        static_timestep=torch.empty(1),
        static_output=torch.empty(1, 2, 4),
        current_context_digest="same",
    )

    def fail_copy(*args, **kwargs):
        raise AssertionError("PrefixContext should not be copied on digest hit")

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.vla.cuda_graph._copy_prefix_context_",
        fail_copy,
    )

    runner._sync_context_if_needed(captured, _prefix_context(2.0, "same"))

    assert captured.static_prefix_context.past_key_values[0][0].eq(1.0).all()


def test_denoise_graph_copies_mutable_prefix_graph_output():
    runner = VLADenoiseGraphRunner(enabled=True)
    static_context = _prefix_context(1.0, None)
    captured = _CapturedDenoiseGraph(
        graph=object(),
        static_prefix_context=static_context,
        static_x_t=torch.empty(1, 2, 4),
        static_timestep=torch.empty(1),
        static_output=torch.empty(1, 2, 4),
    )
    current_context = _prefix_context(2.0, None)
    current_context.layout["mutable_graph_output"] = True
    current_context.prefix_pad_masks[:, -1] = False

    runner._sync_context_if_needed(captured, current_context)

    assert captured.static_prefix_context.past_key_values[0][0].eq(2.0).all()
    assert torch.equal(
        captured.static_prefix_context.prefix_pad_masks,
        current_context.prefix_pad_masks,
    )


def _denoise_signature(prefix_len: int) -> VLADenoiseGraphSignature:
    return VLADenoiseGraphSignature(
        batch_size=1,
        prefix_len=prefix_len,
        prefix_full_attention=False,
        action_horizon=2,
        action_dim=4,
        dtype="float32",
        parallel_layout="single",
    )


def test_denoise_graph_capacity_falls_back_without_capturing_new_signature():
    runner = VLADenoiseGraphRunner(enabled=True, max_entries=1)
    runner._cache.entries[_denoise_signature(32)] = object()
    fake_cuda_tensor = SimpleNamespace(device=SimpleNamespace(type="cuda"))

    result = runner.capture_or_run(
        _denoise_signature(64),
        lambda *_args: "eager",
        _prefix_context(1.0, None),
        fake_cuda_tensor,
        object(),
    )

    assert result == "eager"
    assert list(runner._cache.entries) == [_denoise_signature(32)]


def test_zero_denoise_graph_capacity_disables_runner():
    runner = VLADenoiseGraphRunner(enabled=True, max_entries=0)

    assert not runner.enabled


class _FakeGraph:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


def _fake_capture():
    return SimpleNamespace(graph=_FakeGraph())


def test_graph_cache_evicts_lru_and_releases_graph():
    cache = _BoundedCaptureCache("test", max_entries=2, evict_on_miss=True)
    first = _fake_capture()
    second = _fake_capture()
    third = _fake_capture()
    cache.put("first", first)
    cache.put("second", second)

    assert cache.get("first") is first
    cache.put("third", third)

    assert tuple(cache.entries) == ("first", "third")
    assert second.graph.reset_calls == 1
    assert cache.info().evictions == 1

    cache.clear()
    assert first.graph.reset_calls == 1
    assert third.graph.reset_calls == 1


def test_graph_cache_releases_lru_before_new_capture():
    cache = _BoundedCaptureCache("test", max_entries=1, evict_on_miss=True)
    first = _fake_capture()
    cache.put("first", first)

    cache.prepare_admission("second")

    assert not cache.entries
    assert first.graph.reset_calls == 1
    assert cache.info().evictions == 1


def test_non_evicting_graph_cache_rejects_new_signature_at_capacity():
    cache = _BoundedCaptureCache("test", max_entries=1, evict_on_miss=False)
    first = _fake_capture()
    rejected = _fake_capture()
    cache.put("first", first)

    assert not cache.put("second", rejected)
    assert tuple(cache.entries) == ("first",)
    assert first.graph.reset_calls == 0
    assert rejected.graph.reset_calls == 1


def test_graph_cache_info_tracks_hits_misses_and_failures():
    cache = _BoundedCaptureCache("test", max_entries=1, evict_on_miss=False)
    cache.put("first", _fake_capture())

    assert cache.get("first") is not None
    assert cache.get("missing") is None
    cache.mark_failure()

    info = cache.info()
    assert info.hits == 1
    assert info.misses == 1
    assert info.captures == 1
    assert info.failures == 1


def test_zero_graph_capacity_disables_both_runners():
    assert not VLAPrefixGraphRunner(enabled=True, max_entries=0).enabled
    assert not VLADenoiseGraphRunner(enabled=True, max_entries=0).enabled


def test_prefix_prompt_bucket_preserves_tokens_and_masks():
    tokens = torch.arange(40).view(1, 40)
    token_masks = torch.ones_like(tokens, dtype=torch.bool)

    bucketed_tokens, bucketed_masks, logical_length, bucket = bucket_prompt_tokens(
        tokens,
        token_masks,
        (32, 64, 128, 200),
    )

    assert logical_length == 40
    assert bucket == 64
    assert bucketed_tokens.shape == (1, 64)
    assert torch.equal(bucketed_tokens[:, :40], tokens)
    assert bucketed_tokens[:, 40:].eq(0).all()
    assert bucketed_masks[:, :40].all()
    assert not bucketed_masks[:, 40:].any()


def test_prefix_prompt_bucket_preserves_mask_holes():
    tokens = torch.arange(6).view(1, 6)
    token_masks = torch.tensor([[True, False, True, False, False, False]])

    bucketed_tokens, bucketed_masks, logical_length, bucket = bucket_prompt_tokens(
        tokens,
        token_masks,
        (4, 8),
    )

    assert logical_length == 3
    assert bucket == 4
    assert torch.equal(bucketed_tokens, tokens[:, :4])
    assert torch.equal(bucketed_masks, token_masks[:, :4])


def test_prompt_bucket_selection_and_exact_tail():
    assert select_prompt_token_bucket(33, (32, 64, 128)) == 64
    assert select_prompt_token_bucket(129, (32, 64, 128)) is None

    tokens = torch.arange(140).view(1, 140)
    token_masks = torch.arange(140).view(1, 140) < 129
    exact_tokens, exact_masks, logical_length, bucket = bucket_prompt_tokens(
        tokens,
        token_masks,
        (32, 64, 128),
    )

    assert logical_length == 129
    assert bucket is None
    assert exact_tokens.shape == (1, 129)
    assert exact_masks.all()


def test_effective_token_length_uses_last_visible_position():
    masks = torch.tensor(
        [
            [True, False, False, True, False],
            [True, True, False, False, False],
        ]
    )

    assert effective_token_length(masks) == 4


def test_pi05_graph_config_validation():
    config = Pi05PipelineConfig()
    config.update_pipeline_config(
        {
            "prompt_token_buckets": [16, 48, 96],
            "action_cuda_graph_max_entries": 3,
        }
    )
    assert config.prompt_token_buckets == [16, 48, 96]
    assert config.action_cuda_graph_max_entries == 3

    invalid_configs = (
        ({"prompt_token_buckets": [32, 32]}, "strictly increasing"),
        ({"prompt_token_buckets": [64, 32]}, "strictly increasing"),
        ({"prompt_token_buckets": [0, 32]}, "positive"),
        ({"prompt_token_buckets": [32, 256]}, "max_token_len"),
        ({"prefix_cuda_graph_max_entries": -1}, "prefix_cuda_graph"),
        ({"action_cuda_graph_max_entries": -1}, "action_cuda_graph"),
    )
    for overrides, match in invalid_configs:
        with pytest.raises(ValueError, match=match):
            Pi05PipelineConfig(**overrides)


def test_prefix_cache_key_distinguishes_bucket_layouts_and_mask_holes():
    model = Pi05PolicyModel.__new__(Pi05PolicyModel)
    model.config = Pi05PipelineConfig(prompt_token_buckets=[32, 64])
    model.dtype = torch.bfloat16
    model.model_path = "lerobot/pi05_base"
    model._prompt_token_bucketing_enabled = lambda: True
    common = dict(
        metadata={"camera_order": ("front",)},
        images={"front": torch.zeros(1, 3, 2, 2)},
        image_masks={"front": torch.tensor(True)},
        token_masks=torch.tensor([[True, False, False, True, False]]),
    )
    first = SimpleNamespace(tokens=torch.tensor([[1, 2, 3, 4, 0]]), **common)
    second = SimpleNamespace(tokens=torch.tensor([[1, 2, 3, 9, 0]]), **common)

    exact = model.build_prefix_cache_key(first)
    bucketed = model.build_prefix_cache_key(first, bucket_prompt=True)

    assert exact != bucketed
    assert model.build_prefix_cache_key(first) != model.build_prefix_cache_key(second)


def test_bucket_miss_keeps_action_denoise_eager():
    model = Pi05PolicyModel.__new__(Pi05PolicyModel)
    nn.Module.__init__(model)
    model.action_expert = lambda _context, x_t, _timestep, **_kwargs: x_t + 1
    model.graph_runner = SimpleNamespace(
        capture_or_run=lambda *_args, **_kwargs: pytest.fail(
            "bucket misses must not capture action graphs"
        )
    )
    context = SimpleNamespace(
        layout={"cuda_graph_eligible": False},
        prefix_len=900,
    )
    x_t = torch.zeros(1, 50, 32)

    output = model.denoise_step(
        context,
        x_t,
        torch.ones(1),
        use_cuda_graph=True,
    )

    torch.testing.assert_close(output, torch.ones_like(x_t))


def _observation_with_token_len(token_len: int) -> VLAObservationBatch:
    tokens = torch.arange(200).view(1, 200)
    token_masks = torch.arange(200).view(1, 200) < token_len
    return VLAObservationBatch(
        prompt=["prompt"],
        images={"camera": torch.zeros(1, 3, 4, 4)},
        image_masks={"camera": torch.ones(1, dtype=torch.bool)},
        state=None,
        noise=None,
        tokens=tokens,
        token_masks=token_masks,
        batch_size=1,
        metadata={"camera_order": ("camera",)},
    )


class _RecordingPrefixRunner:
    enabled = True

    def __init__(self):
        self.calls = []

    def capture_or_run(self, signature, _step_fn, inputs):
        self.calls.append((signature, inputs))
        return signature


def _recording_policy(config: Pi05PipelineConfig) -> Pi05PolicyModel:
    model = Pi05PolicyModel.__new__(Pi05PolicyModel)
    nn.Module.__init__(model)
    model.config = config
    model.device = torch.device("cpu")
    model.prefix_graph_runner = _RecordingPrefixRunner()
    model._prompt_token_bucketing_enabled = lambda: bool(config.prompt_token_buckets)
    return model


def test_default_prefix_graph_keeps_exact_prompt_signatures():
    model = _recording_policy(Pi05PipelineConfig())

    signature_33 = model.encode_prefix(_observation_with_token_len(33))
    signature_64 = model.encode_prefix(_observation_with_token_len(64))

    assert signature_33 != signature_64
    assert model.prefix_graph_runner.calls[0][1][-2].shape == (1, 33)
    assert model.prefix_graph_runner.calls[1][1][-2].shape == (1, 64)


def test_prefix_prompt_lengths_share_bucket_graph_signature():
    model = _recording_policy(
        Pi05PipelineConfig(
            prompt_token_buckets=[32, 64, 128, 200],
        )
    )

    signature_33 = model.encode_prefix(_observation_with_token_len(33))
    signature_64 = model.encode_prefix(_observation_with_token_len(64))

    assert signature_33 == signature_64
    for (_, inputs), expected_token_len in zip(
        model.prefix_graph_runner.calls,
        (33, 64),
        strict=True,
    ):
        assert inputs[-2].shape == (1, 64)
        assert inputs[-1].shape == (1, 64)
        assert inputs[-1].sum().item() == expected_token_len


def test_prefix_graph_rejects_tensor_parallel_prefix():
    model = Pi05PolicyModel.__new__(Pi05PolicyModel)
    model.config = Pi05PipelineConfig()
    model.device = torch.device("cuda")
    model.runtime_role = "all"
    model._prefix_language_model = lambda: SimpleNamespace(tensor_parallel=True)

    assert not model._prefix_cuda_graph_enabled()


def test_prefix_graph_rejects_partial_language_offload():
    model = Pi05PolicyModel.__new__(Pi05PolicyModel)
    model.config = Pi05PipelineConfig(
        offload_prefix_language_layer_count_after_prefix=1
    )
    model.device = torch.device("cuda")
    model.runtime_role = "all"
    model._prefix_tensor_parallel_enabled = lambda: False

    assert not model._prefix_cuda_graph_enabled()


def test_runai_direct_gpu_loader_does_not_reject_split_roles(monkeypatch):
    class FakeSafeOpen:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def keys(self):
            return ["action.weight"]

    monkeypatch.setattr(
        pi05_policy_module,
        "safe_open",
        lambda *args, **kwargs: FakeSafeOpen(),
    )

    model = Pi05PolicyModel.__new__(Pi05PolicyModel)
    model.device = torch.device("cuda")
    model.runtime_role = "action"
    model.config = Pi05PipelineConfig()
    model.manifest = Pi05CheckpointManifest(
        model_path="fake",
        safetensor_files=["fake.safetensors"],
    )
    model._should_read_source_key = lambda key: True
    target_state = {
        "action.weight": SimpleNamespace(device=SimpleNamespace(type="cuda")),
    }

    assert model._should_stream_weights_to_gpu(target_state, {})

    model.runtime_role = "idle"
    assert model._should_stream_weights_to_gpu(target_state, {})


def test_pi05_checkpoint_excludes_declared_processor_state(tmp_path):
    from safetensors.torch import save_file

    model_file = tmp_path / "model.safetensors"
    save_file({"action_in_proj.weight": torch.ones(2, 2)}, model_file)
    for kind in ("pre", "post"):
        state_file = f"policy_{kind}processor_step_0.safetensors"
        save_file({"action.mean": torch.zeros(7)}, tmp_path / state_file)
        (tmp_path / f"policy_{kind}processor.json").write_text(
            json.dumps({"steps": [{"state_file": state_file}, {"config": {}}]})
        )

    manifest = Pi05PolicyModel._inspect_checkpoint(str(tmp_path), Pi05PipelineConfig())

    assert manifest.safetensor_files == [str(model_file)]
    assert "action_in_proj.weight" in manifest.component_keys["action_heads"]


@pytest.mark.parametrize("declared_empty_camera", [False, True])
def test_pi05_checkpoint_does_not_duplicate_empty_cameras(
    tmp_path, declared_empty_camera
):
    features = {"observation.images.image": {"type": "VISUAL", "shape": [3, 224, 224]}}
    if declared_empty_camera:
        features["observation.images.empty_camera_0"] = {
            "type": "VISUAL",
            "shape": [3, 224, 224],
        }
    (tmp_path / "config.json").write_text(
        json.dumps({"input_features": features, "empty_cameras": 1})
    )
    config = Pi05PipelineConfig()
    for _ in range(2):
        Pi05PolicyModel._apply_checkpoint_config(str(tmp_path), config)
        assert config.image_keys == ("image", "empty_camera_0")


def test_pi05_loader_maps_unfused_prefix_weights_to_parallel_targets():
    q_key = (
        "paligemma_with_expert.paligemma.model.language_model.layers.0."
        "self_attn.q_proj.weight"
    )
    gate_key = (
        "paligemma_with_expert.paligemma.model.language_model.layers.0."
        "mlp.gate_proj.weight"
    )

    assert (
        "paligemma_with_expert.paligemma.model.language_model.layers.0."
        "self_attn.qkv_proj.weight",
        "q",
    ) in Pi05PolicyModel._candidate_target_weights(q_key)
    assert (
        "paligemma_with_expert.paligemma.model.language_model.layers.0."
        "mlp.gate_up_proj.weight",
        0,
    ) in Pi05PolicyModel._candidate_target_weights(gate_key)


def test_action_parallel_info_reports_single_rank_without_process_group():
    model = Pi05PolicyModel.__new__(Pi05PolicyModel)
    model.runtime_role = "all"

    info = model.action_parallel_info(prefix_context=None)

    assert info == {
        "split_group": False,
        "runtime_role": "all",
        "action_sequence_parallel": False,
    }


def test_pi05_siglip_reuses_srt_model_with_layerwise_groups():
    config = SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        layer_norm_eps=1e-6,
        image_size=4,
        patch_size=2,
        num_channels=3,
        hidden_act="gelu_pytorch_tanh",
    )
    with get_context().override_server_args():
        model = Pi05SiglipVisionModel(
            config,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            qkv_backend="sdpa",
            flatten_batch=False,
            use_data_parallel=True,
        )

    state_keys = set(model.state_dict())
    prefix = "vision_model.encoder.layers.0.self_attn"
    vision_model = model.vision_model
    layer = vision_model.encoder.layers[0]

    assert isinstance(model, SiglipVisionModel)
    assert f"{prefix}.qkv_proj.weight" in state_keys
    assert f"{prefix}.proj.weight" in state_keys
    assert vision_model.embeddings.position_embedding.tp_size == 1
    assert layer.self_attn.tp_size == 1
    assert layer.self_attn.qkv_backend.flatten_batch is False
    assert layer.mlp.fc1.output_size_per_partition == config.intermediate_size
    assert layer.mlp.fc2.input_size_per_partition == config.intermediate_size
    assert isinstance(layer.mlp.act, nn.GELU)
    assert layer.mlp.act.approximate == "tanh"
    assert model.device == vision_model.embeddings.patch_embedding.weight.device
    assert model.layer_names == ["vision_model.encoder.layers"]


def test_pi05_siglip_checkpoint_names_map_to_srt_layers():
    source_prefix = (
        "paligemma_with_expert.paligemma.vision_tower.vision_model."
        "encoder.layers.0.self_attn"
    )
    target_prefix = (
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
        "encoder.layers.0.self_attn"
    )

    assert (
        f"{target_prefix}.qkv_proj.weight",
        "q",
    ) in Pi05PolicyModel._candidate_target_weights(f"{source_prefix}.q_proj.weight")
    assert (
        f"{target_prefix}.proj.weight",
        None,
    ) in Pi05PolicyModel._candidate_target_weights(f"{source_prefix}.out_proj.weight")


def test_prefix_language_embedding_matches_openpi_scale():
    image_embedding = torch.ones(1, 2, 8)
    language_embedding = torch.full((1, 3, 8), 0.25)
    model = SimpleNamespace(
        paligemma_with_expert=SimpleNamespace(
            embed_images=lambda images: [image_embedding],
            embed_language_tokens=lambda tokens: language_embedding,
        )
    )

    embeddings, _, _ = Pi05CoreModel.embed_prefix(
        model,
        images=[torch.zeros(1, 3, 4, 4)],
        image_masks=[torch.ones(1, dtype=torch.bool)],
        tokens=torch.ones(1, 3, dtype=torch.long),
        token_masks=torch.ones(1, 3, dtype=torch.bool),
    )

    torch.testing.assert_close(
        embeddings[:, 2:],
        language_embedding * (language_embedding.shape[-1] ** 0.5),
    )


def test_cached_pi05_sinusoidal_scaling_is_bit_exact():
    time = torch.tensor([0.125, 0.75], dtype=torch.float32)
    dimension = 32
    min_period = 4e-3
    max_period = 4.0
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float64)
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * torch.pi

    expected = create_sinusoidal_pos_embedding(
        time,
        dimension,
        min_period,
        max_period,
    )
    actual = create_sinusoidal_pos_embedding(
        time,
        dimension,
        min_period,
        max_period,
        scaling=scaling,
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("backend", ["auto", "triton_split_kv"])
def test_prepare_denoise_layout_matches_per_step_construction(backend):
    model = Pi05CoreModel.__new__(Pi05CoreModel)
    nn.Module.__init__(model)
    model.config = Pi05PipelineConfig(action_attention_backend=backend)
    prefix_pad_masks = torch.tensor(
        [[True, True, False], [True, False, False]], dtype=torch.bool
    )
    x_t = torch.zeros(2, 4, 7)

    attention_mask, position_ids = model.prepare_denoise_layout(
        prefix_pad_masks,
        x_t,
        action_position_offset=3,
    )

    suffix_pad_masks = torch.ones(2, 4, dtype=torch.bool)
    suffix_att_masks = torch.zeros(2, 4)
    suffix_att_masks[:, 0] = 1
    expected_2d_mask = torch.cat(
        [
            prefix_pad_masks[:, None, :].expand(2, 4, 3),
            make_att_2d_masks(suffix_pad_masks, suffix_att_masks),
        ],
        dim=2,
    )
    expected_mask = model.prepare_attention_masks_4d(expected_2d_mask)
    expected_positions = torch.tensor([[5, 6, 7, 8], [4, 5, 6, 7]])

    assert torch.equal(attention_mask.expand_as(expected_mask), expected_mask)
    assert torch.equal(position_ids, expected_positions)
    full_attention_mask, full_attention_positions = model.prepare_denoise_layout(
        prefix_pad_masks,
        x_t,
        prefix_full_attention=True,
        action_position_offset=3,
    )
    assert full_attention_mask is None
    assert torch.equal(full_attention_positions, expected_positions)


def test_sample_actions_only_hoists_denoise_layout_for_eager():
    model = Pi05PolicyModel.__new__(Pi05PolicyModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(action_horizon=2, action_dim=3)
    model.device = torch.device("cpu")
    model.graph_runner = SimpleNamespace(enabled=True)
    model.loop_graph_runner = SimpleNamespace(enabled=False)
    model._offload_action_expert_between_requests = lambda: False
    model._can_use_action_sequence_parallel = lambda *_args: False
    model.denoise_step = lambda _ctx, x_t, _t, **_kwargs: torch.zeros_like(x_t)
    layout_calls = []
    model.core_model = SimpleNamespace(
        prepare_denoise_layout=lambda *args, **kwargs: (
            layout_calls.append((args, kwargs))
            or (None, torch.zeros(1, 2, dtype=torch.long))
        )
    )
    observation = SimpleNamespace(batch_size=1)
    prefix_context = _prefix_context(1.0, "prompt")
    prefix_context.layout["full_attention"] = True
    noise = torch.zeros(1, 2, 3)

    model.sample_actions(
        observation,
        prefix_context,
        noise=noise,
        num_steps=2,
        use_cuda_graph=True,
    )
    assert not layout_calls

    prefix_context.layout["full_attention"] = False
    model.sample_actions(
        observation,
        prefix_context,
        noise=noise,
        num_steps=2,
        use_cuda_graph=True,
    )
    assert not layout_calls

    prefix_context.layout["cuda_graph_eligible"] = False
    model.sample_actions(
        observation,
        prefix_context,
        noise=noise,
        num_steps=2,
        use_cuda_graph=True,
    )
    assert len(layout_calls) == 1

    layout_calls.clear()
    prefix_context.layout["cuda_graph_eligible"] = True
    model.sample_actions(
        observation,
        prefix_context,
        noise=noise,
        num_steps=2,
        use_cuda_graph=False,
    )
    assert len(layout_calls) == 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_action_loop_graph_refreshes_inputs_and_owns_output(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    num_steps = 7
    model = Pi05PolicyModel.__new__(Pi05PolicyModel)
    nn.Module.__init__(model)
    model.config = Pi05PipelineConfig()
    model.loop_graph_runner = VLATensorGraphRunner(
        "test_action_loop", enabled=True, max_entries=2, evict_on_miss=False
    )
    model.core_model = SimpleNamespace(prepare_denoise_layout=lambda *_: None)

    def velocity(context, x_t, timestep, **kwargs):
        keys, values, _ = context.past_key_values[0]
        return (
            x_t * timestep[:, None, None]
            + keys.sum()
            + values.sum()
            + context.prefix_pad_masks.sum()
        )

    model.action_expert = velocity
    context = _prefix_context(0.01, None)
    context.prefix_pad_masks = context.prefix_pad_masks.to(device)
    context.past_key_values = VLADensePrefixCache(
        (k.to(device), v.to(device), w) for k, v, w in context.past_key_values
    )
    context.layout["mutable_graph_output"] = True
    noise = torch.ones(1, 2, 4, device=device)
    original_noise = noise.clone()
    previous_output = previous_copy = None
    for _ in range(3):
        expected = noise.clone()
        for value in torch.linspace(1.0, 1.0 / num_steps, num_steps, device=device):
            expected.add_(
                velocity(context, expected, value.expand(1)), alpha=-1.0 / num_steps
            )
        actual = model._sample_actions_loop_graph(context, noise, num_steps)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(noise, original_noise, rtol=0, atol=0)
        if previous_output is not None:
            torch.testing.assert_close(previous_output, previous_copy, rtol=0, atol=0)
        previous_output, previous_copy = actual, actual.clone()
        context.past_key_values[0][0].add_(0.02)
        context.past_key_values[0][1].sub_(0.01)
        context.prefix_pad_masks.logical_not_()
    if device == "cuda":
        assert model.loop_graph_runner.cache_info().captures == 1
        assert model.loop_graph_runner.cache_info().failures == 0


def test_adaptive_norm_precomputed_modulation_is_exact():
    torch.manual_seed(12)
    dtype = torch.bfloat16
    norm = PiGemmaRMSNorm(16, cond_dim=16).to(dtype=dtype)
    nn.init.normal_(norm.dense.weight)
    x = torch.randn(2, 5, 16, dtype=dtype)
    cond = torch.randn(2, 16, dtype=dtype)
    expected = norm(x, cond)
    actual = norm(x, modulation=norm.dense(cond))
    for before, after in zip(expected, actual, strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["train", "apply", "load_state_dict"])
def test_action_time_cache_is_invalidated_before_weight_changes(operation):
    model = Pi05PolicyModel.__new__(Pi05PolicyModel)
    nn.Module.__init__(model)
    model.register_parameter("weight", nn.Parameter(torch.ones(2)))
    cleared = []
    for name in ("prefix_graph_runner", "graph_runner", "loop_graph_runner"):
        setattr(
            model, name, SimpleNamespace(clear=lambda name=name: cleared.append(name))
        )
    model._action_loop_schedule_key = (10, 1)
    model._action_loop_schedule = (torch.ones(2),)
    if operation == "train":
        model.train()
    elif operation == "apply":
        model.to(dtype=torch.float64)
    else:
        model.load_state_dict({"weight": torch.zeros(2)})
    assert len(cleared) == 3
    assert model._action_loop_schedule_key is None
    assert model._action_loop_schedule is None


def test_uint8_resize_rounds_before_normalization():
    image = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]]) / 255.0

    resized = _resize_with_pad_image_tensor(
        image,
        (3, 3),
        round_to_uint8=True,
    )

    expected = (
        torch.round(
            torch.nn.functional.interpolate(
                image[None], size=(3, 3), mode="bilinear", align_corners=False
            )[0]
            * 255.0
        )
        / 255.0
    )
    torch.testing.assert_close(resized, expected, rtol=0.0, atol=0.0)


def test_normalized_float_image_is_not_normalized_twice():
    image = np.full((2, 4, 3), -0.5, dtype=np.float32)

    preprocessed = _preprocess_image(image, (4, 4))

    assert preprocessed.shape == (3, 4, 4)
    torch.testing.assert_close(
        preprocessed[:, 1:3],
        torch.full((3, 2, 4), -0.5),
    )
    torch.testing.assert_close(
        preprocessed[:, (0, 3)],
        torch.full((3, 2, 4), -1.0),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA model kernels")
def test_prefix_kv_only_matches_full_forward(monkeypatch):
    """Every cached layer must match even though the final hidden state is unused."""
    from transformers import GemmaConfig

    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )
    from sglang.multimodal_gen.runtime.models.vlas.pi05_core import PiGemmaModel

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.server_args.server_args._global_server_args",
        SimpleNamespace(attention_backend="torch_sdpa", attention_backend_config=None),
    )
    config = GemmaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=32,
        max_position_embeddings=64,
        dtype=torch.bfloat16,
    )
    torch.manual_seed(42)
    model = PiGemmaModel(config).cuda().bfloat16().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.05)
    inputs = torch.randn(2, 7, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(2, 1, 7, 7, device="cuda", dtype=torch.bool)
    mask[0, :, :, -2:] = False
    with torch.inference_mode(), set_forward_context(0, None):
        expected = model(inputs_embeds=inputs, attention_mask=mask, use_cache=True)

        def unused(*args):
            raise AssertionError("KV-only prefix must skip final attention/MLP/norm")

        hooks = [
            module.register_forward_pre_hook(unused)
            for module in (
                model.layers[-1].self_attn.o_proj,
                model.layers[-1].mlp,
                model.norm,
            )
        ]
        try:
            actual = model(
                inputs_embeds=inputs,
                attention_mask=mask,
                use_cache=True,
                prefix_kv_only=True,
            )
        finally:
            for hook in hooks:
                hook.remove()
    assert actual.last_hidden_state is None
    assert expected.last_hidden_state.shape == inputs.shape
    for (keys, values, _), (expected_keys, expected_values, _) in zip(
        actual.past_key_values, expected.past_key_values, strict=True
    ):
        torch.testing.assert_close(keys, expected_keys, rtol=0, atol=0)
        torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
