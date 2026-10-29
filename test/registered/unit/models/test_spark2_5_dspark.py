"""CPU unit tests for the Spark2.5 DSpark adapter."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
from sglang.kernels.spec import KernelBackend
from sglang.srt.configs.model_config import ModelConfig, is_dspark_draft
from sglang.srt.layers.layer_boundary import layer_stack
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import spark2_5
from sglang.srt.models.dspark import DSparkDraftModel, VanillaMarkov
from sglang.srt.models.spark2_5 import Spark2_5ForCausalLM
from sglang.srt.models.spark2_5_dspark import (
    DSparkSpark25DraftModel,
    Spark2_5DSparkAttention,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.mem_cache_utils import routed

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


@pytest.fixture
def single_rank_native():
    # Match the model unit tests: real modules, no process groups or GPU kernels.
    previous_backend = get_fused_op_backend()
    set_fused_op_backend(KernelBackend.TORCH)
    try:
        with (
            get_context().override_server_args(),
            get_parallel().override(tp_size=1, tp_rank=0, tp_group=None),
            patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True),
        ):
            yield
    finally:
        set_fused_op_backend(previous_backend)


def _bundled_spark_config(**overrides):
    fields = dict(
        architectures=["Spark2_5ForCausalLM"],
        num_draft_layers=3,
        dspark_block_size=7,
        dspark_markov_rank=4,
        markov_rank=4,
        markov_head_type="vanilla",
        dspark_noise_token_id=2,
        dspark_target_layer_ids=[0, 2, 5],
        hidden_size=8,
        target_hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        sliding_window=16,
        max_position_embeddings=32,
        layer_types=["sliding_attention", "full_attention"] * 3,
        hybrid_layer_pattern=[1, 0] * 3,
        rope_parameters={
            "sliding_attention": {"rope_theta": 10000.0},
            "full_attention": {"rope_theta": 1000000.0},
        },
        vocab_size=32,
        draft_vocab_size=32,
        rms_norm_eps=1e-6,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _model_config(config=None, *, is_draft=True, algorithm="DSPARK"):
    model_config = object.__new__(ModelConfig)
    model_config.is_draft_model = is_draft
    model_config.speculative_algorithm = algorithm
    model_config.hf_config = config if config is not None else _bundled_spark_config()
    model_config.hf_text_config = model_config.hf_config
    model_config._config_draft_model()
    return model_config


@pytest.mark.parametrize("algorithm", [None, "DSPARK"])
def test_spark_bundled_checkpoint_rewrites_to_dspark_draft(algorithm):
    model_config = _model_config(algorithm=algorithm)

    assert model_config.hf_config.architectures == ["DSparkSpark25DraftModel"]
    assert is_dspark_draft(model_config.hf_config)
    assert model_config.hf_config.num_hidden_layers == 3
    assert model_config.hf_config.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
    ]
    assert model_config.hf_config.hybrid_layer_pattern == [1, 1, 1]
    assert model_config.hf_config.is_hybrid_swa

    config = parse_dspark_draft_config(draft_hf_config=model_config.hf_config)
    assert config.gamma == 7
    assert config.markov_rank == 4
    assert config.target_layer_ids == [0, 2, 5]
    assert config.markov_head_type == "vanilla"
    assert config.mask_token_id == 2


@pytest.mark.parametrize(
    "is_draft,bundled",
    [(False, True), (True, False)],
)
def test_spark_target_or_unbundled_config_keeps_mixed_attention(is_draft, bundled):
    config = _bundled_spark_config()
    if not bundled:
        for name in (
            "dspark_block_size",
            "dspark_markov_rank",
            "dspark_noise_token_id",
            "dspark_target_layer_ids",
        ):
            delattr(config, name)
    model_config = _model_config(config, is_draft=is_draft)
    assert model_config.hf_config.architectures == ["Spark2_5ForCausalLM"]
    assert not is_dspark_draft(model_config.hf_config)
    assert model_config.hf_config.num_hidden_layers == 6
    assert (
        model_config.hf_config.layer_types
        == [
            "sliding_attention",
            "full_attention",
        ]
        * 3
    )
    assert model_config.hf_config.hybrid_layer_pattern == [1, 0] * 3


@pytest.mark.parametrize("nested_kind", ["absent", "dict", "namespace"])
def test_spark_draft_builds_all_sliding_layers_with_matching_context_width(
    single_rank_native, nested_kind
):
    config = _bundled_spark_config(draft_vocab_size=24)
    if nested_kind != "absent":
        nested = dict(target_layer_ids=[1], block_size=7)
        config.dflash_config = (
            nested if nested_kind == "dict" else SimpleNamespace(**nested)
        )
    draft_config = _model_config(config).hf_config
    model = DSparkSpark25DraftModel(draft_config)

    assert model.num_context_features == 3
    assert model.fc.weight.shape == (8, 24)
    assert len(model.fc_norm) == 3
    assert model.config.dflash_config["target_layer_ids"] == [0, 2, 5]
    # DSpark IDs take precedence without mutating the caller's nested config.
    if nested_kind == "absent":
        assert not hasattr(config, "dflash_config")
    elif nested_kind == "dict":
        assert config.dflash_config == dict(target_layer_ids=[1], block_size=7)
    else:
        assert config.dflash_config.target_layer_ids == [1]
    assert len(model.layers) == 3
    for layer in model.layers:
        assert isinstance(layer.self_attn, Spark2_5DSparkAttention)
        assert layer.self_attn.attn.attn_type == AttentionType.ENCODER_ONLY
        assert layer.self_attn.attn.sliding_window_size == 15
        assert not layer.self_attn.out_proj.reduce_results
        assert not layer.mlp.down_proj.reduce_results
    assert isinstance(model.markov_head, VanillaMarkov)
    assert model.markov_head.markov_w1.weight.shape == (24, 4)
    assert model.markov_head.markov_w2.weight.shape == (24, 4)
    assert not model.supports_fused_context_kv


@pytest.mark.parametrize("raw_rank", [None, 2], ids=["prefixed-only", "conflicting"])
def test_spark_markov_head_uses_resolved_dspark_rank(single_rank_native, raw_rank):
    """A prefixed rank must construct the head even without a matching raw rank."""
    config = _bundled_spark_config(markov_rank=raw_rank)
    if raw_rank is None:
        del config.markov_rank
    draft_config = _model_config(config).hf_config

    model = DSparkSpark25DraftModel(draft_config)

    assert model.config.markov_rank == 4
    assert model.markov_head.markov_w1.weight.shape == (32, 4)
    assert model.markov_head.markov_w2.weight.shape == (32, 4)
    if raw_rank is None:
        assert not hasattr(draft_config, "markov_rank")
    else:
        assert draft_config.markov_rank == raw_rank


def test_spark_mlp_matches_exact_gelu_formula(single_rank_native):
    mlp = spark2_5.Spark2_5MLP(4, 3, reduce_results=False)
    with torch.no_grad():
        mlp.gate_up_proj.weight.copy_(torch.arange(24).reshape(6, 4) / 16 - 0.5)
        mlp.down_proj.weight.copy_(torch.arange(12).reshape(4, 3) / 12 - 0.5)
    hidden = torch.linspace(-2, 3, 12).reshape(3, 4)
    gate, up = F.linear(hidden, mlp.gate_up_proj.weight).chunk(2, dim=-1)
    expected = F.linear(F.gelu(gate, approximate="none") * up, mlp.down_proj.weight)
    torch.testing.assert_close(mlp(hidden), expected)


def _projection_model(num_context_features=2):
    model = object.__new__(DSparkSpark25DraftModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(target_hidden_size=4, hidden_size=4)
    model.num_context_features = num_context_features
    model.fc_norm = nn.ModuleList(
        [RMSNorm(4, eps=1e-6, force_native=True) for _ in range(num_context_features)]
    )
    model.fc = nn.Linear(num_context_features * 4, 4, bias=False)
    model.hidden_norm = RMSNorm(4, eps=1e-6, force_native=True)
    return model


@pytest.mark.parametrize("rows", [0, 3])
def test_spark_target_hidden_projection_matches_per_feature_rms_norm(rows):
    model = _projection_model()
    with torch.no_grad():
        model.fc.weight.copy_(torch.arange(32).view(4, 8) / 32 - 0.4)
        model.fc_norm[0].weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        model.fc_norm[1].weight.copy_(torch.tensor([2.0, 1.5, 1.0, 0.5]))
        model.hidden_norm.weight.copy_(torch.tensor([0.8, 1.2, 0.6, 1.4]))
    hidden = torch.linspace(-2, 3, rows * 8).reshape(rows, 8)
    normalized = torch.cat(
        [
            F.rms_norm(chunk, (4,), norm.weight, eps=1e-6)
            for chunk, norm in zip(hidden.chunk(2, dim=-1), model.fc_norm)
        ],
        dim=-1,
    )
    expected = F.rms_norm(
        F.linear(normalized, model.fc.weight),
        (4,),
        model.hidden_norm.weight,
        eps=1e-6,
    )
    torch.testing.assert_close(model.project_target_hidden(hidden), expected)


def test_spark_capture_configuration_requires_single_pipeline_stage():
    # Capture is configured on the target Spark wrapper.  The draft consumes
    # the resulting auxiliary hidden states; it does not own the target taps.
    model = object.__new__(Spark2_5ForCausalLM)
    nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(world_size=1, is_last_rank=True)
    model.capture_aux_hidden_states = False
    model.model = SimpleNamespace(capture_aux_hidden_states=False, layers_to_capture=[])

    model.set_dspark_layers_to_capture([0, 2, 5])

    assert model.capture_aux_hidden_states
    assert model.model.capture_aux_hidden_states
    assert model.model.layers_to_capture == [0, 2, 5]

    model.pp_group.world_size = 2
    with pytest.raises(NotImplementedError, match="PP=1"):
        model.set_dspark_layers_to_capture([0])


def test_spark_capture_preserves_boundary_residuals(single_rank_native, monkeypatch):
    draft = DSparkSpark25DraftModel(_model_config().hf_config)
    target = spark2_5.Spark2_5Model.__new__(spark2_5.Spark2_5Model)
    nn.Module.__init__(target)
    target.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    config = _bundled_spark_config(num_hidden_layers=3)
    with layer_stack():
        target.layers = nn.ModuleList(
            [spark2_5.Spark2_5DecoderLayer(config, layer_id=i) for i in range(3)]
        )
    assert [layer.self_attn.attn.sliding_window_size for layer in target.layers] == [
        15,
        -1,
        15,
    ]
    # Keep the real layer boundaries, replacing only attention/MLP compute.
    for layer in [*target.layers, *draft.layers]:
        monkeypatch.setattr(
            layer.self_attn,
            "forward",
            lambda positions, hidden_states, forward_batch: hidden_states * 0.25,
        )
        monkeypatch.setattr(
            layer.mlp, "forward", lambda hidden_states: hidden_states * 0.5
        )
    target.start_layer, target.end_layer = 0, len(target.layers)
    target.norm = draft.norm
    target.layers_to_capture = [0, 2]
    target.capture_aux_hidden_states = True
    source = torch.linspace(-2, 3, 24).reshape(3, 8)
    positions = torch.arange(3)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        input_ids=positions,
        attn_cp_metadata=None,
        can_run_tbo=False,
        residual_stream=None,
    )

    captured_output, aux = target(
        positions, positions, batch, input_embeds=source.clone()
    )

    expected = source.clone()
    snapshots = []
    for i in range(3):
        expected = expected + F.rms_norm(expected, (8,), eps=1e-6) * 0.25
        expected = expected + F.rms_norm(expected, (8,), eps=1e-6) * 0.5
        if i in (0, 2):
            snapshots.append(expected.clone())
    assert len(aux) == 2
    for actual, snapshot in zip(aux, snapshots):
        torch.testing.assert_close(actual, snapshot)
    torch.testing.assert_close(captured_output, F.rms_norm(expected, (8,), eps=1e-6))
    assert batch.residual_stream is None

    target.capture_aux_hidden_states = False
    output = target(positions, positions, batch, input_embeds=source.clone())
    torch.testing.assert_close(output, captured_output)
    assert batch.residual_stream is None

    draft_output = draft(positions, positions, batch, input_embeds=source.clone())
    assert draft_output.next_token_logits is None
    torch.testing.assert_close(draft_output.hidden_states, captured_output)
    assert batch.residual_stream is None


@pytest.mark.parametrize(
    "preprojected,commit_lens",
    [(False, None), (False, [2, 1]), (False, [0, 0]), (True, None)],
    ids=["all-rows", "partial-commit", "zero-commit", "preprojected"],
)
def test_spark_context_kv_writes_only_committed_rows(preprojected, commit_lens):
    model = _projection_model()
    hidden = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    model.project_target_hidden = Mock(return_value=hidden)
    positions = torch.arange(4)
    attention = SimpleNamespace(
        num_kv_heads=1,
        head_dim=4,
        attn=SimpleNamespace(layer_id=0, k_scale=0.5, v_scale=0.25),
        kv_proj_only=Mock(side_effect=lambda x: (x + 1, x + 2)),
        apply_k_rope=Mock(side_effect=lambda pos, k: k + pos[:, None]),
    )
    model.layers = [SimpleNamespace(self_attn=attention)]
    translated = []

    def translate(loc):
        # Uncommitted locations can be out of bounds and must be masked first.
        assert (loc >= 0).all() and (loc < 20).all()
        translated.append(loc.clone())
        return loc + 20

    pool = routed(SimpleNamespace(set_kv_buffer=Mock()))
    runner_ids = SimpleNamespace(swa_write_ids=translate)
    injector = TargetHiddenKvInjector(
        draft_model=model,
        draft_model_runner=SimpleNamespace(
            token_to_kv_pool=pool, runner_ids=runner_ids
        ),
        model_runner=SimpleNamespace(device=torch.device("cpu")),
        device=torch.device("cpu"),
        verify_num_draft_tokens=2,
        block_pos_offsets=torch.arange(2),
    )
    loc_2d = torch.tensor([[3, 4], [8, 999]])
    if commit_lens is None:
        loc_2d[1, 1] = 9
    injector.inject_target_hidden(
        target_hidden=hidden,
        target_hidden_is_projected=preprojected,
        positions=positions,
        cache_loc=loc_2d.flatten(),
        cache_loc_2d=loc_2d if commit_lens is not None else None,
        commit_lens=torch.tensor(commit_lens) if commit_lens is not None else None,
    )
    if preprojected:
        model.project_target_hidden.assert_not_called()
    else:
        model.project_target_hidden.assert_called_once_with(hidden)
    valid_rows = (
        torch.ones(4, dtype=torch.bool)
        if commit_lens is None
        else (torch.arange(2)[None, :] < torch.tensor(commit_lens)[:, None]).flatten()
    )
    torch.testing.assert_close(
        translated[0], torch.where(valid_rows, loc_2d.flatten(), 0)
    )
    if not valid_rows.any():
        pool.set_kv_buffer.assert_not_called()
        return
    pool.set_kv_buffer.assert_called_once()
    radix_attn, write_loc, key, value, k_scale, v_scale = (
        pool.set_kv_buffer.call_args.args
    )
    assert radix_attn is attention.attn
    assert write_loc.physical
    torch.testing.assert_close(write_loc.loc, loc_2d.flatten()[valid_rows])
    torch.testing.assert_close(write_loc.swa_loc, write_loc.loc + 20)
    torch.testing.assert_close(key, (hidden + 1 + positions[:, None])[valid_rows, None])
    torch.testing.assert_close(value, (hidden + 2)[valid_rows, None])
    assert (k_scale, v_scale) == (0.5, 0.25)


def test_spark_prefix_writer_uses_sliding_pool_locations_and_layer_id():
    pool = SWAKVPool(
        size=32,
        size_swa=32,
        page_size=1,
        dtype=torch.float32,
        head_num=1,
        head_dim=4,
        swa_attention_layer_ids=[0, 2],
        full_attention_layer_ids=[1],
        device="cpu",
        enable_alt_stream=False,
        kv_cache_layout="nhd",
    )
    attention = SimpleNamespace(
        attn=SimpleNamespace(layer_id=2, k_scale=0.5, v_scale=0.25)
    )
    full_loc = torch.tensor([[1, 2], [3, 0]])
    swa_loc = full_loc + 20
    commit_lens = torch.tensor([2, 1])
    key = torch.arange(16, dtype=torch.float32).reshape(4, 1, 4)
    value = key + 1

    with (
        patch("sglang.srt.mem_cache.memory_pool._is_cuda", False),
        patch("sglang.srt.mem_cache.memory_pool._is_hip", False),
    ):
        assert DSparkSpark25DraftModel._set_swa_kv_buffer_prefix_valid(
            pool=pool,
            attn=attention,
            full_loc_2d=full_loc,
            swa_loc_2d=swa_loc,
            commit_lens=commit_lens,
            k=key,
            v=value,
        )

    written_k, written_v = pool.get_kv_buffer(2)
    expected_k = torch.zeros_like(written_k)
    expected_v = torch.zeros_like(written_v)
    expected_k[torch.tensor([21, 22, 23])] = key[:3]
    expected_v[torch.tensor([21, 22, 23])] = value[:3]
    torch.testing.assert_close(written_k, expected_k)
    torch.testing.assert_close(written_v, expected_v)
    for layer_id in (0, 1):
        for buffer in pool.get_kv_buffer(layer_id):
            assert torch.count_nonzero(buffer) == 0


def test_spark_prefix_writer_falls_back_to_full_owner_locations():
    """A fused draft folded into FULL must inject committed rows without SWA ids."""
    model = _projection_model()
    hidden = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    model.project_target_hidden = Mock(return_value=hidden)
    positions = torch.arange(4)
    attention = SimpleNamespace(
        num_kv_heads=1,
        head_dim=4,
        attn=SimpleNamespace(layer_id=0, k_scale=0.5, v_scale=0.25),
        kv_proj_only=Mock(side_effect=lambda x: (x + 1, x + 2)),
        apply_k_rope=Mock(side_effect=lambda pos, k: k + pos[:, None]),
    )
    model.layers = [SimpleNamespace(self_attn=attention)]
    pool = routed(SimpleNamespace(set_kv_buffer=Mock()))
    full_loc_2d = torch.tensor([[3, 4], [8, 999]])

    model.write_target_hidden_kv(
        target_hidden=hidden,
        pool=pool,
        positions=positions,
        cache_loc=full_loc_2d.flatten(),
        cache_loc_2d=full_loc_2d,
        commit_lens=torch.tensor([2, 1]),
        swa_write_ids=lambda _full_loc: None,
    )

    _, write_loc, _, _, _, _ = pool.set_kv_buffer.call_args.args
    assert write_loc.physical
    assert write_loc.swa_loc is None
    torch.testing.assert_close(write_loc.loc, torch.tensor([3, 4, 8]))


@pytest.mark.parametrize(
    "model_cls,projection_name",
    [(DSparkDraftModel, "o_proj"), (DSparkSpark25DraftModel, "out_proj")],
)
def test_context_kv_pruning_releases_each_models_output_projection(
    model_cls, projection_name
):
    model = object.__new__(model_cls)
    nn.Module.__init__(model)
    model.markov_head = nn.Linear(4, 4)
    model.confidence_head = nn.Linear(4, 1)
    layer = nn.Module()
    layer.mlp = nn.Linear(4, 4)
    layer.self_attn = nn.Module()
    layer.self_attn.k_proj = nn.Linear(4, 4)
    setattr(layer.self_attn, projection_name, nn.Linear(4, 4))
    model.layers = nn.ModuleList([layer])

    with patch("torch.cuda.empty_cache"):
        model.prune_to_ctx_kv_injection()

    assert model.markov_head is None
    assert model.confidence_head is None
    assert layer.mlp is None
    assert getattr(layer.self_attn, projection_name) is None
    assert layer.self_attn.k_proj is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
