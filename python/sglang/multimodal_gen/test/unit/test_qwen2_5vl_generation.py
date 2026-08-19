from types import SimpleNamespace

import torch
from torch import nn

import sglang.multimodal_gen.runtime.models.encoders.qwen2_5vl as qwen2_5vl
from sglang.multimodal_gen.configs.models.encoders.qwen_image import Qwen2_5VLConfig
from sglang.multimodal_gen.configs.pipeline_configs.longcat_image import (
    LongCatImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageLayeredPipelineConfig,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen2_5vl import (
    Qwen2_5_VLAttention,
    Qwen2_5_VLForConditionalGeneration,
    _apply_repetition_penalty,
    _make_column_linear,
    _make_row_linear,
    _select_next_token,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen2_5vl_vision import (
    Qwen2_5VLVisionRotaryEmbedding,
    Qwen2_5VLVisionTransformer,
    _vision_position_ids,
    _vision_window_index,
)
from sglang.multimodal_gen.runtime.pipelines.longcat_image import LongCatImagePipeline
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.models.qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionPatchMerger,
    Qwen2_5_VLMLP,
)
from sglang.srt.runtime_context import get_parallel


class _StubQwen2_5VL(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, next_tokens: list[list[int]], eos_token_id=5):
        nn.Module.__init__(self)
        self.model = nn.Module()
        self.model.rope_deltas = torch.tensor([99])
        self.config = SimpleNamespace(eos_token_id=eos_token_id, pad_token_id=0)
        self.generation_config = {
            "do_sample": True,
            "temperature": 0.1,
            "top_k": 1,
            "top_p": 0.001,
            "repetition_penalty": 1.05,
            "eos_token_id": [eos_token_id, 6],
            "pad_token_id": 0,
        }
        self.next_tokens = next_tokens
        self.calls = []

    def forward(self, input_ids, **kwargs):
        call_index = len(self.calls)
        self.calls.append((input_ids.clone(), kwargs))
        logits = torch.full((input_ids.shape[0], 1, 8), -100.0)
        for batch_index, token_id in enumerate(self.next_tokens[call_index]):
            logits[batch_index, 0, token_id] = 100.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=f"cache-{call_index}",
        )


class _AttentionRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.masks = []

    def forward(self, query, key, value, attn_mask=None):
        self.masks.append(attn_mask)
        return query


def test_native_vision_reuses_srt_modules():
    config = SimpleNamespace(
        hidden_size=16,
        intermediate_size=24,
        hidden_act="silu",
        num_heads=2,
        depth=0,
        patch_size=2,
        temporal_patch_size=1,
        in_channels=3,
        spatial_merge_size=2,
        out_hidden_size=12,
        fullatt_block_indexes=[],
        window_size=8,
    )
    with get_parallel().override(tp_size=1, tp_rank=0):
        model = Qwen2_5VLVisionTransformer(config)
        mlp = Qwen2_5_VLMLP(
            16,
            24,
            fuse_gate_up=False,
        )
        fused_mlp = Qwen2_5_VLMLP(16, 24)

    assert isinstance(model.patch_embed, Qwen2_5_VisionPatchEmbed)
    assert isinstance(model.merger, Qwen2_5_VisionPatchMerger)
    assert not mlp.fuse_gate_up
    assert isinstance(mlp.gate_proj, ColumnParallelLinear)
    assert isinstance(mlp.up_proj, ColumnParallelLinear)
    assert mlp.gate_proj.tp_size == mlp.up_proj.tp_size == 1
    assert isinstance(mlp.down_proj, ReplicatedLinear)
    assert isinstance(fused_mlp.down_proj, RowParallelLinear)
    assert mlp.act is not None
    assert isinstance(
        _make_column_linear(16, 24, bias=False, use_tensor_parallel=False),
        ReplicatedLinear,
    )
    assert isinstance(
        _make_row_linear(24, 16, bias=False, use_tensor_parallel=False),
        ReplicatedLinear,
    )


def test_text_mlp_uses_single_rank_when_intermediate_size_is_not_tp_divisible(
    monkeypatch,
):
    monkeypatch.setattr(qwen2_5vl, "Qwen2_5_VLAttention", lambda *_args: nn.Identity())
    monkeypatch.setattr(qwen2_5vl, "_tp_world_size", lambda: 3)
    monkeypatch.setattr(qwen2_5vl, "_tp_rank", lambda: 2)
    config = SimpleNamespace(
        hidden_size=16,
        intermediate_size=25,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        use_sliding_window=False,
        _attn_implementation="flash_attention_2",
        layer_types=["full_attention"],
    )

    layer = qwen2_5vl.Qwen2_5_VLDecoderLayer(config, layer_idx=0)

    assert layer.mlp.tp_size == 1
    assert layer.mlp.tp_rank == 0
    assert isinstance(layer.mlp.gate_proj, ColumnParallelLinear)
    assert isinstance(layer.mlp.up_proj, ColumnParallelLinear)
    assert layer.mlp.gate_proj.tp_rank == layer.mlp.up_proj.tp_rank == 0
    assert isinstance(layer.mlp.down_proj, ReplicatedLinear)


def test_explicit_attention_mask_is_limited_to_cached_generation(monkeypatch):
    attention = Qwen2_5_VLAttention.__new__(Qwen2_5_VLAttention)
    nn.Module.__init__(attention)
    attention.q_proj = nn.Identity()
    attention.k_proj = nn.Identity()
    attention.v_proj = nn.Identity()
    attention.o_proj = nn.Identity()
    attention.num_heads = 1
    attention.num_key_value_heads = 1
    attention.head_dim = 4
    attention.rotary_emb = object()
    attention.attn = _AttentionRecorder()
    monkeypatch.setattr(
        qwen2_5vl,
        "apply_qwen_vl_text_rope",
        lambda _rotary_emb, _position_ids, query, key: (query, key),
    )

    hidden_states = torch.randn(1, 2, 4)
    explicit_mask = torch.zeros(1, 1, 2, 2)
    kwargs = {
        "hidden_states": hidden_states,
        "attention_mask": explicit_mask,
        "position_ids": torch.zeros(3, 1, 2, dtype=torch.long),
    }

    attention(**kwargs, use_cache=False)
    attention(**kwargs, use_cache=True)

    assert attention.attn.masks[0] is None
    assert attention.attn.masks[1] is explicit_mask


def test_repetition_penalty_matches_sign_dependent_scaling():
    logits = torch.tensor([[2.0, -3.0, 4.0]])
    penalized = _apply_repetition_penalty(logits, torch.tensor([[0, 1]]), penalty=2.0)
    torch.testing.assert_close(penalized, torch.tensor([[1.0, -6.0, 4.0]]))


def test_top_k_one_sampling_is_deterministic():
    token = _select_next_token(
        torch.tensor([[1.0, 3.0, 2.0]]),
        torch.tensor([[0]]),
        do_sample=True,
        temperature=0.1,
        top_k=1,
        top_p=0.001,
        repetition_penalty=1.0,
    )
    assert token.tolist() == [1]


def test_generate_reuses_cache_and_only_prefills_vision_once():
    model = _StubQwen2_5VL([[4], [5]])
    pixel_values = torch.ones(1, 3)
    generated = model.generate(
        torch.tensor([[1, 2]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        pixel_values=pixel_values,
        image_grid_thw=torch.tensor([[1, 1, 1]]),
        mm_token_type_ids=torch.tensor([[0, 1]]),
        max_new_tokens=3,
    )

    assert generated.tolist() == [[1, 2, 4, 5]]
    assert len(model.calls) == 2
    assert model.calls[0][1]["pixel_values"] is pixel_values
    assert model.calls[1][1]["pixel_values"] is None
    assert model.calls[0][1]["past_key_values"] is None
    assert model.calls[1][1]["past_key_values"] == "cache-0"
    assert model.calls[0][1]["cache_position"].tolist() == [0, 1]
    assert model.calls[1][1]["cache_position"].tolist() == [2]
    assert model.model.rope_deltas is None


def test_generate_pads_finished_rows_until_the_batch_stops():
    model = _StubQwen2_5VL([[5, 4], [7, 6]])
    generated = model.generate(
        torch.tensor([[1], [2]]),
        max_new_tokens=3,
    )

    assert generated.tolist() == [[1, 5, 0], [2, 4, 6]]


def test_native_vision_indices_preserve_merged_token_groups():
    grid_thw = torch.tensor([[1, 4, 4], [2, 2, 4]])
    position_ids = _vision_position_ids(grid_thw, spatial_merge_size=2)
    window_index, cu_window_seqlens = _vision_window_index(
        grid_thw,
        spatial_merge_size=2,
        window_size=8,
        patch_size=2,
    )

    assert position_ids.shape == (32, 2)
    assert sorted(window_index.tolist()) == list(range(8))
    assert cu_window_seqlens[0].item() == 0
    assert cu_window_seqlens[-1].item() == 32


def test_native_vision_keeps_rotary_trigonometry_in_fp32():
    class PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, 8, bias=False, dtype=torch.bfloat16)

        def forward(self, hidden_states):
            return self.proj(hidden_states)

    class BlockRecorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.position_embedding_dtypes = None

        def forward(self, hidden_states, *, position_embeddings, **_kwargs):
            self.position_embedding_dtypes = tuple(
                embedding.dtype for embedding in position_embeddings
            )
            return hidden_states

    class Merger(nn.Module):
        def forward(self, hidden_states):
            return hidden_states.reshape(-1, 4, hidden_states.shape[-1])[:, 0]

    model = Qwen2_5VLVisionTransformer.__new__(Qwen2_5VLVisionTransformer)
    nn.Module.__init__(model)
    model.spatial_merge_size = 2
    model.spatial_merge_unit = 4
    model.patch_size = 2
    model.window_size = 8
    model.full_attention_layers = frozenset({0})
    model.patch_embed = PatchEmbed()
    model.rotary_pos_emb = Qwen2_5VLVisionRotaryEmbedding(2)
    block = BlockRecorder()
    model.blocks = nn.ModuleList([block])
    model.merger = Merger()
    output = model(
        torch.zeros(16, 1, dtype=torch.bfloat16),
        grid_thw=torch.tensor([[1, 4, 4]]),
    )

    assert output.dtype == torch.bfloat16
    assert block.position_embedding_dtypes == (torch.float32, torch.float32)


def test_qwen_generation_pipelines_load_the_native_component():
    longcat_config = LongCatImagePipelineConfig()

    assert isinstance(longcat_config.text_encoder_configs[0], Qwen2_5VLConfig)
    assert "text_encoder" in LongCatImagePipeline._required_config_modules
    assert Qwen2_5_VLForConditionalGeneration._fsdp_forward_methods == ("generate",)
    assert "model.visual.blocks" in Qwen2_5_VLForConditionalGeneration.layer_names

    for pipeline_config in (longcat_config, QwenImageLayeredPipelineConfig()):
        deployment = pipeline_config.get_model_deployment_config()
        assert deployment.keep_resident_min_available_gb == 70
        assert deployment.keep_resident_components == ("text_encoder", "vae")
