from types import SimpleNamespace

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.encoders.qwen3vl import Qwen3VLArchConfig
from sglang.multimodal_gen.runtime.layers.quantization.configs.quanto_int8_config import (
    QuantoInt8Config,
)
from sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl import (
    MiniMaxH3Qwen3VLEncoder,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import (
    Qwen3VLForConditionalGeneration,
    _make_text_rms_norm,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl_vision import (
    Qwen3VLVisionRotaryEmbedding,
    Qwen3VLVisionTransformer,
    _vision_cu_seqlens,
    _vision_position_ids,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models.qwen3_vl import (
    Qwen3VLMoeVisionPatchMerger,
    Qwen3VLVisionPatchEmbed,
)
from sglang.srt.runtime_context import get_parallel


def test_native_vision_layout_matches_qwen3_merge_order():
    grid_thw = torch.tensor([[1, 4, 6], [2, 2, 4]])

    position_ids = _vision_position_ids(grid_thw, spatial_merge_size=2)
    cu_seqlens = _vision_cu_seqlens(grid_thw)

    assert position_ids.shape == (40, 2)
    assert position_ids[:8].tolist() == [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
    ]
    assert cu_seqlens.tolist() == [0, 24, 32, 40]


def test_qwen3vl_text_reuses_srt_rms_norm():
    norm = _make_text_rms_norm(16, 1e-6)

    assert isinstance(norm, RMSNorm)
    assert norm.cast_x_before_out_mul


def test_native_vision_keeps_checkpoint_parameter_names():
    config = SimpleNamespace(
        hidden_size=16,
        intermediate_size=24,
        hidden_act="gelu_pytorch_tanh",
        num_heads=2,
        depth=0,
        patch_size=2,
        temporal_patch_size=1,
        in_channels=3,
        num_position_embeddings=16,
        spatial_merge_size=2,
        out_hidden_size=12,
        deepstack_visual_indexes=[],
    )
    with get_parallel().override(tp_size=1, tp_rank=0):
        model = Qwen3VLVisionTransformer(config)

    assert isinstance(model.patch_embed, Qwen3VLVisionPatchEmbed)
    assert isinstance(model.merger, Qwen3VLMoeVisionPatchMerger)

    assert set(model.state_dict()) == {
        "patch_embed.proj.weight",
        "patch_embed.proj.bias",
        "pos_embed.weight",
        "merger.norm.weight",
        "merger.norm.bias",
        "merger.linear_fc1.weight",
        "merger.linear_fc1.bias",
        "merger.linear_fc2.weight",
        "merger.linear_fc2.bias",
    }


def test_native_vision_accepts_srt_linear_quantization():
    config = SimpleNamespace(
        hidden_size=16,
        intermediate_size=24,
        hidden_act="gelu_pytorch_tanh",
        num_heads=2,
        depth=1,
        patch_size=2,
        temporal_patch_size=1,
        in_channels=3,
        num_position_embeddings=16,
        spatial_merge_size=2,
        out_hidden_size=12,
        deepstack_visual_indexes=[],
    )
    prefixes = {
        "model.visual.blocks.0.attn.qkv_proj",
        "model.visual.blocks.0.attn.proj",
        "model.visual.blocks.0.mlp.linear_fc1",
        "model.visual.blocks.0.mlp.linear_fc2",
    }
    quant_config = QuantoInt8Config(prefixes)
    with get_parallel().override(tp_size=1, tp_rank=0):
        model = Qwen3VLVisionTransformer(
            config,
            quant_config=quant_config,
            prefix="model.visual",
        )

    assert quant_config.selected == prefixes
    for name, parameter in model.blocks[0].named_parameters():
        if name.endswith("weight") and not name.startswith("norm"):
            assert parameter.dtype == torch.int8


def test_native_vision_keeps_position_math_in_fp32():
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

    model = Qwen3VLVisionTransformer.__new__(Qwen3VLVisionTransformer)
    nn.Module.__init__(model)
    model.spatial_merge_size = 2
    model.patch_embed = PatchEmbed()
    model.pos_embed = nn.Embedding(16, 8, dtype=torch.bfloat16)
    model.num_grid_per_side = 4
    model.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(2)
    block = BlockRecorder()
    model.blocks = nn.ModuleList([block])
    model.merger = Merger()
    model.deepstack_merger_list = nn.ModuleList()
    model._deepstack_merger_by_layer = {}

    grid_thw = torch.tensor([[1, 4, 6]])
    interpolated_position = model._interpolate_position_embeddings(grid_thw)
    output = model(torch.zeros(24, 1, dtype=torch.bfloat16), grid_thw=grid_thw)

    assert interpolated_position.dtype == torch.float32
    assert output.pooler_output.dtype == torch.bfloat16
    assert block.position_embedding_dtypes == (torch.float32, torch.float32)


def test_qwen3_multimodal_encoders_layerwise_offload_vision_blocks():
    assert "model.visual.blocks" in Qwen3VLForConditionalGeneration.layer_names
    assert "model.visual.blocks" in MiniMaxH3Qwen3VLEncoder.layer_names
    assert any(
        condition.__name__ == "is_block"
        for condition in Qwen3VLArchConfig()._fsdp_shard_conditions
    )
