# SPDX-License-Identifier: Apache-2.0
import math

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
from sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn_h3 import (
    MiniMaxH3VSAImpl,
    MiniMaxH3VSAMetadataBuilder,
    _sparse_indices_from_scores,
    compute_topk,
    h3_vsa_prefix_segments,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.pipelines_core.lora.adapter_dense_payload import (
    AdapterDensePayload,
    resolve_dense_key,
    swap_peft_swiglu_fc1_lora_b,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.format_adapter import (
    normalize_lora_state_dict,
)
from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    is_vsa_h3_backend,
)


def test_normalize_drops_whole_parameter_keys():
    state = {
        "transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(4, 2),
        "transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(8, 4),
        "proj_in.diff": torch.ones(3),
        "transformer_blocks.0.attn.to_gate_compress.set_weight": torch.ones(2, 2),
    }
    normalized = normalize_lora_state_dict(state)
    assert "proj_in.diff" not in normalized
    assert "transformer_blocks.0.attn.to_gate_compress.set_weight" not in normalized
    assert "transformer_blocks.0.attn.to_q.lora_A.weight" in normalized


def test_dense_keys_map_and_apply():
    mapping = get_param_names_mapping(MiniMaxH3DiTArchConfig().param_names_mapping)
    assert resolve_dense_key("proj_in.diff", mapping) == (
        "video_patch_proj.weight",
        "add",
    )
    assert resolve_dense_key(
        "transformer_blocks.2.attn.to_gate_compress.set_weight", mapping
    ) == ("blocks.2.attn.to_gate_compress.weight", "set")
    assert (
        resolve_dense_key("transformer_blocks.0.attn.to_q.lora_A.weight", mapping)
        is None
    )

    class Wrapped(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.base_layer = inner

        @property
        def weight(self):
            return self.base_layer.weight

        @property
        def bias(self):
            return self.base_layer.bias

    root = nn.Module()
    root.norm1 = nn.RMSNorm(4)
    root.linear = Wrapped(nn.Linear(4, 4, bias=True))
    root.norm1.weight.data.fill_(2.0)
    root.linear.base_layer.bias.data.fill_(0.5)
    patch = AdapterDensePayload.from_state_dict(
        {
            "norm1.diff": torch.ones(4),
            "linear.diff_b": torch.full((4,), 0.25),
        }
    )
    applied, unmatched = patch.apply_to_module(root)
    assert applied == 2
    assert unmatched == []
    torch.testing.assert_close(root.norm1.weight, torch.full((4,), 3.0))
    torch.testing.assert_close(root.linear.base_layer.bias, torch.full((4,), 0.75))


def test_fasth3_ffn_lora_b_is_value_first_like_fastvideo():
    value = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    gate = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)
    peft = torch.cat([value, gate], dim=0)
    swapped = swap_peft_swiglu_fc1_lora_b(
        "transformer_blocks.0.ff.net.0.proj.lora_B.weight",
        "blocks.0.mlp.fc1.lora_B",
        peft,
    )
    torch.testing.assert_close(swapped, torch.cat([gate, value], dim=0))
    unchanged = swap_peft_swiglu_fc1_lora_b(
        "transformer_blocks.0.attn.to_q.lora_B.weight",
        "blocks.0.attn.qkv_proj.lora_B",
        peft,
    )
    torch.testing.assert_close(unchanged, peft)


def test_vsa_h3_aliases_and_packed_tile64():
    assert is_vsa_h3_backend("vsa_h3")
    assert is_vsa_h3_backend(AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3)
    assert not is_vsa_h3_backend("video_sparse_attn")

    text_len, audio_rows = 32, 16
    latent = (8, 16, 16)
    patch = (1, 2, 2)
    prefix = h3_vsa_prefix_segments(text_len, 0, audio_rows)
    assert prefix == (text_len, audio_rows)
    metadata = MiniMaxH3VSAMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=latent,
        patch_size=patch,
        VSA_sparsity=0.9,
        prefix_segments=prefix,
        device=torch.device("cpu"),
        tile_size=64,
    )
    video_rows = math.prod(t // p for t, p in zip(latent, patch))
    assert metadata.total_seq_length == text_len + audio_rows + video_rows
    assert metadata.tile_elems == 64

    impl = MiniMaxH3VSAImpl(
        num_heads=2,
        head_size=8,
        causal=False,
        softmax_scale=8**-0.5,
        prefix="blocks.0.attn",
    )
    packed = torch.arange(
        metadata.total_seq_length * 2 * 8, dtype=torch.float32
    ).reshape(1, metadata.total_seq_length, 2, 8)
    recovered = impl.postprocess_output(
        impl._tile_bhsd(packed, metadata, "q"), metadata
    )
    torch.testing.assert_close(recovered, packed)


def test_vsa_h3_exempt_prefix_and_rejects_tile_256():
    torch.manual_seed(0)
    scores = torch.randn(1, 2, 6, 6)
    q2k_idx, q2k_num = _sparse_indices_from_scores(scores, 2, 4, 0.5)
    assert int(q2k_num[0, 0, 0]) == 2 + compute_topk(0.5, 4)
    assert torch.equal(
        q2k_idx[..., :2],
        torch.tensor([0, 1], dtype=q2k_idx.dtype).expand_as(q2k_idx[..., :2]),
    )
    try:
        MiniMaxH3VSAMetadataBuilder().build(
            current_timestep=0,
            raw_latent_shape=(4, 8, 8),
            patch_size=(1, 2, 2),
            VSA_sparsity=0.9,
            prefix_segments=(8,),
            device=torch.device("cpu"),
            tile_size=256,
        )
    except ValueError as error:
        assert "tile_size" in str(error)
    else:
        raise AssertionError("tile 256 must be rejected on the Triton-64 path")
