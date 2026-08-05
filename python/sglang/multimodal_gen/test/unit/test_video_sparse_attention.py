import math

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import (
    video_sparse_attn as vsa_module,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn import (
    VSA_TILE_SIZE,
    VideoSparseAttentionImpl,
    VideoSparseAttentionMetadataBuilder,
    _compute_cur_topk,
)


def test_video_sparse_attention_tile_buffer_reuse_and_untile():
    metadata = VideoSparseAttentionMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=(5, 7, 9),
        patch_size=(1, 1, 1),
        VSA_sparsity=0.5,
        device=torch.device("cpu"),
    )

    impl = object.__new__(VideoSparseAttentionImpl)
    total_seq_length = metadata.total_seq_length
    x = torch.arange(2 * total_seq_length * 3 * 4, dtype=torch.float32).reshape(
        2, total_seq_length, 3, 4
    )

    tiled = impl.preprocess_qkv(x, metadata)
    assert metadata.tile_buf is tiled
    assert torch.equal(
        metadata.untile_combined_index,
        metadata.non_pad_index[metadata.reverse_tile_partition_indices],
    )
    assert torch.equal(impl.postprocess_output(tiled, metadata), x)

    next_x = x + 1
    next_tiled = impl.preprocess_qkv(next_x, metadata)
    assert next_tiled.data_ptr() == tiled.data_ptr()
    assert torch.equal(impl.postprocess_output(next_tiled, metadata), next_x)

    pad_mask = torch.ones(next_tiled.shape[1], dtype=torch.bool)
    pad_mask[metadata.non_pad_index.cpu()] = False
    assert torch.all(next_tiled[:, pad_mask] == 0)


def test_vsa_forward_cur_topk_uses_padded_kv_block_count(monkeypatch):
    metadata = VideoSparseAttentionMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=(5, 32, 32),
        patch_size=(1, 1, 1),
        VSA_sparsity=0.75,
        device=torch.device("cpu"),
    )
    num_kv_blocks = metadata.variable_block_sizes.numel()
    block_elements = math.prod(VSA_TILE_SIZE)
    padded_seq_len = num_kv_blocks * block_elements
    expected_topk = math.ceil((1 - metadata.VSA_sparsity) * num_kv_blocks)
    unpadded_topk = math.ceil(
        (1 - metadata.VSA_sparsity) * (metadata.total_seq_length / block_elements)
    )
    captured = {}

    def fake_video_sparse_attn(
        query,
        key,
        value,
        variable_block_sizes,
        topk,
        block_size,
        compress_attn_weight,
    ):
        captured["topk"] = topk
        captured["block_size"] = block_size
        assert torch.equal(variable_block_sizes, metadata.variable_block_sizes)
        return query

    monkeypatch.setattr(vsa_module, "video_sparse_attn", fake_video_sparse_attn)

    query = torch.ones(1, padded_seq_len, 1, 1)
    output = object.__new__(VideoSparseAttentionImpl).forward(
        query, query, query, query, metadata
    )

    assert unpadded_topk < expected_topk
    assert captured["topk"] == expected_topk
    assert captured["block_size"] == VSA_TILE_SIZE
    assert output.shape == query.shape


def test_vsa_cur_topk_clamps_to_valid_block_range():
    metadata = VideoSparseAttentionMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=(5, 32, 32),
        patch_size=(1, 1, 1),
        VSA_sparsity=1.0,
        device=torch.device("cpu"),
    )
    num_kv_blocks = metadata.variable_block_sizes.numel()

    assert _compute_cur_topk(metadata) == 1

    metadata.VSA_sparsity = -0.01
    assert _compute_cur_topk(metadata) == num_kv_blocks
