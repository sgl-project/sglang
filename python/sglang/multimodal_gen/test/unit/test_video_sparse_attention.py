import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn import (
    VideoSparseAttentionImpl,
    VideoSparseAttentionMetadataBuilder,
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


def test_video_sparse_attention_direct_qkvg_tiling_matches_concatenated_path():
    metadata = VideoSparseAttentionMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=(5, 7, 9),
        patch_size=(1, 1, 1),
        VSA_sparsity=0.5,
        device=torch.device("cpu"),
    )
    impl = object.__new__(VideoSparseAttentionImpl)
    query = torch.arange(
        2 * metadata.total_seq_length * 3 * 4, dtype=torch.float32
    ).reshape(2, metadata.total_seq_length, 3, 4)
    tensors = (query, query + 1, query + 2, query + 3)

    expected = impl.preprocess_qkv(torch.cat(tensors, dim=0), metadata).clone()
    tile_buf = metadata.tile_buf
    actual = impl.preprocess_qkvg(*tensors, metadata)

    assert actual is tile_buf
    assert torch.equal(actual, expected)
    for tiled, original in zip(actual.chunk(4, dim=0), tensors):
        assert torch.equal(impl.postprocess_output(tiled, metadata), original)
