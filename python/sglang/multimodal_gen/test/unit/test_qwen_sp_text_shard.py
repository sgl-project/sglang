"""Unit test for masked/batched SP text sharding in the QwenImage DiT.

Exercises ``_mask_shard_text_for_sp`` (used by the masked branch of the
transformer forward under sequence parallelism) with the SP world-size/rank
helpers mocked, so no GPU or distributed init is required. Verifies that the
per-rank text shards + their validity masks reassemble to the original
padded stream, i.e. every real text token is kept on exactly one rank and all
padding is masked out.
"""

from unittest import mock

import torch

from sglang.multimodal_gen.runtime.models.dits import qwen_image as qi


def _run_all_ranks(ehs, freqs_cis, key_mask, image_seq_len, sp_size):
    """Collect (text_shard, joint_mask) from every SP rank."""
    shards = []
    for rank in range(sp_size):
        with mock.patch.object(qi, "get_sp_world_size", return_value=sp_size), \
             mock.patch.object(qi, "get_sp_parallel_rank", return_value=rank):
            ehs_r, _, joint_mask, meta = qi._mask_shard_text_for_sp(
                ehs.clone(), freqs_cis, key_mask.clone(), image_seq_len
            )
        shards.append((ehs_r, joint_mask, meta))
    return shards


def test_mask_shard_reassembles_padded_text():
    torch.manual_seed(0)
    sp_size = 2
    b, t_real, dim, image_seq_len = 2, 7, 8, 4  # t_real=7 not divisible by 2
    ehs = torch.randn(b, t_real, dim)
    # row 0 valid len 7 (full), row 1 valid len 3 (variable-length batch)
    key_mask = torch.zeros(b, t_real, dtype=torch.bool)
    key_mask[0, :7] = True
    key_mask[1, :3] = True

    shards = _run_all_ranks(ehs, None, key_mask, image_seq_len, sp_size)

    local_txt = shards[0][0].shape[1]
    assert local_txt == ((t_real + sp_size - 1) // sp_size)  # padded/sp = 4
    # text shards concatenate back to the padded text (real prefix intact)
    text_cat = torch.cat([s[0] for s in shards], dim=1)
    assert torch.allclose(text_cat[:, :t_real], ehs)

    # per-rank joint mask = [text_local_validity | image_ones]
    text_valid_cat = torch.cat(
        [s[1][:, : s[0].shape[1]] for s in shards], dim=1
    )
    padded_mask = torch.zeros(b, local_txt * sp_size, dtype=torch.bool)
    padded_mask[:, :t_real] = key_mask
    assert torch.equal(text_valid_cat, padded_mask)  # exactly the real tokens kept

    for _, joint_mask, _ in shards:
        assert joint_mask.shape == (b, local_txt + image_seq_len)
        assert joint_mask[:, local_txt:].all()  # image fully valid


def test_even_length_no_padding():
    sp_size = 2
    b, t_real, dim, image_seq_len = 1, 8, 4, 2  # divisible
    ehs = torch.randn(b, t_real, dim)
    key_mask = torch.ones(b, t_real, dtype=torch.bool)
    shards = _run_all_ranks(ehs, None, key_mask, image_seq_len, sp_size)
    assert shards[0][0].shape[1] == 4
    text_cat = torch.cat([s[0] for s in shards], dim=1)
    assert torch.allclose(text_cat, ehs)  # no padding, exact split
    for _, joint_mask, _ in shards:
        assert joint_mask.all()  # all valid (full text + image)


def test_rope_cache_sharded_with_text():
    sp_size = 2
    b, t_real, dim, image_seq_len = 1, 6, 4, 2
    ehs = torch.randn(b, t_real, dim)
    img_cache = torch.randn(image_seq_len, dim)
    txt_cache = torch.randn(t_real, dim)
    key_mask = torch.ones(b, t_real, dtype=torch.bool)
    for rank in range(sp_size):
        with mock.patch.object(qi, "get_sp_world_size", return_value=sp_size), \
             mock.patch.object(qi, "get_sp_parallel_rank", return_value=rank):
            _, freqs_cis, _, _ = qi._mask_shard_text_for_sp(
                ehs.clone(), (img_cache, txt_cache), key_mask.clone(), image_seq_len
            )
        # txt RoPE cache is sharded to match the text; image cache untouched
        assert freqs_cis[1].shape[0] == t_real // sp_size
        assert torch.equal(freqs_cis[0], img_cache)
