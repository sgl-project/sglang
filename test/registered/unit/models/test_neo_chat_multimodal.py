# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from sglang.kernels.ops.attention.extend_attention import (
    _custom_mask_dense_attention_fwd,
    extend_attention_fwd,
)
from sglang.srt.configs.neo_chat import NEOVisionConfig
from sglang.srt.models.neo_chat_mask import (
    build_u1_hybrid_allowed_matrix,
    build_u1_hybrid_backend_mask,
)
from sglang.srt.models.neo_chat_vision import (
    NEOVisionModel,
    build_abs_positions_from_grid_hw,
)
from sglang.srt.multimodal.processors.neo_chat import build_u1_mrope_positions


def test_neo_chat_mrope_positions_match_u1_layout() -> None:
    input_ids = torch.tensor([10, 20, 30, 30, 30, 30, 40, 50])
    positions, delta = build_u1_mrope_positions(
        input_ids,
        img_start_token_id=20,
        img_context_token_id=30,
        grid_hw=torch.tensor([[4, 4]]),
        downsample_ratio=0.5,
    )

    assert positions.tolist() == [
        [0, 1, 2, 2, 2, 2, 3, 4],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0],
    ]
    assert delta.tolist() == [-3]


def test_neo_chat_mrope_precomputes_text_decode_axes() -> None:
    input_ids = torch.tensor([10, 20, 30, 30, 40])
    positions, delta = build_u1_mrope_positions(
        input_ids,
        img_start_token_id=20,
        img_context_token_id=30,
        grid_hw=torch.tensor([[2, 4]]),
        downsample_ratio=0.5,
        future_decode_tokens=3,
    )

    assert positions[:, -3:].tolist() == [
        [4, 5, 6],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert delta.tolist() == [-1]


def test_neo_chat_hybrid_mask_matches_dense_policy() -> None:
    indexes = torch.tensor(
        [
            [0, 1, 2, 2, 2, 3],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
        ]
    )
    image_tag = torch.tensor([False, False, True, True, True, False])
    dense = build_u1_hybrid_allowed_matrix(indexes[0], image_tag)
    flat, indptr = build_u1_hybrid_backend_mask(
        indexes,
        image_tag,
        [6],
        [0],
    )

    assert flat is not None
    assert torch.equal(flat.reshape(6, 6), dense)
    assert indptr.tolist() == [0, 36]
    assert dense[2, 4]
    assert not dense[1, 2]


def test_neo_chat_hybrid_mask_keeps_cached_prefix_visible() -> None:
    indexes = torch.tensor(
        [
            [2, 2, 3],
            [0, 1, 0],
            [0, 0, 0],
        ]
    )
    image_tag = torch.tensor([True, True, False])
    flat, indptr = build_u1_hybrid_backend_mask(
        indexes,
        image_tag,
        [3],
        [5],
    )

    assert flat is not None
    mask = flat.reshape(3, 8)
    assert mask[:, :5].all()
    assert mask[0, 6]
    assert mask[1, 5]
    assert not mask[0, 7]
    assert indptr.tolist() == [0, 24]


def test_neo_chat_hybrid_mask_survives_image_only_in_cached_prefix() -> None:
    indexes = torch.tensor(
        [
            [5],
            [0],
            [0],
        ]
    )
    image_tag = torch.tensor([False])
    flat, indptr = build_u1_hybrid_backend_mask(
        indexes,
        image_tag,
        [1],
        [5],
        force_custom_mask=True,
    )

    assert flat is not None
    assert flat.tolist() == [True] * 6
    assert indptr.tolist() == [0, 6]


def test_neo_chat_grid_positions_are_row_major() -> None:
    abs_x, abs_y = build_abs_positions_from_grid_hw(torch.tensor([[2, 3], [1, 2]]))

    assert abs_x.tolist() == [0, 1, 2, 0, 1, 2, 0, 1]
    assert abs_y.tolist() == [0, 0, 0, 1, 1, 1, 0, 0]


def test_neo_chat_vision_refreshes_fp32_rope_cache_after_cast() -> None:
    config = NEOVisionConfig(
        hidden_size=8,
        llm_hidden_size=8,
        num_channels=3,
        patch_size=2,
        downsample_ratio=0.5,
        max_position_embeddings_vision=16,
    )
    model = NEOVisionModel(config).to(dtype=torch.bfloat16)

    assert model.cos_cached_x.dtype == torch.bfloat16
    model._ensure_fp32_rope_cache()
    assert model.cos_cached_x.dtype == torch.float32
    assert model.sin_cached_y.dtype == torch.float32


def test_short_custom_mask_dense_attention_matches_reference() -> None:
    query = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            [[0.9, 1.0], [1.1, 1.2]],
        ]
    )
    key = torch.tensor(
        [
            [[0.2, 0.1]],
            [[0.4, 0.3]],
            [[0.6, 0.5]],
        ]
    )
    value = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[0.5, 0.5]],
        ]
    )
    output = torch.empty_like(query)
    allowed = torch.tensor(
        [
            [True, False, False],
            [True, True, True],
            [True, True, True],
        ]
    )

    _custom_mask_dense_attention_fwd(
        q_extend=query,
        k_extend=key,
        v_extend=value,
        o_extend=output,
        k_buffer=torch.empty(0, 1, 2),
        v_buffer=torch.empty(0, 1, 2),
        qo_indptr=torch.tensor([0, 3]),
        kv_indptr=torch.tensor([0, 0]),
        kv_indices=torch.empty(0, dtype=torch.long),
        custom_mask=allowed.flatten(),
        mask_indptr=torch.tensor([0, 9]),
        k_scale=1.0,
        v_scale=1.0,
        sm_scale=2**-0.5,
    )

    repeated_key = key.repeat_interleave(2, dim=1)
    repeated_value = value.repeat_interleave(2, dim=1)
    scores = torch.einsum(
        "qhd,khd->hqk",
        query.float(),
        repeated_key.float(),
    )
    scores = scores * (2**-0.5)
    scores = scores.masked_fill(
        ~allowed.unsqueeze(0),
        torch.finfo(scores.dtype).min,
    )
    expected = torch.einsum(
        "hqk,khd->qhd",
        torch.softmax(scores, dim=-1),
        repeated_value,
    )
    torch.testing.assert_close(output, expected)


def test_short_custom_mask_dense_attention_reads_cached_prefix() -> None:
    query = torch.tensor([[[0.5, 0.25]]])
    current_key = torch.tensor([[[0.1, 0.2]]])
    current_value = torch.tensor([[[0.0, 1.0]]])
    key_buffer = torch.tensor(
        [
            [[0.3, 0.4]],
            [[0.7, 0.8]],
        ]
    )
    value_buffer = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.5, 0.5]],
        ]
    )
    output = torch.empty_like(query)
    allowed = torch.tensor([[True, True, True]])

    _custom_mask_dense_attention_fwd(
        q_extend=query,
        k_extend=current_key,
        v_extend=current_value,
        o_extend=output,
        k_buffer=key_buffer,
        v_buffer=value_buffer,
        qo_indptr=torch.tensor([0, 1]),
        kv_indptr=torch.tensor([0, 2]),
        kv_indices=torch.tensor([0, 1]),
        custom_mask=allowed.flatten(),
        mask_indptr=torch.tensor([0, 3]),
        k_scale=1.0,
        v_scale=1.0,
        sm_scale=2**-0.5,
    )

    all_keys = torch.cat([key_buffer, current_key], dim=0)
    all_values = torch.cat([value_buffer, current_value], dim=0)
    scores = torch.einsum(
        "qhd,khd->hqk",
        query.float(),
        all_keys.float(),
    )
    expected = torch.einsum(
        "hqk,khd->qhd",
        torch.softmax(scores * (2**-0.5), dim=-1),
        all_values,
    )
    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_custom_mask_triton_attention_reads_across_query_blocks() -> None:
    device = torch.device("cuda")
    prefix_len = 12
    image_len = 144
    suffix_len = 4
    total_len = prefix_len + image_len + suffix_len
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 16
    scaling = head_dim**-0.5

    t_indexes = torch.arange(total_len, device=device)
    t_indexes[prefix_len : prefix_len + image_len] = prefix_len
    indexes = torch.stack(
        [
            t_indexes,
            torch.zeros(total_len, device=device, dtype=torch.long),
            torch.zeros(total_len, device=device, dtype=torch.long),
        ]
    )
    image_tag = torch.zeros(total_len, device=device, dtype=torch.bool)
    image_tag[prefix_len : prefix_len + image_len] = True

    query = torch.zeros(
        total_len,
        num_q_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    key = torch.zeros(
        total_len,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    value = torch.zeros_like(key)
    value[64 : prefix_len + image_len] = 1
    repeated_value = value.repeat_interleave(
        num_q_heads // num_kv_heads,
        dim=1,
    )
    allowed = build_u1_hybrid_allowed_matrix(indexes[0], image_tag)
    scores = torch.zeros(
        num_q_heads,
        total_len,
        total_len,
        device=device,
        dtype=torch.float32,
    )
    scores.masked_fill_(
        ~allowed.unsqueeze(0),
        torch.finfo(scores.dtype).min,
    )
    expected = torch.einsum(
        "hqk,khd->qhd",
        torch.softmax(scores, dim=-1).to(value.dtype),
        repeated_value,
    )

    custom_mask, mask_indptr = build_u1_hybrid_backend_mask(
        indexes,
        image_tag,
        [total_len],
        [0],
    )
    assert custom_mask is not None
    actual = torch.empty_like(query)
    extend_attention_fwd(
        query,
        key,
        value,
        actual,
        torch.empty(
            1,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=key.dtype,
        ),
        torch.empty(
            1,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=value.dtype,
        ),
        torch.tensor([0, total_len], device=device, dtype=torch.int64),
        torch.tensor([0, 0], device=device, dtype=torch.int64),
        torch.empty(0, device=device, dtype=torch.int64),
        custom_mask,
        True,
        mask_indptr,
        513,
        1.0,
        1.0,
        sm_scale=scaling,
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=0.01, rtol=0.01)
