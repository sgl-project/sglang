"""Focused SM120 FlashAttention-4 regression tests."""

import math

import cutlass.cute as cute
import pytest
import torch

from sglang.kernels.ops.attention.fa4_sm120.policy import (
    low_hd_paged_decode_tile_m,
    visible_decode_seqlen_k,
)
from sglang.kernels.ops.attention.fa4_sm120.runtime import (
    Sm120ForwardHost,
    sm120_forward_host,
)
from sglang.kernels.ops.attention.flash_attention_v4_sm120 import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_flash_attention_v4_sm120_runtime_policy,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=240,
    stage="base-b",
    runner_config="1-gpu-small",
)

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12):
    pytest.skip(
        "SM12x FlashAttention-4 test requires CUDA SM 12.x.",
        allow_module_level=True,
    )


def test_sm120_runtime_policy_delegates_decode_shapes():
    policy = get_flash_attention_v4_sm120_runtime_policy(
        device_capability=(12, 0),
        deterministic=False,
    )
    assert policy.num_splits == 1
    assert policy.decode_num_splits == 0
    assert policy.decode_uses_static_max_seqlen_k

    deterministic_policy = get_flash_attention_v4_sm120_runtime_policy(
        device_capability=(12, 0),
        deterministic=True,
    )
    assert deterministic_policy.num_splits == 1
    assert deterministic_policy.decode_num_splits == 1
    assert deterministic_policy.decode_uses_static_max_seqlen_k

    future_policy = get_flash_attention_v4_sm120_runtime_policy(
        device_capability=(13, 0),
        deterministic=False,
    )
    assert future_policy.num_splits == 0
    assert future_policy.decode_num_splits == 0
    assert not future_policy.decode_uses_static_max_seqlen_k


def test_sm120_preallocated_output_contract_is_validated_before_launch():
    q = torch.empty((1, 1, 1, 32), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((1, 1, 1, 32), device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)

    with pytest.raises(ValueError, match="must not require gradients"):
        flash_attn_with_kvcache(
            q,
            k,
            v,
            out=torch.empty_like(q, requires_grad=True),
        )

    strided_out = torch.empty(
        (1, 1, 1, 64),
        device="cuda",
        dtype=torch.bfloat16,
    )[..., ::2]
    with pytest.raises(ValueError, match="must have stride 1"):
        flash_attn_with_kvcache(q, k, v, out=strided_out)


@pytest.mark.parametrize(
    (
        "head_dim",
        "head_dim_v",
        "visible_seqlen_k",
        "qhead_per_kvhead",
        "expected_tile_m",
    ),
    [
        pytest.param(64, 64, 256, 1, None, id="minimum-exclusive"),
        pytest.param(64, 64, 512, 8, None, id="hd64-short-mqa-fallback"),
        pytest.param(64, 64, 513, 8, 16, id="hd64-long-mqa"),
        pytest.param(128, 128, 512, 8, 32, id="hd128-short-mqa"),
        pytest.param(128, 128, 2048, 1, 16, id="hd128-mha"),
        pytest.param(64, 128, 2048, 1, None, id="asymmetric-fallback"),
        pytest.param(96, 96, 2048, 1, None, id="unqualified-fallback"),
    ],
)
def test_sm120_low_hd_decode_tile_qualification(
    head_dim,
    head_dim_v,
    visible_seqlen_k,
    qhead_per_kvhead,
    expected_tile_m,
):
    assert (
        low_hd_paged_decode_tile_m(
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            paged_kv=True,
            seqlen_q=1,
            visible_seqlen_k=visible_seqlen_k,
            qhead_per_kvhead=qhead_per_kvhead,
        )
        == expected_tile_m
    )


def test_sm120_decode_visible_k_uses_exact_local_window():
    assert (
        visible_decode_seqlen_k(
            8192,
            is_local=False,
            window_size_left=256,
            window_size_right=0,
        )
        == 8192
    )
    assert (
        visible_decode_seqlen_k(
            8192,
            is_local=True,
            window_size_left=256,
            window_size_right=0,
        )
        == 257
    )
    assert (
        visible_decode_seqlen_k(
            128,
            is_local=True,
            window_size_left=256,
            window_size_right=0,
        )
        == 128
    )


@cute.jit
def _coordinate_score_mod(
    score,
    batch_idx,
    head_idx,
    q_idx,
    kv_idx,
    seqlen_info,
    aux_tensors,
):
    """Exercise logical batch/head/token indices, including packed GQA."""
    return (
        score
        + batch_idx * 0.00390625
        + head_idx * 0.0078125
        + q_idx * 0.015625
        - kv_idx * 0.0078125
    )


@cute.jit
def _aux_score_mod(
    score,
    batch_idx,
    head_idx,
    q_idx,
    kv_idx,
    seqlen_info,
    aux_tensors,
):
    """Exercise runtime aux-tensor threading without shape-specific indexing."""
    return score + aux_tensors[0][0]


def _attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    batch_idx: int = 0,
    causal: bool = False,
    window_size: tuple[int | None, int | None] = (None, None),
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    sinks: torch.Tensor | None = None,
    score_mod: str | None = None,
    aux_score: float = 0.0,
    rel_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 attention output and head-major LSE for one sequence."""
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_kv_heads, _ = k.shape
    assert num_q_heads % num_kv_heads == 0
    scale = head_dim**-0.5 if softmax_scale is None else softmax_scale
    k = k.float().repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    v = v.float().repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) * scale

    q_idx = torch.arange(q_len, device=q.device)
    kv_idx = torch.arange(k_len, device=q.device)
    if score_mod == "coordinate":
        scores += (
            batch_idx * 0.00390625
            + torch.arange(num_q_heads, device=q.device)[:, None, None] * 0.0078125
            + q_idx[None, :, None] * 0.015625
            - kv_idx[None, None, :] * 0.0078125
        )
    elif score_mod == "aux":
        scores += aux_score
    elif score_mod is not None:
        raise ValueError(f"unknown score_mod reference: {score_mod}")

    relative_position = q_idx[:, None] + k_len - q_len - kv_idx[None, :]
    if rel_bias is not None:
        bias_index = relative_position.clamp(0, rel_bias.shape[-1] - 1)
        bias = (
            rel_bias.float()
            .permute(1, 0, 2)
            .gather(
                2,
                bias_index[None].expand(num_q_heads, -1, -1),
            )
        )
        bias.masked_fill_(
            ~((relative_position >= 0) & (relative_position < rel_bias.shape[-1]))[
                None
            ],
            0.0,
        )
        scores += bias
    if softcap > 0:
        scores = torch.tanh(scores / softcap) * softcap

    visible = torch.ones_like(relative_position, dtype=torch.bool)
    if causal:
        visible &= relative_position >= 0
    if window_size[0] is not None:
        visible &= relative_position <= window_size[0]
    if window_size[1] is not None:
        visible &= relative_position >= -window_size[1]
    scores.masked_fill_(~visible[None], -torch.inf)

    if sinks is None:
        probabilities = torch.softmax(scores, dim=-1)
        lse = torch.logsumexp(scores, dim=-1)
    else:
        row_max = torch.maximum(scores.amax(dim=-1), sinks.float()[:, None])
        weights = torch.exp(scores - row_max[:, :, None])
        denominator = weights.sum(dim=-1) + torch.exp(sinks.float()[:, None] - row_max)
        probabilities = weights / denominator[:, :, None]
        lse = torch.log(denominator) + row_max
    output = torch.einsum("hqk,khd->qhd", probabilities, v)
    return output, lse


def _assert_attention_close(
    output: torch.Tensor,
    reference: torch.Tensor,
    lse: torch.Tensor | None = None,
    lse_reference: torch.Tensor | None = None,
) -> None:
    error = (output.float() - reference.float()).abs()
    assert error.max().item() < 1e-2
    assert error.mean().item() < 5e-4
    if lse is not None:
        assert lse_reference is not None
        lse_error = (lse.float() - lse_reference.float()).abs()
        assert lse_error.max().item() < 2e-4


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    window_left: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 causal attention-with-sink output and LSE."""
    seq = q.shape[0]
    scale = q.shape[-1] ** -0.5
    scores = (
        torch.einsum(
            "qhd,kd->hqk",
            q.float(),
            k[:, 0].float(),
        )
        * scale
    )
    q_idx = torch.arange(seq, device=q.device)[:, None]
    kv_idx = torch.arange(seq, device=q.device)[None, :]
    mask = kv_idx <= q_idx
    if window_left is not None:
        mask &= kv_idx >= q_idx - window_left
    scores.masked_fill_(~mask[None], -torch.inf)

    row_max = torch.maximum(scores.amax(dim=-1), sinks.float()[:, None])
    weights = torch.exp(scores - row_max[:, :, None])
    denominator = weights.sum(dim=-1) + torch.exp(sinks.float()[:, None] - row_max)
    output = (
        torch.einsum(
            "hqk,kd->qhd",
            weights,
            v[:, 0].float(),
        )
        / denominator.T[:, :, None]
    )
    lse = torch.log(denominator) + row_max
    return output, lse


def _relative_bias_reference(
    q_parts: list[torch.Tensor],
    k_parts: list[torch.Tensor],
    v_parts: list[torch.Tensor],
    bias_parts: list[torch.Tensor],
    *,
    causal: bool,
    window_size: tuple[int | None, int | None],
    softcap: float = 0.0,
) -> torch.Tensor:
    outputs = []
    for q, k, v, rel_bias in zip(q_parts, k_parts, v_parts, bias_parts):
        q_len, k_len = q.shape[0], k.shape[0]
        k = k.float().repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        v = v.float().repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        scores = torch.einsum("qhd,khd->hqk", q.float(), k) * q.shape[-1] ** -0.5
        q_idx = torch.arange(q_len, device=q.device) + k_len - q_len
        kv_idx = torch.arange(k_len, device=q.device)
        rel_dist = q_idx[:, None] - kv_idx[None, :]
        bias_idx = rel_dist.clamp(0, rel_bias.shape[-1] - 1)
        bias = (
            rel_bias.float()
            .permute(1, 0, 2)
            .gather(
                2,
                bias_idx[None].expand(q.shape[1], -1, -1),
            )
        )
        bias.masked_fill_(
            ~((rel_dist >= 0) & (rel_dist < rel_bias.shape[-1]))[None],
            0.0,
        )
        scores += bias
        if softcap > 0:
            scores = torch.tanh(scores / softcap) * softcap
        visible = torch.ones_like(rel_dist, dtype=torch.bool)
        if causal:
            visible &= rel_dist >= 0
        if window_size[0] is not None:
            visible &= rel_dist <= window_size[0]
        if window_size[1] is not None:
            visible &= rel_dist >= -window_size[1]
        scores.masked_fill_(~visible[None], -torch.inf)
        outputs.append(torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), v))
    return torch.cat(outputs)


@pytest.mark.parametrize(
    (
        "dtype",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "head_dim_v",
        "causal",
        "window_size",
        "softcap",
        "has_sink",
        "pack_gqa",
        "preallocate_out",
    ),
    [
        pytest.param(
            torch.bfloat16,
            4,
            4,
            32,
            32,
            False,
            (None, None),
            0.0,
            False,
            False,
            False,
            id="bf16-mha-hd32-global",
        ),
        pytest.param(
            torch.float16,
            6,
            1,
            64,
            96,
            True,
            (None, None),
            0.0,
            True,
            True,
            True,
            id="fp16-mqa-hd64-hdv96-causal-sink",
        ),
        pytest.param(
            torch.bfloat16,
            8,
            2,
            96,
            64,
            False,
            (47, 11),
            4.0,
            False,
            True,
            False,
            id="bf16-gqa-hd96-hdv64-local-softcap",
        ),
        pytest.param(
            torch.bfloat16,
            8,
            2,
            128,
            128,
            True,
            (None, None),
            0.0,
            True,
            False,
            False,
            id="bf16-gqa-hd128-causal-sink-unpacked",
        ),
        pytest.param(
            torch.bfloat16,
            6,
            1,
            192,
            128,
            False,
            (None, None),
            0.0,
            False,
            None,
            False,
            id="bf16-mqa-hd192-hdv128-global-auto-pack",
        ),
        pytest.param(
            torch.float16,
            4,
            4,
            128,
            64,
            False,
            (63, 7),
            0.0,
            False,
            False,
            True,
            id="fp16-mha-hd128-hdv64-local-out",
        ),
    ],
)
def test_sm120_dense_feature_matrix_matches_reference(
    dtype,
    num_q_heads,
    num_kv_heads,
    head_dim,
    head_dim_v,
    causal,
    window_size,
    softcap,
    has_sink,
    pack_gqa,
    preallocate_out,
):
    """Cover dense MHA/GQA/MQA, dtype, shape, mask, sink, and output modes."""
    torch.manual_seed(20260801 + head_dim + head_dim_v)
    batch_size, q_len, k_len = 2, 37, 141
    q = torch.randn(
        batch_size,
        q_len,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    k = torch.randn(
        batch_size,
        k_len,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    v = torch.randn(
        batch_size,
        k_len,
        num_kv_heads,
        head_dim_v,
        device="cuda",
        dtype=dtype,
    )
    sinks = (
        torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)
        if has_sink
        else None
    )
    out_buffer = torch.empty(
        batch_size,
        q_len,
        num_q_heads,
        head_dim_v,
        device="cuda",
        dtype=dtype,
    )
    output, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        sinks=sinks,
        pack_gqa=pack_gqa,
        return_softmax_lse=True,
        out=out_buffer if preallocate_out else None,
    )
    references = [
        _attention_reference(
            q[batch_idx],
            k[batch_idx],
            v[batch_idx],
            batch_idx=batch_idx,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            sinks=sinks,
        )
        for batch_idx in range(batch_size)
    ]
    output_reference = torch.stack([reference[0] for reference in references])
    lse_reference = torch.stack([reference[1] for reference in references])
    _assert_attention_close(output, output_reference, lse, lse_reference)
    if preallocate_out:
        assert output.data_ptr() == out_buffer.data_ptr()


@pytest.mark.parametrize(
    (
        "dtype",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "head_dim_v",
        "causal",
        "window_size",
        "pack_gqa",
        "num_splits",
    ),
    [
        pytest.param(
            torch.bfloat16,
            4,
            4,
            64,
            64,
            False,
            (None, None),
            False,
            1,
            id="bf16-mha-hd64-global",
        ),
        pytest.param(
            torch.float16,
            8,
            2,
            80,
            48,
            True,
            (None, None),
            True,
            2,
            id="fp16-gqa-hd80-hdv48-causal-splitkv",
        ),
        pytest.param(
            torch.bfloat16,
            6,
            1,
            128,
            96,
            False,
            (63, 9),
            None,
            1,
            id="bf16-mqa-hd128-hdv96-local-auto-pack",
        ),
        pytest.param(
            torch.bfloat16,
            8,
            2,
            192,
            128,
            True,
            (127, 0),
            False,
            2,
            id="bf16-gqa-hd192-hdv128-causal-local-splitkv",
        ),
    ],
)
def test_sm120_varlen_feature_matrix_matches_reference(
    dtype,
    num_q_heads,
    num_kv_heads,
    head_dim,
    head_dim_v,
    causal,
    window_size,
    pack_gqa,
    num_splits,
):
    """Cover ragged batches across topology, asymmetric dimensions, and SplitKV."""
    torch.manual_seed(20260802 + head_dim + head_dim_v)
    q_lengths = (19, 53)
    k_lengths = (79, 151)
    q_parts = [
        torch.randn(
            length,
            num_q_heads,
            head_dim,
            device="cuda",
            dtype=dtype,
        )
        for length in q_lengths
    ]
    k_parts = [
        torch.randn(
            length,
            num_kv_heads,
            head_dim,
            device="cuda",
            dtype=dtype,
        )
        for length in k_lengths
    ]
    v_parts = [
        torch.randn(
            length,
            num_kv_heads,
            head_dim_v,
            device="cuda",
            dtype=dtype,
        )
        for length in k_lengths
    ]
    cu_seqlens_q = torch.tensor(
        [0, q_lengths[0], sum(q_lengths)],
        device="cuda",
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0, k_lengths[0], sum(k_lengths)],
        device="cuda",
        dtype=torch.int32,
    )
    output, lse = flash_attn_varlen_func(
        torch.cat(q_parts),
        torch.cat(k_parts),
        torch.cat(v_parts),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        causal=causal,
        window_size=window_size,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        return_softmax_lse=True,
    )
    references = [
        _attention_reference(
            q,
            k,
            v,
            batch_idx=batch_idx,
            causal=causal,
            window_size=window_size,
        )
        for batch_idx, (q, k, v) in enumerate(zip(q_parts, k_parts, v_parts))
    ]
    output_reference = torch.cat([reference[0] for reference in references])
    lse_reference = torch.cat([reference[1] for reference in references], dim=1)
    _assert_attention_close(output, output_reference, lse, lse_reference)


def test_sm120_dense_seqused_qk_matches_reference():
    """Dense storage with per-batch used lengths must ignore allocated padding."""
    torch.manual_seed(20260803)
    batch_size, max_q, max_k = 3, 64, 160
    num_q_heads, num_kv_heads = 8, 2
    head_dim, head_dim_v = 64, 96
    q_lengths = (17, 41, 63)
    k_lengths = (79, 129, 159)
    q = torch.randn(
        batch_size,
        max_q,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        batch_size,
        max_k,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        batch_size,
        max_k,
        num_kv_heads,
        head_dim_v,
        device="cuda",
        dtype=torch.bfloat16,
    )
    seqused_q = torch.tensor(q_lengths, device="cuda", dtype=torch.int32)
    seqused_k = torch.tensor(k_lengths, device="cuda", dtype=torch.int32)
    output, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        causal=True,
        window_size=(95, 0),
        pack_gqa=True,
        return_softmax_lse=True,
    )
    for batch_idx, (q_len, k_len) in enumerate(zip(q_lengths, k_lengths)):
        reference, lse_reference = _attention_reference(
            q[batch_idx, :q_len],
            k[batch_idx, :k_len],
            v[batch_idx, :k_len],
            batch_idx=batch_idx,
            causal=True,
            window_size=(95, 0),
        )
        _assert_attention_close(
            output[batch_idx, :q_len],
            reference,
            lse[batch_idx, :, :q_len],
            lse_reference,
        )


@pytest.mark.parametrize(
    ("score_mod", "reference_kind", "has_aux"),
    [
        pytest.param(
            _coordinate_score_mod,
            "coordinate",
            False,
            id="logical-coordinates",
        ),
        pytest.param(_aux_score_mod, "aux", True, id="aux-tensor"),
    ],
)
def test_sm120_packed_gqa_score_mod_matches_reference(
    score_mod,
    reference_kind,
    has_aux,
):
    """Score modifiers must see logical packed-GQA indices and runtime aux data."""
    torch.manual_seed(20260804)
    q_lengths = (17, 35)
    k_lengths = (79, 143)
    num_q_heads, num_kv_heads, head_dim = 8, 2, 64
    q_parts = [
        torch.randn(
            length,
            num_q_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for length in q_lengths
    ]
    k_parts = [
        torch.randn(
            length,
            num_kv_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for length in k_lengths
    ]
    v_parts = [torch.randn_like(k) for k in k_parts]
    cu_seqlens_q = torch.tensor(
        [0, q_lengths[0], sum(q_lengths)],
        device="cuda",
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0, k_lengths[0], sum(k_lengths)],
        device="cuda",
        dtype=torch.int32,
    )
    aux_value = 0.125
    aux_tensors = (
        [torch.tensor([aux_value], device="cuda", dtype=torch.float32)]
        if has_aux
        else None
    )
    output, lse = flash_attn_varlen_func(
        torch.cat(q_parts),
        torch.cat(k_parts),
        torch.cat(v_parts),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        causal=True,
        pack_gqa=True,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        return_softmax_lse=True,
    )
    references = [
        _attention_reference(
            q,
            k,
            v,
            batch_idx=batch_idx,
            causal=True,
            score_mod=reference_kind,
            aux_score=aux_value,
        )
        for batch_idx, (q, k, v) in enumerate(zip(q_parts, k_parts, v_parts))
    ]
    output_reference = torch.cat([reference[0] for reference in references])
    lse_reference = torch.cat([reference[1] for reference in references], dim=1)
    _assert_attention_close(output, output_reference, lse, lse_reference)

    if has_aux:
        updated_aux_value = -0.25
        aux_tensors[0].fill_(updated_aux_value)
        updated_output, updated_lse = flash_attn_varlen_func(
            torch.cat(q_parts),
            torch.cat(k_parts),
            torch.cat(v_parts),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(q_lengths),
            max_seqlen_k=max(k_lengths),
            causal=True,
            pack_gqa=True,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
            return_softmax_lse=True,
        )
        updated_references = [
            _attention_reference(
                q,
                k,
                v,
                batch_idx=batch_idx,
                causal=True,
                score_mod=reference_kind,
                aux_score=updated_aux_value,
            )
            for batch_idx, (q, k, v) in enumerate(zip(q_parts, k_parts, v_parts))
        ]
        updated_output_reference = torch.cat(
            [reference[0] for reference in updated_references]
        )
        updated_lse_reference = torch.cat(
            [reference[1] for reference in updated_references],
            dim=1,
        )
        _assert_attention_close(
            updated_output,
            updated_output_reference,
            updated_lse,
            updated_lse_reference,
        )


@pytest.mark.parametrize(
    (
        "dtype",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "head_dim_v",
        "causal",
        "window_size",
        "softcap",
        "has_sink",
        "pack_gqa",
        "num_splits",
    ),
    [
        pytest.param(
            torch.bfloat16,
            8,
            2,
            64,
            96,
            True,
            (None, None),
            0.0,
            True,
            True,
            1,
            id="bf16-gqa-hd64-hdv96-causal-sink",
        ),
        pytest.param(
            torch.float16,
            4,
            4,
            128,
            64,
            False,
            (255, 0),
            5.0,
            False,
            False,
            2,
            id="fp16-mha-hd128-hdv64-local-softcap-splitkv",
        ),
    ],
)
def test_sm120_relative_bias_dense_matches_reference(
    dtype,
    num_q_heads,
    num_kv_heads,
    head_dim,
    head_dim_v,
    causal,
    window_size,
    softcap,
    has_sink,
    pack_gqa,
    num_splits,
):
    """Dense sheared bias composes with sink, softcap, SplitKV, LSE, and out."""
    torch.manual_seed(20260805 + head_dim + head_dim_v)
    batch_size, q_len, k_len, rel_extent = 2, 33, 193, 256
    q = torch.randn(
        batch_size,
        q_len,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    k = torch.randn(
        batch_size,
        k_len,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    v = torch.randn(
        batch_size,
        k_len,
        num_kv_heads,
        head_dim_v,
        device="cuda",
        dtype=dtype,
    )
    rel_bias = (
        0.1
        * torch.randn(
            batch_size,
            q_len,
            num_q_heads,
            rel_extent,
            device="cuda",
        )
    ).to(dtype)
    sinks = (
        torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)
        if has_sink
        else None
    )
    out_buffer = torch.empty(
        batch_size,
        q_len,
        num_q_heads,
        head_dim_v,
        device="cuda",
        dtype=dtype,
    )
    output, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        sinks=sinks,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        rel_bias=rel_bias,
        return_softmax_lse=True,
        out=out_buffer,
    )
    references = [
        _attention_reference(
            q[batch_idx],
            k[batch_idx],
            v[batch_idx],
            batch_idx=batch_idx,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            sinks=sinks,
            rel_bias=rel_bias[batch_idx],
        )
        for batch_idx in range(batch_size)
    ]
    output_reference = torch.stack([reference[0] for reference in references])
    lse_reference = torch.stack([reference[1] for reference in references])
    _assert_attention_close(output, output_reference, lse, lse_reference)
    assert output.data_ptr() == out_buffer.data_ptr()


def test_sm120_relative_bias_cuda_graph_replays_and_eager_remains_reusable():
    """The shearing producer and attention consumer must capture as one graph."""
    torch.manual_seed(20260806)
    q_len, k_len, rel_extent = 33, 193, 256
    num_q_heads, num_kv_heads, head_dim = 8, 2, 64
    q = torch.randn(
        1,
        q_len,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        1,
        k_len,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    rel_bias = (
        0.1
        * torch.randn(
            1,
            q_len,
            num_q_heads,
            rel_extent,
            device="cuda",
        )
    ).to(torch.bfloat16)
    rel_bias_prep_cache = {}

    def run():
        return flash_attn_varlen_func(
            q,
            k,
            v,
            causal=True,
            pack_gqa=True,
            rel_bias=rel_bias,
            rel_bias_prep_cache=rel_bias_prep_cache,
        )

    eager_before = run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    eager_after = run()
    torch.cuda.synchronize()

    reference, _ = _attention_reference(
        q[0],
        k[0],
        v[0],
        causal=True,
        rel_bias=rel_bias[0],
    )
    torch.testing.assert_close(eager_before, eager_after, atol=0.0, rtol=0.0)
    torch.testing.assert_close(graph_output, eager_before, atol=0.0, rtol=0.0)
    _assert_attention_close(graph_output[0], reference)


@pytest.mark.parametrize(
    (
        "dtype",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "head_dim_v",
        "q_len",
        "page_size",
        "causal",
        "window_size",
        "pack_gqa",
        "num_splits",
    ),
    [
        pytest.param(
            torch.float16,
            4,
            4,
            32,
            32,
            7,
            64,
            True,
            (None, None),
            False,
            1,
            id="fp16-mha-hd32-page64-causal",
        ),
        pytest.param(
            torch.bfloat16,
            8,
            2,
            64,
            96,
            33,
            128,
            False,
            (127, 15),
            True,
            2,
            id="bf16-gqa-hd64-hdv96-page128-local-splitkv",
        ),
        pytest.param(
            torch.bfloat16,
            6,
            1,
            128,
            64,
            5,
            64,
            False,
            (None, None),
            None,
            2,
            id="bf16-mqa-hd128-hdv64-page64-global-splitkv",
        ),
        pytest.param(
            torch.bfloat16,
            8,
            2,
            96,
            128,
            17,
            32,
            True,
            (None, None),
            False,
            1,
            id="bf16-gqa-hd96-hdv128-page32-causal-unpacked",
        ),
        pytest.param(
            torch.bfloat16,
            8,
            2,
            64,
            64,
            1,
            64,
            True,
            (None, None),
            True,
            0,
            id="bf16-gqa-hd64-page64-decode-auto-split",
        ),
        pytest.param(
            torch.bfloat16,
            8,
            1,
            128,
            128,
            1,
            64,
            True,
            (None, None),
            True,
            0,
            id="bf16-mqa-hd128-page64-decode-auto-split",
        ),
    ],
)
def test_sm120_paged_kv_feature_matrix_matches_reference(
    dtype,
    num_q_heads,
    num_kv_heads,
    head_dim,
    head_dim_v,
    q_len,
    page_size,
    causal,
    window_size,
    pack_gqa,
    num_splits,
):
    """Cover paged MHA/GQA/MQA outside the HD256 decode specialization."""
    torch.manual_seed(20260807 + head_dim + head_dim_v + page_size)
    batch_size, pages_per_sequence = 2, 6
    max_seqlen_k = pages_per_sequence * page_size
    k_lengths = (max_seqlen_k - page_size - 11, max_seqlen_k - 1)
    q = torch.randn(
        batch_size,
        q_len,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    k_cache = torch.randn(
        batch_size * pages_per_sequence,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    v_cache = torch.randn(
        batch_size * pages_per_sequence,
        page_size,
        num_kv_heads,
        head_dim_v,
        device="cuda",
        dtype=dtype,
    )
    page_table = (
        torch.randperm(
            batch_size * pages_per_sequence,
            device="cuda",
            dtype=torch.int64,
        )
        .to(torch.int32)
        .view(batch_size, pages_per_sequence)
    )
    seqused_k = torch.tensor(k_lengths, device="cuda", dtype=torch.int32)
    output, lse = flash_attn_varlen_func(
        q,
        k_cache,
        v_cache,
        seqused_k=seqused_k,
        max_seqlen_q=q_len,
        max_seqlen_k=max_seqlen_k,
        page_table=page_table,
        causal=causal,
        window_size=window_size,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        return_softmax_lse=True,
    )
    references = []
    for batch_idx, k_len in enumerate(k_lengths):
        pages = page_table[batch_idx]
        k = k_cache.index_select(0, pages).flatten(0, 1)[:k_len]
        v = v_cache.index_select(0, pages).flatten(0, 1)[:k_len]
        references.append(
            _attention_reference(
                q[batch_idx],
                k,
                v,
                batch_idx=batch_idx,
                causal=causal,
                window_size=window_size,
            )
        )
    output_reference = torch.stack([reference[0] for reference in references])
    lse_reference = torch.stack([reference[1] for reference in references])
    _assert_attention_close(output, output_reference, lse, lse_reference)

    if num_splits > 1:
        unsplit_output = flash_attn_varlen_func(
            q,
            k_cache,
            v_cache,
            seqused_k=seqused_k,
            max_seqlen_q=q_len,
            max_seqlen_k=max_seqlen_k,
            page_table=page_table,
            causal=causal,
            window_size=window_size,
            num_splits=1,
            pack_gqa=pack_gqa,
        )
        torch.testing.assert_close(output, unsplit_output, atol=4e-3, rtol=0.0)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_kv_heads", [8, 2, 1])
def test_sm120_low_hd_paged_decode_reuses_cached_host_plan(
    monkeypatch,
    head_dim,
    num_kv_heads,
):
    """Low-HD MHA/GQA/MQA decode must reuse its compiled TVM-FFI plan."""
    sm120_forward_host.clear_launch_plans()
    cache_hits = []
    original_try_paged_decode = sm120_forward_host.try_paged_decode

    def record_cache_hit(**kwargs):
        result = original_try_paged_decode(**kwargs)
        cache_hits.append(result is not None)
        return result

    monkeypatch.setattr(
        sm120_forward_host,
        "try_paged_decode",
        record_cache_hit,
    )
    torch.manual_seed(20260731 + head_dim + num_kv_heads)
    batch_size, num_q_heads = 2, 8
    max_seqlen, page_size = 512, 64
    pages_per_request = max_seqlen // page_size
    q = torch.randn(
        batch_size,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        batch_size * pages_per_request,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.arange(
        batch_size * pages_per_request,
        device="cuda",
        dtype=torch.int32,
    ).view(batch_size, pages_per_request)
    cache_seqlens = torch.full(
        (batch_size,),
        max_seqlen,
        device="cuda",
        dtype=torch.int32,
    )
    cu_seqlens_q = torch.arange(
        batch_size + 1,
        device="cuda",
        dtype=torch.int32,
    )
    out = torch.empty_like(q)

    def run():
        return flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen,
            causal=True,
            num_splits=0,
            pack_gqa=None,
            out=out,
        )

    first = run().clone()
    second = run().clone()
    torch.cuda.synchronize()

    assert not any(cache_hits[:-1])
    assert cache_hits[-1]
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("head_dim", "pack_gqa", "causal", "window_size", "softcap"),
    [
        pytest.param(32, True, True, (None, None), 0.0, id="hd32-packed-causal"),
        pytest.param(
            96,
            False,
            True,
            (None, None),
            5.0,
            id="hd96-unpacked-causal-softcap",
        ),
        pytest.param(128, True, False, (255, 0), 0.0, id="hd128-packed-local"),
    ],
)
def test_sm120_relative_bias_varlen_matches_reference(
    head_dim,
    pack_gqa,
    causal,
    window_size,
    softcap,
):
    """Relative bias must follow logical Q/K positions for every SM120 tile."""
    torch.manual_seed(20260731 + head_dim)
    q_lengths = (47, 73)
    k_lengths = (193, 321)
    num_q_heads, num_kv_heads, rel_extent = 8, 2, 256
    q_parts = [
        torch.randn(
            length,
            num_q_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for length in q_lengths
    ]
    k_parts = [
        torch.randn(
            length,
            num_kv_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for length in k_lengths
    ]
    v_parts = [torch.randn_like(k) for k in k_parts]
    bias_parts = [
        (
            0.1
            * torch.randn(
                length,
                num_q_heads,
                rel_extent,
                device="cuda",
            )
        ).to(torch.bfloat16)
        for length in q_lengths
    ]
    cu_seqlens_q = torch.tensor(
        [0, q_lengths[0], sum(q_lengths)],
        device="cuda",
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0, k_lengths[0], sum(k_lengths)],
        device="cuda",
        dtype=torch.int32,
    )

    output = flash_attn_varlen_func(
        torch.cat(q_parts),
        torch.cat(k_parts),
        torch.cat(v_parts),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        pack_gqa=pack_gqa,
        rel_bias=torch.cat(bias_parts),
    )
    reference = _relative_bias_reference(
        q_parts,
        k_parts,
        v_parts,
        bias_parts,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
    )
    error = (output.float() - reference.float()).abs()
    assert error.max().item() < 1e-2
    assert error.mean().item() < 5e-4


def test_sm120_relative_bias_paged_splitkv_is_cache_order_independent():
    """Paged relative coordinates and SplitKV must not collide in the JIT cache."""
    torch.manual_seed(20260731)
    q_lengths = (9, 19)
    k_lengths = (383, 509)
    num_q_heads, num_kv_heads, head_dim = 8, 2, 128
    rel_extent, page_size, pages_per_seq = 256, 128, 4
    q_parts = [
        torch.randn(
            length,
            num_q_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for length in q_lengths
    ]
    bias_parts = [
        (
            0.1
            * torch.randn(
                length,
                num_q_heads,
                rel_extent,
                device="cuda",
            )
        ).to(torch.bfloat16)
        for length in q_lengths
    ]
    k_cache = torch.randn(
        len(q_lengths) * pages_per_seq,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.tensor(
        [[3, 0, 7, 1], [6, 2, 5, 4]],
        device="cuda",
        dtype=torch.int32,
    )
    k_parts = [
        k_cache.index_select(0, pages).flatten(0, 1)[:length]
        for pages, length in zip(page_table, k_lengths)
    ]
    v_parts = [
        v_cache.index_select(0, pages).flatten(0, 1)[:length]
        for pages, length in zip(page_table, k_lengths)
    ]
    q = torch.cat(q_parts)
    rel_bias = torch.cat(bias_parts)
    cu_seqlens_q = torch.tensor(
        [0, q_lengths[0], sum(q_lengths)],
        device="cuda",
        dtype=torch.int32,
    )
    seqused_k = torch.tensor(k_lengths, device="cuda", dtype=torch.int32)

    def run(num_splits):
        return flash_attn_varlen_func(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            page_table=page_table,
            max_seqlen_q=max(q_lengths),
            max_seqlen_k=pages_per_seq * page_size,
            causal=False,
            window_size=(rel_extent - 1, 0),
            num_splits=num_splits,
            pack_gqa=True,
            rel_bias=rel_bias,
        )

    split_output = run(2)
    unsplit_output = run(1)
    split_output_again = run(2)
    reference = _relative_bias_reference(
        q_parts,
        k_parts,
        v_parts,
        bias_parts,
        causal=False,
        window_size=(rel_extent - 1, 0),
    )

    torch.testing.assert_close(
        split_output,
        split_output_again,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        split_output,
        unsplit_output,
        atol=2e-3,
        rtol=0.0,
    )
    error = (split_output.float() - reference.float()).abs()
    assert error.max().item() < 1e-2
    assert error.mean().item() < 5e-4


@pytest.mark.parametrize("window_left", [None, 250])
def test_sm120_varlen_mqa_hd256_learnable_sink(window_left):
    """Cover Q6/KV1 head-dim-256 global and local prefill shapes."""
    torch.manual_seed(1234)
    seq, num_q_heads, head_dim = 512, 6, 256
    q = torch.randn(seq, num_q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(seq, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sinks = torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, seq], dtype=torch.int32, device="cuda")
    window_size = (None, None) if window_left is None else (window_left, 0)
    out_ref, lse_ref = _reference(q, k, v, sinks, window_left)

    outputs = []
    for pack_gqa in (False, True, None):
        out, lse = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq,
            max_seqlen_k=seq,
            softmax_scale=head_dim**-0.5,
            causal=True,
            window_size=window_size,
            sinks=sinks,
            pack_gqa=pack_gqa,
            return_softmax_lse=True,
        )
        output_error = (out.float() - out_ref).abs()
        lse_error = (lse - lse_ref).abs()
        assert output_error.max().item() < 1e-2
        assert output_error.mean().item() < 5e-4
        assert lse_error.max().item() < 5e-5
        outputs.append(out)

    torch.testing.assert_close(outputs[0], outputs[1], atol=0.0, rtol=0.0)
    torch.testing.assert_close(outputs[1], outputs[2], atol=0.0, rtol=0.0)


def test_sm120_forward_only_varlen_cache_is_pack_order_independent(monkeypatch):
    """MHA/GQA plans must be distinct and accept dynamic sequence offsets."""
    sm120_forward_host.clear_launch_plans()
    cache_hits = []
    original_try_varlen = sm120_forward_host.try_varlen

    def record_cache_hit(**kwargs):
        result = original_try_varlen(**kwargs)
        cache_hits.append(result is not None)
        return result

    monkeypatch.setattr(sm120_forward_host, "try_varlen", record_cache_hit)
    torch.manual_seed(1234)
    lengths = (128, 111, 97)
    total, num_q_heads, head_dim = sum(lengths), 8, 64
    q = torch.randn(
        total,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    inputs = {}
    for name, num_kv_heads in (("mha", 8), ("gqa", 2)):
        k = torch.randn(
            total,
            num_kv_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        v = torch.randn_like(k)
        reference = torch.cat(
            [
                _attention_reference(
                    q[start:end],
                    k[start:end],
                    v[start:end],
                    causal=True,
                )[0]
                for start, end in zip(offsets[:-1], offsets[1:])
            ]
        )
        inputs[name] = (k, v, reference)

    for name in ("mha", "gqa", "mha"):
        k, v, reference = inputs[name]
        output = torch.empty_like(q)
        for _ in range(2):
            result = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max(lengths),
                max_seqlen_k=max(lengths),
                softmax_scale=head_dim**-0.5,
                causal=True,
                pack_gqa=name == "gqa",
                out=output,
            )
            assert result.data_ptr() == output.data_ptr()
            _assert_attention_close(result, reference)

    assert cache_hits == [False, True, False, True, True, True]

    # The compiled plan keys tensor metadata, not the contents of cu_seqlens.
    # Repartition the same storage while retaining shape, total, and maximum.
    alternate_lengths = (97, 111, 128)
    alternate_offsets = [0]
    for length in alternate_lengths:
        alternate_offsets.append(alternate_offsets[-1] + length)
    alternate_cu_seqlens = torch.tensor(
        alternate_offsets,
        dtype=torch.int32,
        device="cuda",
    )
    k, v, _ = inputs["mha"]
    alternate_reference = torch.cat(
        [
            _attention_reference(
                q[start:end],
                k[start:end],
                v[start:end],
                causal=True,
            )[0]
            for start, end in zip(alternate_offsets[:-1], alternate_offsets[1:])
        ]
    )
    alternate_output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=alternate_cu_seqlens,
        cu_seqlens_k=alternate_cu_seqlens,
        max_seqlen_q=max(alternate_lengths),
        max_seqlen_k=max(alternate_lengths),
        softmax_scale=head_dim**-0.5,
        causal=True,
        pack_gqa=False,
    )
    assert cache_hits[-1]
    _assert_attention_close(alternate_output, alternate_reference)


def test_sm120_varlen_padding_ctas_are_inert_across_tile_specializations():
    """Padding CTAs must not consume stale SMEM or write another batch's output."""
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    num_q_heads, head_dim = 6, 256

    def make_inputs(lengths, seed):
        torch.manual_seed(seed)
        total = sum(lengths)
        q = torch.randn(
            total,
            num_q_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = torch.randn(total, 1, head_dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        sinks = torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)
        cuts = [0]
        for length in lengths:
            cuts.append(cuts[-1] + length)
        cu_seqlens = torch.tensor(cuts, device="cuda", dtype=torch.int32)
        return q, k, v, sinks, cu_seqlens

    def run(inputs, lengths, pack_gqa):
        q, k, v, sinks, cu_seqlens = inputs
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max(lengths),
            max_seqlen_k=max(lengths),
            softmax_scale=head_dim**-0.5,
            causal=True,
            sinks=sinks,
            pack_gqa=pack_gqa,
            return_softmax_lse=True,
        )

    # Exercise the three SM-normalized selector regions before the padding
    # case. This is the order that exposed stale shared-memory state.
    rows_per_sm = (58, 84, 100)
    for seed, ratio in enumerate(rows_per_sm):
        seq = math.ceil(ratio * num_sms / num_q_heads)
        inputs = make_inputs([seq], seed)
        for pack_gqa in (False, True, False):
            run(inputs, [seq], pack_gqa)
        torch.cuda.synchronize()

    # At 64 query rows/SM, two batches select M64 while the conservative
    # varlen grid contains padding CTAs.
    seq = math.ceil(32 * num_sms / num_q_heads)
    lengths = [seq, seq]
    inputs = make_inputs(lengths, 10)
    q, k, v, sinks, _ = inputs
    references = [
        _reference(
            q[start : start + seq],
            k[start : start + seq],
            v[start : start + seq],
            sinks,
            None,
        )
        for start in (0, seq)
    ]
    out_ref = torch.cat([reference[0] for reference in references], dim=0)
    lse_ref = torch.cat([reference[1] for reference in references], dim=1)

    for pack_gqa in (False, True, None):
        out, lse = run(inputs, lengths, pack_gqa)
        output_error = (out.float() - out_ref).abs()
        lse_error = (lse - lse_ref).abs()
        assert output_error.max().item() < 1e-2
        assert output_error.mean().item() < 5e-4
        assert lse_error.max().item() < 5e-5


@pytest.mark.parametrize("window_left", [None, 192])
def test_sm120_paged_decode_ragged_splits_are_cache_order_independent(
    monkeypatch,
    window_left,
):
    """Ragged SplitKV must remain correct across uniform/ragged cache reuse."""
    sm120_forward_host.clear_launch_plans()
    cache_hits = []
    original_try_paged_decode = sm120_forward_host.try_paged_decode

    def record_cache_hit(**kwargs):
        result = original_try_paged_decode(**kwargs)
        cache_hits.append(result is not None)
        return result

    monkeypatch.setattr(
        sm120_forward_host,
        "try_paged_decode",
        record_cache_hit,
    )
    torch.manual_seed(1234)
    batch_size, num_q_heads, head_dim = 4, 6, 256
    max_seqlen, page_size = 1024, 64
    pages_per_request = max_seqlen // page_size
    q = torch.randn(
        batch_size,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        batch_size * pages_per_request,
        page_size,
        1,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.arange(
        batch_size * pages_per_request,
        device="cuda",
        dtype=torch.int32,
    ).view(batch_size, pages_per_request)
    cu_seqlens_q = torch.arange(
        batch_size + 1,
        device="cuda",
        dtype=torch.int32,
    )
    sinks = torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)

    def run(lengths):
        return flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=torch.tensor(
                lengths,
                device="cuda",
                dtype=torch.int32,
            ),
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen,
            causal=True,
            window_size=((None, None) if window_left is None else (window_left, 0)),
            num_splits=4,
            pack_gqa=True,
            sinks=sinks,
        )

    uniform_lengths = [max_seqlen] * batch_size
    # Exercise both sides of every 64-token tile boundary. N-distributed QK
    # has a different accumulator-column layout, so its mask must not use the
    # ordinary QK path's R2P column mapping.
    ragged_lengths = [1000, 513, 257, 63]
    uniform_first = run(uniform_lengths)
    ragged_first = run(ragged_lengths)
    ragged_second = run(ragged_lengths)
    uniform_second = run(uniform_lengths)

    # The public host fast path and the generic compile path may both probe an
    # empty cache on the first call. Once registered, every reuse must hit.
    assert not any(cache_hits[:-3])
    assert cache_hits[-3:] == [True, True, True]
    torch.testing.assert_close(ragged_first, ragged_second, atol=0.0, rtol=0.0)
    torch.testing.assert_close(uniform_first, uniform_second, atol=0.0, rtol=0.0)

    reference = torch.empty_like(ragged_first, dtype=torch.float32)
    scale = head_dim**-0.5
    for batch_idx, length in enumerate(ragged_lengths):
        pages = page_table[batch_idx]
        start = 0 if window_left is None else max(0, length - 1 - window_left)
        k = k_cache.index_select(0, pages).flatten(0, 1)[start:length, 0].float()
        v = v_cache.index_select(0, pages).flatten(0, 1)[start:length, 0].float()
        scores = q[batch_idx].float() @ k.T * scale
        row_max = torch.maximum(scores.amax(dim=-1), sinks.float())
        weights = torch.exp(scores - row_max[:, None])
        denominator = weights.sum(dim=-1) + torch.exp(sinks.float() - row_max)
        reference[batch_idx] = weights @ v / denominator[:, None]

    error = (ragged_first.float() - reference).abs()
    assert error.max().item() < 1e-2
    assert error.mean().item() < 5e-4


def test_sm120_paged_decode_transpose_is_cache_order_independent():
    """Gather and page-TMA transpose must not share a compiled specialization."""
    torch.manual_seed(20260729)
    batch_size, num_q_heads, head_dim = 1, 6, 256
    max_seqlen, page_size = 2048, 64
    num_pages = max_seqlen // page_size
    q = torch.randn(
        batch_size,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        num_pages,
        page_size,
        1,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.randperm(
        num_pages,
        device="cuda",
        dtype=torch.int64,
    ).to(torch.int32)[None]
    cache_seqlens = torch.tensor(
        [max_seqlen],
        device="cuda",
        dtype=torch.int32,
    )
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    sinks = torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)

    def run(num_splits):
        return flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen,
            causal=True,
            num_splits=num_splits,
            pack_gqa=True,
            sinks=sinks,
        )

    # S32 owns one KV tile per CTA and selects gather. S16 owns two and
    # selects the full-transpose page-TMA class. Alternate both compile/cache
    # entries in the order that previously exposed an illegal access.
    normal_first = run(32)
    transpose_after = run(16)
    transpose_repeat = run(16)
    normal_after = run(32)
    torch.cuda.synchronize()

    torch.testing.assert_close(normal_first, normal_after, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        transpose_after,
        transpose_repeat,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        normal_first,
        transpose_after,
        atol=2e-3,
        rtol=0.0,
    )

    pages = page_table[0]
    k = k_cache.index_select(0, pages).flatten(0, 1)[:, 0].float()
    v = v_cache.index_select(0, pages).flatten(0, 1)[:, 0].float()
    scores = q[0].float() @ k.T * head_dim**-0.5
    row_max = torch.maximum(scores.amax(dim=-1), sinks.float())
    weights = torch.exp(scores - row_max[:, None])
    denominator = weights.sum(dim=-1) + torch.exp(sinks.float() - row_max)
    reference = weights @ v / denominator[:, None]
    error = (transpose_after[0].float() - reference).abs()
    assert error.max().item() < 1e-2
    assert error.mean().item() < 5e-4


@pytest.mark.parametrize(
    ("num_q_heads", "pack_gqa", "expected_transpose", "expected_split_qk"),
    [
        pytest.param(6, True, True, False, id="transpose"),
        pytest.param(16, True, False, True, id="split-qk"),
        pytest.param(1, None, False, True, id="auto-mha-split-qk"),
        pytest.param(6, False, False, False, id="single-qk"),
    ],
)
def test_sm120_paged_decode_graph_pdl_is_correct_and_eager_reusable(
    monkeypatch,
    num_q_heads,
    pack_gqa,
    expected_transpose,
    expected_split_qk,
):
    """Every SplitKV dataflow must safely launch its captured combine early."""
    sm120_forward_host.clear_launch_plans()
    cache_hits = []
    original_try_paged_decode = sm120_forward_host.try_paged_decode

    def record_cache_hit(**kwargs):
        result = original_try_paged_decode(**kwargs)
        cache_hits.append(result is not None)
        return result

    monkeypatch.setattr(
        sm120_forward_host,
        "try_paged_decode",
        record_cache_hit,
    )
    torch.manual_seed(20260730)
    batch_size, head_dim = 1, 256
    max_seqlen, page_size = 2048, 64
    num_pages = max_seqlen // page_size
    captured_plans = []
    original_resolve_plan = Sm120ForwardHost.resolve_plan

    def recording_resolve_plan(**kwargs):
        plan = original_resolve_plan(**kwargs)
        if kwargs["is_stream_capturing"]:
            captured_plans.append(plan)
        return plan

    monkeypatch.setattr(
        Sm120ForwardHost,
        "resolve_plan",
        staticmethod(recording_resolve_plan),
    )
    q = torch.randn(
        batch_size,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        num_pages,
        page_size,
        1,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.randperm(
        num_pages,
        device="cuda",
        dtype=torch.int64,
    ).to(torch.int32)[None]
    cache_seqlens = torch.tensor(
        [max_seqlen],
        device="cuda",
        dtype=torch.int32,
    )
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    sinks = torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)

    def run():
        return flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen,
            causal=True,
            num_splits=16,
            pack_gqa=pack_gqa,
            sinks=sinks,
        )

    eager_before = run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run()
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    eager_after = run()
    torch.cuda.synchronize()

    # Eager warmup and graph capture may each probe through both the public
    # bridge and the generic compile path. Captured execution must not reuse an
    # eager launch plan, while the final eager call must.
    assert not any(cache_hits[:-1])
    assert cache_hits[-1]
    torch.testing.assert_close(eager_before, eager_after, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        graph_output,
        eager_before,
        atol=2e-3,
        rtol=0.0,
    )
    assert captured_plans
    assert all(plan.num_splits > 1 for plan in captured_plans)
    assert all(plan.launch_split_combine_early for plan in captured_plans)
    assert all(plan.transpose_qk_pv is expected_transpose for plan in captured_plans)
    assert all(plan.split_qk_n is expected_split_qk for plan in captured_plans)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
