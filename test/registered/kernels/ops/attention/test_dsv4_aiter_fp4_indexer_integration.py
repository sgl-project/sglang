from __future__ import annotations

import math
import sys

import pytest
import torch

from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis
from sglang.kernels.ops.attention.dsv4.aiter_fp4_indexer import (
    aiter_fp4_paged_mqa_logits,
    aiter_k_indexer_fp4_cache_write,
    aiter_q_indexer_rope_hadamard_fp4_quant,
    prepare_aiter_fp4_indexer_cos_sin,
)
from sglang.kernels.ops.attention.dsv4.compress import CompressorDecodePlan
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

HEADS = 64
HEAD_DIM = 128
GROUPS = 4
GROUP_SIZE = 32
KV_BLOCK_SIZE = 64
PAYLOAD_SENTINEL = 0xA5
SCALE_SENTINEL = 0xD0


def _has_required_aiter_apis() -> bool:
    if not is_hip() or not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx950":
        return False

    import aiter
    from aiter.ops import flydsl

    required_apis = {
        "aiter.dtypes.fp4x2": getattr(getattr(aiter, "dtypes", None), "fp4x2", None),
        "aiter.rope_rotate_activation": getattr(aiter, "rope_rotate_activation", None),
        "aiter.rmsnorm_rope_rotate_activation_fp4quant_kvcache": getattr(
            aiter, "rmsnorm_rope_rotate_activation_fp4quant_kvcache", None
        ),
        "aiter.ops.flydsl.flydsl_pa_mqa_logits_fp4": getattr(
            flydsl, "flydsl_pa_mqa_logits_fp4", None
        ),
        "aiter.ops.flydsl.flydsl_pa_mqa_logits_fp4_prefill": getattr(
            flydsl, "flydsl_pa_mqa_logits_fp4_prefill", None
        ),
    }
    missing = [
        name
        for name, value in required_apis.items()
        if value is None or (name != "aiter.dtypes.fp4x2" and not callable(value))
    ]
    if missing:
        raise RuntimeError(
            f"missing required AITER FP4 indexer APIs: {', '.join(missing)}"
        )
    return True


pytestmark = pytest.mark.skipif(
    not _has_required_aiter_apis(),
    reason="requires HIP on exact gfx950 with AITER FP4 indexer APIs",
)


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=packed.device,
        dtype=torch.float32,
    )
    packed_u8 = packed.view(torch.uint8)
    codes = torch.stack((packed_u8 & 0xF, packed_u8 >> 4), dim=-1)
    return values[codes.long()].flatten(-2)


def _e8m0_to_float(scales: torch.Tensor) -> torch.Tensor:
    return torch.exp2(scales.to(torch.float32) - 127.0)


def _dequant_q(q_fp4: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
    q_values = _unpack_fp4(q_fp4).reshape(-1, HEADS, GROUPS, GROUP_SIZE)
    scales = q_scale[:, 0].permute(0, 3, 2, 1).reshape(-1, HEADS, GROUPS)
    return (q_values * _e8m0_to_float(scales).unsqueeze(-1)).flatten(-2)


def _dequant_paged_k(
    k_payload: torch.Tensor,
    k_scale: torch.Tensor,
    page_table: torch.Tensor,
) -> torch.Tensor:
    payload = k_payload.view(torch.uint8)[page_table.long(), 0].permute(0, 1, 3, 2, 4)
    scale_positions = torch.arange(KV_BLOCK_SIZE, device=k_scale.device)
    scale_positions = (
        scale_positions % 16 * (KV_BLOCK_SIZE // 16) + scale_positions // 16
    )
    scales = k_scale[page_table.long(), 0].index_select(-1, scale_positions)
    scales = scales.permute(0, 1, 3, 2)
    values = _unpack_fp4(payload)
    dequant = values * _e8m0_to_float(scales).unsqueeze(-1)
    return dequant.flatten(-2).reshape(page_table.shape[0], -1, HEAD_DIM)


def _reference_logits(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_payload: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    page_table: torch.Tensor,
    c4_seq_lens: torch.Tensor,
    weight_scale: float,
) -> torch.Tensor:
    q = _dequant_q(q_fp4, q_scale)
    k = _dequant_paged_k(k_payload, k_scale, page_table)
    scores = torch.einsum("thd,tkd->tkh", q, k).relu_()
    logits = (scores * weights[:, None].float()).sum(dim=-1) * weight_scale
    positions = torch.arange(logits.shape[1], device=logits.device)
    return logits.masked_fill(positions >= c4_seq_lens[:, None], float("-inf"))


def _assert_logits(
    actual: torch.Tensor,
    reference: torch.Tensor,
    c4_seq_lens: torch.Tensor,
) -> None:
    assert actual.shape == reference.shape
    for row, seq_len_tensor in enumerate(c4_seq_lens):
        seq_len = int(seq_len_tensor.item())
        assert torch.isfinite(actual[row, :seq_len]).all()
        assert torch.isneginf(actual[row, seq_len:]).all()
        if seq_len == 0:
            continue
        cosine = torch.nn.functional.cosine_similarity(
            actual[row, :seq_len].float(),
            reference[row, :seq_len].float(),
            dim=0,
        )
        assert cosine.item() > 0.99, f"row {row} cosine similarity was {cosine.item()}"


def _slot_payload(k_payload: torch.Tensor, slot: int) -> torch.Tensor:
    page, offset = divmod(slot, KV_BLOCK_SIZE)
    return k_payload[page, 0, :, offset].view(torch.uint8)


def _slot_scale(k_scale: torch.Tensor, slot: int) -> torch.Tensor:
    page, offset = divmod(slot, KV_BLOCK_SIZE)
    scale_offset = (offset % 16) * (KV_BLOCK_SIZE // 16) + offset // 16
    return k_scale[page, 0, :, scale_offset]


@torch.inference_mode()
def _prepare_eager_q_integration():
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260826)
    freqs_cis = precompute_freqs_cis(64, 512, 0, 10000, 1, 32, 1).to(device)
    cos, sin = prepare_aiter_fp4_indexer_cos_sin(freqs_cis)

    decode_q = torch.randn(
        2,
        HEADS,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    decode_positions = torch.tensor([19, 113], device=device, dtype=torch.int64)
    decode_q_fp4, decode_q_scale = aiter_q_indexer_rope_hadamard_fp4_quant(
        decode_q, cos, sin, decode_positions
    )
    assert decode_q_fp4.shape == (2, HEADS, HEAD_DIM // 2)
    assert decode_q_scale.shape == (2, 1, GROUPS, 16, 4)
    assert decode_q_fp4.dtype == torch.float4_e2m1fn_x2
    assert decode_q_scale.dtype == torch.uint8
    assert torch.isfinite(_dequant_q(decode_q_fp4, decode_q_scale)).all()
    assert not torch.equal(
        decode_q_fp4[0].view(torch.uint8), decode_q_fp4[1].view(torch.uint8)
    )
    return device, generator, cos, sin, decode_q, decode_q_fp4, decode_q_scale


@torch.inference_mode()
def test_dsv4_aiter_fp4_indexer_graph_capture_replay() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260827)
    freqs_cis = precompute_freqs_cis(64, 512, 0, 10000, 1, 32, 1).to(device)
    cos, sin = prepare_aiter_fp4_indexer_cos_sin(freqs_cis)

    num_queries = 4
    num_k_rows = 64 + 65 + 1
    q = torch.empty(num_queries, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    q_positions = torch.empty(num_queries, device=device, dtype=torch.int64)
    k = torch.empty(num_k_rows, HEAD_DIM, device=device, dtype=torch.bfloat16)
    norm_weight = torch.empty(HEAD_DIM, device=device, dtype=torch.bfloat16)
    plan_words = torch.zeros((num_k_rows, 4), device=device, dtype=torch.int32)
    plan = CompressorDecodePlan(4, plan_words.view(torch.uint8))
    out_loc = torch.empty(num_k_rows, device=device, dtype=torch.int64)
    k_payload = torch.empty(
        (6, 1, GROUPS, KV_BLOCK_SIZE, GROUP_SIZE // 2),
        device=device,
        dtype=torch.float4_e2m1fn_x2,
    )
    k_scale = torch.empty(
        (6, 1, GROUPS, KV_BLOCK_SIZE), device=device, dtype=torch.uint8
    )
    decode_page_table = torch.empty((num_queries, 2), device=device, dtype=torch.int32)
    decode_c4_seq_lens = torch.empty(num_queries, device=device, dtype=torch.int32)
    decode_weights = torch.empty(
        (num_queries, HEADS), device=device, dtype=torch.bfloat16
    )
    prefill_page_table = torch.empty_like(decode_page_table)
    prefill_c4_seq_lens = torch.empty_like(decode_c4_seq_lens)
    prefill_weights = torch.empty_like(decode_weights)
    weight_scale = 1.0 / math.sqrt(HEAD_DIM * HEADS)

    def load_replay_inputs(replay: int) -> int:
        q.copy_(torch.randn(q.shape, device=device, dtype=q.dtype, generator=generator))
        k.copy_(torch.randn(k.shape, device=device, dtype=k.dtype, generator=generator))
        norm_weight.copy_(
            torch.randn(
                norm_weight.shape,
                device=device,
                dtype=norm_weight.dtype,
                generator=generator,
            )
        )
        decode_weights.copy_(
            (
                0.5
                + torch.rand(decode_weights.shape, device=device, generator=generator)
            ).to(torch.bfloat16)
        )
        prefill_weights.copy_(
            (
                0.5
                + torch.rand(prefill_weights.shape, device=device, generator=generator)
            ).to(torch.bfloat16)
        )
        plan_words.zero_()
        k_payload.view(torch.uint8).fill_(PAYLOAD_SENTINEL)
        k_scale.fill_(SCALE_SENTINEL)

        first_plan = torch.arange(4, 4 * 64 + 1, 4, device=device, dtype=torch.int32)
        second_plan = torch.arange(4, 4 * 65 + 1, 4, device=device, dtype=torch.int32)
        if replay == 0:
            q_positions.copy_(
                torch.tensor([3, 19, 61, 113], device=device, dtype=torch.int64)
            )
            first_slots = torch.arange(64, 128, device=device, dtype=torch.int64)
            second_slots = torch.cat(
                (
                    torch.arange(256, 320, device=device, dtype=torch.int64),
                    torch.tensor([128], device=device, dtype=torch.int64),
                )
            )
            skipped_slot = 129
            plan_words[:, 0].copy_(
                torch.cat(
                    (
                        first_plan,
                        second_plan,
                        torch.tensor([5], device=device, dtype=torch.int32),
                    )
                )
            )
            out_loc.copy_(
                torch.cat(
                    (
                        first_slots,
                        second_slots,
                        torch.tensor([skipped_slot], device=device, dtype=torch.int64),
                    )
                )
            )
            decode_page_table.copy_(
                torch.tensor(
                    [[1, 3], [4, 2], [4, 2], [1, 3]],
                    device=device,
                    dtype=torch.int32,
                )
            )
            decode_c4_seq_lens.copy_(
                torch.tensor([64, 65, 33, 17], device=device, dtype=torch.int32)
            )
            prefill_page_table.copy_(
                torch.tensor(
                    [[1, 3], [1, 3], [4, 2], [0, 0]],
                    device=device,
                    dtype=torch.int32,
                )
            )
            prefill_c4_seq_lens.copy_(
                torch.tensor([17, 64, 65, 0], device=device, dtype=torch.int32)
            )
        else:
            q_positions.copy_(
                torch.tensor([7, 23, 101, 255], device=device, dtype=torch.int64)
            )
            first_slots = torch.arange(320, 384, device=device, dtype=torch.int64)
            second_slots = torch.cat(
                (
                    torch.arange(192, 256, device=device, dtype=torch.int64),
                    torch.tensor([0], device=device, dtype=torch.int64),
                )
            )
            skipped_slot = 1
            plan_words[:, 0].copy_(
                torch.cat(
                    (
                        first_plan.flip(0),
                        second_plan.roll(1),
                        torch.tensor([9], device=device, dtype=torch.int32),
                    )
                )
            )
            out_loc.copy_(
                torch.cat(
                    (
                        first_slots,
                        second_slots,
                        torch.tensor([skipped_slot], device=device, dtype=torch.int64),
                    )
                )
            )
            decode_page_table.copy_(
                torch.tensor(
                    [[5, 2], [3, 0], [5, 2], [3, 0]],
                    device=device,
                    dtype=torch.int32,
                )
            )
            decode_c4_seq_lens.copy_(
                torch.tensor([63, 65, 32, 48], device=device, dtype=torch.int32)
            )
            prefill_page_table.copy_(
                torch.tensor(
                    [[5, 2], [3, 0], [3, 0], [0, 0]],
                    device=device,
                    dtype=torch.int32,
                )
            )
            prefill_c4_seq_lens.copy_(
                torch.tensor([64, 33, 65, 0], device=device, dtype=torch.int32)
            )
        return skipped_slot

    load_replay_inputs(0)
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        for _ in range(2):
            warm_q_fp4, warm_q_scale = aiter_q_indexer_rope_hadamard_fp4_quant(
                q, cos, sin, q_positions
            )
            aiter_k_indexer_fp4_cache_write(
                k=k,
                norm_weight=norm_weight,
                norm_epsilon=1.0e-6,
                cos=cos,
                sin=sin,
                plan=plan,
                out_loc=out_loc,
                k_payload=k_payload,
                k_scale=k_scale,
            )
            aiter_fp4_paged_mqa_logits(
                q_fp4=warm_q_fp4,
                q_scale=warm_q_scale,
                k_payload=k_payload,
                k_scale=k_scale,
                weights=decode_weights,
                page_table=decode_page_table,
                c4_seq_lens=decode_c4_seq_lens,
                weight_scale=weight_scale,
                is_decode=True,
            )
            aiter_fp4_paged_mqa_logits(
                q_fp4=warm_q_fp4,
                q_scale=warm_q_scale,
                k_payload=k_payload,
                k_scale=k_scale,
                weights=prefill_weights,
                page_table=prefill_page_table,
                c4_seq_lens=prefill_c4_seq_lens,
                weight_scale=weight_scale,
                is_decode=False,
            )
    torch.cuda.current_stream().wait_stream(warm_stream)
    torch.cuda.synchronize()

    load_replay_inputs(0)
    k_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(k_graph):
        aiter_k_indexer_fp4_cache_write(
            k=k,
            norm_weight=norm_weight,
            norm_epsilon=1.0e-6,
            cos=cos,
            sin=sin,
            plan=plan,
            out_loc=out_loc,
            k_payload=k_payload,
            k_scale=k_scale,
        )

    decode_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(decode_graph):
        decode_q_fp4, decode_q_scale = aiter_q_indexer_rope_hadamard_fp4_quant(
            q, cos, sin, q_positions
        )
        decode_logits = aiter_fp4_paged_mqa_logits(
            q_fp4=decode_q_fp4,
            q_scale=decode_q_scale,
            k_payload=k_payload,
            k_scale=k_scale,
            weights=decode_weights,
            page_table=decode_page_table,
            c4_seq_lens=decode_c4_seq_lens,
            weight_scale=weight_scale,
            is_decode=True,
        )

    prefill_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(prefill_graph):
        prefill_q_fp4, prefill_q_scale = aiter_q_indexer_rope_hadamard_fp4_quant(
            q, cos, sin, q_positions
        )
        prefill_logits = aiter_fp4_paged_mqa_logits(
            q_fp4=prefill_q_fp4,
            q_scale=prefill_q_scale,
            k_payload=k_payload,
            k_scale=k_scale,
            weights=prefill_weights,
            page_table=prefill_page_table,
            c4_seq_lens=prefill_c4_seq_lens,
            weight_scale=weight_scale,
            is_decode=False,
        )

    previous_decode = None
    previous_prefill = None
    previous_q_payload = None
    for replay in range(2):
        skipped_slot = load_replay_inputs(replay)
        k_graph.replay()
        decode_graph.replay()
        prefill_graph.replay()
        torch.cuda.synchronize()

        decode_snapshot = decode_logits.clone()
        prefill_snapshot = prefill_logits.clone()
        q_payload_snapshot = decode_q_fp4.view(torch.uint8).clone()
        assert (_slot_payload(k_payload, skipped_slot) == PAYLOAD_SENTINEL).all()
        assert (_slot_scale(k_scale, skipped_slot) == SCALE_SENTINEL).all()

        eager_q_fp4, eager_q_scale = aiter_q_indexer_rope_hadamard_fp4_quant(
            q, cos, sin, q_positions
        )
        torch.testing.assert_close(
            decode_q_fp4.view(torch.uint8), eager_q_fp4.view(torch.uint8)
        )
        torch.testing.assert_close(decode_q_scale, eager_q_scale)
        torch.testing.assert_close(
            prefill_q_fp4.view(torch.uint8), eager_q_fp4.view(torch.uint8)
        )
        torch.testing.assert_close(prefill_q_scale, eager_q_scale)

        eager_k_payload = torch.empty_like(k_payload)
        eager_k_scale = torch.empty_like(k_scale)
        eager_k_payload.view(torch.uint8).fill_(PAYLOAD_SENTINEL)
        eager_k_scale.fill_(SCALE_SENTINEL)
        aiter_k_indexer_fp4_cache_write(
            k=k,
            norm_weight=norm_weight,
            norm_epsilon=1.0e-6,
            cos=cos,
            sin=sin,
            plan=plan,
            out_loc=out_loc,
            k_payload=eager_k_payload,
            k_scale=eager_k_scale,
        )
        torch.testing.assert_close(
            k_payload.view(torch.uint8), eager_k_payload.view(torch.uint8)
        )
        torch.testing.assert_close(k_scale, eager_k_scale)

        eager_decode = aiter_fp4_paged_mqa_logits(
            q_fp4=eager_q_fp4,
            q_scale=eager_q_scale,
            k_payload=eager_k_payload,
            k_scale=eager_k_scale,
            weights=decode_weights,
            page_table=decode_page_table,
            c4_seq_lens=decode_c4_seq_lens,
            weight_scale=weight_scale,
            is_decode=True,
        )
        eager_prefill = aiter_fp4_paged_mqa_logits(
            q_fp4=eager_q_fp4,
            q_scale=eager_q_scale,
            k_payload=eager_k_payload,
            k_scale=eager_k_scale,
            weights=prefill_weights,
            page_table=prefill_page_table,
            c4_seq_lens=prefill_c4_seq_lens,
            weight_scale=weight_scale,
            is_decode=False,
        )
        torch.testing.assert_close(decode_snapshot, eager_decode)
        torch.testing.assert_close(prefill_snapshot, eager_prefill)
        _assert_logits(
            decode_snapshot,
            _reference_logits(
                eager_q_fp4,
                eager_q_scale,
                eager_k_payload,
                eager_k_scale,
                decode_weights,
                decode_page_table,
                decode_c4_seq_lens,
                weight_scale,
            ),
            decode_c4_seq_lens,
        )
        _assert_logits(
            prefill_snapshot,
            _reference_logits(
                eager_q_fp4,
                eager_q_scale,
                eager_k_payload,
                eager_k_scale,
                prefill_weights,
                prefill_page_table,
                prefill_c4_seq_lens,
                weight_scale,
            ),
            prefill_c4_seq_lens,
        )

        if previous_decode is not None:
            assert not torch.allclose(decode_snapshot[0, :16], previous_decode[0, :16])
            assert not torch.allclose(
                prefill_snapshot[1, :16], previous_prefill[1, :16]
            )
            assert not torch.equal(q_payload_snapshot, previous_q_payload)
        previous_decode = decode_snapshot
        previous_prefill = prefill_snapshot
        previous_q_payload = q_payload_snapshot


@torch.inference_mode()
def test_dsv4_aiter_fp4_indexer_q_k_decode_prefill_integration() -> None:
    device, generator, cos, sin, decode_q, decode_q_fp4, decode_q_scale = (
        _prepare_eager_q_integration()
    )

    first_sequence_slots = torch.arange(
        1 * KV_BLOCK_SIZE,
        2 * KV_BLOCK_SIZE,
        device=device,
        dtype=torch.int64,
    )
    second_sequence_slots = torch.cat(
        (
            torch.arange(
                4 * KV_BLOCK_SIZE,
                5 * KV_BLOCK_SIZE,
                device=device,
                dtype=torch.int64,
            ),
            torch.tensor([2 * KV_BLOCK_SIZE], device=device, dtype=torch.int64),
        )
    )
    skipped_slot = 2 * KV_BLOCK_SIZE + 1
    out_loc = torch.cat(
        (
            first_sequence_slots,
            second_sequence_slots,
            torch.tensor([skipped_slot], device=device, dtype=torch.int64),
        )
    )
    assert (
        second_sequence_slots[63] // KV_BLOCK_SIZE
        != second_sequence_slots[64] // KV_BLOCK_SIZE
    )

    boundary_seq_lens = torch.arange(4, 4 * 65 + 1, 4, device=device, dtype=torch.int64)
    seq_lens = torch.cat(
        (
            boundary_seq_lens[:64],
            boundary_seq_lens,
            torch.tensor([5], device=device, dtype=torch.int64),
        )
    )
    req_pool_indices = torch.arange(seq_lens.numel(), device=device)
    plan = CompressorDecodePlan.generate_legacy(4, req_pool_indices, seq_lens)
    assert isinstance(plan, CompressorDecodePlan)

    k = torch.randn(
        seq_lens.numel(),
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    norm_weight = torch.randn(
        HEAD_DIM, device=device, dtype=torch.bfloat16, generator=generator
    )
    k_payload = torch.empty(
        (6, 1, GROUPS, KV_BLOCK_SIZE, GROUP_SIZE // 2),
        device=device,
        dtype=torch.float4_e2m1fn_x2,
    )
    k_scale = torch.empty(
        (6, 1, GROUPS, KV_BLOCK_SIZE), device=device, dtype=torch.uint8
    )
    k_payload.view(torch.uint8).fill_(PAYLOAD_SENTINEL)
    k_scale.fill_(SCALE_SENTINEL)

    aiter_k_indexer_fp4_cache_write(
        k=k,
        norm_weight=norm_weight,
        norm_epsilon=1.0e-6,
        cos=cos,
        sin=sin,
        plan=plan,
        out_loc=out_loc,
        k_payload=k_payload,
        k_scale=k_scale,
    )
    torch.cuda.synchronize()

    assert (_slot_payload(k_payload, skipped_slot) == PAYLOAD_SENTINEL).all()
    assert (_slot_scale(k_scale, skipped_slot) == SCALE_SENTINEL).all()
    for slot in (int(first_sequence_slots[0]), int(second_sequence_slots[-1])):
        assert (_slot_payload(k_payload, slot) != PAYLOAD_SENTINEL).any()
        assert (_slot_scale(k_scale, slot) != SCALE_SENTINEL).any()
    assert (k_payload[0].view(torch.uint8) == PAYLOAD_SENTINEL).all()
    assert (k_scale[0] == SCALE_SENTINEL).all()
    assert (k_payload[3].view(torch.uint8) == PAYLOAD_SENTINEL).all()
    assert (k_scale[3] == SCALE_SENTINEL).all()
    assert (k_payload[5].view(torch.uint8) == PAYLOAD_SENTINEL).all()
    assert (k_scale[5] == SCALE_SENTINEL).all()
    assert (k_payload[2, :, :, 1:].view(torch.uint8) == PAYLOAD_SENTINEL).all()
    assert (k_scale[2, :, :, 1:] == SCALE_SENTINEL).all()

    page_table = torch.tensor([[1, 3], [4, 2]], device=device, dtype=torch.int32)
    c4_seq_lens = torch.tensor([64, 65], device=device, dtype=torch.int32)
    weights = (
        0.5
        + torch.rand(2, HEADS, device=device, dtype=torch.float32, generator=generator)
    ).to(torch.bfloat16)
    weight_scale = 1.0 / math.sqrt(HEAD_DIM * HEADS)
    decode_logits = aiter_fp4_paged_mqa_logits(
        q_fp4=decode_q_fp4,
        q_scale=decode_q_scale,
        k_payload=k_payload,
        k_scale=k_scale,
        weights=weights,
        page_table=page_table,
        c4_seq_lens=c4_seq_lens,
        weight_scale=weight_scale,
        is_decode=True,
    )
    decode_reference = _reference_logits(
        decode_q_fp4,
        decode_q_scale,
        k_payload,
        k_scale,
        weights,
        page_table,
        c4_seq_lens,
        weight_scale,
    )
    _assert_logits(decode_logits, decode_reference, c4_seq_lens)
    assert not torch.allclose(decode_logits[0, :64], decode_logits[1, :64])

    extra_q = torch.randn(
        2,
        HEADS,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    prefill_q = torch.stack((extra_q[0], decode_q[0], extra_q[1], decode_q[1]))
    prefill_positions = torch.tensor([7, 19, 61, 113], device=device, dtype=torch.int64)
    prefill_q_fp4, prefill_q_scale = aiter_q_indexer_rope_hadamard_fp4_quant(
        prefill_q, cos, sin, prefill_positions
    )
    torch.testing.assert_close(
        prefill_q_fp4.view(torch.uint8)[[1, 3]], decode_q_fp4.view(torch.uint8)
    )
    torch.testing.assert_close(prefill_q_scale[[1, 3]], decode_q_scale)

    prefill_page_table = page_table.repeat_interleave(2, dim=0)
    prefill_c4_seq_lens = torch.tensor(
        [63, 64, 64, 65], device=device, dtype=torch.int32
    )
    extra_weights = (
        0.5
        + torch.rand(2, HEADS, device=device, dtype=torch.float32, generator=generator)
    ).to(torch.bfloat16)
    prefill_weights = torch.stack(
        (extra_weights[0], weights[0], extra_weights[1], weights[1])
    ).contiguous()
    prefill_logits = aiter_fp4_paged_mqa_logits(
        q_fp4=prefill_q_fp4,
        q_scale=prefill_q_scale,
        k_payload=k_payload,
        k_scale=k_scale,
        weights=prefill_weights,
        page_table=prefill_page_table,
        c4_seq_lens=prefill_c4_seq_lens,
        weight_scale=weight_scale,
        is_decode=False,
    )
    prefill_reference = _reference_logits(
        prefill_q_fp4,
        prefill_q_scale,
        k_payload,
        k_scale,
        prefill_weights,
        prefill_page_table,
        prefill_c4_seq_lens,
        weight_scale,
    )
    _assert_logits(prefill_logits, prefill_reference, prefill_c4_seq_lens)
    for prefill_row, decode_row in ((1, 0), (3, 1)):
        seq_len = int(c4_seq_lens[decode_row].item())
        cosine = torch.nn.functional.cosine_similarity(
            prefill_logits[prefill_row, :seq_len].float(),
            decode_logits[decode_row, :seq_len].float(),
            dim=0,
        )
        assert cosine.item() > 0.99
        assert torch.isneginf(prefill_logits[prefill_row, seq_len:]).all()


@torch.inference_mode()
def test_dsv4_aiter_fp4_indexer_prefill_above_persistent_grid_floor() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260828)
    num_queries = 513
    seq_len = 37
    freqs_cis = precompute_freqs_cis(64, 512, 0, 10000, 1, 32, 1).to(device)
    cos, sin = prepare_aiter_fp4_indexer_cos_sin(freqs_cis)
    q = torch.randn(
        num_queries,
        HEADS,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    positions = torch.arange(num_queries, device=device, dtype=torch.int64) % 512
    q_fp4, q_scale = aiter_q_indexer_rope_hadamard_fp4_quant(q, cos, sin, positions)
    k_payload = torch.randint(
        0,
        256,
        (1, 1, GROUPS, KV_BLOCK_SIZE, GROUP_SIZE // 2),
        device=device,
        dtype=torch.uint8,
        generator=generator,
    ).view(torch.float4_e2m1fn_x2)
    k_scale = torch.full(
        (1, 1, GROUPS, KV_BLOCK_SIZE),
        127,
        device=device,
        dtype=torch.uint8,
    )
    weights = (
        0.5 + torch.rand(num_queries, HEADS, device=device, generator=generator)
    ).to(torch.bfloat16)
    page_table = torch.zeros((num_queries, 1), device=device, dtype=torch.int32)
    c4_seq_lens = torch.full((num_queries,), seq_len, device=device, dtype=torch.int32)
    weight_scale = 1.0 / math.sqrt(HEAD_DIM * HEADS)

    logits = aiter_fp4_paged_mqa_logits(
        q_fp4=q_fp4,
        q_scale=q_scale,
        k_payload=k_payload,
        k_scale=k_scale,
        weights=weights,
        page_table=page_table,
        c4_seq_lens=c4_seq_lens,
        weight_scale=weight_scale,
        is_decode=False,
    )
    final_reference = _reference_logits(
        q_fp4[-1:],
        q_scale[-1:],
        k_payload,
        k_scale,
        weights[-1:],
        page_table[-1:],
        c4_seq_lens[-1:],
        weight_scale,
    )
    _assert_logits(logits[-1:], final_reference, c4_seq_lens[-1:])


@torch.inference_mode()
def test_dsv4_aiter_fp4_indexer_68k_lookahead_guard() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260829)
    c4_seq_len = 17_000
    num_pages = math.ceil(c4_seq_len / KV_BLOCK_SIZE)

    q_fp4 = torch.randint(
        0,
        256,
        (1, HEADS, HEAD_DIM // 2),
        device=device,
        dtype=torch.uint8,
        generator=generator,
    ).view(torch.float4_e2m1fn_x2)
    q_scale = torch.full((1, 1, GROUPS, 16, 4), 127, device=device, dtype=torch.uint8)
    k_payload = torch.randint(
        0,
        256,
        (num_pages, 1, GROUPS, KV_BLOCK_SIZE, GROUP_SIZE // 2),
        device=device,
        dtype=torch.uint8,
        generator=generator,
    ).view(torch.float4_e2m1fn_x2)
    k_scale = torch.full(
        (num_pages, 1, GROUPS, KV_BLOCK_SIZE),
        127,
        device=device,
        dtype=torch.uint8,
    )
    weights = torch.ones((1, HEADS), device=device, dtype=torch.bfloat16)
    page_table = torch.arange(num_pages, device=device, dtype=torch.int32)[None]
    c4_seq_lens = torch.tensor([c4_seq_len], device=device, dtype=torch.int32)

    for is_decode in (True, False):
        logits = aiter_fp4_paged_mqa_logits(
            q_fp4=q_fp4,
            q_scale=q_scale,
            k_payload=k_payload,
            k_scale=k_scale,
            weights=weights,
            page_table=page_table,
            c4_seq_lens=c4_seq_lens,
            weight_scale=1.0 / math.sqrt(HEAD_DIM * HEADS),
            is_decode=is_decode,
        )
        torch.cuda.synchronize()

        assert logits.shape == (1, num_pages * KV_BLOCK_SIZE)
        assert torch.isfinite(logits[0, :c4_seq_len]).all()
        assert torch.isneginf(logits[0, c4_seq_len:]).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
