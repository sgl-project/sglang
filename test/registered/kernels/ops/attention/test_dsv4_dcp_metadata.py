import math
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
from sglang.kernels.ops.attention.deepseek_v4_rope import (
    precompute_freqs_cis,
)
from sglang.kernels.ops.attention.dsv4.compress import (
    CompressorDecodePlan,
    compress_norm_rope_store,
)
from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
    init_compression_metadata,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_triton,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_prefill import (
    _sparse_attn_v4_paged_prefill_triton,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.runtime import (
    build_decode_streams,
    build_prefill_indices,
    update_dcp_csa_stream,
)
from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DeepseekV4HipRadixBackend,
)
from sglang.srt.layers.attention.dsv4.dcp import (
    local_compressed_lens,
    local_swa_lens,
    localize_compressed_indices,
    select_dcp_attn_sink,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=15, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DSV4 DCP metadata tests require a GPU."
)

DEVICE = "cuda"


@pytest.mark.parametrize("seq_lens_cpu_is_final", [False, True])
def test_dspark_target_verify_cpu_seq_len_contract(
    seq_lens_cpu_is_final: bool,
) -> None:
    backend = object.__new__(DeepseekV4HipRadixBackend)
    backend.target_verify_num_draft_tokens = 5
    backend.cuda_int32_kwargs = {"device": DEVICE, "dtype": torch.int32}
    backend._move_to_device = lambda values: torch.tensor(
        values, dtype=torch.int32, device=DEVICE
    )

    captured = {}
    backend.init_forward_metadata_prefill = lambda **kwargs: captured.update(kwargs)

    prefix_lens = [97, 255]
    final_lens = [length + 5 for length in prefix_lens]
    input_cpu_lens = final_lens if seq_lens_cpu_is_final else prefix_lens
    req_pool_indices = torch.tensor([3, 7], dtype=torch.int64, device=DEVICE)
    backend.init_forward_metadata_target_verify_old(
        max_seq_len=max(final_lens),
        req_pool_indices=req_pool_indices,
        seq_lens=torch.tensor(prefix_lens, dtype=torch.int32, device=DEVICE),
        seq_lens_cpu=input_cpu_lens,
        out_cache_loc=torch.zeros(10, dtype=torch.int64, device=DEVICE),
        seq_lens_cpu_is_final=seq_lens_cpu_is_final,
    )

    assert captured["seq_lens_cpu"] == final_lens
    torch.testing.assert_close(
        captured["seq_lens"],
        torch.tensor(final_lens, dtype=torch.int32, device=DEVICE),
    )
    causal_lens, _ = backend.expand_prefill_casually(
        num_tokens=10,
        seq_lens=captured["seq_lens_cpu"],
        extend_seq_lens=captured["extend_seq_lens_cpu"],
        req_pool_indices=req_pool_indices,
        padded_num_tokens=None,
    )
    torch.testing.assert_close(
        causal_lens,
        torch.tensor(
            list(range(98, 103)) + list(range(256, 261)),
            dtype=torch.int32,
            device=DEVICE,
        ),
    )


def test_dspark_graph_replay_normalizes_padded_cpu_seq_lens() -> None:
    backend = object.__new__(DeepseekV4HipRadixBackend)
    backend.target_verify_num_draft_tokens = 5
    backend.is_dspark_draft = True
    backend.MAX_SEQ_LEN_FOR_CAPTURE = 1024
    backend.cuda_int32_kwargs = {"device": DEVICE, "dtype": torch.int32}
    backend._move_to_device = lambda values: torch.tensor(
        values, dtype=torch.int32, device=DEVICE
    )

    captured = {}
    backend.init_forward_metadata_prefill = lambda **kwargs: captured.update(kwargs)
    backend.replay_cuda_graph_metadata_from = lambda **kwargs: None
    forward_batch = SimpleNamespace(
        batch_size=2,
        forward_mode=ForwardMode.TARGET_VERIFY,
        actual_forward_mode=ForwardMode.TARGET_VERIFY,
        positions=torch.zeros(10, dtype=torch.int64, device=DEVICE),
        req_pool_indices=torch.tensor([3, 0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([97, 1], dtype=torch.int32, device=DEVICE),
        seq_lens_sum=103,
        seq_lens_cpu=torch.tensor([102, 1], dtype=torch.int64),
        out_cache_loc=torch.zeros(5, dtype=torch.int64, device=DEVICE),
        num_padding=1,
        spec_info=None,
    )

    backend.init_forward_metadata_out_graph(forward_batch)

    assert captured["seq_lens_cpu"] == [102, 6]
    torch.testing.assert_close(
        captured["seq_lens"],
        torch.tensor([102, 6], dtype=torch.int32, device=DEVICE),
    )
    causal_lens, _ = backend.expand_prefill_casually(
        num_tokens=10,
        seq_lens=captured["seq_lens_cpu"],
        extend_seq_lens=captured["extend_seq_lens_cpu"],
        req_pool_indices=forward_batch.req_pool_indices,
        padded_num_tokens=None,
    )
    torch.testing.assert_close(
        causal_lens,
        torch.tensor(
            list(range(98, 103)) + list(range(2, 7)),
            dtype=torch.int32,
            device=DEVICE,
        ),
    )


def test_dcp_sink_shift_is_cuda_graph_capture_safe() -> None:
    dcp_size = 8
    local_heads = 16
    attn_sink = torch.arange(128, dtype=torch.float32, device=DEVICE)
    shifted_sink = torch.empty_like(attn_sink)
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        shifted_sink.copy_(
            select_dcp_attn_sink(
                attn_sink,
                local_heads,
                attn_tp_rank=0,
                dcp_size=dcp_size,
                dcp_rank=0,
            )
            - math.log(float(dcp_size))
        )

    attn_sink.add_(3.0)
    graph.replay()
    torch.testing.assert_close(
        shifted_sink,
        attn_sink - math.log(float(dcp_size)),
    )


@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
def test_compressed_metadata_matches_dcp_domain(dcp_size: int) -> None:
    physical_page_size = 256
    seq_lens = torch.tensor(
        [4, 8, 128, 256, 2048, 2049], dtype=torch.int32, device=DEVICE
    )
    positions = seq_lens - 1
    logical_page_size = physical_page_size * dcp_size
    raw_out_loc = torch.tensor(
        [
            logical_page_size + 3,
            logical_page_size + 7,
            2 * logical_page_size + 127,
            3 * logical_page_size + 255,
            7 * logical_page_size + 2047,
            7 * logical_page_size + 2048,
        ],
        dtype=torch.int64,
        device=DEVICE,
    )
    num_pages = 16
    page_table = torch.arange(
        10, 10 + seq_lens.numel() * num_pages, dtype=torch.int32, device=DEVICE
    ).reshape(seq_lens.numel(), num_pages)

    for rank in range(dcp_size):
        result = init_compression_metadata(
            seq_lens,
            positions,
            raw_out_loc,
            page_table,
            physical_page_size,
            compute_page_indices=True,
            dcp_size=dcp_size,
            dcp_rank=rank,
        )
        c4_out, _, c4_lens, _, c128_out, _, c128_lens, _, c128_pages = result

        expected_c4 = localize_compressed_indices(raw_out_loc, 4, dcp_size, rank)
        expected_c128 = localize_compressed_indices(raw_out_loc, 128, dcp_size, rank)
        c4_boundaries = seq_lens % 4 == 0
        c128_boundaries = seq_lens % 128 == 0
        expected_c4_out = torch.where(
            c4_boundaries,
            expected_c4.local,
            torch.zeros_like(expected_c4.local),
        )
        expected_c128_out = torch.where(
            c128_boundaries,
            expected_c128.local,
            torch.zeros_like(expected_c128.local),
        )

        torch.testing.assert_close(c4_out, expected_c4_out)
        torch.testing.assert_close(c128_out, expected_c128_out)
        torch.testing.assert_close(
            c4_lens,
            local_compressed_lens(seq_lens, 4, dcp_size, rank).to(torch.int32),
        )
        torch.testing.assert_close(
            c128_lens,
            local_compressed_lens(seq_lens, 128, dcp_size, rank).to(torch.int32),
        )

        c128_page_size = physical_page_size // 128
        for row, local_len in enumerate(c128_lens.tolist()):
            for index in range(c128_pages.shape[1]):
                if index >= local_len:
                    assert c128_pages[row, index].item() == -1
                    continue
                page = index // c128_page_size
                offset = index % c128_page_size
                expected = page_table[row, page].item() * c128_page_size + offset
                assert c128_pages[row, index].item() == expected


def test_unified_bf16_store_skips_negative_locations() -> None:
    torch.manual_seed(0)
    num_tokens = 3
    head_dim = 512
    compress_ratio = 4
    kv = torch.randn(num_tokens, head_dim, dtype=torch.bfloat16, device=DEVICE)
    norm_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=DEVICE)
    seq_lens = torch.arange(
        compress_ratio,
        (num_tokens + 1) * compress_ratio,
        compress_ratio,
        dtype=torch.int64,
        device=DEVICE,
    )
    req_pool_indices = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    plan = CompressorDecodePlan.generate_legacy(
        compress_ratio, req_pool_indices, seq_lens
    )
    out_loc = torch.tensor([-1, 2, -1], dtype=torch.int64, device=DEVICE)
    cache = torch.full((4, head_dim), 7.0, dtype=torch.bfloat16, device=DEVICE)
    before = cache.clone()
    freqs_cis = precompute_freqs_cis(
        64, int(seq_lens.max().item()) + 1, 0, 10000, 1, 32, 1
    ).to(DEVICE)

    compress_norm_rope_store(
        kv,
        plan,
        norm_weight=norm_weight,
        norm_eps=1.0e-6,
        freq_cis=freqs_cis,
        out_loc=out_loc,
        kvcache=cache.view(torch.uint8),
        page_size=1,
        bf16_store=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(cache[0], before[0])
    torch.testing.assert_close(cache[1], before[1])
    assert not torch.equal(cache[2], before[2])
    torch.testing.assert_close(cache[3], before[3])


@pytest.mark.parametrize("kv_splits", [1, 8])
def test_partitioned_decode_lse_merge_matches_unsharded(kv_splits: int) -> None:
    torch.manual_seed(1 + kv_splits)
    dcp_size = 8
    num_tokens, num_heads, head_dim = 2, 16, 512
    kv_lens = [67, 53]
    total_kv = sum(kv_lens)
    q = torch.randn(
        num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=DEVICE
    )
    kv = torch.randn(total_kv, head_dim, dtype=torch.bfloat16, device=DEVICE)
    full_indices = torch.arange(total_kv, dtype=torch.int32, device=DEVICE)
    full_indptr = torch.tensor(
        [0, kv_lens[0], total_kv], dtype=torch.int32, device=DEVICE
    )
    sink = torch.randn(num_heads, dtype=torch.float32, device=DEVICE)
    scale = head_dim**-0.5

    reference_out, reference_lse = _sparse_attn_v4_paged_decode_triton(
        q,
        kv,
        full_indices,
        full_indptr,
        sink,
        scale,
        kv_splits=kv_splits,
        return_lse=True,
    )

    partial_outputs = []
    partial_lses = []
    shifted_sink = sink - torch.log(
        torch.tensor(float(dcp_size), dtype=torch.float32, device=DEVICE)
    )
    for rank in range(dcp_size):
        rank_indices = []
        rank_indptr = [0]
        offset = 0
        for length in kv_lens:
            segment = full_indices[offset : offset + length]
            owned = segment[segment % dcp_size == rank]
            rank_indices.append(owned)
            rank_indptr.append(rank_indptr[-1] + owned.numel())
            offset += length
        rank_indices = torch.cat(rank_indices)
        rank_indptr = torch.tensor(rank_indptr, dtype=torch.int32, device=DEVICE)
        partial_out, partial_lse = _sparse_attn_v4_paged_decode_triton(
            q,
            kv,
            rank_indices,
            rank_indptr,
            shifted_sink,
            scale,
            kv_splits=kv_splits,
            return_lse=True,
        )
        partial_outputs.append(partial_out)
        partial_lses.append(partial_lse)

    combined_out, combined_lse = dcp_lse_combine_triton(
        torch.stack(partial_outputs),
        torch.stack(partial_lses),
        is_lse_base_on_e=True,
        return_lse=True,
    )
    torch.testing.assert_close(
        combined_out.float(), reference_out.float(), atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(combined_lse, reference_lse, atol=3e-3, rtol=3e-3)


def test_partitioned_prefill_lse_merge_counts_replicated_terms_once() -> None:
    torch.manual_seed(19)
    dcp_size = 8
    num_tokens, num_heads, head_dim = 2, 16, 512
    prefix_lens = [23, 17]
    extend_lens = [5, 3]
    total_prefix = sum(prefix_lens)
    total_extend = sum(extend_lens)
    q = torch.randn(
        num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=DEVICE
    )
    prefix_kv = torch.randn(total_prefix, head_dim, dtype=torch.bfloat16, device=DEVICE)
    extend_kv = torch.randn(total_extend, head_dim, dtype=torch.bfloat16, device=DEVICE)
    prefix_indices = torch.arange(total_prefix, dtype=torch.int32, device=DEVICE)
    prefix_indptr = torch.tensor(
        [0, prefix_lens[0], total_prefix], dtype=torch.int32, device=DEVICE
    )
    extend_indices = torch.arange(total_extend, dtype=torch.int32, device=DEVICE)
    extend_indptr = torch.tensor(
        [0, extend_lens[0], total_extend], dtype=torch.int32, device=DEVICE
    )
    sink = torch.randn(num_heads, dtype=torch.float32, device=DEVICE)
    scale = head_dim**-0.5

    reference_out, reference_lse = _sparse_attn_v4_paged_prefill_triton(
        q,
        prefix_kv,
        prefix_indices,
        prefix_indptr,
        extend_kv,
        extend_indices,
        extend_indptr,
        sink,
        scale,
        return_lse=True,
    )

    partial_outputs = []
    partial_lses = []
    replicated_logit_shift = -torch.log(
        torch.tensor(float(dcp_size), dtype=torch.float32, device=DEVICE)
    ).item()
    for rank in range(dcp_size):
        rank_indices = []
        rank_indptr = [0]
        offset = 0
        for length in prefix_lens:
            segment = prefix_indices[offset : offset + length]
            owned = segment[segment % dcp_size == rank]
            rank_indices.append(owned)
            rank_indptr.append(rank_indptr[-1] + owned.numel())
            offset += length
        rank_indices = torch.cat(rank_indices)
        rank_indptr = torch.tensor(rank_indptr, dtype=torch.int32, device=DEVICE)
        partial_out, partial_lse = _sparse_attn_v4_paged_prefill_triton(
            q,
            prefix_kv,
            rank_indices,
            rank_indptr,
            extend_kv,
            extend_indices,
            extend_indptr,
            sink,
            scale,
            return_lse=True,
            replicated_logit_shift=replicated_logit_shift,
        )
        partial_outputs.append(partial_out)
        partial_lses.append(partial_lse)

    combined_out, combined_lse = dcp_lse_combine_triton(
        torch.stack(partial_outputs),
        torch.stack(partial_lses),
        is_lse_base_on_e=True,
        return_lse=True,
    )
    torch.testing.assert_close(
        combined_out.float(), reference_out.float(), atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(combined_lse, reference_lse, atol=3e-3, rtol=3e-3)


def test_dcp_decode_streams_partition_swa_and_c128() -> None:
    state_slot = torch.tensor([2, 5], dtype=torch.int32, device=DEVICE)
    positions = torch.tensor([127, 132], dtype=torch.int32, device=DEVICE)
    hca_pages = torch.tensor(
        [[10, 11, -1, -1], [20, 21, 22, -1]],
        dtype=torch.int32,
        device=DEVICE,
    )
    win, ring_stride, swa_pages = 128, 133, 2048
    seen_swa = [set(), set()]
    seen_hca = [set(), set()]

    for rank in range(8):
        swa_i, swa_p, hca_i, hca_p, csa_i, csa_p = build_decode_streams(
            state_slot=state_slot,
            positions=positions,
            swa_len=torch.tensor([128, 128], dtype=torch.int32, device=DEVICE),
            hca_len=torch.tensor([2, 3], dtype=torch.int32, device=DEVICE),
            csa_len=torch.zeros(2, dtype=torch.int32, device=DEVICE),
            hca_page_indices=hca_pages,
            csa_width=0,
            win=win,
            ring_stride=ring_stride,
            swa_pages=swa_pages,
            dcp_size=8,
            dcp_rank=rank,
        )
        assert csa_i.numel() == positions.numel() * win
        torch.testing.assert_close(csa_p, swa_p)
        for row, position in enumerate(positions.tolist()):
            swa_segment = swa_i[swa_p[row] : swa_p[row + 1]].tolist()
            hca_segment = hca_i[hca_p[row] : hca_p[row + 1]].tolist()
            expected_positions = [
                key_position
                for key_position in range(max(0, position - win + 1), position + 1)
                if key_position % 8 == rank
            ]
            expected_swa = [
                state_slot[row].item() * ring_stride + key_position % ring_stride
                for key_position in expected_positions
            ]
            assert swa_segment == expected_swa
            assert hca_segment[: len(expected_swa)] == expected_swa
            seen_swa[row].update(swa_segment)
            seen_hca[row].update(hca_segment[len(expected_swa) :])

    for row, position in enumerate(positions.tolist()):
        all_positions = range(max(0, position - win + 1), position + 1)
        expected_swa = {
            state_slot[row].item() * ring_stride + key_position % ring_stride
            for key_position in all_positions
        }
        assert seen_swa[row] == expected_swa
        expected_hca = {
            value + swa_pages for value in hca_pages[row].tolist() if value >= 0
        }
        assert seen_hca[row] == expected_hca


def test_dcp_csa_stream_packs_local_global_topk_winners() -> None:
    state_slot = torch.tensor([3, 4], dtype=torch.int32, device=DEVICE)
    positions = torch.tensor([130, 255], dtype=torch.int32, device=DEVICE)
    c4_pages = torch.tensor(
        [[100, 101, -1], [200, -1, -1]], dtype=torch.int32, device=DEVICE
    )
    swa_len = torch.tensor([16, 16], dtype=torch.int32, device=DEVICE)
    c4_len = torch.tensor([2, 1], dtype=torch.int32, device=DEVICE)
    indices = torch.empty(
        positions.numel() * (128 + c4_pages.shape[1]),
        dtype=torch.int32,
        device=DEVICE,
    )
    indptr = torch.empty(positions.numel() + 1, dtype=torch.int32, device=DEVICE)
    update_dcp_csa_stream(
        indices=indices,
        indptr=indptr,
        state_slot=state_slot,
        positions=positions,
        swa_len=swa_len,
        c4_page_indices=c4_pages,
        c4_len=c4_len,
        win=128,
        ring_stride=133,
        swa_pages=2048,
        dcp_size=8,
        dcp_rank=3,
    )
    for row, position in enumerate(positions.tolist()):
        segment = indices[indptr[row] : indptr[row + 1]].tolist()
        start = max(0, position - 127)
        expected_swa = [
            state_slot[row].item() * 133 + key_position % 133
            for key_position in range(start, position + 1)
            if key_position % 8 == 3
        ]
        expected_c4 = [value + 2048 for value in c4_pages[row].tolist() if value >= 0]
        assert segment == expected_swa + expected_c4


def test_dcp_prefill_indices_partition_only_persisted_prefix() -> None:
    state_slot = torch.tensor([2, 2], dtype=torch.int32, device=DEVICE)
    positions = torch.tensor([139, 151], dtype=torch.int64, device=DEVICE)
    chunk_start = torch.tensor([128, 128], dtype=torch.int64, device=DEVICE)
    cu_q = torch.tensor([0, 0], dtype=torch.int64, device=DEVICE)
    c128_pages = torch.tensor(
        [[10, 11, -1], [20, -1, -1]], dtype=torch.int32, device=DEVICE
    )
    win, ring_stride, swa_pages = 128, 133, 2048
    all_prefix_swa = [set(), set()]

    for rank in range(8):
        prefix, prefix_indptr, extend, extend_indptr = build_prefill_indices(
            compress_ratio=128,
            state_slot=state_slot,
            positions=positions,
            chunk_start=chunk_start,
            cu_q=cu_q,
            win=win,
            ring_stride=ring_stride,
            swa_pages=swa_pages,
            c128_page_indices=c128_pages,
            c4_sparse_page_indices=None,
            dcp_size=8,
            dcp_rank=rank,
        )
        for row, position in enumerate(positions.tolist()):
            prefix_segment = prefix[
                prefix_indptr[row] : prefix_indptr[row + 1]
            ].tolist()
            compressed = [
                value + swa_pages for value in c128_pages[row].tolist() if value >= 0
            ]
            assert prefix_segment[-len(compressed) :] == compressed
            prefix_swa = prefix_segment[: -len(compressed)]
            expected_positions = [
                key_position
                for key_position in range(max(0, position - win + 1), 128)
                if key_position % 8 == rank
            ]
            expected_swa = [
                state_slot[row].item() * ring_stride + key_position % ring_stride
                for key_position in expected_positions
            ]
            assert prefix_swa == expected_swa
            all_prefix_swa[row].update(prefix_swa)

            extend_segment = extend[
                extend_indptr[row] : extend_indptr[row + 1]
            ].tolist()
            expected_extend_len = position - chunk_start[row].item() + 1
            assert extend_segment == list(range(expected_extend_len))

    for row, position in enumerate(positions.tolist()):
        expected = {
            state_slot[row].item() * ring_stride + key_position % ring_stride
            for key_position in range(max(0, position - win + 1), 128)
        }
        assert all_prefix_swa[row] == expected


def test_dcp_decode_and_csa_streams_replay_live_cuda_graph_inputs() -> None:
    dcp_size, dcp_rank = 8, 5
    win, ring_stride, swa_pages = 128, 133, 2048
    state_slot = torch.tensor([2, 4], dtype=torch.int32, device=DEVICE)
    positions = torch.tensor([127, 132], dtype=torch.int32, device=DEVICE)
    hca_pages = torch.tensor(
        [[10, 11, -1], [20, 21, -1]], dtype=torch.int32, device=DEVICE
    )
    hca_len = torch.tensor([2, 2], dtype=torch.int32, device=DEVICE)
    c4_pages = torch.tensor(
        [[100, 101, -1], [200, -1, -1]], dtype=torch.int32, device=DEVICE
    )
    c4_len = torch.tensor([2, 1], dtype=torch.int32, device=DEVICE)

    def eager_streams():
        swa_len = local_swa_lens(positions + 1, win, dcp_size, dcp_rank).to(torch.int32)
        streams = build_decode_streams(
            state_slot=state_slot,
            positions=positions,
            swa_len=swa_len,
            hca_len=hca_len,
            csa_len=c4_len,
            hca_page_indices=hca_pages,
            csa_width=c4_pages.shape[1],
            win=win,
            ring_stride=ring_stride,
            swa_pages=swa_pages,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
        )
        update_dcp_csa_stream(
            indices=streams[4],
            indptr=streams[5],
            state_slot=state_slot,
            positions=positions,
            swa_len=swa_len,
            c4_page_indices=c4_pages,
            c4_len=c4_len,
            win=win,
            ring_stride=ring_stride,
            swa_pages=swa_pages,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
        )
        return streams

    captured = eager_streams()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fresh = eager_streams()
        for destination, source in zip(captured, fresh):
            destination.copy_(source)

    positions.copy_(torch.tensor([131, 255], dtype=torch.int32, device=DEVICE))
    hca_pages.copy_(
        torch.tensor([[30, -1, -1], [40, 41, 42]], dtype=torch.int32, device=DEVICE)
    )
    hca_len.copy_(torch.tensor([1, 3], dtype=torch.int32, device=DEVICE))
    c4_pages.copy_(
        torch.tensor([[300, -1, -1], [400, 401, -1]], dtype=torch.int32, device=DEVICE)
    )
    c4_len.copy_(torch.tensor([1, 2], dtype=torch.int32, device=DEVICE))
    graph.replay()
    torch.cuda.synchronize()
    expected = eager_streams()

    for actual_indices, actual_indptr, expected_indices, expected_indptr in (
        (captured[0], captured[1], expected[0], expected[1]),
        (captured[2], captured[3], expected[2], expected[3]),
        (captured[4], captured[5], expected[4], expected[5]),
    ):
        torch.testing.assert_close(actual_indptr, expected_indptr)
        for row in range(positions.numel()):
            actual = actual_indices[actual_indptr[row] : actual_indptr[row + 1]]
            wanted = expected_indices[expected_indptr[row] : expected_indptr[row + 1]]
            torch.testing.assert_close(actual, wanted)
