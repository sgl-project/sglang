from __future__ import annotations

from typing import NamedTuple

import pytest
import torch

from sglang.kernels.ops.kvcache.hisparse_mtp import (
    HiSparseMTPCacheState,
    HiSparseMTPMissWorkspace,
    load_cache_to_device_buffer_mtp_mla,
)
from sglang.srt.utils import is_npu, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or is_npu() or is_xpu(),
    reason="HiSparse MTP swap tests require a CUDA GPU.",
)

DEVICE = "cuda"
TOKEN_SCALE = 1_000_003


class _SwapState(NamedTuple):
    device_buffer_tokens: torch.Tensor
    device_buffer_locs: torch.Tensor
    host_cache_locs: torch.Tensor
    host_cache: torch.Tensor
    device_buffer: torch.Tensor
    cache_state: HiSparseMTPCacheState
    miss_workspace: HiSparseMTPMissWorkspace


def _make_ring_hash(
    num_reqs: int, hot_buffer_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    hash_size = 1 << (2 * hot_buffer_size - 1).bit_length()
    primary = torch.full((num_reqs, hash_size), -1, dtype=torch.int64, device=DEVICE)
    secondary = torch.full_like(primary, -1)
    tokens = torch.arange(hot_buffer_size, dtype=torch.int64, device=DEVICE)
    hash_slots = ((tokens * 2654435761) & (hash_size - 1)).to(torch.long)
    slot_bits = (hot_buffer_size - 1).bit_length()
    packed_entries = (tokens << slot_bits) | tokens
    primary[:, hash_slots] = packed_entries
    return primary, secondary


def _make_state(
    *,
    num_reqs: int,
    hot_buffer_size: int,
    page_size: int,
    scratch_size: int,
    seq_len: int,
    item_words: int,
    metadata_occurrences: int,
) -> _SwapState:
    buffer_size = hot_buffer_size + page_size
    device_buffer_tokens = torch.full(
        (num_reqs, buffer_size), -1, dtype=torch.int32, device=DEVICE
    )
    device_buffer_tokens[:, :hot_buffer_size] = torch.arange(
        hot_buffer_size, dtype=torch.int32, device=DEVICE
    )

    physical_tokens_per_req = buffer_size + scratch_size
    request_bases = (
        torch.arange(num_reqs, dtype=torch.int32, device=DEVICE).view(-1, 1)
        * physical_tokens_per_req
    )
    device_buffer_locs = (
        request_bases
        + torch.arange(buffer_size, dtype=torch.int32, device=DEVICE).view(1, -1)
    ).contiguous()
    scratch_locs = (
        request_bases
        + buffer_size
        + torch.arange(scratch_size, dtype=torch.int32, device=DEVICE).view(1, -1)
    ).contiguous()

    host_cache_locs = torch.arange(seq_len, dtype=torch.int64, device=DEVICE)
    host_cache_locs = host_cache_locs.view(1, -1).repeat(num_reqs, 1).contiguous()
    host_cache = torch.empty((seq_len, item_words), dtype=torch.int64, pin_memory=True)
    host_cache.copy_(
        torch.arange(seq_len, dtype=torch.int64).view(-1, 1) * TOKEN_SCALE
        + torch.arange(item_words, dtype=torch.int64).view(1, -1)
    )

    device_buffer = torch.full(
        (num_reqs * physical_tokens_per_req, item_words),
        -1,
        dtype=torch.int64,
        device=DEVICE,
    )
    hot_locs = device_buffer_locs[:, :hot_buffer_size].to(torch.long)
    device_buffer[hot_locs] = host_cache[:hot_buffer_size].to(DEVICE)

    hash_primary, hash_secondary = _make_ring_hash(num_reqs, hot_buffer_size)
    cache_state = HiSparseMTPCacheState(
        hash_primary=hash_primary,
        hash_secondary=hash_secondary,
        ring_state=torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        ref_epochs=torch.zeros(
            (num_reqs, hot_buffer_size), dtype=torch.int32, device=DEVICE
        ),
    )
    miss_workspace = HiSparseMTPMissWorkspace(
        locs=scratch_locs,
        metadata=torch.full(
            (num_reqs, 5 * metadata_occurrences),
            -1,
            dtype=torch.int32,
            device=DEVICE,
        ),
        counters=torch.zeros(4 * num_reqs, dtype=torch.int32, device=DEVICE),
    )
    return _SwapState(
        device_buffer_tokens=device_buffer_tokens,
        device_buffer_locs=device_buffer_locs,
        host_cache_locs=host_cache_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        cache_state=cache_state,
        miss_workspace=miss_workspace,
    )


def _run_swap(
    *,
    top_k_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    state: _SwapState,
    out: torch.Tensor | None = None,
    req_pool_indices: torch.Tensor | None = None,
    num_real_reqs: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.full_like(top_k_tokens, -1)
    else:
        out.fill_(-1)
    num_reqs = top_k_tokens.size(0)
    if req_pool_indices is None:
        req_pool_indices = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    if num_real_reqs is None:
        num_real_reqs = torch.tensor([num_reqs], dtype=torch.int32, device=DEVICE)
    load_cache_to_device_buffer_mtp_mla(
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=state.device_buffer_tokens,
        host_cache_locs=state.host_cache_locs,
        device_buffer_locs=state.device_buffer_locs,
        host_cache=state.host_cache,
        device_buffer=state.device_buffer,
        top_k_device_locs=out,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        cache_state=state.cache_state,
        miss_workspace=state.miss_workspace,
        num_real_reqs=num_real_reqs,
    )
    return out


def _assert_output_matches_tokens(
    state: _SwapState, out: torch.Tensor, tokens: torch.Tensor
) -> None:
    actual = state.device_buffer[out.to(torch.long)]
    expected = tokens.to(torch.int64).unsqueeze(-1) * TOKEN_SCALE + torch.arange(
        state.device_buffer.size(-1), dtype=torch.int64, device=DEVICE
    )
    torch.testing.assert_close(actual, expected)


class TestHiSparseMTPSwap(CustomTestCase):
    def test_deduplicates_repeated_misses_and_copies_full_items(self) -> None:
        hot_size, page_size = 4096, 64
        num_steps, top_k, item_words = 4, 2048, 72
        total_occurrences = num_steps * top_k
        state = _make_state(
            num_reqs=1,
            hot_buffer_size=hot_size,
            page_size=page_size,
            scratch_size=hot_size,
            seq_len=16384,
            item_words=item_words,
            metadata_occurrences=total_occurrences,
        )

        miss_count = 196
        hits = torch.arange(top_k - miss_count, dtype=torch.int32, device=DEVICE)
        shared_misses = hot_size + torch.arange(
            miss_count, dtype=torch.int32, device=DEVICE
        )
        step = torch.cat((hits, shared_misses))
        top_k_tokens = step.view(1, 1, -1).repeat(1, num_steps, 1)
        seq_lens = torch.full((num_steps,), 16384, dtype=torch.int32, device=DEVICE)

        out = _run_swap(top_k_tokens=top_k_tokens, seq_lens=seq_lens, state=state)
        torch.cuda.synchronize()

        _assert_output_matches_tokens(state, out, top_k_tokens)
        self.assertEqual(int(state.miss_workspace.counters[0].item()), miss_count)
        repeated_miss_locs = out[0, :, -miss_count:]
        self.assertTrue(torch.all(repeated_miss_locs == repeated_miss_locs[0]).item())

    def test_copies_782_cross_step_unique_misses(self) -> None:
        hot_size, page_size = 4096, 64
        num_steps, top_k, item_words = 4, 2048, 72
        total_occurrences = num_steps * top_k
        state = _make_state(
            num_reqs=1,
            hot_buffer_size=hot_size,
            page_size=page_size,
            scratch_size=hot_size,
            seq_len=16384,
            item_words=item_words,
            metadata_occurrences=total_occurrences,
        )

        steps = []
        next_miss = hot_size
        for step_idx, miss_count in enumerate((196, 196, 195, 195)):
            hits = torch.roll(
                torch.arange(hot_size, dtype=torch.int32, device=DEVICE),
                step_idx * 137,
            )[: top_k - miss_count]
            misses = torch.arange(
                next_miss,
                next_miss + miss_count,
                dtype=torch.int32,
                device=DEVICE,
            )
            next_miss += miss_count
            steps.append(torch.cat((hits, misses)))
        top_k_tokens = torch.stack(steps).unsqueeze(0).contiguous()
        seq_lens = torch.full((num_steps,), 16384, dtype=torch.int32, device=DEVICE)
        out = _run_swap(
            top_k_tokens=top_k_tokens,
            seq_lens=seq_lens,
            state=state,
        )
        torch.cuda.synchronize()

        _assert_output_matches_tokens(state, out, top_k_tokens)
        self.assertEqual(int(state.miss_workspace.counters[0].item()), 782)

    def test_resolves_all_speculative_extra_page_slots_without_host_io(self) -> None:
        hot_size, page_size = 4096, 64
        num_steps, top_k = 4, 2048
        total_occurrences = num_steps * top_k
        seq_len = 8192
        state = _make_state(
            num_reqs=1,
            hot_buffer_size=hot_size,
            page_size=page_size,
            scratch_size=hot_size,
            seq_len=seq_len,
            item_words=72,
            metadata_occurrences=total_occurrences,
        )

        draft_tokens = torch.arange(
            seq_len - num_steps, seq_len, dtype=torch.int32, device=DEVICE
        )
        extra_offsets = torch.tensor([0, 7, 31, 63], device=DEVICE)
        extra_locs = state.device_buffer_locs[0, hot_size + extra_offsets].to(
            torch.long
        )
        state.device_buffer_tokens[0, hot_size + extra_offsets] = draft_tokens
        state.device_buffer[extra_locs] = state.host_cache[
            draft_tokens.to(device="cpu", dtype=torch.long)
        ].to(DEVICE)
        state.host_cache_locs[0, draft_tokens.to(torch.long)] = -1

        hits = torch.arange(top_k - 1, dtype=torch.int32, device=DEVICE)
        top_k_tokens = torch.stack(
            [torch.cat((hits, draft_tokens[step : step + 1])) for step in range(4)]
        ).unsqueeze(0)
        seq_lens = draft_tokens + 1

        out = _run_swap(top_k_tokens=top_k_tokens, seq_lens=seq_lens, state=state)
        torch.cuda.synchronize()

        _assert_output_matches_tokens(state, out, top_k_tokens)
        torch.testing.assert_close(out[0, :, -1].to(torch.long), extra_locs)
        self.assertEqual(int(state.miss_workspace.counters[0].item()), 0)

    def test_full_union_overflow_preserves_all_8192_outputs(self) -> None:
        hot_size, page_size = 4096, 64
        num_steps, top_k = 4, 2048
        total_occurrences = num_steps * top_k
        state = _make_state(
            num_reqs=1,
            hot_buffer_size=hot_size,
            page_size=page_size,
            scratch_size=total_occurrences - hot_size,
            seq_len=16384,
            item_words=72,
            metadata_occurrences=total_occurrences,
        )
        top_k_tokens = (
            hot_size + torch.arange(total_occurrences, dtype=torch.int32, device=DEVICE)
        ).view(1, num_steps, top_k)
        seq_lens = torch.full((num_steps,), 16384, dtype=torch.int32, device=DEVICE)

        out = _run_swap(top_k_tokens=top_k_tokens, seq_lens=seq_lens, state=state)
        torch.cuda.synchronize()

        _assert_output_matches_tokens(state, out, top_k_tokens)
        self.assertEqual(torch.unique(out).numel(), total_occurrences)
        self.assertEqual(
            int(state.miss_workspace.counters[0].item()), total_occurrences
        )

    def test_packed_ring_supports_glm52_native_context_length(self) -> None:
        hot_size, page_size = 4096, 64
        num_steps, top_k = 4, 2048
        total_occurrences = num_steps * top_k
        seq_len = 1_048_648
        state = _make_state(
            num_reqs=1,
            hot_buffer_size=hot_size,
            page_size=page_size,
            scratch_size=hot_size,
            seq_len=seq_len,
            item_words=1,
            metadata_occurrences=total_occurrences,
        )
        top_k_tokens = torch.arange(top_k, dtype=torch.int32, device=DEVICE).view(
            1, 1, -1
        )
        top_k_tokens = top_k_tokens.repeat(1, num_steps, 1)
        high_token = seq_len - 1
        top_k_tokens[:, :, -1] = high_token
        seq_lens = torch.full((num_steps,), seq_len, dtype=torch.int32, device=DEVICE)

        out = _run_swap(top_k_tokens=top_k_tokens, seq_lens=seq_lens, state=state)
        torch.cuda.synchronize()

        _assert_output_matches_tokens(state, out, top_k_tokens)
        self.assertTrue(out.ge(0).all().item())

    def test_cuda_graph_replay_preserves_valid_locations(self) -> None:
        hot_size, page_size = 4096, 64
        num_steps, top_k = 4, 2048
        total_occurrences = num_steps * top_k
        state = _make_state(
            num_reqs=1,
            hot_buffer_size=hot_size,
            page_size=page_size,
            scratch_size=hot_size,
            seq_len=65536,
            item_words=72,
            metadata_occurrences=total_occurrences,
        )
        top_k_tokens = torch.arange(top_k, dtype=torch.int32, device=DEVICE).repeat(
            num_steps, 1
        )
        for step, miss_count in enumerate((164, 102, 61, 20)):
            top_k_tokens[step, -miss_count:] = torch.arange(
                8192 + step * top_k,
                8192 + step * top_k + miss_count,
                dtype=torch.int32,
                device=DEVICE,
            )
        top_k_tokens = top_k_tokens.unsqueeze(0).contiguous()
        seq_lens = torch.tensor(
            [65533, 65534, 65535, 65536], dtype=torch.int32, device=DEVICE
        )

        _run_swap(top_k_tokens=top_k_tokens, seq_lens=seq_lens, state=state)
        torch.cuda.synchronize()
        graph_out = torch.full_like(top_k_tokens, -1)
        req_pool_indices = torch.arange(1, dtype=torch.int32, device=DEVICE)
        num_real_reqs = torch.tensor([1], dtype=torch.int32, device=DEVICE)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run_swap(
                top_k_tokens=top_k_tokens,
                seq_lens=seq_lens,
                state=state,
                out=graph_out,
                req_pool_indices=req_pool_indices,
                num_real_reqs=num_real_reqs,
            )

        for _ in range(4):
            graph.replay()
        torch.cuda.synchronize()

        _assert_output_matches_tokens(state, graph_out, top_k_tokens)
        self.assertTrue(graph_out.ge(0).all().item())

    def test_rejects_invalid_step_shape_before_compilation(self) -> None:
        state = _make_state(
            num_reqs=1,
            hot_buffer_size=4096,
            page_size=64,
            scratch_size=4096,
            seq_len=8192,
            item_words=1,
            metadata_occurrences=8192,
        )
        with self.assertRaisesRegex(ValueError, "2-4 steps"):
            _run_swap(
                top_k_tokens=torch.zeros(
                    (1, 1, 2048), dtype=torch.int32, device=DEVICE
                ),
                seq_lens=torch.tensor([8192], dtype=torch.int32, device=DEVICE),
                state=state,
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
