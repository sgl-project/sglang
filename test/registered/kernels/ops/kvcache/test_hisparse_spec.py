from __future__ import annotations

from typing import NamedTuple

import pytest
import torch

from sglang.kernels.ops.kvcache.hisparse import (
    HiSparseSpecState,
    copy_cache_planned_mla,
    load_cache_to_device_buffer_spec_mla,
)
from sglang.srt.utils import is_npu, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or is_npu() or is_xpu(),
    reason="HiSparse speculative swap tests require a CUDA GPU.",
)

DEVICE = "cuda"
TOKEN_SCALE = 1_000_003


class _SwapState(NamedTuple):
    device_buffer_tokens: torch.Tensor
    device_buffer_locs: torch.Tensor
    host_cache_locs: torch.Tensor
    host_cache: torch.Tensor
    device_buffer: torch.Tensor
    swap_state: HiSparseSpecState


def _make_cache_index(num_reqs: int, hot_buffer_size: int) -> torch.Tensor:
    hash_size = 1 << (2 * hot_buffer_size - 1).bit_length()
    cache_index = torch.full(
        (num_reqs, 2, hash_size), -1, dtype=torch.int64, device=DEVICE
    )
    tokens = torch.arange(hot_buffer_size, dtype=torch.int64, device=DEVICE)
    hash_slots = ((tokens * 2654435761) & (hash_size - 1)).to(torch.long)
    packed_entries = (tokens << 32) | tokens
    cache_index[:, 0, hash_slots] = packed_entries
    return cache_index


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

    scratch_state = torch.full(
        (num_reqs + 1, max(4 * num_reqs, 5 * metadata_occurrences)),
        -1,
        dtype=torch.int32,
        device=DEVICE,
    )
    scratch_state[0].zero_()
    swap_state = HiSparseSpecState(
        cache_index=_make_cache_index(num_reqs, hot_buffer_size),
        cache_policy=torch.zeros(
            (num_reqs + 1, hot_buffer_size),
            dtype=torch.int32,
            device=DEVICE,
        ),
        scratch_locs=scratch_locs,
        scratch_state=scratch_state,
    )
    return _SwapState(
        device_buffer_tokens=device_buffer_tokens,
        device_buffer_locs=device_buffer_locs,
        host_cache_locs=host_cache_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        swap_state=swap_state,
    )


def _run_swap(
    *,
    top_k_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    state: _SwapState,
    out: torch.Tensor | None = None,
    req_pool_indices: torch.Tensor | None = None,
    num_real_reqs: torch.Tensor | None = None,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.full_like(top_k_tokens, -1)
    else:
        out.fill_(-1)
    num_reqs = top_k_tokens.size(0)
    if req_pool_indices is None:
        req_pool_indices = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)
    if num_real_reqs is None:
        num_real_reqs = torch.tensor([num_reqs], dtype=torch.int32, device=DEVICE)
    load_cache_to_device_buffer_spec_mla(
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=state.device_buffer_tokens,
        host_cache_locs=state.host_cache_locs,
        device_buffer_locs=state.device_buffer_locs,
        host_cache=state.host_cache,
        device_buffer=state.device_buffer,
        top_k_device_locs=out,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        state=state.swap_state,
        num_real_reqs=num_real_reqs,
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
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


class TestHiSparseSpec(CustomTestCase):
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
        self.assertEqual(int(state.swap_state.scratch_state[0, 0].item()), miss_count)
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
        self.assertEqual(int(state.swap_state.scratch_state[0, 0].item()), 782)

    def test_records_union_plan_for_shared_layer_io(self) -> None:
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
        for step_idx, step_miss_count in enumerate((196, 196, 195, 195)):
            hits = torch.roll(
                torch.arange(hot_size, dtype=torch.int32, device=DEVICE),
                step_idx * 137,
            )[: top_k - step_miss_count]
            misses = torch.arange(
                next_miss,
                next_miss + step_miss_count,
                dtype=torch.int32,
                device=DEVICE,
            )
            next_miss += step_miss_count
            steps.append(torch.cat((hits, misses)))
        top_k_tokens = torch.stack(steps).unsqueeze(0).contiguous()
        seq_lens = torch.full((num_steps,), 16384, dtype=torch.int32, device=DEVICE)
        miss_src = torch.full(
            (1, total_occurrences), -1, dtype=torch.int64, device=DEVICE
        )
        miss_dst = torch.full(
            (1, total_occurrences), -1, dtype=torch.int32, device=DEVICE
        )
        miss_count = torch.full((1,), -1, dtype=torch.int32, device=DEVICE)

        _run_swap(
            top_k_tokens=top_k_tokens,
            seq_lens=seq_lens,
            state=state,
            miss_src=miss_src,
            miss_dst=miss_dst,
            miss_count=miss_count,
        )
        shared_layer_buffer = torch.full_like(state.device_buffer, -1)
        copy_cache_planned_mla(
            miss_src=miss_src,
            miss_dst=miss_dst,
            miss_count=miss_count,
            num_real_reqs=torch.ones(1, dtype=torch.int32, device=DEVICE),
            host_cache=state.host_cache,
            device_buffer=shared_layer_buffer,
            item_size_bytes=state.host_cache.stride(0)
            * state.host_cache.element_size(),
        )
        torch.cuda.synchronize()

        self.assertEqual(int(miss_count.item()), 782)
        count = int(miss_count.item())
        src = miss_src[0, :count].to(torch.long)
        dst = miss_dst[0, :count].to(torch.long)
        torch.testing.assert_close(
            shared_layer_buffer[dst], state.host_cache[src.cpu()].to(DEVICE)
        )
        torch.testing.assert_close(shared_layer_buffer[dst], state.device_buffer[dst])

    def test_padded_request_clears_stale_plan_count(self) -> None:
        hot_size, page_size = 4096, 64
        num_steps, top_k = 4, 2048
        total_occurrences = num_steps * top_k
        state = _make_state(
            num_reqs=2,
            hot_buffer_size=hot_size,
            page_size=page_size,
            scratch_size=hot_size,
            seq_len=8192,
            item_words=1,
            metadata_occurrences=total_occurrences,
        )
        top_k_tokens = torch.arange(top_k, dtype=torch.int32, device=DEVICE).view(
            1, 1, -1
        )
        top_k_tokens = top_k_tokens.repeat(2, num_steps, 1).contiguous()
        seq_lens = torch.full((2 * num_steps,), 8192, dtype=torch.int32, device=DEVICE)
        miss_src = torch.full(
            (2, total_occurrences), -1, dtype=torch.int64, device=DEVICE
        )
        miss_dst = torch.full(
            (2, total_occurrences), -1, dtype=torch.int32, device=DEVICE
        )
        miss_count = torch.full((2,), 123, dtype=torch.int32, device=DEVICE)

        _run_swap(
            top_k_tokens=top_k_tokens,
            seq_lens=seq_lens,
            state=state,
            num_real_reqs=torch.ones(1, dtype=torch.int32, device=DEVICE),
            miss_src=miss_src,
            miss_dst=miss_dst,
            miss_count=miss_count,
        )
        torch.cuda.synchronize()

        self.assertEqual(int(miss_count[1].item()), 0)

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
        self.assertEqual(int(state.swap_state.scratch_state[0, 0].item()), 0)

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

        miss_src = torch.full(
            (1, total_occurrences), -1, dtype=torch.int64, device=DEVICE
        )
        miss_dst = torch.full(
            (1, total_occurrences), -1, dtype=torch.int32, device=DEVICE
        )
        miss_count = torch.full((1,), -1, dtype=torch.int32, device=DEVICE)
        out = _run_swap(
            top_k_tokens=top_k_tokens,
            seq_lens=seq_lens,
            state=state,
            miss_src=miss_src,
            miss_dst=miss_dst,
            miss_count=miss_count,
        )
        torch.cuda.synchronize()

        _assert_output_matches_tokens(state, out, top_k_tokens)
        self.assertEqual(torch.unique(out).numel(), total_occurrences)
        self.assertEqual(
            int(state.swap_state.scratch_state[0, 0].item()), total_occurrences
        )
        self.assertEqual(int(miss_count.item()), total_occurrences)
        self.assertTrue(miss_src.ge(0).all().item())
        self.assertTrue(miss_dst.ge(0).all().item())

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

        # The first call admits the high token into the packed hash. The
        # second call must resolve it as a hot hit rather than truncating the
        # packed int64 entry and repeating Host-to-GPU IO.
        out = _run_swap(top_k_tokens=top_k_tokens, seq_lens=seq_lens, state=state)
        torch.cuda.synchronize()
        _assert_output_matches_tokens(state, out, top_k_tokens)
        self.assertEqual(int(state.swap_state.scratch_state[0, 0].item()), 0)

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
        req_pool_indices = torch.arange(1, dtype=torch.int64, device=DEVICE)
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
