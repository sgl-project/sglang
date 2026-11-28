"""Exercise the B12 union with real 656-byte target KV and adversarial misses."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-small")
from test_hisparse_spec import _assert_output_matches_tokens, _make_state, _run_swap


@pytest.mark.parametrize("steps", [1, 2, 4, 8, 12])
@pytest.mark.parametrize("pattern", ["repeated", "disjoint"])
@pytest.mark.parametrize("use_graph", [False, True])
def test_union_keeps_every_query_location_valid(steps, pattern, use_graph):
    hot, topk, page = 4096, 2048, 64
    occurrences = steps * topk
    metadata = (
        32768
        if steps == 1
        else (
            (1 << (occurrences - 1).bit_length())
            if occurrences > 16384
            else occurrences
        )
    )
    state = _make_state(
        num_reqs=1,
        hot_buffer_size=hot,
        page_size=page,
        scratch_size=20480 if steps == 1 else max(1024, occurrences - hot),
        seq_len=65536,
        item_words=82,
        metadata_occurrences=metadata,
    )
    if pattern == "disjoint":
        tokens = torch.arange(
            hot, hot + occurrences, dtype=torch.int32, device="cuda"
        ).view(1, steps, topk)
    else:
        tokens = (
            torch.arange(hot, hot + topk, dtype=torch.int32, device="cuda")
            .view(1, 1, topk)
            .repeat(1, steps, 1)
        )
    tokens = tokens.contiguous()
    seq_lens = torch.full((steps,), 65536, dtype=torch.int32, device="cuda")
    request_rows = torch.zeros(1, dtype=torch.int64, device="cuda")
    real_requests = torch.ones(1, dtype=torch.int32, device="cuda")
    kwargs = dict(req_pool_indices=request_rows, num_real_reqs=real_requests)
    output = _run_swap(top_k_tokens=tokens, seq_lens=seq_lens, state=state, **kwargs)
    torch.cuda.synchronize()
    _assert_output_matches_tokens(state, output, tokens)
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run_swap(
                top_k_tokens=tokens,
                seq_lens=seq_lens,
                state=state,
                out=output,
                **kwargs,
            )
        # Change to a fully cold union without changing any address/shape.
        tokens.add_(28672 if pattern == "disjoint" else 8192)
        graph.replay()
        torch.cuda.synchronize()
        _assert_output_matches_tokens(state, output, tokens)
    # A second swap may replace the ring, but it must preserve all locations
    # needed by the current union until the caller has consumed them.
    output = _run_swap(
        top_k_tokens=tokens, seq_lens=seq_lens, state=state, out=output, **kwargs
    )
    torch.cuda.synchronize()
    _assert_output_matches_tokens(state, output, tokens)


@pytest.mark.parametrize("use_graph", [False, True])
def test_target_only_tail_reuses_b12_workspace_and_resolves_new_token(use_graph):
    hot, topk, page = 4096, 2048, 64
    state = _make_state(
        num_reqs=1,
        hot_buffer_size=hot,
        page_size=page,
        scratch_size=20480,
        seq_len=65536,
        item_words=82,
        metadata_occurrences=32768,
    )
    # Include hot hits, a cold host range, newest persistent token at HOT,
    # and the current verify token in HOT+1, still absent from Host.
    tokens = torch.arange(hot, hot + topk, dtype=torch.int32, device="cuda").view(
        1, 1, topk
    )
    tokens[0, 0, :4] = torch.tensor(
        [0, 100, 60000, 60001], dtype=torch.int32, device="cuda"
    )
    for slot, token in [(hot, 60000), (hot + 1, 60001)]:
        location = state.device_buffer_locs[0, slot].long()
        state.device_buffer_tokens[0, slot] = token
        state.device_buffer[location].copy_(state.host_cache[token].cuda())
    # Poison Host values for resident tokens; passing must use the actual
    # persistent/extra GPU row, rather than reread a not-yet-committed host row.
    expected = state.host_cache[tokens.long().cpu()].cuda()
    state.host_cache[60000:60002].fill_(-777)
    lengths = torch.tensor([60002], dtype=torch.int32, device="cuda")
    rows = torch.zeros(1, dtype=torch.int64, device="cuda")
    real = torch.ones(1, dtype=torch.int32, device="cuda")
    output = _run_swap(
        top_k_tokens=tokens,
        seq_lens=lengths,
        state=state,
        req_pool_indices=rows,
        num_real_reqs=real,
    )
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run_swap(
                top_k_tokens=tokens,
                seq_lens=lengths,
                state=state,
                out=output,
                req_pool_indices=rows,
                num_real_reqs=real,
            )
        graph.replay()
    torch.cuda.synchronize()
    actual = state.device_buffer[output.long()]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
