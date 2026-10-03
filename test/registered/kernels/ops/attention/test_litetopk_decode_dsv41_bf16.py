"""Paired BF16 exact histogram and real serving routes for next_n 1..6."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    stage="base-b-kernel-unit",
    runner_config="4-gpu-b200",
    disabled="Requires paired DeepGEMM BF16 exact-histogram build",
)

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_TEST_DSV41_BF16_EXACT") != "1",
    reason="requires the paired sgl-gemm BF16 exact-histogram build",
)


def _inputs(requests, next_n, length):
    rows, pages = requests * next_n, (length + 127) // 128
    q = torch.randint(0, 256, (rows, 1, 32, 64), device="cuda", dtype=torch.uint8).view(
        torch.int8
    )
    sf = torch.full((rows, 1, 32), 0x7D7D7D7D, device="cuda", dtype=torch.int32)
    cache = torch.randint(
        0, 256, (requests * pages, 128 * 68), device="cuda", dtype=torch.uint8
    )
    cache[:, 128 * 64 :] = 125
    ids = torch.arange(requests, device="cuda", dtype=torch.int32).repeat_interleave(
        next_n
    )
    table = torch.randperm(requests * pages, device="cuda").int().view(requests, pages)
    table = table.repeat_interleave(next_n, 0).contiguous()
    lens = (
        length
        - next_n
        + 1
        + torch.arange(rows, device="cuda", dtype=torch.int32) % next_n
    ).view(rows, 1)
    return dict(
        q=(q, sf),
        kv_cache=cache.view(requests * pages, 128, 1, 68),
        weights=torch.randn((rows, 32), device="cuda", dtype=torch.bfloat16),
        context_lens=lens,
        block_table=table,
        indices=ids,
        tokens_per_request=next_n,
        max_context_len=pages * 128,
    )


def _schedule(dg, data):
    return dg.get_paged_mqa_logits_bf16_metadata(
        data["context_lens"],
        128,
        dg.get_num_sms(),
        indices=data["indices"],
        tokens_per_request=data["tokens_per_request"],
    )


def _reference(dg, data, schedule):
    return dg.fp4_paged_mqa_logits_bf16(
        data["q"],
        data["kv_cache"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        schedule,
        data["max_context_len"],
        indices=data["indices"],
        tokens_per_request=data["tokens_per_request"],
    )


def _check(scores, reference, out, data):
    for row, length in enumerate(data["context_lens"].view(-1).tolist()):
        actual = scores[row, :length].contiguous().view(torch.int16)
        expected = reference[row, :length].contiguous().view(torch.int16)
        assert torch.equal(actual, expected)
        bits = expected.long() & 0xFFFF
        keys = torch.where(bits >= 0x8000, bits ^ 0xFFFF, bits | 0x8000)
        columns = torch.arange(length, device="cuda")
        slots = data["block_table"][row, columns // 128].long() * 128 + columns % 128
        composite = (keys << 31) | (0x7FFFFFFF - slots)
        keep = min(length, 512)
        want = slots[composite.topk(keep).indices].sort().values
        assert torch.equal(out[row, :keep].long().sort().values, want)
        assert bool((out[row, keep:] == -1).all())


def _at_rest(storage):
    state = storage.workspace[: storage.max_rows * 16].view(torch.int32).view(-1, 4)
    assert not bool(storage.histogram.any())
    assert not bool(state[:, :2].any())
    assert torch.equal(state[:, 2], state[:, 3])
    assert not bool(storage.workspace[storage.max_rows * 16 :].any())


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("next_n", [1, 2, 3, 4, 5, 6])
def test_shared_requests_and_dynamic_graph(pdl, next_n):
    import deep_gemm as dg

    from sglang.kernels.experimental.litetopk_decode.bf16 import Bf16Dsv41DecodeStorage

    torch.manual_seed(20261003 + next_n)
    old_pdl = dg.get_pdl()
    dg.set_pdl(pdl)
    try:
        requests, length = 4, 262144
        rows = requests * next_n
        storage = Bf16Dsv41DecodeStorage(24, torch.device("cuda"))
        plan = storage.plan(rows)
        data = _inputs(requests, next_n, length)
        schedule = _schedule(dg, data)
        out = torch.empty((rows, 512), dtype=torch.int32, device="cuda")
        scores, slots = plan(**data, schedule_metadata=schedule, out=out)
        _check(scores, _reference(dg, data, schedule), slots, data)
        _at_rest(storage)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            scores, slots = plan(**data, schedule_metadata=_schedule(dg, data), out=out)
        torch.cuda.current_stream().wait_stream(stream)
        for new_length in (262144, 513, 16384, 65536):
            data["context_lens"].copy_(
                (
                    new_length
                    - next_n
                    + 1
                    + torch.arange(rows, device="cuda", dtype=torch.int32) % next_n
                ).view(rows, 1)
            )
            graph.replay()
            _check(scores, _reference(dg, data, _schedule(dg, data)), slots, data)
            _at_rest(storage)
    finally:
        dg.set_pdl(old_pdl)


def _metadata(data):
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata

    return PagedIndexerMetadata(
        page_size=256,
        compressed_page_size=128,
        page_table=data["block_table"],
        compressed_seq_lens=data["context_lens"].view(-1),
        use_topk_v2=True,
        force_deep_gemm_metadata=True,
        compress_ratio=2,
    )


def _serving(monkeypatch, data):
    from sglang.srt.layers.attention.dsv4.v41_indexer import full_topk, sparse_table
    from sglang.srt.layers.attention.dsv4.v41_indexer.litetopk import LiteTopKDecode
    from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import DeepGEMMDecodeData
    from sglang.srt.layers.attention.dsv4.v41_indexer.types import Selection

    # Only projection/packing is substituted; the indexers and GPU kernels run.
    packed = DeepGEMMDecodeData(*data["q"], data["weights"].float(), data["kv_cache"])
    for module in (full_topk, sparse_table):
        monkeypatch.setattr(module, "get_deep_gemm_decode_data", lambda *_: packed)
    hook = LiteTopKDecode(device=torch.device("cuda", 0), check=False, check_file=None)
    full = full_topk.FullTopKIndexer(
        token_to_kv_pool=None,
        req_to_token=None,
        use_deep_gemm_prefill=False,
        use_deep_gemm_decode=True,
        litetopk=hook,
    )
    sparse = sparse_table.SparseTableBackend(
        token_to_kv_pool=None,
        req_to_token=None,
        page_size=256,
        candidate_topk_blocks=512,
        candidate_block_size=8,
        litetopk=hook,
    )
    inputs = SimpleNamespace(
        paged_metadata=_metadata(data),
        is_verify=data["tokens_per_request"] > 1,
        req_rows=data["indices"],
    )
    out = Selection(
        torch.empty(
            data["context_lens"].numel(), 512, device="cuda", dtype=torch.int32
        ),
        None,
    )
    return hook, full, sparse, inputs, out


@pytest.mark.parametrize("next_n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("publish", [False, True])
def test_serving_full_and_sparse_dynamic_graph(monkeypatch, next_n, publish):
    import deep_gemm as dg

    from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
        amax_topk_blocks,
        candidate_row_lens,
    )
    from sglang.kernels.ops.attention.dsv4.candidate_table import sort_candidate_blocks

    if publish and not hasattr(dg, "get_paged_sparse_mqa_logits_metadata"):
        pytest.skip("SparseTable requires SGLang's existing sparse DeepGEMM extension")
    torch.manual_seed(631 + next_n)
    data = _inputs(2, next_n, 16384)
    hook, full, sparse, inputs, out = _serving(monkeypatch, data)

    def forward():
        hook.prepare_metadata(inputs.paged_metadata, inputs.req_rows, next_n)
        if publish:
            table = sparse.publish_decode(inputs, out)
            torch.cuda.current_stream().wait_event(table.ready)
            return table
        full.topk_decode(inputs, out)
        return None

    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        table = forward()
    for length, split_runs in ((16384, False), (513, True), (8193, False)):
        rows = data["context_lens"].numel()
        data["context_lens"].copy_(
            (
                length
                - next_n
                + 1
                + torch.arange(rows, device="cuda", dtype=torch.int32) % next_n
            ).view(-1, 1)
        )
        ids = torch.arange(rows, device="cuda", dtype=torch.int32)
        data["indices"].copy_(ids if split_runs else ids // next_n)
        # Each same-ID run still shares its page table. Physical order changes.
        data["block_table"].copy_(data["block_table"].flip(1))
        graph.replay()
        reference = _reference(dg, data, _schedule(dg, data))
        _check(reference, reference, out.page_indices, data)
        _at_rest(hook._storage)
        if publish:
            lens = data["context_lens"].view(-1)
            counts, valid = candidate_row_lens(lens, 512)
            expected = amax_topk_blocks(reference.float(), lens, counts, 512)
            phys = sort_candidate_blocks(expected, lens, data["block_table"], 128)
            assert torch.equal(table.blocks, expected)
            assert torch.equal(table.phys_blocks, phys)
            assert torch.equal(table.valid_lens, valid)


def test_serving_fallback_and_metadata_copy(monkeypatch):
    from sglang.srt.layers.attention.dsv4.v41_indexer.types import Selection

    data = _inputs(2, 6, 511)
    hook, full, _, inputs, out = _serving(monkeypatch, data)
    md = inputs.paged_metadata
    hook.prepare_metadata(md, inputs.req_rows, 6)
    full.topk_decode(inputs, out)
    other = _metadata(_inputs(2, 6, 511))
    hook.prepare_metadata(other, inputs.req_rows.clone(), 6)
    ptrs = (md.bf16_schedule.data_ptr(), md.bf16_indices.data_ptr())
    md.copy_(other)
    assert ptrs == (md.bf16_schedule.data_ptr(), md.bf16_indices.data_ptr())
    assert torch.equal(md.bf16_schedule, other.bf16_schedule)
    assert torch.equal(md.bf16_indices, other.bf16_indices)
    # Unsupported CPU hint clears a previous schedule and runs the real legacy path.
    hook.prepare_metadata(md, inputs.req_rows, 7)
    assert md.bf16_schedule is None
    full.topk_decode(inputs, out)
    fallback = out.page_indices.clone()
    full.litetopk = None
    full.topk_decode(inputs, out)
    assert torch.equal(fallback.sort().values, out.page_indices.sort().values)
    # Raw-position consumers must also use the legacy path, even with a schedule.
    full.litetopk = hook
    hook.prepare_metadata(md, inputs.req_rows, 6)
    raw_out = Selection(out.page_indices, torch.empty_like(out.page_indices))
    full.topk_decode(inputs, raw_out)
    assert torch.equal(fallback.sort().values, out.page_indices.sort().values)
    pos = raw_out.raw_indices.long()
    slots = md.page_table.gather(1, pos.clamp_min(0) // 128).long() * 128 + pos % 128
    slots.masked_fill_(pos < 0, -1)
    assert torch.equal(slots, out.page_indices.long())


def test_serving_check_mode(monkeypatch):
    from sglang.srt.layers.attention.dsv4.v41_indexer.litetopk import LiteTopKDecode

    data = _inputs(2, 6, 511)
    _, full, _, inputs, out = _serving(monkeypatch, data)
    hook = LiteTopKDecode(device=torch.device("cuda", 0), check=True, check_file=None)
    full.litetopk = hook

    def forward():
        hook.prepare_metadata(inputs.paged_metadata, inputs.req_rows, 6)
        full.topk_decode(inputs, out)

    forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    graph.replay()
    torch.cuda.synchronize()
    assert hook._check_counts.tolist() == [24, 0, 2]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_candidate_amax(dtype):
    from sglang.kernels.ops.attention.dsv4.candidate_blocks import amax8_varlen

    scores = torch.randn((8, 16400), device="cuda", dtype=dtype)
    scores[:, 7] = torch.nan
    lens = torch.tensor(
        [0, 1, 7, 8, 9, 513, 8193, 16391], device="cuda", dtype=torch.int32
    )
    out = torch.full((8, 2050), -321.0, device="cuda")
    amax8_varlen(scores, lens, out=out)
    ref = scores.float().nan_to_num(nan=-torch.inf).view(8, -1, 8).amax(-1)
    for row, length in enumerate(lens.tolist()):
        n = (length + 7) // 8
        if n:
            ref[row, n - 1] = torch.inf
        assert torch.equal(out[row, :n], ref[row, :n])
        assert bool((out[row, n:] == -321).all())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
