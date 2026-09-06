"""QSA prefill synchronization and compressed-key pack tests."""

import ast
import inspect
import textwrap

import pytest
import torch

from sglang.srt.layers.attention.qsa.kernel import (
    qsa_pack_prefill_compressed_keys,
    qsa_prefill_all_visible_indices,
)
from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _forbidden_calls(function, names: set[str]) -> list[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in names
    ]


def test_qsa_prefill_hot_paths_have_no_explicit_host_materialization():
    hot_paths = (
        QSAIndexer.project_qk,
        QSAIndexer.apply_rope,
        QSAIndexerMetadata.get_prefill_mqa_inputs,
        QwenSparseAttnBackend._metadata_from_forward_batch,
    )
    for function in hot_paths:
        assert not _forbidden_calls(
            function, {"item", "tolist", "cpu"}
        ), function.__qualname__
    for function in (QSAIndexer.project_qk, QSAIndexer.apply_rope):
        assert not _forbidden_calls(function, {"max"}), function.__qualname__


def test_qsa_prefill_pack_has_no_eager_gather_chain():
    assert not _forbidden_calls(
        QSAIndexerMetadata.get_prefill_mqa_inputs,
        {"tolist", "index_select", "cat"},
    )


def _fragmented_slot_table(sequence_lens, page_size=64):
    max_length = max(sequence_lens)
    table = torch.zeros(
        (len(sequence_lens), max_length), dtype=torch.int32, device="cuda"
    )
    next_page = 3
    max_page = 0
    for request, length in enumerate(sequence_lens):
        cursor = 0
        while cursor < length:
            page = next_page
            next_page += 2 + (request & 1)
            count = min(page_size, length - cursor)
            table[request, cursor : cursor + count] = page * page_size + torch.arange(
                count, dtype=torch.int32, device="cuda"
            )
            cursor += count
            max_page = max(max_page, page)
    return table, max_page


@pytest.mark.parametrize(
    "sequence_lens",
    (
        (1, 2, 3),
        (3, 4, 7),
        (33,),
        (83, 42),
        (76, 68, 67),
        (33, 45, 56, 64, 68, 75, 88, 101),
        (8192,),
    ),
)
def test_qsa_prefill_compressed_pack_matches_indirect_gather(sequence_lens):
    ratio, heads, head_dim, page_size = 4, 1, 128, 64
    slot_table, max_page = _fragmented_slot_table(sequence_lens, page_size)
    compressed_rows = (max_page + 1) * (page_size // ratio)
    compressed_k = torch.randn(
        compressed_rows,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    block_lens = [length // ratio for length in sequence_lens]
    cu_cpu = [0]
    for length in block_lens:
        cu_cpu.append(cu_cpu[-1] + length)
    seq = torch.tensor(sequence_lens, dtype=torch.int32, device="cuda")
    cu = torch.tensor(cu_cpu, dtype=torch.int32, device="cuda")
    output = torch.empty(
        (cu_cpu[-1], heads, head_dim), dtype=compressed_k.dtype, device="cuda"
    )

    packed = qsa_pack_prefill_compressed_keys(
        compressed_k, slot_table, seq, cu, output, ratio
    )
    assert packed.data_ptr() == output.data_ptr()
    expected = []
    for request, block_count in enumerate(block_lens):
        raw_slots = slot_table[request, : block_count * ratio : ratio]
        expected.append(compressed_k.index_select(0, raw_slots.long() // ratio))
    expected_packed = torch.cat(expected) if expected else output.clone()
    torch.testing.assert_close(packed, expected_packed, rtol=0, atol=0)


def test_qsa_prefill_compressed_scratch_is_reused():
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend._qsa_prefill_compressed_scratch = {}
    device = torch.device("cuda")
    large = backend._get_qsa_prefill_compressed_scratch(
        1024, 1, 128, torch.bfloat16, device
    )
    small = backend._get_qsa_prefill_compressed_scratch(
        52, 1, 128, torch.bfloat16, device
    )
    assert small.data_ptr() == large.data_ptr()


@pytest.mark.parametrize(
    "sequence_lens,query_ranges",
    (
        ((1, 3, 4), None),
        ((33,), None),
        ((80, 36), None),
        ((76, 68, 67), None),
        ((1024, 2048), ((1008, 1024), (2032, 2048))),
        ((2047, 2048), ((2039, 2047), (2040, 2048))),
    ),
)
def test_qsa_prefill_all_visible_indices_matches_reference(sequence_lens, query_ranges):
    token_topk, compress_ratio = 2048, 4
    ranges = (
        tuple((0, length) for length in sequence_lens)
        if query_ranges is None
        else query_ranges
    )
    position_parts = []
    request_parts = []
    for request, (start, end) in enumerate(ranges):
        position_parts.append(
            torch.arange(start, end, dtype=torch.int64, device="cuda")
        )
        request_parts.append(
            torch.full((end - start,), request, dtype=torch.int32, device="cuda")
        )
    positions = torch.cat(position_parts)
    request_ids = torch.cat(request_parts)
    lengths = torch.tensor(sequence_lens, dtype=torch.int32, device="cuda")
    output = torch.empty(
        (positions.numel(), token_topk + compress_ratio - 1),
        dtype=torch.int32,
        device="cuda",
    )

    actual = qsa_prefill_all_visible_indices(
        positions,
        request_ids,
        lengths,
        output,
        token_topk,
        compress_ratio,
    )
    columns = torch.arange(output.shape[1], dtype=torch.int32, device="cuda")
    row_lengths = lengths.index_select(0, request_ids.long()).long()
    visible = torch.minimum(positions + 1, row_lengths).clamp(max=token_topk)
    expected = torch.where(
        columns[None, :] < visible[:, None], columns[None, :], -1
    ).contiguous()

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qsa_prefill_indices_scratch_is_reused():
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend._qsa_prefill_indices_scratch = {}
    device = torch.device("cuda")
    large = backend._get_qsa_prefill_indices_scratch(1024, 2051, device)
    small = backend._get_qsa_prefill_indices_scratch(52, 2051, device)
    assert small.data_ptr() == large.data_ptr()
