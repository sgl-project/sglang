"""MiniMax-M3 mono row tables: causal verify rows, graph-replay freshness, padding."""

import pytest
import torch

from sglang.srt.models.minimax_m3_mono.metadata import prepare_token_rows
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")

pytestmark = pytest.mark.skipif(not torch.version.hip, reason="ROCm mono metadata")


def inputs():
    pages = torch.tensor(
        [[1, 2, 3, 4], [7, 4, 9, 3], [10, 11, 12, 13], [6, 2, 8, 5]],
        dtype=torch.int32,
        device="cuda",
    )
    table = (
        pages[:, :, None] * 128 + torch.arange(128, dtype=torch.int32, device="cuda")
    ).reshape(4, 512)
    requests = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    prefixes = torch.tensor([127, 255], dtype=torch.int64, device="cuda")
    columns = torch.tensor([0, 128, 256, 384], dtype=torch.int64, device="cuda")
    blocks = torch.empty((8, 4), dtype=torch.int32, device="cuda")
    lengths = torch.empty(8, dtype=torch.int32, device="cuda")
    slots = torch.arange(128, 136, dtype=torch.int64, device="cuda")
    return table, requests, prefixes, columns, blocks, lengths, slots


def test_verify_causal_rows_with_shuffled_pages():
    args = inputs()
    blocks, lengths = prepare_token_rows(
        *args, query_len=4, is_verify=True, max_context_len=512
    )
    assert lengths.tolist() == [128, 129, 130, 131, 256, 257, 258, 259]
    assert blocks.tolist() == [[6, 2, 8, 5]] * 4 + [[7, 4, 9, 3]] * 4


def test_graph_replay_refreshes_prefixes_and_request_mapping():
    args = inputs()
    prepare_token_rows(*args, query_len=4, is_verify=True, max_context_len=512)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare_token_rows(*args, query_len=4, is_verify=True, max_context_len=512)
    args[1].copy_(torch.tensor([0, 2], dtype=torch.int32, device="cuda"))
    args[2].copy_(torch.tensor([383, 507], dtype=torch.int64, device="cuda"))
    graph.replay()
    assert args[5].tolist() == [384, 385, 386, 387, 508, 509, 510, 511]
    assert args[4].tolist() == [[1, 2, 3, 4]] * 4 + [[10, 11, 12, 13]] * 4


def test_decode_keeps_its_lengths_and_padding():
    table, requests, prefixes, columns, blocks, lengths, slots = inputs()
    prefixes.copy_(torch.tensor([7, 0], device="cuda"))
    slots[1] = 0
    prepare_token_rows(
        table,
        requests,
        prefixes,
        columns,
        blocks[:2],
        lengths[:2],
        slots[:2],
        query_len=1,
        is_verify=False,
        max_context_len=512,
    )
    assert lengths[:2].tolist() == [7, 0]
    assert blocks[:2].tolist() == [[6, 2, 8, 5], [0, 0, 0, 0]]


def test_verify_padding_uses_slots_not_prefix_length():
    args = inputs()
    args[2].zero_()
    args[6][4:] = 0
    blocks, lengths = prepare_token_rows(
        *args, query_len=4, is_verify=True, max_context_len=512
    )
    # A live first verification has a zero accepted prefix too. Only the
    # reserved dummy cache slots distinguish it from the padded request.
    assert lengths.tolist() == [1, 2, 3, 4, 0, 0, 0, 0]
    assert blocks.tolist() == [[6, 2, 8, 5]] * 4 + [[0, 0, 0, 0]] * 4


def test_graph_replay_refreshes_padding_membership():
    args = inputs()
    args[6][4:] = -1
    prepare_token_rows(*args, query_len=4, is_verify=True, max_context_len=512)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare_token_rows(*args, query_len=4, is_verify=True, max_context_len=512)
    args[6][:4] = 0
    args[6][4:] = 256
    args[2].copy_(torch.tensor([63, 383], dtype=torch.int64, device="cuda"))
    graph.replay()
    assert args[5].tolist() == [0, 0, 0, 0, 384, 385, 386, 387]
    assert args[4].tolist() == [[0, 0, 0, 0]] * 4 + [[7, 4, 9, 3]] * 4


def test_terminal_lookahead_stays_within_the_model_context():
    args = inputs()
    args[2].copy_(torch.tensor([510, 511], dtype=torch.int64, device="cuda"))
    _, lengths = prepare_token_rows(
        *args, query_len=4, is_verify=True, max_context_len=512
    )
    assert lengths.tolist() == [511, 512, 512, 512, 512, 512, 512, 512]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
