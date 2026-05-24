import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_index_cache_module():
    path = (
        Path(__file__).parents[3]
        / "python/sglang/srt/models/deepseek_v4_index_cache.py"
    )
    spec = importlib.util.spec_from_file_location("deepseek_v4_index_cache", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


index_cache = _load_index_cache_module()
get_index_cache_policy = index_cache.get_index_cache_policy


def test_dsv4_index_cache_c4_layer_ids_match_returned_topk_slots():
    config = SimpleNamespace(compress_ratios=[0, 4, 128, 4, 0, 4])

    assert index_cache.get_c4_layer_ids(config) == [1, 3, 5]


def test_dsv4_index_cache_freq_uses_c4_layer_sequence():
    config = SimpleNamespace(
        compress_ratios=[0, 0, 4, 128, 4, 128, 4, 0],
        index_topk_freq=2,
        index_topk_pattern=None,
    )

    assert get_index_cache_policy(config, 0, 0) == (None, None)
    assert get_index_cache_policy(config, 2, 4) == (False, True)
    assert get_index_cache_policy(config, 4, 4) == (True, False)
    assert get_index_cache_policy(config, 6, 4) == (False, False)


def test_dsv4_index_cache_freq_supports_consecutive_skips():
    config = SimpleNamespace(
        compress_ratios=[4, 128, 4, 128, 4, 128, 4],
        index_topk_freq=3,
        index_topk_pattern=None,
    )

    assert get_index_cache_policy(config, 0, 4) == (False, True)
    assert get_index_cache_policy(config, 2, 4) == (True, True)
    assert get_index_cache_policy(config, 4, 4) == (True, False)
    assert get_index_cache_policy(config, 6, 4) == (False, False)


def test_dsv4_index_cache_accepts_c4_indexed_pattern():
    config = SimpleNamespace(
        compress_ratios=[0, 0, 4, 128, 4, 128, 4, 0],
        index_topk_freq=1,
        index_topk_pattern="FSF",
    )

    assert get_index_cache_policy(config, 2, 4) == (False, True)
    assert get_index_cache_policy(config, 4, 4) == (True, False)
    assert get_index_cache_policy(config, 6, 4) == (False, False)


def test_dsv4_index_cache_accepts_layer_indexed_pattern():
    config = SimpleNamespace(
        compress_ratios=[0, 0, 4, 128, 4, 128, 4, 0],
        index_topk_freq=1,
        index_topk_pattern="FFFFSFFF",
    )

    assert get_index_cache_policy(config, 2, 4) == (False, True)
    assert get_index_cache_policy(config, 4, 4) == (True, False)
    assert get_index_cache_policy(config, 6, 4) == (False, False)


def test_dsv4_index_cache_ignores_nextn_and_non_c4_layers():
    config = SimpleNamespace(
        compress_ratios=[0, 0, 4, 128, 4],
        index_topk_freq=2,
        index_topk_pattern=None,
    )

    assert get_index_cache_policy(config, 2, 4, is_nextn=True) == (None, None)
    assert get_index_cache_policy(config, 3, 128) == (None, None)


def test_dsv4_index_cache_policy_allows_hisa():
    config = SimpleNamespace(
        compress_ratios=[0, 0, 4, 128, 4],
        index_topk_freq=2,
        index_topk_pattern=None,
        use_hisa=True,
        hisa_k_block_size=128,
        hisa_block_topk=64,
    )

    assert get_index_cache_policy(config, 2, 4) == (False, True)
    assert get_index_cache_policy(config, 4, 4) == (True, False)


def test_dsv4_index_cache_rejects_short_layer_indexed_pattern():
    config = SimpleNamespace(
        compress_ratios=[0, 0, 4, 128, 4, 128, 4, 0],
        index_topk_freq=1,
        index_topk_pattern="FFFF",
    )

    with pytest.raises(ValueError, match="does not cover next C4 layer"):
        get_index_cache_policy(config, 2, 4)

    with pytest.raises(ValueError, match="does not cover C4 layer"):
        get_index_cache_policy(config, 4, 4)


def test_dsv4_index_cache_rejects_layer_indexed_pattern_shorter_than_first_c4():
    config = SimpleNamespace(
        compress_ratios=[0, 0, 4, 128, 4, 128, 4, 0],
        index_topk_freq=1,
        index_topk_pattern="F",
    )

    with pytest.raises(ValueError, match="does not cover first C4 layer"):
        get_index_cache_policy(config, 2, 4)


def test_dsv4_index_cache_rejects_invalid_pattern_entries():
    config = SimpleNamespace(
        compress_ratios=[0, 0, 4, 128, 4, 128, 4, 0],
        index_topk_freq=1,
        index_topk_pattern="FXS",
    )

    with pytest.raises(ValueError, match="unsupported entries"):
        get_index_cache_policy(config, 2, 4)


def test_dsv4_index_cache_rejects_c4_indexed_pattern_that_skips_first_c4():
    config = SimpleNamespace(
        compress_ratios=[0, 0, 4, 128, 4, 128, 4, 0],
        index_topk_freq=1,
        index_topk_pattern="SFF",
    )

    with pytest.raises(ValueError, match="first C4 layer as F"):
        get_index_cache_policy(config, 2, 4)


def test_dsv4_index_cache_rejects_layer_indexed_pattern_that_skips_first_c4():
    config = SimpleNamespace(
        compress_ratios=[0, 0, 4, 128, 4, 128, 4, 0],
        index_topk_freq=1,
        index_topk_pattern="FFSFFFFF",
    )

    with pytest.raises(ValueError, match="first C4 layer as F"):
        get_index_cache_policy(config, 2, 4)


def test_dsv4_index_cache_reuse_guards():
    prev = object()

    assert index_cache.should_reuse_index_cache(True, prev, None)
    assert not index_cache.should_reuse_index_cache(False, prev, None)
    assert not index_cache.should_reuse_index_cache(True, None, None)
    assert not index_cache.should_reuse_index_cache(True, prev, object())

    assert index_cache.should_return_index_cache(True, None)
    assert not index_cache.should_return_index_cache(False, None)
    assert not index_cache.should_return_index_cache(None, None)
    assert not index_cache.should_return_index_cache(True, object())


def test_dsv4_index_cache_min_seq_len_gate_uses_raw_token_length():
    seq_lens = torch.tensor([10, 25000], dtype=torch.int32)

    assert index_cache.index_cache_enabled_for_seq_lens(seq_lens, 0)
    assert index_cache.index_cache_enabled_for_seq_lens(seq_lens, 100000)
    assert not index_cache.index_cache_enabled_for_seq_lens(seq_lens, 100001)
    assert not index_cache.index_cache_enabled_for_seq_lens(
        torch.empty(0, dtype=torch.int32),
        1,
    )


def test_dsv4_index_cache_graph_gate_disables_short_context_graphs():
    config = SimpleNamespace(
        index_topk_freq=2,
        index_topk_pattern=None,
        index_topk_min_seq_len=75000,
    )

    assert index_cache.should_disable_cuda_graph_for_index_cache_gate(
        config,
        torch.tensor([1000, 12000], dtype=torch.int32),
    )
    assert not index_cache.should_disable_cuda_graph_for_index_cache_gate(
        config,
        torch.tensor([20000], dtype=torch.int32),
    )


def test_dsv4_index_cache_graph_gate_ignores_disabled_indexcache():
    config = SimpleNamespace(
        index_topk_freq=1,
        index_topk_pattern=None,
        index_topk_min_seq_len=75000,
    )

    assert not index_cache.should_disable_cuda_graph_for_index_cache_gate(
        config,
        torch.tensor([1], dtype=torch.int32),
    )


def test_dsv4_index_cache_config_defaults_to_long_context_gate():
    path = Path(__file__).parents[3] / "python/sglang/srt/configs/deepseek_v4.py"
    module = ast.parse(path.read_text())
    default_value = None
    for node in ast.walk(module):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "index_topk_min_seq_len"
        ):
            default_value = ast.literal_eval(node.value)
            break

    assert default_value == 75000


def test_dsv4_index_cache_rebuilds_physical_indices_from_raw_cache():
    source = SimpleNamespace(
        c4_sparse_raw_indices=torch.tensor([[0, 63, 64, 130, -1]], dtype=torch.int32),
    )
    target = SimpleNamespace(
        c4_sparse_page_indices=torch.full((1, 5), -1, dtype=torch.int32),
        c4_sparse_raw_indices=torch.full((1, 5), -1, dtype=torch.int32),
        page_table=torch.tensor([[10, 11, 12]], dtype=torch.int32),
    )
    indexer_metadata = SimpleNamespace(
        c4_seq_lens=torch.tensor([131], dtype=torch.int32),
        c4_page_size=64,
    )

    cached = index_cache.make_index_cache_from_metadata(source)
    copied_raw = index_cache.assign_index_cache_to_metadata(
        cached, target, indexer_metadata
    )

    assert target.c4_sparse_page_indices.tolist() == [[640, 703, 704, 770, -1]]
    assert target.c4_sparse_raw_indices is source.c4_sparse_raw_indices
    assert copied_raw is target.c4_sparse_raw_indices


def test_dsv4_index_cache_clamps_invalid_raw_indices_before_gather():
    source = SimpleNamespace(
        c4_sparse_raw_indices=torch.tensor([[0, 9999, -1]], dtype=torch.int32),
    )
    target = SimpleNamespace(
        c4_sparse_page_indices=torch.full((1, 3), -1, dtype=torch.int32),
        c4_sparse_raw_indices=torch.full((1, 3), -1, dtype=torch.int32),
        page_table=torch.tensor([[10, 11]], dtype=torch.int32),
    )
    indexer_metadata = SimpleNamespace(
        c4_seq_lens=torch.tensor([64], dtype=torch.int32),
        c4_page_size=64,
    )

    cached = index_cache.make_index_cache_from_metadata(source)
    index_cache.assign_index_cache_to_metadata(cached, target, indexer_metadata)

    assert target.c4_sparse_page_indices.tolist() == [[640, -1, -1]]


def test_dsv4_index_cache_rejects_missing_raw_indices():
    source = SimpleNamespace(
        c4_sparse_page_indices=torch.tensor([[3, 4, -1]], dtype=torch.int32),
        c4_sparse_raw_indices=None,
    )

    with pytest.raises(AssertionError, match="raw index cache is required"):
        index_cache.make_index_cache_from_metadata(source)


def test_dsv4_index_cache_rejects_shape_mismatch():
    source = SimpleNamespace(
        c4_sparse_raw_indices=torch.tensor([[8, 9]], dtype=torch.int32),
    )

    target = SimpleNamespace(
        c4_sparse_page_indices=torch.full((1, 2), -1, dtype=torch.int32),
        c4_sparse_raw_indices=torch.full((1, 3), -1, dtype=torch.int32),
        page_table=torch.tensor([[0]], dtype=torch.int32),
    )
    indexer_metadata = SimpleNamespace(
        c4_seq_lens=torch.tensor([3], dtype=torch.int32),
        c4_page_size=64,
    )
    cached = index_cache.make_index_cache_from_metadata(source)
    with pytest.raises(AssertionError, match="raw index cache shape mismatch"):
        index_cache.assign_index_cache_to_metadata(cached, target, indexer_metadata)
