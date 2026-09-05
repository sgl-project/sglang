import ast
import inspect
import textwrap

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer, litetopk
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _async_cp_store_tree():
    source = textwrap.dedent(
        inspect.getsource(dsa_indexer.Indexer._fused_q_prepare_and_cp_store)
    )
    return ast.parse(source)


def test_cp_packed_index_k_defaults_off():
    assert not litetopk.CP_PACKED_INDEX_K_AG


def test_paged_candidates_full_and_tail_capacity(monkeypatch):
    monkeypatch.setattr(litetopk, "_PAGED_CAND_BUFS", {})
    monkeypatch.setattr(litetopk, "_PAGED_POOL_PAGES_PER_ROW", 12)
    dev = torch.device("meta")
    backing = None
    for rows in (4096, 4088, 4096):
        buffers = litetopk._paged_cand_bufs(rows, 1048512, dev)
        assert buffers["overflow_val"].shape == (12 * rows, 4096)
        assert buffers["overflow_idx"].shape == (12 * rows, 4096)
        assert buffers["page_table"].shape == (rows, 254)
        assert buffers["page_table"].is_contiguous()
        # The observed 1M tail peak was 35,372 pages; eight pages/row failed.
        assert buffers["overflow_val"].shape[0] > 35372
        current = litetopk._PAGED_CAND_BUFS[str(dev)]["ov"]
        if backing is not None:
            assert current is backing
        backing = current


def test_cp_packed_index_k_has_execution_marker():
    strings = {
        node.value
        for node in ast.walk(_async_cp_store_tree())
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert (
        "LITETOPK_CP_PACKED_INDEX_K_AG_EXECUTED local_fp8_pack_before_CP_allgather %s"
    ) in strings
    assert "rank_major_direct_store" in strings


def test_cp_packed_index_k_preserves_one_collective_per_branch():
    tree = _async_cp_store_tree()
    packed_branch = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if (
            isinstance(node.test, ast.Attribute)
            and node.test.attr == "CP_PACKED_INDEX_K_AG"
        ):
            packed_branch = node
            break
    assert packed_branch is not None

    def called_attrs(nodes):
        calls = []
        for node in nodes:
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                if isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
                elif isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
        return calls

    packed_calls = called_attrs(packed_branch.body)
    legacy_calls = called_attrs(packed_branch.orelse)
    assert packed_calls.count("try_materialize_rank_major_indexer_k_cache") == 1
    assert packed_calls.count("materialize_full_indexer_k_cache") == 1
    assert legacy_calls.count("materialize_full_indexer_k_cache") == 1
    assert "fused_quantize_index_k_packed" in packed_calls
    assert "_store_packed_index_k_cache" in packed_calls
    assert "_store_rank_major_packed_index_k_cache" in packed_calls
    assert "_store_index_k_cache" in legacy_calls


def test_cp_packed_index_k_stays_on_comm_stream_scope():
    tree = _async_cp_store_tree()
    matching_scopes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        calls = set()
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if isinstance(child.func, ast.Attribute):
                calls.add(child.func.attr)
            elif isinstance(child.func, ast.Name):
                calls.add(child.func.id)
        if {
            "fused_quantize_index_k_packed",
            "try_materialize_rank_major_indexer_k_cache",
            "materialize_full_indexer_k_cache",
            "_store_packed_index_k_cache",
            "_store_rank_major_packed_index_k_cache",
        } <= calls:
            matching_scopes.append(node)
    assert len(matching_scopes) == 1


def test_rank_major_source_mapping_inverts_interleave_gather():
    for cp_size in (2, 4, 8):
        rows_per_rank = 17
        logical = list(range(cp_size * rows_per_rank))
        rank_major = [
            logical[local_row * cp_size + rank]
            for rank in range(cp_size)
            for local_row in range(rows_per_rank)
        ]
        restored = [
            rank_major[
                (row & (cp_size - 1)) * rows_per_rank
                + (row >> (cp_size.bit_length() - 1))
            ]
            for row in logical
        ]
        assert restored == logical
