"""CPU numerical tests for DSA's interleaved DCP index cache and global top-k.

The pure helpers are loaded directly so this test also runs without CUDA or
the server's optional dependencies: python -m pytest <this file>.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[5]


def _load_helpers():
    spec = importlib.util.spec_from_file_location(
        "dsa_dcp_helpers",
        ROOT / "python/sglang/srt/layers/attention/dsa/dcp_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dcp = _load_helpers()


def _load_function(path, name, namespace, owner=None):
    # Exercise dispatch without importing GPU-only kernels on the CPU runner.
    tree = ast.parse((ROOT / path).read_text())
    if owner:
        tree = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == owner
        )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec("from __future__ import annotations\n" + ast.unparse(function), namespace)
    return namespace[name]


def test_decode_returns_lse_and_normalizes_empty_shards():
    namespace = dict(
        torch=torch,
        get_parallel=lambda: SimpleNamespace(dcp_enabled=True),
        concat_mla_absorb_q_general=lambda q, rope: torch.cat([q, rope], dim=-1),
    )
    path = "python/sglang/srt/layers/attention/dsa_backend.py"
    decode = _load_function(
        path, "forward_decode", namespace, "DeepseekSparseAttnBackend"
    )
    flashmla = _load_function(
        path, "_forward_flashmla_kv", namespace, "DeepseekSparseAttnBackend"
    )
    table = torch.tensor([[64, 65, -1, -1], [-1, -1, -1, -1]], dtype=torch.int32)
    backend = SimpleNamespace(
        _resolve_kpool_tail_backend=lambda indices, impl: impl,
        _check_kpool_tail_backend=lambda *args: None,
        dsa_decode_impl="flashmla_kv",
        forward_metadata=SimpleNamespace(
            dsa_cache_seqlens_int32=torch.tensor([2, 0]),
            flashmla_metadata=SimpleNamespace(flashmla_metadata=None, num_splits=None),
        ),
        token_to_kv_pool=SimpleNamespace(
            get_key_buffer=lambda _: torch.zeros(128, 656)
        ),
        hisparse_coordinator=None,
        use_fused_topk=True,
        _pad_topk_indices=lambda indices, _: indices,
        _get_fused_topk_page_table=lambda indices: indices,
        flashmla_kv_num_q_heads=64,
        real_page_size=64,
        kv_cache_dim=656,
        dsa_kv_cache_store_fp8=True,
        dsa_index_topk=4,
    )
    backend._forward_flashmla_kv = MethodType(flashmla, backend)

    def kernel(**kwargs):
        assert kwargs["q"].shape == (2, 1, 64, 576)
        out = torch.ones(2, 1, 64, 512)
        lse = torch.full((2, 64, 1), 2.0)
        out[1] = float("nan")
        lse[1] = float("nan")
        return out, lse

    with patch.dict(
        sys.modules,
        {"sgl_kernel.flash_mla": SimpleNamespace(flash_mla_with_kvcache=kernel)},
    ):
        result = decode(
            backend,
            torch.ones(2, 4, 576),
            None,
            None,
            SimpleNamespace(
                is_cross_attention=False,
                layer_id=0,
                tp_q_head_num=4,
                v_head_dim=512,
                head_dim=576,
                scaling=0.1,
            ),
            SimpleNamespace(),
            topk_indices=table,
        )
    assert isinstance(result, tuple)
    output, lse = result
    assert output.shape == (2, 1, 4, 512)
    assert lse.shape == (2, 4)
    assert (output[0] == 1).all() and (output[1] == 0).all()
    assert (lse[0] == 2).all() and torch.isneginf(lse[1]).all()


def test_dsa_flashmla_merge_uses_natural_logarithms():
    is_base_e = _load_function(
        "python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py",
        "is_mla_dcp_lse_base_on_e",
        {},
    )
    assert is_base_e("dsa") and is_base_e("nsa")
    assert not is_base_e("flashinfer")


@pytest.mark.parametrize("size", [1, 2, 3, 4, 8])
def test_cache_layout_matches_allocator_owner_rule(size):
    # Non-contiguous physical pages, including an incomplete final page.
    page_size = 64
    logical = torch.arange(137)
    pages = torch.tensor([5, 2, 9])
    loc = pages[logical // (page_size * size)] * page_size * size
    loc += logical % (page_size * size)
    lens = torch.arange(138)
    for rank in range(size):
        actual = dcp.localize_dcp_indexer_write_loc(loc, dcp_size=size, dcp_rank=rank)
        expected = torch.tensor(
            [int(v) // size if int(v) % size == rank else 0 for v in loc]
        )
        torch.testing.assert_close(actual, expected)
        actual_lens = dcp.localize_dcp_indexer_seq_lens(
            lens, dcp_size=size, dcp_rank=rank
        )
        assert actual_lens.tolist() == [
            sum(p % size == rank for p in range(length)) for length in lens
        ]
        for length in [1, 7, 64, 65, 137]:
            table = dcp.localize_dcp_indexer_page_table(
                loc[None, :length], dcp_size=size, dcp_rank=rank
            )
            owned = loc[:length][loc[:length] % size == rank] // size
            torch.testing.assert_close(table[0, : len(owned)], owned)
            assert table.shape[1] >= 1


def _distributed_topk(logits, lengths, size, topk, **kwargs):
    candidates = [
        dcp.dcp_topk_candidates(
            logits[:, rank::size],
            lengths,
            topk=topk,
            dcp_size=size,
            dcp_rank=rank,
            **kwargs,
        )
        for rank in range(size)
    ]
    scores = torch.cat([c[0] for c in candidates], dim=1)
    positions = torch.cat([c[1] for c in candidates], dim=1)
    return [
        dcp.merge_dcp_topk_candidates(
            scores, positions, topk=topk, dcp_size=size, dcp_rank=rank
        )
        for rank in range(size)
    ]


@pytest.mark.parametrize("size", [2, 3, 4, 8])
def test_global_topk_and_attention_match_unsharded_reference(size):
    torch.manual_seed(42)
    lengths = torch.tensor([0, 1, 7, 31, 137])
    logits = torch.rand(5, 137)
    # All best candidates for the long request live on one rank. Independently
    # choosing k per rank would incorrectly attend to k * size tokens.
    logits[-1, ::size] += 10
    topk = 8
    shards = _distributed_topk(logits, lengths, size, topk)
    attention_logits = torch.randn(5, 137)
    values = torch.randn(137, 4)
    for row, length in enumerate(lengths.tolist()):
        expected = set(logits[row, :length].topk(min(topk, length)).indices.tolist())
        selected = set()
        partials, lses = [], []
        for rank, shard in enumerate(shards):
            indices = shard[row][shard[row] >= 0].long() * size + rank
            selected.update(indices.tolist())
            if indices.numel():
                scores = attention_logits[row, indices]
                partials.append(scores.softmax(0) @ values[indices])
                lses.append(scores.logsumexp(0))
        assert selected == expected
        if expected:
            indices = torch.tensor(sorted(expected))
            expected_output = (
                attention_logits[row, indices].softmax(0) @ values[indices]
            )
            actual_output = torch.stack(lses).softmax(0) @ torch.stack(partials)
            torch.testing.assert_close(actual_output, expected_output)


def test_forced_tokens_use_global_positions_and_padding_is_ignored():
    logits = torch.arange(48, dtype=torch.float32).repeat(2, 1)
    lengths = torch.tensor([3, 45])
    shards = _distributed_topk(
        logits, lengths, 4, 8, num_init_tokens=2, num_local_tokens=3
    )
    for row, length in enumerate(lengths.tolist()):
        selected = {
            int(local) * 4 + rank
            for rank, shard in enumerate(shards)
            for local in shard[row]
            if local >= 0
        }
        assert len(selected) == min(8, length)
        assert {0, 1, *range(max(0, length - 3), length)} <= selected
        assert max(selected) < length
