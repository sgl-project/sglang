from types import SimpleNamespace

import pytest

from sglang.srt.layers.attention.qsa.config import parse_qsa_profile
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.speculative.draft_utils import qsa_draft_extend_graph_backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("wrapped", [False, True])
def test_compressed_chain_draft_uses_full_attention_backend(wrapped):
    backend = QwenSparseAttnBackend()
    backend.qsa_profile = parse_qsa_profile(
        SimpleNamespace(
            indexer_n_heads=4,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=2048,
            indexer_compress_ratio=4,
        )
    )
    wrapper = SimpleNamespace(full_attn_backend=backend) if wrapped else backend
    assert qsa_draft_extend_graph_backend(wrapper, 1) is backend


@pytest.mark.parametrize(
    "variant,enabled,topk",
    [("tokenwise", False, 1), ("compressed", False, 1), ("compressed", True, 2)],
)
def test_unsupported_profiles_and_trees_remain_eager(variant, enabled, topk):
    backend = QwenSparseAttnBackend()
    backend.qsa_profile = SimpleNamespace(
        variant=variant, draft_extend_cuda_graph=enabled
    )
    assert qsa_draft_extend_graph_backend(backend, topk) is None


def test_unrelated_or_unconfigured_backends_are_not_selected():
    assert qsa_draft_extend_graph_backend(None, 1) is None
    assert qsa_draft_extend_graph_backend(SimpleNamespace(), 1) is None
    assert qsa_draft_extend_graph_backend(QwenSparseAttnBackend(), 1) is None
