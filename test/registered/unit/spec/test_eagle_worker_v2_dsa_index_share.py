"""Regression tests for PP+MTP on dense and sparse MLA backends."""

from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _IndexerCapableBackend(AttentionBackend):
    def supports_dsa_indexer_metadata(self) -> bool:
        return True


class _FakeDraftBackendFactory:
    captured_seed = None

    def __init__(self, *args, seed_dsa_topk_from_draft_extend=False, **kwargs):
        type(self).captured_seed = seed_dsa_topk_from_draft_extend

    def create_decode_backend(self):
        return object()

    def create_draft_extend_backend(self):
        return None


def _make_worker(attn_backend):
    worker = object.__new__(EagleDraftWorker)
    worker.topk = 1
    worker.speculative_num_steps = 2
    worker.server_args = SimpleNamespace()
    worker.draft_runner = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                index_share_for_mtp_iteration=True,
                index_topk=2048,
            )
        ),
        attn_backend=attn_backend,
    )
    return worker


class TestAttentionBackendIndexerCapability(CustomTestCase):
    def test_default_backend_does_not_support_indexer_metadata(self):
        self.assertFalse(AttentionBackend().supports_dsa_indexer_metadata())

    def test_sparse_dsa_backend_supports_indexer_metadata(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        self.assertTrue(backend.supports_dsa_indexer_metadata())

    def test_hybrid_requires_prefill_and_decode_indexer_support(self):
        backend = object.__new__(HybridAttnBackend)
        backend.prefill_backend = _IndexerCapableBackend()
        backend.decode_backend = _IndexerCapableBackend()
        self.assertTrue(backend.supports_dsa_indexer_metadata())

        backend.decode_backend = AttentionBackend()
        self.assertFalse(backend.supports_dsa_indexer_metadata())

    def test_tbo_delegates_indexer_support_to_primary(self):
        backend = object.__new__(TboAttnBackend)
        backend.primary = _IndexerCapableBackend()
        self.assertTrue(backend.supports_dsa_indexer_metadata())


class TestEagleDenseIndexerSharing(CustomTestCase):
    def test_dense_backend_disables_index_sharing(self):
        worker = _make_worker(AttentionBackend())
        worker._init_dsa_index_share_state()

        self.assertFalse(worker.index_share_for_mtp_iteration)
        self.assertFalse(worker.seed_dsa_topk_from_draft_extend)

    def test_sparse_backend_keeps_index_sharing(self):
        worker = _make_worker(_IndexerCapableBackend())
        worker._init_dsa_index_share_state()

        self.assertTrue(worker.index_share_for_mtp_iteration)
        self.assertTrue(worker.seed_dsa_topk_from_draft_extend)

    def test_backend_factory_receives_refreshed_seed_state(self):
        worker = _make_worker(_IndexerCapableBackend())
        worker.seed_dsa_topk_from_draft_extend = False
        _FakeDraftBackendFactory.captured_seed = None

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.DraftBackendFactory",
            _FakeDraftBackendFactory,
        ):
            worker.init_attention_backend()

        self.assertTrue(_FakeDraftBackendFactory.captured_seed)


if __name__ == "__main__":
    import unittest

    unittest.main()
