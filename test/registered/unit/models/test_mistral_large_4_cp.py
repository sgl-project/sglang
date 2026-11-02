import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention import dsa_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSharedKVRaggedPrefill(CustomTestCase):
    def _run_ragged(self, implementation, *, shared, q8_fusion=False):
        provider_cache = torch.full((2, 1, 4), 7.0)
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)
            return kwargs["q_all"] if "q_all" in kwargs else kwargs["q_nope"]

        backend = dsa_backend.DeepseekSparseAttnBackend.__new__(
            dsa_backend.DeepseekSparseAttnBackend
        )
        backend.__dict__.update(
            dsa_prefill_impl=implementation,
            dsa_decode_impl="fa3",
            use_mha=False,
            use_fused_topk=True,
            hisparse_coordinator=None,
            has_shared_kv_consumers=shared,
            forward_metadata=SimpleNamespace(
                page_table_1_flattened=None,
                page_table_1=torch.tensor([[0, 1]]),
                indexer_seq_lens_cpu=torch.tensor([2]),
                dsa_cache_seqlens_int32=torch.tensor([1, 2]),
            ),
            token_to_kv_pool=SimpleNamespace(
                layer_router=SimpleNamespace(
                    read=lambda _: SimpleNamespace(key=lambda: provider_cache)
                )
            ),
            _resolve_kpool_tail_backend=lambda indices, impl: impl,
            _check_kpool_tail_backend=lambda *args: None,
            get_topk_transform_method=lambda _: dsa_backend.TopkTransformMethod.RAGGED,
            _pad_topk_indices=lambda indices, n: indices,
            _get_fused_topk_page_table=lambda indices: indices,
            _q8kv8_kv_cat_fusion=q8_fusion,
            _forward_flashmla_sparse=capture,
            _forward_flashmla_sparse_q8kv8=capture,
        )
        layer = SimpleNamespace(
            is_cross_attention=False,
            layer_id=35,
            tp_q_head_num=1,
            v_head_dim=2,
            head_dim=4,
            scaling=1.0,
        )
        mode = SimpleNamespace(
            is_target_verify=lambda: False, is_draft_extend_v2=lambda: False
        )
        batch = SimpleNamespace(forward_mode=mode, extend_prefix_lens_cpu=[0])
        zeros = torch.zeros((2, 1, 2))
        with (
            mock.patch.object(
                dsa_backend,
                "dequantize_k_cache_paged_dsa",
                side_effect=lambda cache, _: cache,
            ),
            mock.patch.object(
                dsa_backend,
                "concat_mla_absorb_q_general",
                side_effect=lambda q, rope: torch.cat((q, rope), dim=-1),
            ),
        ):
            dsa_backend.DeepseekSparseAttnBackend.forward_extend(
                backend,
                zeros,
                zeros.clone(),
                zeros.clone(),
                layer,
                batch,
                save_kv_cache=False,
                q_rope=zeros.clone(),
                k_rope=zeros.clone(),
                topk_indices=torch.tensor([[0, -1], [0, 1]], dtype=torch.int32),
            )
        return provider_cache, captured

    def test_fresh_ragged_prefill_reads_expected_kv(self):
        """Shared consumers use provider KV; ordinary models keep fresh inputs."""
        provider, captured = self._run_ragged("flashmla_sparse", shared=True)
        self.assertIs(captured["kv_cache"], provider)

        for fusion in (False, True):
            with self.subTest(fusion=fusion):
                provider, captured = self._run_ragged(
                    "flashmla_sparse_q8", shared=True, q8_fusion=fusion
                )
                self.assertIs(captured["paged_kv_cache"], provider)
                self.assertIsNone(captured["kv_bf16"])
                self.assertIsNotNone(captured["page_table_1_flattened"])

        provider, captured = self._run_ragged("flashmla_sparse", shared=False)
        self.assertIsNot(captured["kv_cache"], provider)
        self.assertEqual(captured["kv_cache"].abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
