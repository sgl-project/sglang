"""Keep heterogeneous target/draft MLA bytes in separate host pools."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.mem_cache.hybrid_cache.host_pool_config import packed_mla_rows_match
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, HiCacheDraftMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestMLADraftLayout(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(page_size=128)
        override.install()
        self.addCleanup(override.restore)
        self.enterContext(get_parallel().override(attn_dcp_rank=0))

    def pool(self, *, dtype=torch.bfloat16, layers=1, rope=64, size=128, indexed=False):
        return NPUMLATokenToKVPool(
            size=size,
            page_size=128,
            dtype=dtype,
            kv_lora_rank=512,
            qk_rope_head_dim=rope,
            layer_num=layers,
            device="cpu",
            enable_memory_saver=False,
            index_head_dim=128 if indexed else None,
        )

    def plan(self, target, *drafts):
        runner = SimpleNamespace(
            token_to_kv_pool=target, spec_algorithm=SpeculativeAlgorithm.EAGLE
        )
        worker = SimpleNamespace(
            target_worker=SimpleNamespace(model_runner=runner),
            _draft_model_runners=lambda: tuple(
                SimpleNamespace(
                    token_to_kv_pool=draft,
                    model_config=SimpleNamespace(num_nextn_predict_layers=1),
                    runner_ids=SimpleNamespace(is_fused_draft=False),
                )
                for draft in drafts
            ),
        )
        with patch(
            "sglang.srt.speculative.base_spec_worker.get_memory",
            return_value=SimpleNamespace(enable_hierarchical_cache=True),
        ):
            result = BaseSpecWorker._build_hicache_draft_plan(worker)
        return result, runner

    def test_matching_rows_pack_despite_capacity_and_layer_count(self):
        target = self.pool(layers=2, size=256)
        draft = self.pool()
        self.assertTrue(packed_mla_rows_match(target, (draft,)))
        plan, runner = self.plan(target, draft)
        self.assertEqual(plan.mode, HiCacheDraftMode.PACKED)
        self.assertEqual(runner.mtp_draft_device_pools, (draft,))

    def test_different_dtype_selects_sidecar(self):
        target = self.pool(dtype=torch.float16, layers=2)
        draft = self.pool()
        self.assertFalse(packed_mla_rows_match(target, (draft,)))
        plan, runner = self.plan(target, draft)
        self.assertEqual(plan.mode, HiCacheDraftMode.SIDECAR)
        self.assertEqual(plan.device_pools, (draft,))
        self.assertEqual(runner.mtp_draft_device_pools, ())

    def test_different_rope_width_selects_sidecar(self):
        plan, _ = self.plan(self.pool(rope=32), self.pool())
        self.assertEqual(plan.mode, HiCacheDraftMode.SIDECAR)

    def test_draft_indexer_gets_its_own_host_pool(self):
        plan, runner = self.plan(self.pool(indexed=True), self.pool(indexed=True))
        self.assertEqual(plan.mode, HiCacheDraftMode.SIDECAR)
        self.assertEqual(runner.mtp_draft_device_pools, ())

    def test_multiple_heterogeneous_drafts_fail_before_packing(self):
        with self.assertRaisesRegex(NotImplementedError, "one draft sidecar"):
            self.plan(self.pool(dtype=torch.float16), self.pool(), self.pool())

    def test_non_mla_planning_uses_existing_path(self):
        self.assertTrue(packed_mla_rows_match(object(), (object(),)))


if __name__ == "__main__":
    unittest.main()
