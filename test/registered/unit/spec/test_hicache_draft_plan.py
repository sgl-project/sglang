import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.speculative.base_spec_worker import (
    BaseSpecWorker,
    HiCacheDraftMode,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSparkHiCacheDraftPlan(CustomTestCase):
    @staticmethod
    def _build_plan(*, draft_architecture: str, dcp_enabled: bool):
        draft_pool = object()
        target_model_runner = SimpleNamespace(
            model_config=SimpleNamespace(is_deepseek_v4_arch=True),
            mtp_draft_device_pools=(object(),),
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
        )
        draft_runner = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[draft_architecture])
            ),
            token_to_kv_pool=draft_pool,
        )
        worker = SimpleNamespace(
            target_worker=SimpleNamespace(model_runner=target_model_runner),
            _draft_model_runners=lambda: (draft_runner,),
        )

        with (
            patch(
                "sglang.srt.speculative.base_spec_worker.get_memory",
                return_value=SimpleNamespace(enable_hierarchical_cache=True),
            ),
            patch(
                "sglang.srt.speculative.base_spec_worker.get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_decode_retraction_backup=None
                ),
            ),
            patch(
                "sglang.srt.speculative.base_spec_worker.get_parallel",
                return_value=SimpleNamespace(dcp_enabled=dcp_enabled),
            ),
        ):
            plan = BaseSpecWorker._build_hicache_draft_plan(worker)

        return plan, target_model_runner, draft_pool

    def test_packs_dsv4_dspark_without_draft_sidecar(self):
        plan, target_model_runner, draft_pool = self._build_plan(
            draft_architecture="DeepseekV4ForCausalLMDSpark",
            dcp_enabled=True,
        )

        self.assertEqual(plan.mode, HiCacheDraftMode.PACKED)
        self.assertEqual(plan.device_pools, (draft_pool,))
        self.assertEqual(target_model_runner.mtp_draft_device_pools, (draft_pool,))

    def test_rejects_non_packed_dsv4_dspark_with_dcp(self):
        with self.assertRaisesRegex(NotImplementedError, "packed DSpark draft model"):
            self._build_plan(
                draft_architecture="SomeOtherDraftForCausalLM",
                dcp_enabled=True,
            )

    def test_preserves_non_dcp_dspark_sidecar(self):
        plan, target_model_runner, draft_pool = self._build_plan(
            draft_architecture="SomeOtherDraftForCausalLM",
            dcp_enabled=False,
        )

        self.assertEqual(plan.mode, HiCacheDraftMode.SIDECAR)
        self.assertEqual(plan.device_pools, (draft_pool,))
        self.assertEqual(target_model_runner.mtp_draft_device_pools, ())


if __name__ == "__main__":
    unittest.main()
