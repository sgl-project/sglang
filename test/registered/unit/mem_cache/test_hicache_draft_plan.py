"""HiCache must select separate host rows before allocating incompatible MTP KV."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, HiCacheDraftMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def mla_pool(dtype=torch.uint8, dim=656, *, hybrid=False):
    pool = object.__new__(MLATokenToKVPool)
    pool.store_dtype = dtype
    pool.kv_cache_dim = dim
    if hybrid:
        wrapper = object.__new__(HybridLinearKVPool)
        wrapper.full_kv_pool = pool
        return wrapper
    return pool


def build_plan(
    target_pool,
    draft_pools,
    *,
    algorithm=SpeculativeAlgorithm.EAGLE,
    nextn=1,
    enabled=True,
    backup="none",
    architecture="Glm5NextForCausalLMMTP",
):
    target = SimpleNamespace(
        token_to_kv_pool=target_pool,
        spec_algorithm=algorithm,
        mtp_draft_device_pools=("stale",),
    )
    runners = tuple(
        SimpleNamespace(
            token_to_kv_pool=pool,
            model_config=SimpleNamespace(
                num_nextn_predict_layers=nextn,
                hf_config=SimpleNamespace(architectures=[architecture]),
            ),
        )
        for pool in draft_pools
    )
    worker = SimpleNamespace(
        target_worker=SimpleNamespace(model_runner=target),
        _draft_model_runners=lambda: runners,
    )
    prefix = "sglang.srt.speculative.base_spec_worker."
    with (
        patch(
            prefix + "get_memory",
            return_value=SimpleNamespace(enable_hierarchical_cache=enabled),
        ),
        patch(
            prefix + "get_disagg",
            return_value=SimpleNamespace(
                disaggregation_decode_retraction_backup=backup
            ),
        ),
    ):
        plan = BaseSpecWorker._build_hicache_draft_plan(worker)
    return plan, target


class TestHiCacheDraftPlan(unittest.TestCase):
    def test_fp4_target_or_draft_is_rejected_before_selecting_a_plan(self):
        fp4 = object.__new__(MLATokenToKVPoolFP4)
        fp4.store_dtype = torch.uint8
        fp4.kv_cache_dim = 656
        for target, draft in (
            (fp4, mla_pool(torch.bfloat16, 576)),
            (mla_pool(torch.bfloat16, 576), fp4),
            (fp4, mla_pool()),
            (mla_pool(), fp4),
        ):
            for hybrid in (False, True):
                with self.subTest(
                    target_fp4=target is fp4,
                    hybrid=hybrid,
                    draft_dtype=draft.store_dtype,
                ):
                    if hybrid:
                        wrapper = object.__new__(HybridLinearKVPool)
                        wrapper.full_kv_pool = target
                        target_pool = wrapper
                    else:
                        target_pool = target
                    with self.assertRaisesRegex(NotImplementedError, "FP4 MLA KV"):
                        build_plan(target_pool, (draft,))

    def test_mismatched_mla_uses_sidecar(self):
        for target_hybrid in (False, True):
            for draft_hybrid in (False, True):
                for dtype, dim in (
                    (torch.uint8, 528),
                    (torch.bfloat16, 656),
                    (torch.bfloat16, 576),
                    (torch.bfloat16, 328),
                ):
                    with self.subTest(
                        target_hybrid=target_hybrid,
                        draft_hybrid=draft_hybrid,
                        dtype=dtype,
                        dim=dim,
                    ):
                        draft = mla_pool(dtype, dim, hybrid=draft_hybrid)
                        plan, target = build_plan(
                            mla_pool(hybrid=target_hybrid), (draft,)
                        )
                        self.assertEqual(plan.mode, HiCacheDraftMode.SIDECAR)
                        self.assertEqual(plan.device_pools, (draft,))
                        self.assertEqual(target.mtp_draft_device_pools, ())

    def test_matching_mla_keeps_all_packed_depths(self):
        for hybrid in (False, True):
            for count in (1, 3):
                with self.subTest(hybrid=hybrid, count=count):
                    drafts = tuple(mla_pool(hybrid=hybrid) for _ in range(count))
                    plan, target = build_plan(mla_pool(hybrid=hybrid), drafts)
                    self.assertEqual(plan.mode, HiCacheDraftMode.PACKED)
                    self.assertEqual(plan.device_pools, drafts)
                    self.assertEqual(target.mtp_draft_device_pools, drafts)

    def test_later_mtp_depth_cannot_be_silently_dropped(self):
        drafts = (mla_pool(), mla_pool(torch.bfloat16, 576))
        with self.assertRaisesRegex(
            NotImplementedError, "sidecar fallback supports one"
        ):
            build_plan(mla_pool(), drafts)

    def test_disabled_cache_and_no_draft_clear_packed_state(self):
        for drafts, enabled in (((mla_pool(),), False), ((), True)):
            with self.subTest(enabled=enabled):
                plan, target = build_plan(mla_pool(), drafts, enabled=enabled)
                self.assertEqual(plan.mode, HiCacheDraftMode.NONE)
                self.assertEqual(target.mtp_draft_device_pools, ())

    def test_host_retraction_backup_also_uses_fallback(self):
        plan, target = build_plan(
            mla_pool(),
            (mla_pool(torch.bfloat16, 576),),
            enabled=False,
            backup="host_pool",
        )
        self.assertEqual(plan.mode, HiCacheDraftMode.SIDECAR)
        self.assertEqual(target.mtp_draft_device_pools, ())

    def test_non_mtp_eagle_keeps_legacy_sidecar(self):
        drafts = (mla_pool(), mla_pool())
        plan, target = build_plan(mla_pool(), drafts, nextn=0)
        self.assertEqual(plan.mode, HiCacheDraftMode.SIDECAR)
        self.assertEqual(plan.device_pools, drafts[:1])
        self.assertEqual(target.mtp_draft_device_pools, ())

    def test_non_mla_dspark_packing_is_unchanged(self):
        drafts = (object(),)
        plan, target = build_plan(
            object(),
            drafts,
            algorithm=SpeculativeAlgorithm.DSPARK,
            nextn=0,
            architecture="DeepseekV4ForCausalLMDSpark",
        )
        self.assertEqual(plan.mode, HiCacheDraftMode.PACKED)
        self.assertEqual(target.mtp_draft_device_pools, drafts)


if __name__ == "__main__":
    unittest.main()
