"""UBTree compact-state isolation, constructor and memory-budget regressions."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2StateDType,
)
from sglang.srt.disaggregation.decode import HybridMambaDecodeReqToTokenPool
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool
from sglang.srt.mem_cache.ubtree_state import ubtree_spec_state_bytes_per_req
from sglang.srt.runtime_context import get_context, get_schedule
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _params():
    return KimiLinearCacheParams(
        shape=KimiLinearStateShape.create(tp_world_size=1, num_heads=2, head_dim=8),
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
        layers=[0, 1],
    )


def _tensor_bytes(tensor):
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


def _scratch_bytes(state):
    return sum(_tensor_bytes(t) for t in state.intermediate_conv_window) + sum(
        _tensor_bytes(getattr(state, name))
        for name in (
            "intermediate_ssm",
            "tree_rawv",
            "tree_rawk",
            "tree_g",
            "tree_beta",
        )
    )


class TestUBTreeState(CustomTestCase):
    def setUp(self):
        # Allocate tiny tensors on CPU while exercising NVIDIA's state layout.
        for patcher in (
            patch("sglang.srt.mem_cache.memory_pool._is_cpu", False),
            patch(
                "sglang.srt.mem_cache.ubtree_state.current_platform.is_cuda",
                return_value=True,
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_pd_constructor_and_draft_clone_keep_regular_state(self):
        with get_context().override_server_args():
            for width in (None, 2, 17, 32, 49, 65):
                with self.subTest(width=width):
                    pool = HybridMambaDecodeReqToTokenPool(
                        size=1,
                        pre_alloc_size=1,
                        mamba_size=2,
                        max_context_len=16,
                        device="cpu",
                        enable_memory_saver=False,
                        cache_params=_params(),
                        mamba_layer_ids=[0, 1],
                        speculative_num_draft_tokens=width,
                        speculative_eagle_topk=2,
                        enable_mamba_extra_buffer=False,
                        enable_overlap_schedule=False,
                    )
                    self.assertFalse(pool.mamba_pool.factorized_kda_tree)
                    if width is not None:
                        self.assertIsNotNone(
                            pool.mamba_pool.mamba_cache.intermediate_ssm
                        )
                    clone = pool.clone_with_new_mamba(
                        mamba_size=2,
                        mamba_spec_state_size=1,
                        cache_params=_params(),
                        device="cpu",
                        enable_mamba_extra_buffer=False,
                        draft_model_idx=0,
                        speculative_num_draft_tokens=32,
                        speculative_eagle_topk=2,
                    )
                    self.assertFalse(clone.mamba_pool.factorized_kda_tree)
                    self.assertIs(clone.req_to_token, pool.req_to_token)

    def test_ubtree_capability_does_not_leak_to_draft_clone(self):
        with get_context().override_server_args():
            pool = HybridReqToTokenPool(
                size=1,
                mamba_size=2,
                mamba_spec_state_size=1,
                max_context_len=16,
                device="cpu",
                enable_memory_saver=False,
                cache_params=_params(),
                mamba_layer_ids=[0, 1],
                enable_mamba_extra_buffer=False,
                speculative_num_draft_tokens=32,
                speculative_eagle_topk=2,
                enable_ubtree_factorized_kda=True,
            )
            self.assertTrue(pool.mamba_pool.factorized_kda_tree)
            clone = pool.clone_with_new_mamba(
                mamba_size=2,
                mamba_spec_state_size=1,
                cache_params=_params(),
                device="cpu",
                enable_mamba_extra_buffer=False,
                draft_model_idx=0,
                speculative_num_draft_tokens=32,
                speculative_eagle_topk=2,
            )
            self.assertFalse(clone.mamba_pool.factorized_kda_tree)
            self.assertIsNotNone(clone.mamba_pool.mamba_cache.intermediate_ssm)

    def test_factorized_scratch_budget_matches_allocated_tensors(self):
        for width in (2, 17, 32, 33, 49, 65):
            for enabled, requests in ((True, 1), (False, 1), (True, 2)):
                with self.subTest(width=width, enabled=enabled, requests=requests):
                    pool = MambaPool(
                        size=3,
                        spec_state_size=requests,
                        cache_params=_params(),
                        mamba_layer_ids=[0, 1],
                        device="cpu",
                        speculative_num_draft_tokens=width,
                        speculative_eagle_topk=2,
                        enable_ubtree_factorized_kda=enabled,
                    )
                    compact = enabled and requests == 1
                    self.assertEqual(pool.factorized_kda_tree, compact)
                    if compact:
                        self.assertIsNone(pool.mamba_cache.intermediate_ssm)
                        expected = ubtree_spec_state_bytes_per_req(_params(), width) * 2
                    else:
                        self.assertIsNone(pool.mamba_cache.tree_rawv)
                        expected = (
                            _params().mamba_cache_per_req * width * (requests + 1)
                        )
                    self.assertEqual(_scratch_bytes(pool.mamba_cache), expected)

    def test_capacity_solver_charges_actual_compact_scratch_in_all_modes(self):
        total_bytes = 64 << 20
        for width in (2, 17, 32, 33, 49, 65):
            for mode in ("fixed", "no_radix", "auto"):
                with self.subTest(width=width, mode=mode):
                    config = SimpleNamespace(
                        mambaish_config=SimpleNamespace(mamba2_cache_params=_params()),
                        extra_mamba_cache_bytes_per_req=0,
                        fused_entry_bytes=lambda name: None,
                        spec_algorithm=SpeculativeAlgorithm.from_string("UBTREE"),
                        is_draft_worker=False,
                        attn_dp_size=1,
                        pp_size=1,
                        _calculate_mamba_ratio=lambda: 1,
                    )
                    with get_context().override_server_args(
                        speculative_algorithm="UBTREE",
                        speculative_num_draft_tokens=width,
                        speculative_eagle_topk=2,
                        max_running_requests=1,
                        max_mamba_cache_size=3 if mode == "fixed" else None,
                        disable_radix_cache=mode == "no_radix",
                        mamba_full_memory_ratio=0.5,
                        enable_linear_replayssm_spec=False,
                    ):
                        remaining = KVCacheConfigurator._handle_max_mamba_cache(
                            config, total_bytes / (1 << 30)
                        )
                        slots = get_schedule().max_mamba_cache_size
                    scratch = ubtree_spec_state_bytes_per_req(_params(), width) * 2
                    if mode == "auto":
                        expected_slots = (
                            int(
                                ((total_bytes - scratch) / 3)
                                // _params().mamba_cache_per_req
                            )
                            - 1
                        )
                        self.assertEqual(slots, expected_slots)
                    else:
                        self.assertEqual(slots, 3 if mode == "fixed" else 1)
                    self.assertEqual(
                        round(remaining * (1 << 30)),
                        total_bytes
                        - scratch
                        - (slots + 1) * _params().mamba_cache_per_req,
                    )


if __name__ == "__main__":
    unittest.main()
