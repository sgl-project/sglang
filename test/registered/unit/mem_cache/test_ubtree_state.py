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
    tensors = [*state.intermediate_conv_window, state.intermediate_ssm]
    if isinstance(state, MambaPool.UBTreeSpeculativeState):
        tensors.extend(
            (state.tree_rawv, state.tree_rawk, state.tree_g, state.tree_beta)
        )
    return sum(_tensor_bytes(t) for t in tensors)


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
            for width in (None, 2, 49):
                with self.subTest(width=width):
                    pool = HybridMambaDecodeReqToTokenPool(
                        size=1,
                        pre_alloc_size=1,
                        mamba_size=2,
                        row_width=16,
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
                row_width=16,
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
        for width in (2, 49):
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
                        expected = (
                            _params().mamba_cache_per_req * width * (requests + 1)
                        )
                    self.assertEqual(_scratch_bytes(pool.mamba_cache), expected)

    def test_capacity_solver_charges_actual_compact_scratch_in_all_modes(self):
        total_bytes = 64 << 20
        for width in (2, 49):
            for mode in ("fixed", "no_radix", "auto"):
                with self.subTest(width=width, mode=mode):
                    config = SimpleNamespace(
                        mambaish_config=SimpleNamespace(mamba2_cache_params=_params()),
                        extra_mamba_cache_bytes_per_req=0,
                        extra_mamba_cache_bytes_per_request_row=0,
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

    def test_worker_info_reserves_spec_decode_reach(self):
        from sglang.srt.managers.tp_worker import TpModelWorker

        worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                effective_logical_max_total_num_tokens=512,
                logical_max_total_num_tokens=512,
                max_running_requests=1,
                forward_stream=None,
                req_to_token_pool=SimpleNamespace(size=1, row_width=4096),
                token_to_kv_pool=SimpleNamespace(size=512),
                token_to_kv_pool_allocator=None,
            ),
            model_config=SimpleNamespace(context_len=2048),
            random_seed=0,
            device="cpu",
            dllm_algorithm=None,
        )
        for width, expected_reach in ((49, 98), (24, 60)):
            with (
                self.subTest(width=width),
                get_context().override_server_args(
                    speculative_algorithm="UBTREE",
                    speculative_num_steps=15,
                    speculative_eagle_topk=2,
                    speculative_num_draft_tokens=width,
                    page_size=1,
                ),
            ):
                info = TpModelWorker.get_worker_info(worker)
                expected_max_req = 512 - expected_reach
                self.assertEqual(info[4], expected_max_req)
                self.assertEqual(info[5], expected_max_req - 6)

        worker.model_runner.effective_logical_max_total_num_tokens = 106
        worker.model_runner.logical_max_total_num_tokens = 106
        worker.model_runner.token_to_kv_pool.size = 106
        with get_context().override_server_args(
            speculative_algorithm="UBTREE",
            speculative_num_steps=15,
            speculative_eagle_topk=2,
            speculative_num_draft_tokens=49,
            page_size=1,
        ):
            info = TpModelWorker.get_worker_info(worker)
            self.assertEqual((info[4], info[5]), (8, 2))
            worker.model_runner.effective_logical_max_total_num_tokens = 105
            worker.model_runner.logical_max_total_num_tokens = 105
            worker.model_runner.token_to_kv_pool.size = 105
            with self.assertRaisesRegex(ValueError, "106"):
                TpModelWorker.get_worker_info(worker)


if __name__ == "__main__":
    unittest.main()
