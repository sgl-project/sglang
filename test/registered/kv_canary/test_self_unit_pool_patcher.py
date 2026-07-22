from __future__ import annotations

import unittest

import torch

from sglang.kernels.ops.kv_canary.consts import MAX_REAL_KV_SOURCES, RealKvHashMode
from sglang.kernels.ops.kv_canary.verify import (
    CANARY_SLOT_BYTES,
    CanaryLaunchTag,
    RealKvSource,
    VerifyOrWriteContext,
    VerifyPlan,
    launch_canary_verify_kernel,
)
from sglang.srt.kv_canary.buffer_group import PoolKind
from sglang.srt.kv_canary.pool_patcher.api import attach_canary_buffers
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.fixtures import (
    DEFAULT_DEVICE,
    make_base_config,
    make_mha_pool,
    make_swa_pool,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=45, suite="extra-a-test-1-gpu-small-amd")


class PoolPatcherHelper:
    def setUp(self):
        self.device = DEFAULT_DEVICE
        self.config = make_base_config()


class TestAttachCanaryBuffers(PoolPatcherHelper, CustomTestCase):
    def test_canary_buffer_group_allocate_full_only(self):
        """Verify MHA pools allocate only full canary buffers."""
        pool = make_mha_pool(self.device, num_slots=16, dim=8, layer_num=2)
        groups_tuple = attach_canary_buffers(
            pool=pool,
            config=self.config,
            device=self.device,
            kv_token_id_vs_position_offset=0,
        )
        groups = {g.kind: g for g in groups_tuple}
        self.assertEqual(set(groups.keys()), {PoolKind.FULL})
        group = groups[PoolKind.FULL]
        self.assertEqual(group.k_head.shape, (16, CANARY_SLOT_BYTES))
        self.assertEqual(group.k_tail.shape, (16, CANARY_SLOT_BYTES))
        self.assertIsNotNone(group.v_head)
        self.assertIsNotNone(group.v_tail)
        self.assertEqual(group.v_head.shape, (16, CANARY_SLOT_BYTES))

    def test_canary_buffer_group_allocate_full_and_swa(self):
        """Verify SWA pools allocate full and SWA canary buffers."""
        pool = make_swa_pool(self.device, full_slots=16, swa_slots=8)
        groups_tuple = attach_canary_buffers(
            pool=pool,
            config=self.config,
            device=self.device,
            kv_token_id_vs_position_offset=0,
        )
        groups = {g.kind: g for g in groups_tuple}
        self.assertEqual(set(groups.keys()), {PoolKind.FULL, PoolKind.SWA})
        self.assertEqual(groups[PoolKind.FULL].k_head.shape[0], 16)
        self.assertEqual(groups[PoolKind.SWA].k_head.shape[0], 8)
        self.assertIsNotNone(groups[PoolKind.SWA].swa_index_lut)
        self.assertIsNone(groups[PoolKind.FULL].swa_index_lut)


class TestRealKvSources(PoolPatcherHelper, CustomTestCase):
    def test_real_kv_sources_above_4_raises(self):
        """Verify too many real KV sources are rejected."""
        tensor = torch.zeros(4, 16, dtype=torch.uint8, device=self.device)
        sources = tuple(
            RealKvSource(
                tensor=tensor, page_size=1, num_bytes_per_token=16, read_bytes=16
            )
            for _ in range(MAX_REAL_KV_SOURCES + 1)
        )
        canary_buf = torch.zeros(
            4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=self.device
        )
        plan = VerifyPlan.allocate(verify_capacity=1, device=self.device)
        violation_ring = torch.zeros(2, 8, dtype=torch.int64, device=self.device)
        violation_write_index = torch.zeros(1, dtype=torch.int32, device=self.device)
        slot_run_counter = torch.zeros(1, dtype=torch.int64, device=self.device)
        kernel_run_counter = torch.zeros(1, dtype=torch.int64, device=self.device)

        enable_chain_position_assert = torch.ones(
            1, dtype=torch.int32, device=self.device
        )

        with self.assertRaises(ValueError):
            launch_canary_verify_kernel(
                context=VerifyOrWriteContext(
                    canary_buf=canary_buf,
                    kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
                    violation_ring=violation_ring,
                    violation_write_index=violation_write_index,
                    slot_run_counter=slot_run_counter,
                    kernel_run_counter=kernel_run_counter,
                    real_kv_sources=sources,
                    real_kv_hash_mode=RealKvHashMode.NONE,
                    enable_chain_position_assert=enable_chain_position_assert,
                ),
                plan=plan,
                check_verify_expected_token=True,
            )


class TestPoolPatcherBufferInfos(PoolPatcherHelper, CustomTestCase):
    def test_get_contiguous_buf_infos_inserts_canary_entries(self):
        """Verify contiguous buffer metadata includes canary entries after patching."""
        for patched in (False, True):
            with self.subTest(patched=patched):
                pool = make_mha_pool(self.device, num_slots=16, dim=8, layer_num=2)
                ptrs_before, _, _ = pool.get_contiguous_buf_infos()
                n_before = len(ptrs_before)

                if patched:
                    attach_canary_buffers(
                        pool=pool,
                        config=self.config,
                        device=self.device,
                        kv_token_id_vs_position_offset=0,
                    )
                    ptrs_after, _, _ = pool.get_contiguous_buf_infos()
                    self.assertEqual(len(ptrs_after), n_before + 4)
                else:
                    ptrs_after, _, _ = pool.get_contiguous_buf_infos()
                    self.assertEqual(ptrs_after, ptrs_before)

    def test_swa_attach_splices_full_into_contiguous_and_swa_into_state(self):
        """Verify SWA patching splices canary buffers into both buffer lists."""
        pool = make_swa_pool(self.device, full_slots=16, swa_slots=8)
        contiguous_before, _, _ = pool.get_contiguous_buf_infos()
        state_before, _, _ = pool.get_state_buf_infos()

        attach_canary_buffers(
            pool=pool,
            config=self.config,
            device=self.device,
            kv_token_id_vs_position_offset=0,
        )

        contiguous_after, _, _ = pool.get_contiguous_buf_infos()
        state_after, _, _ = pool.get_state_buf_infos()
        self.assertEqual(len(contiguous_after), len(contiguous_before) + 4)
        self.assertEqual(len(state_after), len(state_before) + 4)

    def test_pd_layout_canary_inserted_correctly(self):
        """Verify PD (prefill-decode disaggregation) canary buffers are inserted in layout order."""
        pool = make_mha_pool(self.device, num_slots=16, dim=8, layer_num=2)
        k_ptrs_orig = [b.data_ptr() for b in pool.k_buffer]
        v_ptrs_orig = [b.data_ptr() for b in pool.v_buffer]

        groups_tuple = attach_canary_buffers(
            pool=pool,
            config=self.config,
            device=self.device,
            kv_token_id_vs_position_offset=0,
        )
        group = {g.kind: g for g in groups_tuple}[PoolKind.FULL]
        ptrs_after, _, _ = pool.get_contiguous_buf_infos()

        canary_k_ptrs = [group.k_head.data_ptr(), group.k_tail.data_ptr()]
        canary_v_ptrs = [group.v_head.data_ptr(), group.v_tail.data_ptr()]
        self.assertEqual(ptrs_after[0], canary_k_ptrs[0])
        self.assertEqual(ptrs_after[1 : 1 + len(k_ptrs_orig)], k_ptrs_orig)
        k_tail_idx = 1 + len(k_ptrs_orig)
        self.assertEqual(ptrs_after[k_tail_idx], canary_k_ptrs[1])
        self.assertEqual(ptrs_after[k_tail_idx + 1], canary_v_ptrs[0])
        v_start = k_tail_idx + 2
        self.assertEqual(ptrs_after[v_start : v_start + len(v_ptrs_orig)], v_ptrs_orig)
        self.assertEqual(ptrs_after[-1], canary_v_ptrs[1])


class TestCanaryBufferBudget(PoolPatcherHelper, CustomTestCase):
    def test_canary_buf_per_token_bytes_within_budget(self):
        """Verify canary per-token storage stays below the real KV budget."""
        pool = make_mha_pool(self.device, num_slots=16, dim=64, layer_num=2)
        groups_tuple = attach_canary_buffers(
            pool=pool,
            config=self.config,
            device=self.device,
            kv_token_id_vs_position_offset=0,
        )
        group = {g.kind: g for g in groups_tuple}[PoolKind.FULL]

        slot_stride_bytes = group.k_head.stride(0) * group.k_head.element_size()
        self.assertLessEqual(slot_stride_bytes, CANARY_SLOT_BYTES)

        real_kv_per_token_bytes = (
            pool.k_buffer[0].stride(0) * pool.k_buffer[0].element_size()
        )
        self.assertLess(slot_stride_bytes, real_kv_per_token_bytes)


if __name__ == "__main__":
    unittest.main()
