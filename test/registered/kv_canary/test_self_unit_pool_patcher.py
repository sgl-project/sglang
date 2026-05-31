from __future__ import annotations

import unittest

from sglang.jit_kernel.kv_canary.verify import CANARY_SLOT_BYTES
from sglang.srt.kv_canary.buffer_group import PoolKind
from sglang.srt.kv_canary.pool_patcher.api import attach_canary_buffers
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.fixtures import (
    DEFAULT_DEVICE,
    make_base_config,
    make_mha_pool,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-small")


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
