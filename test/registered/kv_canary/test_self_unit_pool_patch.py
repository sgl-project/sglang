from __future__ import annotations

import unittest

import torch

from sglang.jit_kernel.kv_canary.verify import (
    _MAX_REAL_KV_SOURCES,
    CANARY_SLOT_BYTES,
    RealKvSource,
)
from sglang.srt.kv_canary.buffer_group import PoolKind
from sglang.srt.kv_canary.pool_patch.api import (
    attach_canary_buffers,
    get_canary_buffer_groups,
)
from sglang.srt.kv_canary.pool_patch.utils import make_row_source
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.fixtures import (
    CPU_DEVICE,
    make_base_config,
    make_dsv4_pool,
    make_mha_pool,
    make_mla_pool,
    make_swa_pool,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-large")


class TestSelfUnitPoolPatch(CustomTestCase):
    def setUp(self):
        self.device = CPU_DEVICE
        self.config = make_base_config()

    def test_canary_buffer_group_allocate_full_only(self):
        pool = make_mha_pool(self.device, num_slots=16, dim=8, layer_num=2)
        attach_canary_buffers(pool=pool, config=self.config, device=self.device)
        groups = get_canary_buffer_groups(pool)
        self.assertEqual(set(groups.keys()), {PoolKind.FULL})
        group = groups[PoolKind.FULL]
        self.assertEqual(group.k_head.shape, (16, CANARY_SLOT_BYTES))
        self.assertEqual(group.k_tail.shape, (16, CANARY_SLOT_BYTES))
        self.assertIsNotNone(group.v_head)
        self.assertIsNotNone(group.v_tail)
        self.assertEqual(group.v_head.shape, (16, CANARY_SLOT_BYTES))

    def test_canary_buffer_group_allocate_full_and_swa(self):
        pool = make_swa_pool(self.device, full_slots=16, swa_slots=8)
        attach_canary_buffers(pool=pool, config=self.config, device=self.device)
        groups = get_canary_buffer_groups(pool)
        self.assertEqual(set(groups.keys()), {PoolKind.FULL, PoolKind.SWA})
        self.assertEqual(groups[PoolKind.FULL].k_head.shape[0], 16)
        self.assertEqual(groups[PoolKind.SWA].k_head.shape[0], 8)
        self.assertIsNotNone(groups[PoolKind.SWA].swa_index_lut)
        self.assertIsNone(groups[PoolKind.FULL].swa_index_lut)

    def test_mla_pool_no_v_half(self):
        pool = make_mla_pool(self.device, num_slots=16, dim=16)
        attach_canary_buffers(pool=pool, config=self.config, device=self.device)
        group = get_canary_buffer_groups(pool)[PoolKind.FULL]
        self.assertIsNone(group.v_head)
        self.assertIsNone(group.v_tail)
        self.assertEqual(group.real_kv_sources_v, ())
        self.assertFalse(group.has_v_half)

    def test_dsv4_pool_real_kv_sources_count(self):
        pool = make_dsv4_pool(self.device, full_slots=16, swa_slots=8, page_size=128)
        attach_canary_buffers(pool=pool, config=self.config, device=self.device)
        groups = get_canary_buffer_groups(pool)
        for group in groups.values():
            self.assertLessEqual(len(group.real_kv_sources_k), _MAX_REAL_KV_SOURCES)
            self.assertLessEqual(len(group.real_kv_sources_v), _MAX_REAL_KV_SOURCES)

    def test_real_kv_sources_below_4(self):
        layer = torch.zeros(8, 16, dtype=torch.float16, device=self.device)
        sources = make_row_source(layer_buffer=layer, read_bytes=4)
        self.assertGreater(len(sources), 0)
        self.assertLessEqual(len(sources), _MAX_REAL_KV_SOURCES)

    def test_real_kv_sources_above_4_raises(self):
        from sglang.jit_kernel.kv_canary.verify import (
            CanaryLaunchTag,
            RealKvHashMode,
            VerifyPlan,
            canary_verify_step,
        )

        tensor = torch.zeros(4, 4, dtype=torch.uint8, device=self.device)
        sources = tuple(
            RealKvSource(
                tensor=tensor, page_size=1, num_bytes_per_token=4, read_bytes=2
            )
            for _ in range(_MAX_REAL_KV_SOURCES + 1)
        )
        canary_buf = torch.zeros(
            4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=self.device
        )
        plan = VerifyPlan.allocate(verify_capacity=1, device=self.device)
        violation_ring = torch.zeros(2, 8, dtype=torch.int64, device=self.device)
        violation_write_index = torch.zeros(1, dtype=torch.int32, device=self.device)
        slot_run_counter = torch.zeros(1, dtype=torch.int64, device=self.device)
        kernel_run_counter = torch.zeros(1, dtype=torch.int64, device=self.device)

        with self.assertRaises(ValueError):
            canary_verify_step(
                canary_buf=canary_buf,
                plan=plan,
                kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
                violation_ring=violation_ring,
                violation_write_index=violation_write_index,
                slot_run_counter=slot_run_counter,
                kernel_run_counter=kernel_run_counter,
                real_kv_sources=sources,
                real_kv_hash_mode=RealKvHashMode.OFF,
            )

    def test_get_contiguous_buf_infos_inserts_canary_entries(self):
        for patched in (False, True):
            with self.subTest(patched=patched):
                pool = make_mha_pool(self.device, num_slots=16, dim=8, layer_num=2)
                ptrs_before, _, _ = pool.get_contiguous_buf_infos()
                n_before = len(ptrs_before)

                if patched:
                    attach_canary_buffers(
                        pool=pool, config=self.config, device=self.device
                    )
                    ptrs_after, _, _ = pool.get_contiguous_buf_infos()
                    self.assertEqual(len(ptrs_after), n_before + 4)
                else:
                    ptrs_after, _, _ = pool.get_contiguous_buf_infos()
                    self.assertEqual(ptrs_after, ptrs_before)

    def test_get_state_buf_infos_inserts_canary_entries(self):
        for patched in (False, True):
            with self.subTest(patched=patched):
                pool = make_swa_pool(self.device, full_slots=16, swa_slots=8)
                ptrs_before, _, _ = pool.get_state_buf_infos()
                n_before = len(ptrs_before)

                if patched:
                    attach_canary_buffers(
                        pool=pool, config=self.config, device=self.device
                    )
                    ptrs_after, _, _ = pool.get_state_buf_infos()
                    self.assertEqual(len(ptrs_after), n_before + 8)
                else:
                    ptrs_after, _, _ = pool.get_state_buf_infos()
                    self.assertEqual(ptrs_after, ptrs_before)

    def test_pd_layout_canary_inserted_correctly(self):
        pool = make_mha_pool(self.device, num_slots=16, dim=8, layer_num=2)
        k_ptrs_orig = [b.data_ptr() for b in pool.k_buffer]
        v_ptrs_orig = [b.data_ptr() for b in pool.v_buffer]

        attach_canary_buffers(pool=pool, config=self.config, device=self.device)
        group = get_canary_buffer_groups(pool)[PoolKind.FULL]
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

    def test_pd_naive_first_last_insert_fails_layout_check(self):
        pool = make_mha_pool(self.device, num_slots=16, dim=8, layer_num=2)
        k_ptrs_orig = [b.data_ptr() for b in pool.k_buffer]
        v_ptrs_orig = [b.data_ptr() for b in pool.v_buffer]

        attach_canary_buffers(pool=pool, config=self.config, device=self.device)
        group = get_canary_buffer_groups(pool)[PoolKind.FULL]
        ptrs_after, _, _ = pool.get_contiguous_buf_infos()

        naive_layout = (
            [group.k_head.data_ptr(), group.k_tail.data_ptr()]
            + k_ptrs_orig
            + v_ptrs_orig
            + [group.v_head.data_ptr(), group.v_tail.data_ptr()]
        )
        self.assertNotEqual(ptrs_after, naive_layout)

    def test_canary_buf_per_token_bytes_within_budget(self):
        pool = make_mha_pool(self.device, num_slots=16, dim=64, layer_num=2)
        attach_canary_buffers(pool=pool, config=self.config, device=self.device)
        group = get_canary_buffer_groups(pool)[PoolKind.FULL]

        slot_stride_bytes = group.k_head.stride(0) * group.k_head.element_size()
        self.assertLessEqual(slot_stride_bytes, CANARY_SLOT_BYTES)

        real_kv_per_token_bytes = (
            pool.k_buffer[0].stride(0) * pool.k_buffer[0].element_size()
        )
        self.assertLess(slot_stride_bytes, real_kv_per_token_bytes)


if __name__ == "__main__":
    unittest.main()
