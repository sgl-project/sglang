"""CPU coverage for KPool request-slot selection and paged query rows."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa import kpool_plan
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _NoDeviceReadTensor(torch.Tensor):
    def tolist(self):
        raise AssertionError("Reading device request slots would synchronize")


def _batch(extend_lens=(3, 5), seq_lens=(5, 11), slots=(9, 2)):
    indices = torch.tensor(slots, dtype=torch.int64)
    return SimpleNamespace(
        batch_size=len(slots),
        extend_seq_lens_cpu=list(extend_lens),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
        req_pool_indices=indices.as_subclass(_NoDeviceReadTensor),
        req_pool_indices_cpu=indices,
    )


def _expected_plan():
    return kpool_plan._KPoolCpuPlan(
        pool_batch_idx=[0, 1],
        pool_req=[9, 2],
        pool_pool_id=[0, 1],
        pool_n_from_tail=[2, 2],
        pool_chunk_src=[0, 3],
        pool_tail_logical_base=[0, 4],
        tail_req=[9, 2],
        tail_dst_logical_start=[4, 8],
        tail_chunk_src=[2, 5],
        tail_n_write=[1, 3],
        ragged_q_len=[3, 5],
        ragged_pool_pages=[1, 1],
        cu_pages_excl=[0, 1],
        cu_q_len_excl=[0, 3],
        total_pool_pages=2,
    )


class TestKPoolPlannerCpuMirror(unittest.TestCase):
    def test_mirror_avoids_device_read_and_preserves_compression_rows(self):
        for tensor_lengths in (False, True):
            with self.subTest(tensor_lengths=tensor_lengths):
                batch = _batch()
                if tensor_lengths:
                    batch.extend_seq_lens_cpu = torch.tensor(batch.extend_seq_lens_cpu)
                plan = kpool_plan._kpool_cpu_plan(batch, 4, 64)
                self.assertEqual(plan, _expected_plan())

    def test_absent_or_none_mirror_uses_device_slots(self):
        for absent in (False, True):
            with self.subTest(absent=absent):
                batch = _batch()
                batch.req_pool_indices = batch.req_pool_indices_cpu
                if absent:
                    del batch.req_pool_indices_cpu
                else:
                    batch.req_pool_indices_cpu = None
                plan = kpool_plan._kpool_cpu_plan(batch, 4, 64)
                self.assertEqual(plan, _expected_plan())

    def test_empty_mirror_needs_no_device_read(self):
        plan = kpool_plan._kpool_cpu_plan(_batch((), (), ()), 4, 64)
        self.assertEqual(plan, kpool_plan._KPoolCpuPlan())

    def test_paged_query_rows_use_local_lengths_and_explicit_output_size(self):
        batch = _batch()
        cpu_plan = kpool_plan._kpool_cpu_plan(
            batch,
            4,
            64,
            local_extend_seq_lens_cpu=[1, 2],
            local_seq_lens_cpu=[3, 8],
        )
        expected = _expected_plan()
        expected.ragged_q_len = [1, 2]
        expected.ragged_pool_pages = [0, 1]
        expected.cu_pages_excl = [0, 0]
        expected.cu_q_len_excl = [0, 1]
        expected.total_pool_pages = 1
        self.assertEqual(cpu_plan, expected)

        original_tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return original_tensor(*args, **kwargs)

        page_table = torch.zeros(2, 4, dtype=torch.int32)
        req_to_token = torch.zeros(10, 256, dtype=torch.int32)
        seq_lens = torch.tensor([3, 7, 8], dtype=torch.int32)
        local_slots = torch.tensor([9, 2], dtype=torch.int64)
        with (
            envs.SGLANG_DSA_FUSE_TOPK.override(True),
            patch.object(torch, "tensor", side_effect=cpu_tensor),
            patch.object(
                torch, "repeat_interleave", wraps=torch.repeat_interleave
            ) as repeat,
            patch.object(
                kpool_plan,
                "kpool_build_ragged_layout",
                return_value=(torch.empty(0), torch.empty(0), torch.empty(0)),
            ),
            patch.object(kpool_plan, "dsa_use_prefill_cp", return_value=False),
            patch.object(kpool_plan, "_RAGGED_SCRATCH_K_U8", None),
            patch.object(kpool_plan, "_RAGGED_SCRATCH_K_SCALE", None),
            patch.object(
                kpool_plan,
                "get_req_to_token_pool",
                return_value=SimpleNamespace(req_to_token=req_to_token),
            ),
        ):
            plan = kpool_plan._kpool_plan_to_gpu(
                cpu_plan,
                batch,
                page_table,
                page_table,
                seq_lens,
                local_slots,
                4,
                64,
                TopkTransformMethod.PAGED,
            )

        repeat.assert_called_once()
        self.assertEqual(repeat.call_args.kwargs["output_size"], 3)
        self.assertEqual(plan.ragged_paged_page_table_row_index.tolist(), [9, 2, 2])
        self.assertEqual(plan.ragged_paged_page_table_row_index.dtype, torch.int32)
        self.assertIs(plan.ragged_paged_page_table, req_to_token)


class TestNPUKPoolWritePlanAdapter(CustomTestCase):
    def test_reused_plan_kernel_keeps_npu_physical_page_ids(self):
        """NPU page IDs must survive the generic kernel's pooled-page division."""
        plan = kpool_plan._alloc_kpool_write_plan_buffers(
            max_bs=2,
            num_draft_tokens=3,
            kpool=4,
            device=torch.device("cpu"),
            is_verify=True,
        )
        metadata = SimpleNamespace(kpool_write_plan=plan)
        mode = SimpleNamespace(
            is_target_verify=lambda: True,
            is_decode_or_idle=lambda: False,
            is_draft_extend_v2=lambda: False,
        )

        def run_generic_plan(
            write_start,
            req_pool_indices,
            real_page_table,
            *,
            req_out,
            write_start_out,
            tail_logical_start_out,
            write_loc_out,
            pool_seqlens_per_q_out,
            seqlens_per_q_out,
            kpool,
            num_draft_tokens,
            index_page_size,
        ):
            for batch in range(write_start.shape[0]):
                start = int(write_start[batch])
                pool_id = start // kpool
                token_page_row = (pool_id // index_page_size) * kpool
                page = int(real_page_table[batch * num_draft_tokens, token_page_row])
                write_loc_out[batch, 0] = (
                    page // kpool * index_page_size + pool_id % index_page_size
                )

        with (
            patch.object(kpool_plan, "is_npu", return_value=True),
            patch.object(
                kpool_plan,
                "update_kpool_write_plan_cuda_graph",
                side_effect=run_generic_plan,
            ),
        ):
            kpool_plan.update_kpool_write_plan_npu(
                metadata,
                write_start=torch.tensor([5, 517], dtype=torch.int32),
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                real_page_table=torch.tensor(
                    [[1, 0, 0, 0, 0], [2, 0, 0, 0, 7]], dtype=torch.int32
                ),
                pool_size=4,
                real_page_size=128,
                num_draft_tokens=3,
                forward_mode=mode,
                slots_per_page=128,
            )

        self.assertEqual(plan.write_loc[:, 0].tolist(), [129, 897])


if __name__ == "__main__":
    unittest.main()
