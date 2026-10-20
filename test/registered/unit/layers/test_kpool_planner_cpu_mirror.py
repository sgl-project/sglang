"""CPU coverage for KPool request-slot selection and paged query rows."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa import kpool_plan
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


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


def _replay_extend(prefix_lens, extend_lens, kpool, tail_size, pools, tails):
    """Run one extend's kpool plan over token positions, indexing the tail and
    the chunk as the assemble and tail-scatter kernels do."""
    slots = list(range(len(extend_lens)))
    seq_lens = [p + n for p, n in zip(prefix_lens, extend_lens)]
    plan = kpool_plan._kpool_cpu_plan(_batch(extend_lens, seq_lens, slots), kpool, 64)
    chunk = [
        (req, pos)
        for req, p, n in zip(slots, prefix_lens, extend_lens)
        for pos in range(p, p + n)
    ]
    for req, pool_id, n_tail, src, base in zip(
        plan.pool_req,
        plan.pool_pool_id,
        plan.pool_n_from_tail,
        plan.pool_chunk_src,
        plan.pool_tail_logical_base,
    ):
        group = [tails[req].get((base + s) % tail_size) for s in range(n_tail)]
        for s in range(n_tail, kpool):
            chunk_req, pos = chunk[src + s - n_tail]
            assert chunk_req == req
            group.append(pos)
        pools[(req, pool_id)] = group
    for req, dst, src, n in zip(
        plan.tail_req,
        plan.tail_dst_logical_start,
        plan.tail_chunk_src,
        plan.tail_n_write,
    ):
        for s in range(n):
            assert chunk[src + s][0] == req
            tails[req][(dst + s) % tail_size] = chunk[src + s][1]


class TestKPoolPlannerCpuMirror(unittest.TestCase):
    def test_extend_split_before_its_last_token_pools_the_same_tokens(self):
        """A prefill draft extend may run as every token but the last, then the
        last token alone; each pooled group must read the same positions and
        the tail must end the same as with one extend."""
        prefix_lens = [0, 0, 0, 0, 3, 5, 6, 2]
        extend_lens = [2, 4, 5, 9, 2, 7, 6, 2]
        for tail_size in (4, 10):
            with self.subTest(tail_size=tail_size):

                def history():
                    # Tails already hold the prefix's open group.
                    tails = [
                        {pos % tail_size: pos for pos in range(p - p % 4, p)}
                        for p in prefix_lens
                    ]
                    return {}, tails

                pools, tails = history()
                _replay_extend(prefix_lens, extend_lens, 4, tail_size, pools, tails)
                split_pools, split_tails = history()
                _replay_extend(
                    prefix_lens,
                    [n - 1 for n in extend_lens],
                    4,
                    tail_size,
                    split_pools,
                    split_tails,
                )
                _replay_extend(
                    [p + n - 1 for p, n in zip(prefix_lens, extend_lens)],
                    [1] * len(extend_lens),
                    4,
                    tail_size,
                    split_pools,
                    split_tails,
                )
                self.assertEqual(split_pools, pools)
                for (req, pool_id), group in pools.items():
                    self.assertEqual(group, list(range(4 * pool_id, 4 * pool_id + 4)))
                # Only the open group's slots are read later; closed ones may
                # keep stale entries on either path.
                for req, (p, n) in enumerate(zip(prefix_lens, extend_lens)):
                    open_group = range(p + n - (p + n) % 4, p + n)
                    for split_view in (tails, split_tails):
                        self.assertEqual(
                            [split_view[req][pos % tail_size] for pos in open_group],
                            list(open_group),
                        )

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


if __name__ == "__main__":
    unittest.main()
