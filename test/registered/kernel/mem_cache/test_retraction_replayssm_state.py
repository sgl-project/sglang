import unittest
from array import array

import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.disaggregation.decode import HybridMambaDecodeReqToTokenPool
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.common import retraction_backup, retraction_restore
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestRetractionReplaySSMState(CustomTestCase):
    def _make_request(self, dtype, carries_mamba):
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=512,
            n_groups=2,
            num_heads=4,
            head_dim=128,
            state_size=128,
            conv_kernel=4,
        )
        pool = HybridMambaDecodeReqToTokenPool(
            size=4,
            mamba_size=8,
            pre_alloc_size=1,
            enable_overlap_schedule=False,
            max_context_len=32,
            device="cuda",
            enable_memory_saver=False,
            cache_params=Mamba2CacheParams(
                shape=shape,
                dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=dtype),
                layers=[0, 1],
            ),
            mamba_layer_ids=[0, 1],
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
            speculative_eagle_topk=1,
            enable_linear_replayssm_spec=True,
            linear_replayssm_cache_len=16,
        )
        kv = HybridLinearKVPool(
            size=32,
            dtype=torch.bfloat16,
            page_size=1,
            head_num=1,
            head_dim=64,
            full_attention_layer_ids=[2],
            device="cuda",
            mamba_pool=pool.mamba_pool,
        )
        allocator = TokenToKVPoolAllocator(
            size=32,
            dtype=torch.bfloat16,
            device="cuda",
            kvcache=kv if carries_mamba else kv.full_kv_pool,
            need_sort=False,
        )
        req = Req("retraction", "", array("q", [1, 2, 3]), SamplingParams())
        req.output_ids.extend(range(8))
        pool.alloc([req])
        return req, pool, allocator, kv.full_kv_pool

    def test_cpu_retraction_preserves_committed_recurrence(self):
        cases = [
            (True, torch.float32, 0, 0),
            (True, torch.float32, 3, 0),
            (False, torch.float32, 3, 0),
            (True, torch.float32, 5, 14),
            (False, torch.float32, 5, 14),
            (True, torch.bfloat16, 0, 7),
        ]
        for carries_mamba, dtype, pending, base in cases:
            with self.subTest(
                carries_mamba=carries_mamba, dtype=dtype, pending=pending, base=base
            ):
                torch.manual_seed(42)
                req, pool, allocator, kv = self._make_request(dtype, carries_mamba)
                mp = pool.mamba_pool
                state = mp.mamba_cache
                row, slot = req.kv.req_pool_idx, int(req.kv.mamba_pool_idx)
                self.assertNotEqual(row, slot)
                state.temporal.normal_(std=0.1)
                state.conv[0].fill_(0.5)
                for high, low in (
                    (state.replayssm_d, state.replayssm_rawv),
                    (state.replayssm_k, state.replayssm_rawk),
                ):
                    values = torch.randn_like(high, dtype=torch.float32) * 0.1
                    if high is state.replayssm_k:
                        values /= (values.square().sum(-1, keepdim=True) + 1e-6).sqrt()
                    high.copy_(values)
                    low.copy_(values - high.float())
                state.replayssm_g.fill_(-0.1)
                mp.replayssm_spec_write_pos[row] = pending
                mp.replayssm_cache_base[row] = base

                # Sequential recurrence is independent of the materializer's
                # matrix product; records after the committed prefix are excluded.
                expected = state.temporal[:, slot].cpu().double()
                for offset in range(pending):
                    pos = (base + offset) % 16
                    keys = (
                        state.replayssm_k[:, row, :, pos].cpu().double()
                        + state.replayssm_rawk[:, row, :, pos].cpu().double()
                    ).repeat_interleave(2, dim=1)
                    updates = (
                        state.replayssm_d[:, row, :, pos].cpu().double()
                        + state.replayssm_rawv[:, row, :, pos].cpu().double()
                    )
                    decay = state.replayssm_g[:, row, :, pos].cpu().double().exp()
                    expected = expected * decay[..., None, None]
                    expected += updates[..., :, None] * keys[..., None, :]

                other_slots = [i for i in range(state.temporal.shape[1]) if i != slot]
                other_rows = [
                    i for i in range(mp.replayssm_spec_write_pos.numel()) if i != row
                ]
                untouched = state.temporal[:, other_slots].clone()
                cursors = mp.replayssm_spec_write_pos[other_rows].clone()
                bases = mp.replayssm_cache_base[other_rows].clone()
                n = req.seqlen - 1
                locations = allocator.alloc(n)
                pool.write((row, slice(0, n)), locations.to(torch.int32))
                req.kv.kv_allocated_len = req.kv.kv_committed_len = n
                kv.k_buffer[0][locations] = 0.25
                kv.v_buffer[0][locations] = 0.75

                for _ in range(2):
                    self.assertTrue(
                        retraction_backup(req, None, pool, allocator, "cpu_tensor")
                    )
                self.assertTrue(torch.equal(state.temporal[:, other_slots], untouched))
                self.assertTrue(
                    torch.equal(mp.replayssm_spec_write_pos[other_rows], cursors)
                )
                self.assertTrue(torch.equal(mp.replayssm_cache_base[other_rows], bases))

                allocator.free(locations)
                pool.free_mamba_cache(req)
                pool.free(req)
                req.reset_for_retract()
                held_rows = pool.alloc_rows(1)
                pool.alloc([req])
                self.assertNotEqual(req.kv.req_pool_idx, row)
                pool.free_rows(held_rows)
                locations = allocator.alloc(n)
                pool.write(
                    (req.kv.req_pool_idx, slice(0, n)), locations.to(torch.int32)
                )
                req.kv.kv_allocated_len = req.kv.kv_committed_len = n
                state.temporal[:, req.kv.mamba_pool_idx] = -0.125
                state.conv[0][:, req.kv.mamba_pool_idx] = 0
                retraction_restore(req, None, pool, allocator, "cpu_tensor")

                torch.testing.assert_close(
                    state.temporal[:, req.kv.mamba_pool_idx].cpu().double(),
                    expected,
                    atol=1e-5,
                    rtol=1e-4,
                )
                self.assertIsNone(req.kv.retraction_backup)
                self.assertEqual(
                    int(mp.replayssm_spec_write_pos[req.kv.req_pool_idx]), 0
                )
                self.assertTrue(
                    torch.all(state.conv[0][:, req.kv.mamba_pool_idx] == 0.5)
                )
                self.assertTrue(torch.all(kv.k_buffer[0][locations] == 0.25))
                self.assertTrue(torch.all(kv.v_buffer[0][locations] == 0.75))


if __name__ == "__main__":
    unittest.main()
