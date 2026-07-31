# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.kernels.ops.attention.fla.chunk_delta_h import (
    prepare_kda_state_io_indices,
)
from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestKimiK3KDATemporalCOW(unittest.TestCase):
    H = 24
    K = 128
    V = 128
    SLOTS = 12

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(20260731)
        torch.cuda.set_device(0)

    def _make_state(self, envelope_strided: bool) -> torch.Tensor:
        elements_per_slot = self.H * self.V * self.K
        if not envelope_strided:
            return (
                torch.randn(
                    self.SLOTS,
                    self.H,
                    self.V,
                    self.K,
                    device="cuda",
                    dtype=torch.float32,
                )
                * 0.01
            )

        slot_stride = elements_per_slot + 4096
        storage = torch.randn(
            self.SLOTS * slot_stride,
            device="cuda",
            dtype=torch.float32,
        )
        return torch.as_strided(
            storage,
            size=(self.SLOTS, self.H, self.V, self.K),
            stride=(slot_stride, self.V * self.K, self.K, 1),
        )

    def _make_inputs(self, lengths):
        tokens = sum(lengths)
        shape = (1, tokens, self.H, self.K)
        q = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
        k = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
        v = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
        g = torch.randn(shape, device="cuda", dtype=torch.float32) * 0.1 - 1.0
        beta = torch.sigmoid(
            torch.randn(1, tokens, self.H, device="cuda", dtype=torch.float32)
        )
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
        return q, k, v, g, beta, cu_seqlens

    @staticmethod
    def _clone_preserving_strides(state):
        clone = torch.empty_strided(
            state.shape,
            state.stride(),
            device=state.device,
            dtype=state.dtype,
        )
        clone.copy_(state)
        return clone

    def _run_case(self, lengths, sources, envelope_strided):
        destinations = list(range(6, 6 + len(sources)))
        src = torch.tensor(sources, device="cuda", dtype=torch.int32)
        dst = torch.tensor(destinations, device="cuda", dtype=torch.int32)
        q, k, v, g, beta, cu_seqlens = self._make_inputs(lengths)
        initial = self._make_state(envelope_strided)

        baseline_state = self._clone_preserving_strides(initial)
        baseline_state[dst.long()] = baseline_state[src.long()]
        expected = chunk_kda(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            initial_state=baseline_state,
            initial_state_indices=dst,
            cu_seqlens=cu_seqlens,
        )

        candidate_state = self._clone_preserving_strides(initial)
        source_before = candidate_state[src.long()].clone()
        unrelated_before = candidate_state[5].clone()
        source_map = torch.arange(self.SLOTS, device="cuda", dtype=torch.int32)
        source_map[dst.long()] = src
        state_io_indices = prepare_kda_state_io_indices(
            dst,
            source_map,
            torch.empty(len(dst), device="cuda", dtype=torch.int32),
        )
        actual = chunk_kda(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            initial_state=candidate_state,
            initial_state_indices=dst,
            initial_state_io_indices=state_io_indices,
            cu_seqlens=cu_seqlens,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-3)
        torch.testing.assert_close(
            candidate_state[dst.long()],
            baseline_state[dst.long()],
            rtol=1e-2,
            atol=2e-3,
        )
        torch.testing.assert_close(
            candidate_state[src.long()], source_before, rtol=0, atol=0
        )
        torch.testing.assert_close(candidate_state[5], unrelated_before, rtol=0, atol=0)
        torch.testing.assert_close(source_map[dst.long()], dst, rtol=0, atol=0)

    def test_dense_single_token_and_shared_branches(self):
        self._run_case([1], [0], envelope_strided=False)
        self._run_case([1, 1, 1, 1], [0, 0, 0, 0], envelope_strided=False)

    def test_envelope_strided_variable_length(self):
        self._run_case([1, 17], [0, 1], envelope_strided=True)

    def test_packed_owner_resolution_spans_multiple_programs(self):
        slots = 1024
        count = 513
        dst = torch.arange(count, device="cuda", dtype=torch.int32)
        source_map = torch.arange(slots, device="cuda", dtype=torch.int32)
        expected = (dst + 17) % slots
        source_map[dst.long()] = expected
        output = torch.empty(count, device="cuda", dtype=torch.int32)

        actual = prepare_kda_state_io_indices(dst, source_map, output)
        actual_src = torch.bitwise_and(
            torch.bitwise_right_shift(actual, 16), 0xFFFF
        )
        actual_dst = torch.bitwise_and(actual, 0xFFFF)
        torch.testing.assert_close(actual_src, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_dst, dst, rtol=0, atol=0)
        torch.testing.assert_close(source_map[dst.long()], dst, rtol=0, atol=0)

    def test_pool_publish_reset_and_slot_reuse(self):
        layers = 2
        slots = 6
        pool = object.__new__(MambaPool)
        pool.mamba_cache = MambaPool.State(
            conv=[
                torch.randn(
                    layers,
                    slots,
                    3,
                    32,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
            ],
            temporal=torch.randn(
                layers,
                slots,
                4,
                8,
                8,
                device="cuda",
                dtype=torch.float32,
            ),
        )
        pool.temporal_cow_source_map = None
        pool.replayssm_write_pos = torch.ones(slots, device="cuda", dtype=torch.int32)
        pool.replayssm_cache_base = torch.ones(slots, device="cuda", dtype=torch.int32)
        pool.replayssm_is_flush = torch.ones(slots, device="cuda", dtype=torch.int8)
        pool.debug_memory_pool = False
        pool.__dict__["_conv_fuse_ok"] = False

        source_map = pool.enable_temporal_cow()
        src = torch.tensor([1], device="cuda", dtype=torch.int32)
        dst = torch.tensor([4], device="cuda", dtype=torch.int32)
        temporal_dst_before = pool.mamba_cache.temporal[:, dst.long()].clone()

        pool.prepare_temporal_cow(src, dst)
        torch.testing.assert_close(
            pool.mamba_cache.conv[0][:, dst.long()],
            pool.mamba_cache.conv[0][:, src.long()],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            pool.mamba_cache.temporal[:, dst.long()],
            temporal_dst_before,
            rtol=0,
            atol=0,
        )
        self.assertEqual(int(source_map[dst].item()), int(src.item()))
        self.assertEqual(int(pool.replayssm_write_pos[dst].item()), 0)
        self.assertEqual(int(pool.replayssm_cache_base[dst].item()), 0)
        self.assertEqual(int(pool.replayssm_is_flush[dst].item()), 0)

        state_io_indices = prepare_kda_state_io_indices(
            dst,
            source_map,
            torch.empty(len(dst), device="cuda", dtype=torch.int32),
        )
        packed = int(state_io_indices[0].item())
        self.assertEqual((packed >> 16) & 0xFFFF, int(src.item()))
        self.assertEqual(packed & 0xFFFF, int(dst.item()))
        self.assertEqual(int(source_map[dst].item()), int(dst.item()))

        pool.copy_from(src, dst)
        torch.testing.assert_close(
            pool.mamba_cache.temporal[:, dst.long()],
            pool.mamba_cache.temporal[:, src.long()],
            rtol=0,
            atol=0,
        )
        self.assertEqual(int(source_map[dst].item()), int(dst.item()))

        pool.clear_slots(dst)
        self.assertEqual(
            int(torch.count_nonzero(pool.mamba_cache.temporal[:, dst.long()])),
            0,
        )
        self.assertEqual(
            int(torch.count_nonzero(pool.mamba_cache.conv[0][:, dst.long()])),
            0,
        )
        self.assertEqual(int(source_map[dst].item()), int(dst.item()))

    def test_runner_uses_copy_free_for_triton_extend_and_split_prefill(self):
        mamba_pool = SimpleNamespace(
            temporal_cow_source_map=torch.arange(8, device="cuda", dtype=torch.int32),
            prepare_temporal_cow=Mock(),
            copy_from=Mock(),
            clear_slots=Mock(),
        )
        req_pool = object.__new__(HybridReqToTokenPool)
        req_pool.mamba_pool = mamba_pool
        req_pool.mamba_ckpt_pool = None

        runner = object.__new__(ModelRunner)
        runner.req_to_token_pool = req_pool
        runner.is_draft_worker = False
        runner.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"])
        )

        def batch(mode):
            return SimpleNamespace(
                forward_mode=mode,
                mamba_clear_indices=None,
                mamba_cow_src_indices=torch.tensor(
                    [1], device="cuda", dtype=torch.int32
                ),
                mamba_cow_dst_indices=torch.tensor(
                    [5], device="cuda", dtype=torch.int32
                ),
            )

        triton_backend = SimpleNamespace(is_triton=lambda: True)
        with patch(
            "sglang.srt.layers.attention.linear.utils.get_linear_attn_prefill_backend",
            return_value=triton_backend,
        ):
            extend = batch(ForwardMode.EXTEND)
            runner._maybe_execute_deferred_mamba_cow_and_clear(extend)
            mamba_pool.prepare_temporal_cow.assert_called_once()
            mamba_pool.copy_from.assert_not_called()

            split = batch(ForwardMode.SPLIT_PREFILL)
            runner._maybe_execute_deferred_mamba_cow_and_clear(split)
            self.assertEqual(mamba_pool.prepare_temporal_cow.call_count, 2)
            mamba_pool.copy_from.assert_not_called()

    def test_changing_source_cuda_graph_replay(self):
        dst = torch.tensor([6], device="cuda", dtype=torch.int32)
        state_seed = self._make_state(envelope_strided=False)
        state = state_seed.clone()
        source_map = torch.arange(self.SLOTS, device="cuda", dtype=torch.int32)
        state_io_indices = torch.empty(1, device="cuda", dtype=torch.int32)
        prepare_kda_state_io_indices(dst, source_map, state_io_indices)
        static_inputs = list(self._make_inputs([1]))

        def candidate():
            return chunk_kda(
                q=static_inputs[0],
                k=static_inputs[1],
                v=static_inputs[2],
                g=static_inputs[3],
                beta=static_inputs[4],
                initial_state=state,
                initial_state_indices=dst,
                initial_state_io_indices=state_io_indices,
                cu_seqlens=static_inputs[5],
            )

        candidate()
        torch.cuda.synchronize()
        state.copy_(state_seed)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = candidate()

        for source in (0, 1):
            fresh_inputs = self._make_inputs([1])
            for static, fresh in zip(static_inputs[:5], fresh_inputs[:5]):
                static.copy_(fresh)
            state.copy_(state_seed)
            source_map.copy_(torch.arange(self.SLOTS, device="cuda", dtype=torch.int32))
            source_map[dst.long()] = source
            prepare_kda_state_io_indices(dst, source_map, state_io_indices)

            baseline_state = state_seed.clone()
            baseline_state[dst.long()] = baseline_state[source].clone()
            expected = chunk_kda(
                q=fresh_inputs[0].clone(),
                k=fresh_inputs[1].clone(),
                v=fresh_inputs[2].clone(),
                g=fresh_inputs[3].clone(),
                beta=fresh_inputs[4].clone(),
                initial_state=baseline_state,
                initial_state_indices=dst,
                cu_seqlens=fresh_inputs[5],
            )
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(graph_output, expected, rtol=1e-2, atol=2e-3)
            torch.testing.assert_close(
                state[dst.long()],
                baseline_state[dst.long()],
                rtol=1e-2,
                atol=2e-3,
            )


if __name__ == "__main__":
    unittest.main()
