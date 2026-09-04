"""KPool metadata refreshes outside CUDA graphs, without loading a model."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import dsa_backend
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "Test requires CUDA")
class TestDSAKPoolMetadata(CustomTestCase):
    BS = 4
    WIDTH = 4096
    NEXT_N = 6

    def setUp(self):
        super().setUp()
        major, _ = torch.cuda.get_device_capability()
        if major not in (9, 10):
            self.skipTest("DeepGEMM paged metadata requires Hopper or Blackwell")
        # Only dispatch/configuration are stubbed: metadata construction, GPU
        # kernels, DeepGEMM schedules and CUDA graph replay all run normally.
        self.enterContext(
            patch.object(
                dsa_backend,
                "get_platform",
                return_value=SimpleNamespace(is_sm100=major == 10),
            )
        )
        self.enterContext(
            patch.object(DeepseekSparseAttnBackend, "set_dsa_prefill_impl")
        )

    def _backend(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.device = torch.device("cuda")
        backend.device_sm_major = torch.cuda.get_device_capability()[0]
        backend.num_q_heads = 64
        backend.real_page_size = 64
        backend.dsa_index_kpool = 4
        backend.dsa_index_topk = 2048
        backend.speculative_num_draft_tokens = self.NEXT_N
        backend.dsa_decode_impl = "tilelang"
        backend.hisparse_coordinator = None
        backend.use_fused_topk = False
        backend.dsa_topk_backend = SimpleNamespace(should_use_topk_v2=lambda: True)
        backend.token_to_kv_pool = SimpleNamespace(slots_per_page=64)
        backend.req_to_token = torch.arange(
            8 * self.WIDTH, dtype=torch.int32, device="cuda"
        ).view(8, self.WIDTH)
        backend._arange_buf = torch.arange(128, dtype=torch.int32, device="cuda")
        backend.init_cuda_graph_state(self.BS, self.BS * self.NEXT_N)
        return backend

    def _initialize(self, backend, mode):
        seq_lens = torch.tensor([7, 63, 255, 2049], device="cuda")
        req_indices = torch.tensor([0, 1, 2, 3], device="cuda")
        backend._apply_cuda_graph_metadata(
            self.BS, req_indices, seq_lens, seq_lens.cpu(), mode, None
        )
        return backend.decode_cuda_graph_metadata[self.BS]

    def _expected(self, seq_lens, mode):
        cache = seq_lens.to(torch.int32)
        if mode.is_target_verify():
            cache = cache + self.NEXT_N
        if mode.is_decode_or_idle():
            expanded = cache
        else:
            expanded = (
                cache[:, None]
                - self.NEXT_N
                + torch.arange(1, self.NEXT_N + 1, device="cuda", dtype=torch.int32)
            ).flatten()
        # Selected history is pool aligned; a partial live pool is always kept.
        dsa_lengths = torch.tensor(
            [min(n - n % 4, 2048) + n % 4 for n in expanded.tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        return cache, expanded, dsa_lengths

    def _assert_schedules(self, metadata, cache, expanded, mode):
        import deep_gemm

        from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2

        # These short sequences route no items to the cluster pool. Only the
        # header is initialized; unused plan rows contain allocator scratch.
        expected_plan = plan_topk_v2(expanded)
        self.assertEqual(expected_plan[0, 1].item(), 0)
        torch.testing.assert_close(metadata.topk_v2_plan[0], expected_plan[0])

        if mode.is_target_verify() and torch.cuda.get_device_capability()[0] == 10:
            context_lens = cache[:, None].expand(-1, self.NEXT_N).contiguous()
        else:
            context_lens = expanded.view(-1, 1)
        torch.testing.assert_close(metadata.paged_mqa_ctx_lens_2d, context_lens)
        torch.testing.assert_close(
            metadata.paged_mqa_schedule_metadata,
            deep_gemm.get_paged_mqa_logits_metadata(
                context_lens, 64, deep_gemm.get_num_sms()
            ),
        )
        pool_lens = (expanded // 4).view(-1, 1).clamp(min=1)
        pool_schedule = (
            metadata.pooled_paged_mqa_schedule_metadata
            if mode.is_decode_or_idle()
            else metadata.kpool_write_plan.pool_schedule_metadata
        )
        torch.testing.assert_close(
            pool_schedule,
            deep_gemm.get_paged_mqa_logits_metadata(
                pool_lens, 64, deep_gemm.get_num_sms()
            ),
        )

    def test_decode_verify_and_draft_extend_graph_replay(self):
        for mode in (
            ForwardMode.DECODE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND_V2,
        ):
            with self.subTest(mode=mode):
                backend = self._backend()
                metadata = self._initialize(backend, mode)
                observed = torch.empty_like(metadata.dsa_cache_seqlens_int32)
                observed_write_start = torch.empty_like(
                    metadata.kpool_write_plan.write_start
                )
                forward_batch = SimpleNamespace(forward_mode=mode, batch_size=self.BS)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    backend.init_forward_metadata_in_graph(forward_batch)
                    observed.copy_(metadata.dsa_cache_seqlens_int32)
                    observed_write_start.copy_(metadata.kpool_write_plan.write_start)

                for lengths, requests in (
                    ([8, 64, 256, 2050], [7, 6, 5, 4]),
                    ([7, 65, 259, 2051], [1, 3, 0, 2]),
                ):
                    seq_lens = torch.tensor(lengths, device="cuda")
                    req_indices = torch.tensor(requests, device="cuda")
                    spec_info = SimpleNamespace(
                        num_accept_tokens=torch.tensor([1, 2, 5, 6], device="cuda")
                    )
                    schedule_ptr = metadata.paged_mqa_schedule_metadata.data_ptr()
                    topk_plan_ptr = metadata.topk_v2_plan.data_ptr()
                    metadata.topk_v2_plan.fill_(-1)
                    backend._apply_cuda_graph_metadata(
                        self.BS, req_indices, seq_lens, None, mode, spec_info
                    )
                    graph.replay()
                    cache, expanded, expected = self._expected(seq_lens, mode)
                    torch.testing.assert_close(observed, expected)
                    torch.testing.assert_close(metadata.cache_seqlens_int32, cache)
                    torch.testing.assert_close(metadata.dsa_seqlens_expanded, expanded)
                    torch.testing.assert_close(
                        metadata.dsa_cu_seqlens_k[1:],
                        expected.cumsum(0, dtype=torch.int32),
                    )
                    page_table = backend.req_to_token[req_indices]
                    if not mode.is_decode_or_idle():
                        page_table = page_table.repeat_interleave(self.NEXT_N, dim=0)
                    torch.testing.assert_close(metadata.page_table_1, page_table)
                    torch.testing.assert_close(
                        metadata.real_page_table, page_table[:, ::64] // 64
                    )
                    write_start = cache - (
                        1 if mode.is_decode_or_idle() else self.NEXT_N
                    )
                    torch.testing.assert_close(observed_write_start, write_start)
                    torch.testing.assert_close(
                        metadata.kpool_write_plan.req, req_indices
                    )
                    self._assert_schedules(metadata, cache, expanded, mode)
                    if mode.is_draft_extend_v2():
                        torch.testing.assert_close(
                            metadata.kpool_write_plan.effective_n_per_batch,
                            spec_info.num_accept_tokens.to(torch.int32),
                        )
                    self.assertEqual(
                        metadata.paged_mqa_schedule_metadata.data_ptr(), schedule_ptr
                    )
                    self.assertEqual(metadata.topk_v2_plan.data_ptr(), topk_plan_ptr)
                    self.assertIs(backend.forward_metadata, metadata)

    def test_mtp_precomputed_metadata_preserves_live_tail(self):
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                source, target = self._backend(), self._backend()
                self._initialize(source, mode)
                metadata = self._initialize(target, mode)
                seq_lens = torch.tensor([7, 65, 259, 2051], device="cuda")
                req_indices = torch.tensor([7, 5, 3, 1], device="cuda")
                precomputed = source._precompute_replay_metadata(
                    self.BS, req_indices, seq_lens, None, mode
                )
                metadata.topk_v2_plan.fill_(-1)
                target.init_forward_metadata_replay_cuda_graph_from_precomputed(
                    self.BS, precomputed, mode
                )
                cache, expanded, expected = self._expected(seq_lens, mode)
                torch.testing.assert_close(metadata.cache_seqlens_int32, cache)
                torch.testing.assert_close(metadata.dsa_seqlens_expanded, expanded)
                torch.testing.assert_close(metadata.dsa_cache_seqlens_int32, expected)
                torch.testing.assert_close(metadata.kpool_write_plan.req, req_indices)
                self._assert_schedules(metadata, cache, expanded, mode)
                self.assertIs(target.forward_metadata, metadata)


if __name__ == "__main__":
    unittest.main()
