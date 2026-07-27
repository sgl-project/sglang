"""Correctness tests for DSV4 target-verify live-prefix metadata.

The target-verify CUDA graph allocates metadata at its static capture capacity,
but only the prefix selected by the current device-side sequence length is
live. These tests pin the scalar formulas and the live page-table/C128
prefixes at a fixed 1M-token capture capacity, including replay from a long
batch to an empty one and back without inspecting intentionally stale tails.
They also check the production activation gate and that DeepGEMM ignores
poisoned capture-capacity page-table columns outside the live prefix.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.attention.dsa import deepgemm_paged_mqa_logits_native
from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
    init_compression_metadata,
)
from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
    BuildPageTablePositions,
)
from sglang.srt.layers.attention.dsa.utils import (
    fp8_mqa_logits_ceil_to_ue8m0,
    fp8_mqa_logits_make_fused_kv,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# DSV4 target verification is deployed on Blackwell. Keep this focused kernel
# regression on the SM100 runner that exercises the optimized path.
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

PAGE_SIZE = 256
SWA_WINDOW = 128
MAX_SEQ_LEN = 1 << 20
NUM_PAGES = MAX_SEQ_LEN // PAGE_SIZE
SEQUENCE_LENGTHS = (0, 1, 127, 128, 129, 255, 256, 257, 383, 384, 511, 512, 513)
MULTI_BLOCK_LENGTHS = (65_537, MAX_SEQ_LEN - 1)
INDEXER_BLOCK_SIZE = 64
INDEXER_HEAD_DIM = 128
INDEXER_NUM_HEADS = 32


class TestDSV4LivePrefixMetadata(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        if torch.cuda.get_device_capability()[0] < 10:
            raise unittest.SkipTest(
                "DSV4 target-verify live-prefix metadata is covered on SM100+"
            )
        cls.device = torch.device("cuda")

    def _make_page_inputs(self, seq_lens: tuple[int, ...]):
        num_q = len(seq_lens)
        req_to_token = torch.full(
            (num_q, MAX_SEQ_LEN),
            -1,
            dtype=torch.int32,
            device=self.device,
        )

        # Give every request distinct physical pages and nonzero within-page
        # token offsets. The expected page table must sample exactly the page
        # boundaries and integer-divide those token locations by PAGE_SIZE.
        physical_pages = (
            torch.arange(
                num_q * NUM_PAGES,
                dtype=torch.int32,
                device=self.device,
            ).view(num_q, NUM_PAGES)
            + 100
        )
        req_to_token[:, ::PAGE_SIZE] = physical_pages * PAGE_SIZE + 17

        # The non-identity request order catches row-index/request-index mixups.
        req_pool_indices = torch.arange(
            num_q - 1, -1, -1, dtype=torch.int64, device=self.device
        )
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
        return req_to_token, req_pool_indices, seq_lens_tensor

    @staticmethod
    def _run_page_metadata(req_to_token, req_pool_indices, seq_lens):
        return BuildPageTablePositions.triton(
            req_to_token=req_to_token,
            req_pool_indices_repeated=req_pool_indices,
            seq_lens_casual=seq_lens,
            max_seq_len=MAX_SEQ_LEN,
            page_size=PAGE_SIZE,
            swa_window=SWA_WINDOW,
            live_prefix_only=True,
        )

    @staticmethod
    def _run_compression_metadata(page_metadata, raw_out_loc):
        return init_compression_metadata(
            page_metadata.seq_lens_casual,
            page_metadata.positions_casual,
            raw_out_loc,
            page_metadata.page_table,
            PAGE_SIZE,
            compute_page_indices=True,
            live_prefix_only=True,
        )

    def _assert_page_metadata(
        self,
        seq_lens: tuple[int, ...],
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        metadata,
    ):
        self.assertEqual(metadata.page_table.shape, (len(seq_lens), NUM_PAGES))
        self.assertEqual(metadata.seq_lens_casual.cpu().tolist(), list(seq_lens))
        self.assertEqual(
            metadata.positions_casual.cpu().tolist(),
            [seq_len - 1 for seq_len in seq_lens],
        )
        self.assertEqual(
            metadata.swa_topk_lengths.cpu().tolist(),
            [min(seq_len, SWA_WINDOW) for seq_len in seq_lens],
        )

        req_to_token_cpu = req_to_token.cpu()
        req_pool_indices_cpu = req_pool_indices.cpu().tolist()
        for row, (req, seq_len) in enumerate(zip(req_pool_indices_cpu, seq_lens)):
            live_pages = min(
                (max(seq_len, 0) + PAGE_SIZE - 1) // PAGE_SIZE,
                metadata.page_table.shape[1],
            )
            expected = [
                int(req_to_token_cpu[req, page * PAGE_SIZE]) // PAGE_SIZE
                for page in range(live_pages)
            ]
            self.assertEqual(
                metadata.page_table[row, :live_pages].cpu().tolist(),
                expected,
                f"page-table live prefix mismatch for row={row}, seq_len={seq_len}",
            )
            # The suffix is capture-capacity storage, not current metadata.
            # A shorter replay is allowed to leave values from an earlier
            # longer replay there, so deliberately do not inspect it.

    def _assert_compression_metadata(
        self,
        seq_lens: tuple[int, ...],
        raw_out_loc: torch.Tensor,
        page_table: torch.Tensor,
        outputs,
    ):
        (
            c4_out_loc,
            c4_positions,
            c4_seq_lens_raw,
            c4_seq_lens_clamp1,
            c128_out_loc,
            c128_positions,
            c128_seq_lens_raw,
            c128_seq_lens_clamp1,
            c128_page_indices,
        ) = outputs
        self.assertIsNotNone(c128_page_indices)
        self.assertEqual(
            c128_page_indices.shape,
            (len(seq_lens), NUM_PAGES * (PAGE_SIZE // 128)),
        )

        raw_out_loc_cpu = raw_out_loc.cpu().tolist()
        expected_c4_raw = [seq_len // 4 for seq_len in seq_lens]
        expected_c128_raw = [seq_len // 128 for seq_len in seq_lens]
        self.assertEqual(
            c4_out_loc.cpu().tolist(),
            [
                raw_loc // 4 if seq_len % 4 == 0 else 0
                for raw_loc, seq_len in zip(raw_out_loc_cpu, seq_lens)
            ],
        )
        self.assertEqual(
            c4_positions.cpu().tolist(),
            [(seq_len - 1) & ~3 for seq_len in seq_lens],
        )
        self.assertEqual(c4_seq_lens_raw.cpu().tolist(), expected_c4_raw)
        self.assertEqual(
            c4_seq_lens_clamp1.cpu().tolist(),
            [max(raw_len, 1) for raw_len in expected_c4_raw],
        )
        self.assertEqual(
            c128_out_loc.cpu().tolist(),
            [
                raw_loc // 128 if seq_len % 128 == 0 else 0
                for raw_loc, seq_len in zip(raw_out_loc_cpu, seq_lens)
            ],
        )
        self.assertEqual(
            c128_positions.cpu().tolist(),
            [(seq_len - 1) & ~127 for seq_len in seq_lens],
        )
        self.assertEqual(c128_seq_lens_raw.cpu().tolist(), expected_c128_raw)
        self.assertEqual(
            c128_seq_lens_clamp1.cpu().tolist(),
            [max(raw_len, 1) for raw_len in expected_c128_raw],
        )

        assert c128_page_indices is not None
        c128_page_size = PAGE_SIZE // 128
        capacity = c128_page_indices.shape[1]
        page_table_cpu = page_table.cpu()
        for row, raw_len in enumerate(expected_c128_raw):
            live_c128 = min(max(raw_len, 1), capacity)
            if raw_len == 0:
                expected = [-1]
            else:
                expected = [
                    int(page_table_cpu[row, offset // c128_page_size]) * c128_page_size
                    + offset % c128_page_size
                    for offset in range(live_c128)
                ]
            self.assertEqual(
                c128_page_indices[row, :live_c128].cpu().tolist(),
                expected,
                f"C128 live prefix mismatch for row={row}, seq_len={seq_lens[row]}",
            )
            # As above, the suffix may be stale capture-capacity storage.

    def test_boundary_lengths_match_exact_live_prefixes(self):
        seq_lens_values = SEQUENCE_LENGTHS + MULTI_BLOCK_LENGTHS
        req_to_token, req_pool_indices, seq_lens = self._make_page_inputs(
            seq_lens_values
        )
        raw_out_loc = (
            torch.arange(
                1,
                len(seq_lens_values) + 1,
                dtype=torch.int64,
                device=self.device,
            )
            * 512
        )

        page_metadata = self._run_page_metadata(
            req_to_token, req_pool_indices, seq_lens
        )
        compression_outputs = self._run_compression_metadata(page_metadata, raw_out_loc)
        torch.cuda.synchronize()

        self._assert_page_metadata(
            seq_lens_values,
            req_to_token,
            req_pool_indices,
            page_metadata,
        )
        self._assert_compression_metadata(
            seq_lens_values,
            raw_out_loc,
            page_metadata.page_table,
            compression_outputs,
        )

    def test_cuda_graph_replay_long_short_long(self):
        long_seq_len = MAX_SEQ_LEN - 1
        req_to_token, req_pool_indices, seq_lens = self._make_page_inputs(
            (long_seq_len,)
        )
        raw_out_loc = torch.tensor([512], dtype=torch.int64, device=self.device)

        # Compile both Triton kernels before capture.
        self._run_compression_metadata(
            self._run_page_metadata(req_to_token, req_pool_indices, seq_lens),
            raw_out_loc,
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            page_metadata = self._run_page_metadata(
                req_to_token, req_pool_indices, seq_lens
            )
            compression_outputs = self._run_compression_metadata(
                page_metadata, raw_out_loc
            )

        # Materialize the newly allocated graph outputs with the first long
        # replay; capture records the work but does not promise initialized
        # contents in those buffers.
        graph.replay()
        torch.cuda.synchronize()
        self._assert_page_metadata(
            (long_seq_len,), req_to_token, req_pool_indices, page_metadata
        )
        self._assert_compression_metadata(
            (long_seq_len,),
            raw_out_loc,
            page_metadata.page_table,
            compression_outputs,
        )

        # The empty replay must refresh every scalar and write the required
        # C128 -1 sentinel at element zero. Page-table and C128 tails still
        # contain long-replay data and are intentionally outside the contract.
        seq_lens.fill_(0)
        graph.replay()
        torch.cuda.synchronize()
        self._assert_page_metadata((0,), req_to_token, req_pool_indices, page_metadata)
        self._assert_compression_metadata(
            (0,),
            raw_out_loc,
            page_metadata.page_table,
            compression_outputs,
        )

        seq_lens.fill_(long_seq_len)
        graph.replay()
        torch.cuda.synchronize()
        self._assert_page_metadata(
            (long_seq_len,), req_to_token, req_pool_indices, page_metadata
        )
        self._assert_compression_metadata(
            (long_seq_len,),
            raw_out_loc,
            page_metadata.page_table,
            compression_outputs,
        )

    def test_deepgemm_ignores_poisoned_page_table_tail(self):
        """DeepGEMM must consume only pages selected by the live C4 lengths."""
        import deep_gemm

        batch_size = 1
        verify_width = 4
        max_seq_len = 4 * INDEXER_BLOCK_SIZE
        num_physical_pages = 4

        torch.manual_seed(1234)
        q_fp8 = torch.randn(
            batch_size,
            verify_width,
            INDEXER_NUM_HEADS,
            INDEXER_HEAD_DIM,
            device=self.device,
            dtype=torch.bfloat16,
        ).to(torch.float8_e4m3fn)
        kv_bf16 = torch.randn(
            num_physical_pages,
            INDEXER_BLOCK_SIZE,
            INDEXER_HEAD_DIM,
            device=self.device,
            dtype=torch.bfloat16,
        )
        kv_amax = kv_bf16.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
        kv_scales = fp8_mqa_logits_ceil_to_ue8m0(kv_amax / 448.0).squeeze(-1)
        kv_fp8 = (kv_bf16 / kv_scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
        kv_fused = fp8_mqa_logits_make_fused_kv(
            kv_fp8,
            kv_scales,
            INDEXER_BLOCK_SIZE,
            INDEXER_HEAD_DIM,
        )
        weights = torch.randn(
            batch_size * verify_width,
            INDEXER_NUM_HEADS,
            device=self.device,
            dtype=torch.float32,
        )

        # Four causal target-verify queries, all of which need exactly the
        # first two 64-token C4 pages. Columns two and three are capture
        # capacity only.
        context_lens = torch.tensor(
            [[125, 126, 127, 128]], dtype=torch.int32, device=self.device
        )
        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens,
            INDEXER_BLOCK_SIZE,
            deep_gemm.get_num_sms(),
        )
        valid_page_table = torch.arange(
            num_physical_pages, dtype=torch.int32, device=self.device
        ).unsqueeze(0)
        poisoned_page_table = valid_page_table.clone()
        poisoned_page_table[0, 2:] = torch.tensor(
            [num_physical_pages + 123, 2**30 - 1],
            dtype=torch.int32,
            device=self.device,
        )

        def run(page_table):
            expanded_page_table = page_table.repeat_interleave(verify_width, dim=0)
            return deepgemm_paged_mqa_logits_native(
                deep_gemm.fp8_paged_mqa_logits,
                q_fp8.view(
                    batch_size * verify_width,
                    INDEXER_NUM_HEADS,
                    INDEXER_HEAD_DIM,
                ),
                kv_fused,
                weights,
                context_lens,
                expanded_page_table,
                schedule_metadata,
                max_seq_len,
                q_offset=batch_size * verify_width,
                B=batch_size,
                next_n=verify_width,
            )

        valid_logits = run(valid_page_table)
        poisoned_logits = run(poisoned_page_table)
        torch.cuda.synchronize()

        for query, live_len in enumerate(context_lens[0].cpu().tolist()):
            valid_live = valid_logits[query, :live_len]
            poisoned_live = poisoned_logits[query, :live_len]
            self.assertTrue(torch.isfinite(poisoned_live).all().item())
            torch.testing.assert_close(
                poisoned_live,
                valid_live,
                rtol=0,
                atol=0,
            )

    def test_production_activation_gate_is_fail_closed(self):
        from sglang.srt.layers.attention import deepseek_v4_backend

        backend = object.__new__(deepseek_v4_backend.DeepseekV4AttnBackend)

        def gate(
            *,
            cuda=True,
            sm100=True,
            xpu=False,
            fp4=False,
            tilelang=False,
            aiter=False,
            torch_fallback=False,
        ):
            backend.enable_deepseek_v4_fp4_indexer = fp4
            with (
                mock.patch.object(deepseek_v4_backend, "_is_cuda", cuda),
                mock.patch.object(deepseek_v4_backend, "_is_sm100", sm100),
                mock.patch.object(deepseek_v4_backend, "_is_xpu", xpu),
                mock.patch.object(
                    deepseek_v4_backend.envs.SGLANG_OPT_USE_TILELANG_INDEXER,
                    "get",
                    return_value=tilelang,
                ),
                mock.patch.object(
                    deepseek_v4_backend.envs.SGLANG_OPT_USE_AITER_INDEXER,
                    "get",
                    return_value=aiter,
                ),
                mock.patch.object(
                    deepseek_v4_backend.envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH,
                    "get",
                    return_value=torch_fallback,
                ),
            ):
                return backend._can_use_live_prefix_target_verify_metadata()

        self.assertTrue(gate())
        for override in (
            {"cuda": False},
            {"sm100": False},
            {"xpu": True},
            {"fp4": True},
            {"tilelang": True},
            {"aiter": True},
            {"torch_fallback": True},
        ):
            with self.subTest(override=override):
                self.assertFalse(gate(**override))


if __name__ == "__main__":
    unittest.main()
