"""Correctness tests for DSV4 target-verify live-prefix metadata.

The target-verify CUDA graph allocates metadata at its static capture capacity,
but only the prefix selected by the current device-side sequence length is
live. These tests pin the scalar formulas and the live page-table/C128
prefixes at a fixed 1M-token capture capacity, including replay from a long
batch to an empty one and back without inspecting undefined tails. They also
check the production activation gate and that DeepGEMM/FlashMLA ignore
poisoned capture-capacity metadata outside the live prefix.
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
            # The suffix is undefined capture-capacity storage, not current
            # metadata, so deliberately do not inspect it.

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
            # As above, the suffix is undefined capture-capacity storage.

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

    def test_padded_metadata_rows_mask_missing_cache_write_locations(self):
        # CP-v2 pads causal metadata rows without padding the cache-write
        # locations. Exercise that contract with both the retained full-tail
        # path and the target-verify live-prefix grid.
        seq_lens_values = (0, 128, 256)
        seq_lens = torch.tensor(seq_lens_values, dtype=torch.int32, device=self.device)
        positions = seq_lens - 1
        page_table = (
            torch.arange(
                len(seq_lens_values) * NUM_PAGES,
                dtype=torch.int32,
                device=self.device,
            ).view(len(seq_lens_values), NUM_PAGES)
            + 100
        )
        raw_out_loc = torch.tensor([512, 1024], dtype=torch.int64, device=self.device)

        for live_prefix_only in (False, True):
            with self.subTest(live_prefix_only=live_prefix_only):
                outputs = init_compression_metadata(
                    seq_lens,
                    positions,
                    raw_out_loc,
                    page_table,
                    PAGE_SIZE,
                    compute_page_indices=True,
                    live_prefix_only=live_prefix_only,
                )
                torch.cuda.synchronize()

                self.assertEqual(outputs[0].shape, raw_out_loc.shape)
                self.assertEqual(outputs[4].shape, raw_out_loc.shape)
                self._assert_compression_metadata(
                    seq_lens_values,
                    raw_out_loc,
                    page_table,
                    outputs,
                )

    def test_torch_and_triton_match_page_table_contract(self):
        max_seq_len = 2048
        seq_lens_values = (-1, 0, 1, 255, 256, 257, 1025, 2048, 4096)
        num_q = len(seq_lens_values)
        req_to_token = torch.arange(
            num_q * max_seq_len,
            dtype=torch.int32,
            device=self.device,
        ).view(num_q, max_seq_len)
        req_pool_indices = torch.arange(
            num_q - 1, -1, -1, dtype=torch.int64, device=self.device
        )
        seq_lens = torch.tensor(seq_lens_values, dtype=torch.int32, device=self.device)

        for live_prefix_only in (False, True):
            with self.subTest(live_prefix_only=live_prefix_only):
                kwargs = dict(
                    req_to_token=req_to_token,
                    req_pool_indices_repeated=req_pool_indices,
                    seq_lens_casual=seq_lens,
                    max_seq_len=max_seq_len,
                    page_size=PAGE_SIZE,
                    swa_window=SWA_WINDOW,
                    live_prefix_only=live_prefix_only,
                )
                reference = BuildPageTablePositions.torch(**kwargs)
                actual = BuildPageTablePositions.triton(**kwargs)
                torch.cuda.synchronize()

                torch.testing.assert_close(
                    actual.seq_lens_casual, reference.seq_lens_casual
                )
                torch.testing.assert_close(
                    actual.positions_casual, reference.positions_casual
                )
                torch.testing.assert_close(
                    actual.swa_topk_lengths, reference.swa_topk_lengths
                )
                if not live_prefix_only:
                    torch.testing.assert_close(actual.page_table, reference.page_table)
                    continue

                for row, seq_len in enumerate(seq_lens_values):
                    live_pages = min(
                        (max(seq_len, 0) + PAGE_SIZE - 1) // PAGE_SIZE,
                        reference.page_table.shape[1],
                    )
                    torch.testing.assert_close(
                        actual.page_table[row, :live_pages],
                        reference.page_table[row, :live_pages],
                    )
                    self.assertTrue(
                        torch.all(reference.page_table[row, live_pages:] == -1).item()
                    )

    def test_pad_last_dim_avoids_already_aligned_copy(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import _pad_last_dim

        aligned = torch.arange(2 * 64, dtype=torch.int32, device=self.device).view(
            2, 64
        )
        result = _pad_last_dim(aligned)
        self.assertIs(result, aligned)
        self.assertEqual(result.data_ptr(), aligned.data_ptr())
        self.assertIsNone(_pad_last_dim(None))

        unaligned = torch.arange(2 * 65, dtype=torch.int32, device=self.device).view(
            2, 65
        )
        padded = _pad_last_dim(unaligned)
        self.assertEqual(padded.shape, (2, 128))
        torch.testing.assert_close(padded[:, :65], unaligned)
        self.assertTrue(torch.all(padded[:, 65:] == -1).item())

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

    def test_cuda_graph_replay_refreshes_short_prefix_for_new_request(self):
        long_seq_len = MAX_SEQ_LEN - 1
        short_seq_len = 129
        req_to_token = torch.full(
            (2, MAX_SEQ_LEN),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        page_offsets = torch.arange(NUM_PAGES, dtype=torch.int32, device=self.device)
        physical_pages = torch.stack(
            (100 + page_offsets, 100_000 + page_offsets),
            dim=0,
        )
        req_to_token[:, ::PAGE_SIZE] = physical_pages * PAGE_SIZE + 17

        req_pool_indices = torch.tensor([0], dtype=torch.int64, device=self.device)
        seq_lens = torch.tensor([long_seq_len], dtype=torch.int32, device=self.device)
        raw_out_loc = torch.tensor([512], dtype=torch.int64, device=self.device)

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
        long_a_first_page = int(page_metadata.page_table[0, 0])
        c128_page_indices = compression_outputs[-1]
        assert c128_page_indices is not None
        long_a_first_c128 = int(c128_page_indices[0, 0])

        # Keep the captured tensors and output buffers, but switch both the
        # request identity and its sequence length. The single live page/C128
        # entry must come from request B; retaining request A's prefix would
        # still look structurally valid, so distinct mappings make it visible.
        req_pool_indices.fill_(1)
        seq_lens.fill_(short_seq_len)
        graph.replay()
        torch.cuda.synchronize()
        self._assert_page_metadata(
            (short_seq_len,), req_to_token, req_pool_indices, page_metadata
        )
        self._assert_compression_metadata(
            (short_seq_len,),
            raw_out_loc,
            page_metadata.page_table,
            compression_outputs,
        )
        short_b_first_page = int(page_metadata.page_table[0, 0])
        short_b_first_c128 = int(c128_page_indices[0, 0])
        self.assertNotEqual(short_b_first_page, long_a_first_page)
        self.assertNotEqual(short_b_first_c128, long_a_first_c128)

        req_pool_indices.fill_(0)
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
        self.assertEqual(int(page_metadata.page_table[0, 0]), long_a_first_page)
        self.assertEqual(int(c128_page_indices[0, 0]), long_a_first_c128)

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

    def test_flashmla_ignores_poisoned_c128_suffix(self):
        """Native FlashMLA must bound C128 reads by the live top-k length."""
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DSV4RawVerifyMetadata,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.test.kits.attention_unittest.attention_methods.dsv4_attention import (
            DSV4_PAGE_SIZE,
            DSV4AttentionCase,
            _populate_extra_kv_cache,
            _populate_swa_kv_cache,
            build_dsv4_attention_fixture,
        )
        from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
            _make_eagle_verify_input,
            _prepare_target_verify_batch,
        )

        num_extra_entries = 32
        case = DSV4AttentionCase(
            name="dsv4_target_verify_c128_live_prefix_consumer",
            backend="dsv4",
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(128,),
            extend_lens=(4,),
            compress_ratio=128,
        )
        fixture = build_dsv4_attention_fixture(self, case, device=str(self.device))
        max_context_len = fixture.runner.req_to_token_pool.req_to_token.shape[1]
        self.assertEqual(fixture.backend.max_context_len, max_context_len)
        _populate_swa_kv_cache(
            fixture,
            max_context_len=max_context_len,
            device=str(self.device),
        )
        _populate_extra_kv_cache(
            fixture,
            layer_id=0,
            num_entries=num_extra_entries,
        )
        _prepare_target_verify_batch(fixture.forward_batch, case, str(self.device))
        fixture.forward_batch.spec_info = _make_eagle_verify_input(
            case,
            fixture.forward_batch,
            topk=1,
            device=str(self.device),
        )
        q_input, _ = fixture.actual_module.project(fixture.input_hidden)

        with torch.no_grad(), forward_context(
            ForwardContext(attn_backend=fixture.backend)
        ):
            eager_raw_metadata = fixture.backend._build_forward_metadata(
                fixture.forward_batch
            )
            self.assertIsInstance(eager_raw_metadata, DSV4RawVerifyMetadata)
            self.assertFalse(eager_raw_metadata.live_prefix_only)

            fixture.backend.init_cuda_graph_state(
                max_bs=case.batch_size,
                max_num_tokens=case.num_input_tokens,
            )
            fixture.backend.init_forward_metadata_out_graph(
                fixture.forward_batch, in_capture=True
            )
            raw_metadata = fixture.backend.forward_metadata
            self.assertIsInstance(raw_metadata, DSV4RawVerifyMetadata)
            self.assertTrue(raw_metadata.live_prefix_only)
            fixture.backend.init_forward_metadata_in_graph(fixture.forward_batch)
            core_metadata = fixture.backend.forward_metadata.core_attn_metadata
            c128_indices = core_metadata.c128_page_indices
            c128_lengths = core_metadata.c128_topk_lengths_clamp1
            self.assertIsNotNone(c128_indices)
            self.assertIsNotNone(c128_lengths)
            assert c128_indices is not None
            assert c128_lengths is not None

            self.assertEqual(c128_indices.shape[0], case.num_input_tokens)
            self.assertGreater(c128_indices.shape[1], 1)
            self.assertTrue(torch.all(c128_lengths >= 1).item())
            self.assertTrue(
                torch.all(c128_lengths < c128_indices.shape[1]).item(),
                "the case must leave a non-live C128 capacity suffix",
            )
            live_mask = torch.arange(
                c128_indices.shape[1], device=c128_indices.device
            ).unsqueeze(0).expand_as(c128_indices) < c128_lengths.unsqueeze(1)

            live_values = c128_indices[live_mask].clone()
            c128_indices.masked_fill_(~live_mask, 0)
            baseline_output = fixture.backend.forward(
                q=q_input,
                k=q_input,
                v=q_input,
                layer=fixture.actual_module.attn,
                forward_batch=fixture.forward_batch,
                compress_ratio=case.compress_ratio,
                save_kv_cache=False,
                attn_sink=fixture.actual_module.attn_sink,
            )

            # The pinned native FlashMLA implementation may vector-load an
            # aligned index tile, but it must apply c128_lengths before
            # generating any KV address. An out-of-range suffix catches a
            # dereference-before-mask regression as well as accidental
            # attention to the undefined capacity tail.
            c128_indices.masked_fill_(~live_mask, torch.iinfo(torch.int32).max)
            torch.testing.assert_close(
                c128_indices[live_mask],
                live_values,
                rtol=0,
                atol=0,
            )
            self.assertTrue(
                torch.all(
                    c128_indices[~live_mask] == torch.iinfo(torch.int32).max
                ).item()
            )
            poisoned_output = fixture.backend.forward(
                q=q_input,
                k=q_input,
                v=q_input,
                layer=fixture.actual_module.attn,
                forward_batch=fixture.forward_batch,
                compress_ratio=case.compress_ratio,
                save_kv_cache=False,
                attn_sink=fixture.actual_module.attn_sink,
            )

        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(poisoned_output).all().item())
        torch.testing.assert_close(
            poisoned_output,
            baseline_output,
            rtol=0,
            atol=0,
        )

    def test_production_activation_gate_is_fail_closed(self):
        from sglang.srt.layers.attention import deepseek_v4_backend

        backend = object.__new__(deepseek_v4_backend.DeepseekV4AttnBackend)

        def gate(
            *,
            sm100=True,
            xpu=False,
            fp4=False,
            tilelang=False,
            aiter=False,
            torch_fallback=False,
            topk_torch=False,
            topk_v2=False,
            hisparse=False,
            cuda_graph=True,
        ):
            backend.enable_deepseek_v4_fp4_indexer = fp4
            backend.hisparse_coordinator = object() if hisparse else None
            with (
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
                mock.patch.object(
                    deepseek_v4_backend.envs.SGLANG_TOPK_TRANSFORM_512_TORCH,
                    "get",
                    return_value=topk_torch,
                ),
                mock.patch.object(
                    deepseek_v4_backend.envs.SGLANG_OPT_USE_TOPK_V2,
                    "get",
                    return_value=topk_v2,
                ),
            ):
                # __init__ resolves the frozen terms once; mirror that here so
                # the mocks are in scope for the compute, not just the gate.
                backend.live_prefix_metadata_supported = (
                    backend._compute_live_prefix_metadata_supported()
                )
                return backend._can_use_live_prefix_target_verify_metadata(
                    use_prefill_cuda_graph=cuda_graph
                )

        self.assertTrue(gate())
        # Regression guard, not a mirror: top-k v2 is the default page-table
        # consumer and bounds its reads by the live C4 length, so it must NOT
        # gate the optimization off. The gate deliberately does not read
        # SGLANG_OPT_USE_TOPK_V2; adding it to the conjunction turns this red.
        self.assertTrue(gate(topk_v2=True))

        for override in (
            {"cuda_graph": False},
            {"sm100": False},
            {"xpu": True},
            {"fp4": True},
            {"tilelang": True},
            {"aiter": True},
            {"torch_fallback": True},
            {"topk_torch": True},
            {"hisparse": True},
        ):
            with self.subTest(override=override):
                self.assertFalse(gate(**override))


if __name__ == "__main__":
    unittest.main()
