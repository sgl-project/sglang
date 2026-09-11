"""DSV4 attention correctness — SWA + C4/C128 coverage.

Covers eager EXTEND/DECODE plus CUDA-graph-style capture/replay for the
SWA-only (compress_ratio=0) path of `DeepseekV4AttnBackend` through flash_mla
with the production packed FP8-nope/BF16-rope SWA cache, plus math-faithful
EAGER coverage for the C4 (compress_ratio=4) and C128 (compress_ratio=128)
paths. The C4/C128 cases bypass the production `Compressor`/`C4Indexer`
modules (writing the extra K cache directly via the pack+set path and
seeding `c4_sparse_page_indices` for the un-run indexer) but compare the
flash_mla `extra_k_cache` integration against an independent PyTorch SWA +
extra-K softmax reference. Compressor math correctness (i.e. verifying the
gate+norm+rotate compression itself) is a deferred follow-up.
"""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

_FLASH_MLA_AVAILABLE = (
    importlib.util.find_spec("sgl_kernel") is not None
    and importlib.util.find_spec("sgl_kernel.flash_mla") is not None
)

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dsv4_attention import (  # noqa: E402
    DSV4_PAGE_SIZE,
    DSV4AttentionCase,
    make_dsv4_cases,
    run_dsv4_attention_case,
    run_dsv4_compress_attention_case,
    run_dsv4_draft_extend_attention_case,
    run_dsv4_target_verify_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (  # noqa: E402
    run_dsv4_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_runner import (  # noqa: E402
    run_dsv4_eagle_draft_cuda_graph_runner_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (  # noqa: E402
    run_dsv4_eagle_verify_cuda_graph_case,
)

register_cuda_ci(est_time=25, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
@unittest.skipIf(not _FLASH_MLA_AVAILABLE, "flash_mla is required for DSV4 SWA")
class TestDSV4AttentionBackendCorrectness(CustomTestCase):
    CASES = make_dsv4_cases("dsv4")
    CUDA_GRAPH_DECODE_CASES = (
        DSV4AttentionCase(
            name="runner_cuda_graph_dsv4_decode_within_window",
            backend="dsv4",
            forward_mode=ForwardMode.DECODE,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(64,),
        ),
        DSV4AttentionCase(
            name="runner_cuda_graph_dsv4_decode_multi_request",
            backend="dsv4",
            forward_mode=ForwardMode.DECODE,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(32, 96),
        ),
        DSV4AttentionCase(
            name="runner_cuda_graph_dsv4_c4_decode",
            backend="dsv4",
            forward_mode=ForwardMode.DECODE,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(64,),
            compress_ratio=4,
        ),
        DSV4AttentionCase(
            name="runner_cuda_graph_dsv4_c128_decode",
            backend="dsv4",
            forward_mode=ForwardMode.DECODE,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(128,),
            compress_ratio=128,
        ),
    )
    # SWA + C4 / SWA + C128 cases. Each pre-populates the extra K cache directly
    # via `set_extra_key_buffer`, lets `init_forward_metadata` populate the
    # compression metadata (and seeds `c4_sparse_page_indices` manually for C4
    # since the un-run indexer leaves it at -1), then compares the flash_mla
    # output to an independent PyTorch SWA + extra-K softmax reference.
    COMPRESS_CASES = (
        DSV4AttentionCase(
            name="dsv4_c4_extend",
            backend="dsv4",
            forward_mode=ForwardMode.EXTEND,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(64,),
            extend_lens=(16,),
            compress_ratio=4,
        ),
        DSV4AttentionCase(
            name="dsv4_c4_decode",
            backend="dsv4",
            forward_mode=ForwardMode.DECODE,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(64,),
            compress_ratio=4,
        ),
        DSV4AttentionCase(
            name="dsv4_c128_extend",
            backend="dsv4",
            forward_mode=ForwardMode.EXTEND,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(128,),
            extend_lens=(16,),
            compress_ratio=128,
        ),
        # DeepSeek V4.1 ratio 1 / 2: latents in the c1 / c2 FlashMLA-layout pools,
        # attended through the same extra-cache path as C4.
        DSV4AttentionCase(
            name="dsv4_c1_extend",
            backend="dsv4",
            forward_mode=ForwardMode.EXTEND,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(64,),
            extend_lens=(16,),
            compress_ratio=1,
        ),
        DSV4AttentionCase(
            name="dsv4_c1_decode",
            backend="dsv4",
            forward_mode=ForwardMode.DECODE,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(64, 200),
            compress_ratio=1,
        ),
        DSV4AttentionCase(
            name="dsv4_c2_extend",
            backend="dsv4",
            forward_mode=ForwardMode.EXTEND,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            # Odd lengths: the ratio-2 causal count (pos + 1) // 2 rounds down.
            prefix_lens=(33,),
            extend_lens=(7,),
            compress_ratio=2,
        ),
        DSV4AttentionCase(
            name="dsv4_c2_decode",
            backend="dsv4",
            forward_mode=ForwardMode.DECODE,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(65,),
            compress_ratio=2,
        ),
        DSV4AttentionCase(
            name="dsv4_c128_decode",
            backend="dsv4",
            forward_mode=ForwardMode.DECODE,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(128,),
            compress_ratio=128,
        ),
    )

    def test_swa_only_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsv4_attention_case(self, case)

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_DECODE_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsv4_cuda_graph_decode_case(self, case)

    # EAGLE target_verify (chain only — DSV4 asserts topk <= 1). One case per
    # compress_ratio so SWA, SWA+C4, and SWA+C128 all run through the
    # per-draft-token causal-within-SWA + extra-K reference.
    TARGET_VERIFY_CASES = (
        DSV4AttentionCase(
            name="dsv4_swa_eagle_verify_chain",
            backend="dsv4",
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(64, 96),
            extend_lens=(3, 3),
        ),
        DSV4AttentionCase(
            name="dsv4_c4_eagle_verify_chain",
            backend="dsv4",
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(64, 96),
            extend_lens=(3, 3),
            compress_ratio=4,
        ),
        DSV4AttentionCase(
            name="dsv4_c128_eagle_verify_chain",
            backend="dsv4",
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(128, 160),
            extend_lens=(3, 3),
            compress_ratio=128,
        ),
    )

    def test_compress_attention_cases(self):
        # Pinned to the dense extend path; the sparse prefill path is covered
        # by test_compress_attention_cases_sparse_prefill below.
        for case in self.COMPRESS_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                compress_ratio=case.compress_ratio,
            ):
                run_dsv4_compress_attention_case(self, case, sparse_prefill=False)

    def test_compress_attention_cases_sparse_prefill(self):
        # `_forward_prefill_sparse` extend path; decode never reaches it.
        for case in self.COMPRESS_CASES:
            if not case.forward_mode.is_extend_without_speculative():
                continue
            with self.subTest(
                case=case.name,
                backend=case.backend,
                compress_ratio=case.compress_ratio,
            ):
                run_dsv4_compress_attention_case(self, case, sparse_prefill=True)

    def test_eagle_target_verify_chain_cases(self):
        for case in self.TARGET_VERIFY_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                compress_ratio=case.compress_ratio,
            ):
                run_dsv4_target_verify_attention_case(self, case, topk=1)

    # CUDA-graph capture/replay for EAGLE target_verify across SWA + C4 + C128.
    EAGLE_VERIFY_CUDA_GRAPH_CASES = (
        DSV4AttentionCase(
            name="runner_cuda_graph_dsv4_swa_eagle_verify_chain",
            backend="dsv4",
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(64, 96),
            extend_lens=(3, 3),
        ),
        DSV4AttentionCase(
            name="runner_cuda_graph_dsv4_c4_eagle_verify_chain",
            backend="dsv4",
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(64, 96),
            extend_lens=(3, 3),
            compress_ratio=4,
        ),
        DSV4AttentionCase(
            name="runner_cuda_graph_dsv4_c128_eagle_verify_chain",
            backend="dsv4",
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(128, 160),
            extend_lens=(3, 3),
            compress_ratio=128,
        ),
    )

    def test_runner_mode_eagle_verify_cuda_graph_cases(self):
        for case in self.EAGLE_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                compress_ratio=case.compress_ratio,
            ):
                run_dsv4_eagle_verify_cuda_graph_case(
                    self, case, topk=1, force_gpu_only_seq_lens=True
                )

    def test_eagle_draft_extend_without_cpu_seq_lens(self):
        case = DSV4AttentionCase(
            name="dsv4_swa_eagle_draft_extend_gpu_only_seq_lens",
            backend="dsv4",
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(64, 96),
            extend_lens=(4, 4),
        )
        run_dsv4_draft_extend_attention_case(
            self,
            case,
            force_gpu_only_seq_lens=True,
        )

    # Production EAGLE draft graph runner (chain only, SWA only). The runner
    # routes through `DeepseekV4MultiStepBackend` (one `DeepseekV4AttnBackend`
    # per draft step), captures a fixed batch, and replays distinct request
    # metadata. The fixture's `ProjectedDSV4Attention.forward` writes K via
    # `set_swa_key_buffer_radix` exactly like the production model.
    PRODUCTION_EAGLE_DRAFT_RUNNER_CASES = (
        DSV4AttentionCase(
            name="runner_production_eagle_draft_dsv4_swa_chain",
            backend="dsv4",
            forward_mode=ForwardMode.DECODE,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(32, 64),
        ),
    )

    def test_runner_mode_production_eagle_draft_cuda_graph_runner_cases(self):
        for case in self.PRODUCTION_EAGLE_DRAFT_RUNNER_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsv4_eagle_draft_cuda_graph_runner_case(
                    self,
                    case,
                    force_gpu_only_seq_lens=True,
                )


class TestDSV4BreakableCudaGraphMetadataContract(CustomTestCase):
    """CPU-only checks for the DSV4 BCG metadata replay contract."""

    @staticmethod
    def _make_sparse_prefill_cache(max_seq_len):
        from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
            SparsePrefillChunkCache,
        )

        int32 = dict(dtype=torch.int32)
        return SparsePrefillChunkCache(
            num_reqs=2,
            num_qo_tokens=2,
            max_seq_len=max_seq_len,
            swa_window_size=128,
            swa_page_size=128,
            seq_lens=torch.tensor([max_seq_len, max_seq_len], **int32),
            query_start_loc=torch.tensor([0, 1, 2], **int32),
            swa_token_ids=torch.empty(0, **int32),
            swa_first_pos=torch.zeros(2, **int32),
            swa_gather_lens=torch.zeros(2, **int32),
            swa_offsets=torch.zeros(3, **int32),
        )

    def _make_core_metadata(self, base: int):
        from sglang.srt.layers.attention.deepseek_v4_backend import DSV4AttnMetadata

        metadata = DSV4AttnMetadata(
            page_size=256,
            page_table=torch.tensor(
                [[base + 1, base + 2], [base + 3, base + 4]], dtype=torch.int32
            ),
            raw_out_loc=torch.tensor([base + 5, base + 6], dtype=torch.int32),
            cuda_int32_kwargs={"dtype": torch.int32},
            seq_lens_casual=torch.tensor([base + 7, base + 8], dtype=torch.int32),
            positions_casual=torch.tensor([base + 9, base + 10], dtype=torch.int32),
            swa_page_indices=torch.tensor(
                [[base + 11, base + 12], [base + 13, base + 14]], dtype=torch.int32
            ),
            swa_topk_lengths=torch.tensor([base + 15, base + 16], dtype=torch.int32),
            index_topk=128,
        )
        metadata.c4_out_loc = torch.tensor([base + 17, base + 18], dtype=torch.int32)
        metadata.c128_out_loc = torch.tensor([base + 19, base + 20], dtype=torch.int32)
        metadata.c4_topk_lengths_raw = torch.tensor(
            [base + 21, base + 22], dtype=torch.int32
        )
        metadata.c4_topk_lengths_clamp1 = torch.tensor(
            [base + 23, base + 24], dtype=torch.int32
        )
        metadata.c4_sparse_topk_lengths = torch.tensor(
            [base + 25, base + 26], dtype=torch.int32
        )
        metadata.c4_sparse_page_indices = torch.tensor(
            [[base + 27, base + 28], [base + 29, base + 30]], dtype=torch.int32
        )
        metadata.c4_sparse_raw_indices = torch.tensor(
            [[base + 31, base + 32], [base + 33, base + 34]], dtype=torch.int32
        )
        metadata.c128_page_indices = torch.tensor(
            [[base + 35, base + 36], [base + 37, base + 38]], dtype=torch.int32
        )
        metadata.c128_topk_lengths_clamp1 = torch.tensor(
            [base + 39, base + 40], dtype=torch.int32
        )
        metadata.c0_flashmla_metadata = object()
        metadata.c4_flashmla_metadata = object()
        metadata.c128_flashmla_metadata = object()
        return metadata

    def test_bcg_is_explicit_and_dsv4_backend_opt_in_only(self):
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )
        from sglang.srt.server_args import ServerArgs

        # cg-refactor folded the legacy enable_breakable_cuda_graph flag
        # into cuda_graph_config. Verify the per-phase backend selectors
        # default to None (i.e. nothing opted into BREAKABLE without an
        # explicit CLI flag).
        sa = ServerArgs(model_path="dummy")
        self.assertNotEqual(sa.cuda_graph_backend_decode, "breakable")
        self.assertNotEqual(sa.cuda_graph_backend_prefill, "breakable")
        self.assertFalse(
            AttentionBackend.use_captured_forward_metadata_for_breakable_cuda_graph
        )
        self.assertTrue(
            DeepseekV4AttnBackend.use_captured_forward_metadata_for_breakable_cuda_graph
        )

    def test_prefill_snapshot_declares_pre_replay_boundary(self):
        from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
            DSV4Metadata,
        )

        backend = object.__new__(DeepseekV4AttnBackend)
        backend.forward_metadata = DSV4Metadata(
            self._make_core_metadata(0), indexer_metadata=None
        )
        self.assertIs(
            backend.shared_read_ends(ForwardMode.EXTEND),
            SharedReadEnds.UNKNOWN,
        )

        backend.forward_metadata.prefill_shared_reads_snapshotted = True
        self.assertIs(
            backend.shared_read_ends(ForwardMode.EXTEND),
            SharedReadEnds.PRE_REPLAY,
        )

    def test_snapshot_builds_cache_only_for_sparse_prefill(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _LARGE_INDEXER_QUERY_THRESHOLD,
            DeepseekV4AttnBackend,
            DSV4Metadata,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        cache = object()
        for num_qo_tokens, builds in (
            (_LARGE_INDEXER_QUERY_THRESHOLD, False),
            (_LARGE_INDEXER_QUERY_THRESHOLD + 1, True),
        ):
            with self.subTest(num_qo_tokens=num_qo_tokens):
                backend = object.__new__(DeepseekV4AttnBackend)
                backend.model_runner = SimpleNamespace(
                    spec_algorithm=SpeculativeAlgorithm.DFLASH
                )
                backend.forward_metadata = DSV4Metadata(
                    self._make_core_metadata(0), indexer_metadata=None
                )
                backend._build_sparse_prefill_chunk_cache = mock.Mock(
                    return_value=cache
                )

                with (
                    envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(True),
                    envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(False),
                    mock.patch(
                        "sglang.srt.layers.attention.deepseek_v4_backend.get_platform",
                        return_value=SimpleNamespace(is_sm120=False),
                    ),
                ):
                    backend.prepare_prefill_shared_read_snapshot(
                        batch, num_qo_tokens=num_qo_tokens
                    )

                metadata = backend.forward_metadata
                if builds:
                    backend._build_sparse_prefill_chunk_cache.assert_called_once_with(
                        batch, num_qo_tokens=num_qo_tokens
                    )
                    self.assertIs(metadata.sparse_prefill_cache, cache)
                else:
                    backend._build_sparse_prefill_chunk_cache.assert_not_called()
                    self.assertIsNone(metadata.sparse_prefill_cache)
                # Dense declares the boundary too; it reads only the metadata
                # that init_forward_metadata already snapshotted.
                self.assertTrue(metadata.prefill_shared_reads_snapshotted)

    def test_sparse_prefill_snapshot_marks_success_only_after_build(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
            DSV4Metadata,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        backend = object.__new__(DeepseekV4AttnBackend)
        backend.model_runner = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.DFLASH
        )
        backend.forward_metadata = DSV4Metadata(
            self._make_core_metadata(0), indexer_metadata=None
        )
        backend.forward_metadata.prefill_shared_reads_snapshotted = True
        backend._build_sparse_prefill_chunk_cache = mock.Mock(
            side_effect=RuntimeError("snapshot failed")
        )
        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)

        with (
            envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(True),
            envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(True),
            mock.patch(
                "sglang.srt.layers.attention.deepseek_v4_backend.get_platform",
                return_value=SimpleNamespace(is_sm120=False),
            ),
            self.assertRaisesRegex(RuntimeError, "snapshot failed"),
        ):
            backend.prepare_prefill_shared_read_snapshot(batch, num_qo_tokens=12288)

        self.assertFalse(backend.forward_metadata.prefill_shared_reads_snapshotted)

    def test_refresh_replay_metadata_preserves_captured_tensor_storage(self):
        capture_metadata = self._make_core_metadata(0)
        replay_metadata = self._make_core_metadata(1000)

        tensor_copy_fields = [
            "raw_out_loc",
            "seq_lens_casual",
            "positions_casual",
            "c4_out_loc",
            "c128_out_loc",
            "c4_topk_lengths_raw",
            "c4_topk_lengths_clamp1",
            "c4_sparse_topk_lengths",
        ]
        reference_assign_fields = [
            "page_table",
            "swa_page_indices",
            "swa_topk_lengths",
            "c128_page_indices",
            "c128_topk_lengths_clamp1",
            "c0_flashmla_metadata",
            "c4_flashmla_metadata",
            "c128_flashmla_metadata",
        ]

        captured_tensor_objects = {
            field: getattr(capture_metadata, field) for field in tensor_copy_fields
        }
        captured_sparse_pages = capture_metadata.c4_sparse_page_indices
        captured_sparse_pages_value = captured_sparse_pages.clone()

        capture_metadata.refresh_for_breakable_cuda_graph_replay_(replay_metadata)

        for field in tensor_copy_fields:
            self.assertIs(
                getattr(capture_metadata, field), captured_tensor_objects[field]
            )
            self.assertTrue(
                torch.equal(
                    getattr(capture_metadata, field), getattr(replay_metadata, field)
                ),
                f"{field} should be copied from the replay metadata",
            )

        for field in reference_assign_fields:
            self.assertIs(
                getattr(capture_metadata, field),
                getattr(replay_metadata, field),
                f"{field} should use the replay metadata reference",
            )

        self.assertIs(capture_metadata.c4_sparse_page_indices, captured_sparse_pages)
        self.assertTrue(
            torch.equal(
                capture_metadata.c4_sparse_page_indices, captured_sparse_pages_value
            )
        )

    def test_backend_replay_keeps_captured_metadata_active(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
            DSV4Metadata,
        )

        capture_metadata = DSV4Metadata(
            self._make_core_metadata(0), indexer_metadata=None
        )
        capture_metadata.sparse_prefill_cache = object()
        capture_metadata.prefill_shared_reads_snapshotted = True
        replay_metadata = DSV4Metadata(
            self._make_core_metadata(1000), indexer_metadata=None
        )
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.MAX_SEQ_LEN_FOR_CAPTURE = 4096
        calls = []

        def fake_build_forward_metadata(
            forward_batch, *, max_seq_len_override, use_prefill_cuda_graph
        ):
            calls.append((forward_batch, max_seq_len_override, use_prefill_cuda_graph))
            return replay_metadata

        backend._build_forward_metadata = fake_build_forward_metadata
        forward_batch = SimpleNamespace(name="live")
        static_forward_batch = SimpleNamespace(name="static")

        backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
            capture_metadata,
            forward_batch,
            static_forward_batch=static_forward_batch,
        )

        self.assertIs(calls[0][0], static_forward_batch)
        self.assertEqual(calls[0][1], backend.MAX_SEQ_LEN_FOR_CAPTURE)
        self.assertTrue(calls[0][2])
        self.assertIs(backend.forward_metadata, capture_metadata)
        self.assertIsNone(capture_metadata.sparse_prefill_cache)
        self.assertFalse(capture_metadata.prefill_shared_reads_snapshotted)
        self.assertTrue(
            torch.equal(
                capture_metadata.core_attn_metadata.seq_lens_casual,
                replay_metadata.core_attn_metadata.seq_lens_casual,
            )
        )

    def test_sparse_prefill_workspace_reuses_and_grows(self):
        from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
            SparsePrefillWorkspace,
        )

        workspace = SparsePrefillWorkspace(torch.device("cpu"))
        first = workspace.get(3)
        reused = workspace.get(2)
        grown = workspace.get(7)

        self.assertEqual(first.shape, (3, 1, 512))
        self.assertEqual(reused.data_ptr(), first.data_ptr())
        self.assertEqual(grown.shape, (7, 1, 512))
        self.assertNotEqual(grown.data_ptr(), first.data_ptr())
        self.assertEqual(workspace._buffer.data_ptr(), grown.data_ptr())

    def test_sparse_prefill_c4_uses_live_extent(self):
        page_table = torch.zeros((2, 4096), dtype=torch.int32)
        for max_seq_len in (3, 4, 255, 256, 259, 260):
            with self.subTest(max_seq_len=max_seq_len):
                cache = self._make_sparse_prefill_cache(max_seq_len)
                gather = cache.ensure_compressed(4, page_table, c_page_size=64)
                expected_extent = max(max_seq_len // 4, 1)
                self.assertEqual(gather.flat_token_ids.numel(), 2 * expected_extent)
                self.assertEqual(gather.compressed_base.tolist(), [0, expected_extent])

    def test_sparse_prefill_c128_uses_live_extent(self):
        from sglang.srt.layers.attention.dsv4 import sparse_prefill_utils

        page_indices = torch.full((2, 8192), -1, dtype=torch.int32)
        for max_seq_len in (127, 128, 255, 256):
            with self.subTest(max_seq_len=max_seq_len):
                cache = self._make_sparse_prefill_cache(max_seq_len)
                expected_extent = max(max_seq_len // 128, 1)
                combined = (
                    torch.empty((2, 256), dtype=torch.int32),
                    torch.empty(2, dtype=torch.int32),
                )
                with mock.patch.object(
                    sparse_prefill_utils,
                    "combine_topk_swa_indices",
                    return_value=combined,
                ) as combine:
                    cache.ensure_c128(page_indices)

                self.assertEqual(cache.c128_flat_token_ids.numel(), 2 * expected_extent)
                self.assertEqual(combine.call_args.kwargs["topk"], expected_extent)
                self.assertEqual(
                    combine.call_args.kwargs["topk_indices"].shape,
                    (2, expected_extent),
                )


class TestDSV41SM90CandidateSlots(CustomTestCase):
    def test_ragged_verify_metadata_preserves_low_ratio_row_mapping(self):
        from sglang.srt.layers.attention import deepseek_v4_backend as module

        backend = object.__new__(module.DeepseekV4AttnBackend)
        backend.speculative_num_draft_tokens = 6
        backend.needs_cpu_seq_lens = False
        backend.has_c4 = False
        backend.has_c128 = False
        backend.MAX_SEQ_LEN_FOR_CAPTURE = 384
        backend.req_to_token = torch.empty((3, 384), dtype=torch.int32)
        backend.token_to_kv_pool = SimpleNamespace()

        repeated = torch.tensor([2, 2, 0, 0, 0, 1], dtype=torch.int64)
        positions = torch.tensor([260, 261, 6, 7, 8, 129], dtype=torch.int32)
        core = SimpleNamespace(low_ratios=(), positions_casual=positions)
        backend._expand_prefill_casually_vectorized = mock.Mock(
            return_value=(positions + 1, repeated)
        )
        backend.make_core_attn_metadata = mock.Mock(return_value=core)

        raw = module.DSV4RawVerifyMetadata(
            req_pool_indices=torch.tensor([2, 0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([260, 6, 129], dtype=torch.int32),
            out_cache_loc=torch.arange(6, dtype=torch.int64),
            extend_seq_lens=torch.tensor([2, 3, 1], dtype=torch.int32),
            extend_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
            verify_lens=torch.tensor([2, 3, 1], dtype=torch.int32),
            total_verify_tokens=6,
        )

        metadata = backend.make_forward_metadata_from_raw_verify(raw)

        torch.testing.assert_close(metadata.low_ratio_req_indices, repeated)
        torch.testing.assert_close(
            metadata.low_ratio_pos_i64, positions.to(torch.int64)
        )

    def _run_candidate_case(
        self, forward_mode, req, length_steps, *, use_topk_v2=False
    ):
        from sglang.srt.layers.attention import deepseek_v4_backend as module

        width, ratio, topk = 384, 2, 3
        pages = torch.tensor([[7, 5, 1], [6, 4, 2], [9, 8, 3]])
        tokens = torch.arange(width * ratio)
        mapping = pages[:, tokens // 256] * 256 + tokens % 256
        page_indices = torch.full((len(req), topk), -1, dtype=torch.int32)
        raw_indices = torch.full_like(page_indices, -1)
        core = SimpleNamespace(
            sparse_page_indices=lambda _: page_indices,
            sparse_raw_indices=lambda _: raw_indices,
        )
        backend = object.__new__(module.DeepseekV4AttnBackend)
        backend.forward_metadata = module.DSV4Metadata(
            core_attn_metadata=core,
            indexer_metadata=None,
            c2_indexer_metadata=SimpleNamespace(
                max_c4_seq_len=width,
                use_topk_v2=use_topk_v2,
                c4_seq_lens=torch.full((len(req),), width, dtype=torch.int32),
                topk_metadata=torch.empty(0),
            ),
        )
        backend.req_to_token = mapping
        backend.candidate_masks = None
        table = torch.zeros((1, 64 * 68), dtype=torch.uint8)
        backend.token_to_kv_pool = SimpleNamespace(
            get_index_k_with_scale_buffer=lambda _: table
        )
        indexer = SimpleNamespace(
            queries=lambda q, _: q,
            head_weights=lambda x: x,
            candidate_topk_blocks=2,
            candidate_block_size=4,
            index_topk=topk,
        )
        layer = SimpleNamespace(
            indexer=indexer,
            freqs_cis=torch.zeros(width * ratio, 1),
            compress_ratio=ratio,
            layer_id=20,
        )
        forward_batch = SimpleNamespace(forward_mode=forward_mode)
        columns = torch.arange(width)

        def score(q, weights, slots, lens, table, page_size):
            # Distinct physical slots receive distinct scores for each query.
            return (slots.float() * q).masked_fill(
                torch.arange(slots.shape[1]) >= lens[:, None], -torch.inf
            )

        def score_mapping(
            q,
            weights,
            req_to_token,
            req_rows,
            lens,
            table,
            page_size,
            ratio,
            width,
            candidate_block_size=0,
        ):
            logical = torch.arange(width)
            slots = req_to_token[req_rows[:, None], logical * ratio] // ratio
            logits = score(q, weights, slots, lens, table, page_size)
            if not candidate_block_size:
                return logits
            block_scores = logits.unflatten(-1, (-1, candidate_block_size)).amax(dim=-1)
            last = (lens - 1) // candidate_block_size
            block_scores = block_scores.masked_fill(
                torch.arange(block_scores.shape[1]) == last[:, None], torch.inf
            )
            block_lens = (lens + candidate_block_size - 1) // candidate_block_size
            return logits, block_scores, block_lens

        def topk_v2(scores, lens, page_table, out, page_size, metadata):
            self.assertEqual(lens.dtype, torch.int32)
            out.fill_(-1)
            k = min(out.shape[1], scores.shape[1])
            out[:, :k] = scores.topk(k, dim=-1, sorted=False).indices.to(torch.int32)

        def publish_candidates(
            scores,
            lens,
            req_to_token,
            req_rows,
            *,
            topk_blocks,
            block_size,
            ratio,
            block_scores=None,
            block_lens=None,
        ):
            from sglang.srt.layers.attention.dsv4.indexer import (
                select_candidate_block_indices,
            )

            blocks, reachable = select_candidate_block_indices(
                scores, lens[:, None], topk_blocks, block_size
            )
            blocks = blocks.masked_fill(~reachable, width).sort(dim=-1).values
            positions = (
                blocks[:, :, None] * block_size + torch.arange(block_size)
            ).flatten(1)
            valid = positions < lens[:, None]
            slots = (
                req_to_token[req_rows[:, None], positions.clamp_max(width - 1) * ratio]
                // ratio
            ).masked_fill(~valid, 0)
            return (
                positions,
                slots,
                valid.sum(dim=-1).to(torch.int32),
                torch.empty(0),
            )

        def finalize_topk(
            selected,
            scores,
            score_lens,
            req_to_token,
            req_rows,
            page_out,
            raw_out,
            *,
            ratio,
            candidate_positions=None,
            candidate_slots=None,
        ):
            selected = selected.to(torch.int64)
            selected_scores = scores.gather(1, selected.clamp(0, scores.shape[1] - 1))
            valid = (
                (selected >= 0)
                & (selected < score_lens[:, None])
                & (selected_scores > -torch.inf)
            )
            selected = selected.masked_fill(~valid, scores.shape[1]).sort(1).values
            valid = selected < score_lens[:, None]
            if candidate_positions is None:
                logical = selected
                slots = (
                    req_to_token[
                        req_rows[:, None],
                        logical.clamp_max(scores.shape[1] - 1) * ratio,
                    ]
                    // ratio
                )
            else:
                logical = candidate_positions.gather(
                    1, selected.clamp_max(scores.shape[1] - 1)
                )
                slots = candidate_slots.gather(
                    1, selected.clamp_max(scores.shape[1] - 1)
                )
            page_out.fill_(-1)
            page_out[:, : selected.shape[1]] = slots.masked_fill(~valid, -1)
            if raw_out is not None:
                raw_out.fill_(-1)
                raw_out[:, : selected.shape[1]] = logical.masked_fill(~valid, -1)

        for lengths in length_steps:
            lens = torch.tensor(lengths)
            pos = (lens * ratio - 1).clamp_min(0)
            mapping.copy_(mapping.roll(1, dims=0))
            full_slots = mapping[req[:, None], columns * ratio] // ratio
            source_scores = score(
                torch.ones(len(req), 1), None, full_slots, lens, table, 64
            )
            candidates = module.select_candidate_blocks(
                source_scores, lens[:, None], 2, 4
            )
            with (
                mock.patch.object(
                    module, "fp4_index_logits_decode", side_effect=score
                ) as compact_logits,
                mock.patch.object(
                    module,
                    "fp4_index_logits_req_to_token",
                    side_effect=score_mapping,
                ) as full_logits,
                mock.patch.object(
                    module, "topk_transform_paged_v2", side_effect=topk_v2
                ) as full_topk,
                mock.patch(
                    "sglang.kernels.ops.attention.dsv4.candidate_blocks.candidate_slots",
                    side_effect=publish_candidates,
                ),
                mock.patch(
                    "sglang.kernels.ops.attention.dsv4.candidate_blocks.finalize_candidate_topk",
                    side_effect=finalize_topk,
                ),
            ):
                for i in range(5):
                    indexer.is_candidate_source = i == 0
                    indexer.uses_candidates = i > 0
                    layer.layer_id = 20 + i * 4
                    q = torch.full((len(req), 1), (-1) ** i * (i + 1))
                    backend._low_ratio_index_topk_sm90(
                        layer, q, q, req, pos, forward_batch
                    )
                    expected_scores = score(q, None, full_slots, lens, table, 64)
                    if i:
                        expected_scores.masked_fill_(~candidates, -torch.inf)
                    values, chosen = expected_scores.topk(topk, dim=1)
                    chosen = (
                        chosen.masked_fill(values == -torch.inf, width).sort(1).values
                    )
                    valid = chosen < width
                    expected_pages = full_slots.gather(
                        1, chosen.clamp_max(width - 1)
                    ).masked_fill(~valid, -1)
                    torch.testing.assert_close(
                        page_indices, expected_pages.to(torch.int32)
                    )
                    torch.testing.assert_close(
                        raw_indices, chosen.masked_fill(~valid, -1).to(torch.int32)
                    )
                    if i == 0:
                        published = backend.forward_metadata.sm90_candidates
                    else:
                        self.assertIs(
                            backend.forward_metadata.sm90_candidates, published
                        )
                        positions, slots, counts, _ = published
                        self.assertIs(compact_logits.call_args.args[2], slots)
                        torch.testing.assert_close(
                            positions < lens[:, None],
                            torch.arange(slots.shape[1]) < counts[:, None],
                        )
                        torch.testing.assert_close(
                            slots,
                            full_slots.gather(
                                1, positions.clamp_max(width - 1)
                            ).masked_fill(positions >= lens[:, None], 0),
                        )
                self.assertEqual(full_logits.call_args.args[-1], width)
                self.assertEqual(full_topk.call_count, 5 if use_topk_v2 else 4)
                self.assertEqual(
                    [call.args[2].shape[1] for call in compact_logits.call_args_list],
                    [8, 8, 8, 8],
                )

    def test_decode_candidates_preserve_slots_across_layers_and_steps(self):
        self._run_candidate_case(
            ForwardMode.DECODE,
            torch.tensor([2, 0, 1]),
            ([261, 7, 0], [130, 260, 3]),
        )

    def test_static_verify_candidates_preserve_per_token_rows(self):
        self._run_candidate_case(
            ForwardMode.TARGET_VERIFY,
            torch.tensor([2, 2, 0, 0, 1, 1]),
            ([261, 262, 7, 8, 130, 131], [132, 133, 260, 261, 3, 4]),
            use_topk_v2=True,
        )

    def test_ragged_verify_candidates_preserve_per_token_rows(self):
        self._run_candidate_case(
            ForwardMode.TARGET_VERIFY,
            torch.tensor([2, 2, 0, 0, 0, 1]),
            ([261, 262, 7, 8, 9, 130], [132, 133, 260, 261, 262, 3]),
        )


class TestDSV4SwaOutCacheLocResolution(CustomTestCase):
    """`get_swa_out_cache_loc`: cached fast path vs store-time fallback.

    The KV-store consumers run in paths that never invoke
    `init_forward_metadata_in_graph` (eager idle, runners that only run the
    out-graph prep) or whose batch is re-padded after init (DP attention).
    The resolver must use the per-forward cached value only when it is
    provably current and fall back to translating `out_cache_loc` otherwise.
    """

    def _make_backend(self, mapping: torch.Tensor):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        backend = object.__new__(DeepseekV4AttnBackend)
        backend.forward_metadata = None
        backend.token_to_kv_pool = SimpleNamespace(
            translate_loc_from_full_to_swa=lambda loc: mapping[loc]
        )
        return backend

    @staticmethod
    def _make_fb(out_cache_loc: torch.Tensor, forward_mode: ForwardMode):
        return SimpleNamespace(out_cache_loc=out_cache_loc, forward_mode=forward_mode)

    @staticmethod
    def _set_cached(backend, cached: torch.Tensor):
        backend.forward_metadata = SimpleNamespace(
            core_attn_metadata=SimpleNamespace(swa_out_cache_loc=cached)
        )

    def test_no_metadata_falls_back_to_translate(self):
        mapping = torch.arange(10, dtype=torch.int64) * 2
        backend = self._make_backend(mapping)
        fb = self._make_fb(torch.tensor([3, 4]), ForwardMode.DECODE)
        out = backend.get_swa_out_cache_loc(fb)
        self.assertEqual(out.dtype, torch.int32)
        self.assertEqual(out.tolist(), [6, 8])

    def test_current_cached_value_is_used(self):
        mapping = torch.arange(10, dtype=torch.int64) * 2
        backend = self._make_backend(mapping)
        cached = torch.tensor([6, 8], dtype=torch.int32)
        self._set_cached(backend, cached)
        fb = self._make_fb(torch.tensor([3, 4]), ForwardMode.DECODE)
        self.assertIs(backend.get_swa_out_cache_loc(fb), cached)

    def test_stale_shape_falls_back_to_translate(self):
        # DP padding rebinds out_cache_loc to a longer tensor after init;
        # a pre-pad cached value must not be used.
        mapping = torch.arange(10, dtype=torch.int64) * 2
        backend = self._make_backend(mapping)
        self._set_cached(backend, torch.tensor([6, 8], dtype=torch.int32))
        fb = self._make_fb(torch.tensor([3, 4, 0, 0]), ForwardMode.DRAFT_EXTEND_V2)
        out = backend.get_swa_out_cache_loc(fb)
        self.assertEqual(out.tolist(), [6, 8, 0, 0])

    def test_idle_never_uses_cached_value(self):
        # Idle forwards skip attn init, so any metadata is left over from a
        # previous forward; writing dummy tokens to its locations would
        # corrupt live KV. Idle must translate the zero-padded out_cache_loc.
        mapping = torch.arange(10, dtype=torch.int64) * 2
        backend = self._make_backend(mapping)
        self._set_cached(backend, torch.tensor([6, 8], dtype=torch.int32))
        fb = self._make_fb(torch.tensor([0, 0]), ForwardMode.IDLE)
        out = backend.get_swa_out_cache_loc(fb)
        self.assertEqual(out.tolist(), [0, 0])


if __name__ == "__main__":
    unittest.main()
