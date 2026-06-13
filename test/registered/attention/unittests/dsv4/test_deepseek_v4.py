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
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_FLASH_MLA_AVAILABLE = importlib.util.find_spec("flash_mla") is not None

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
from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_extend_runner import (  # noqa: E402
    run_dsv4_eagle_draft_extend_cuda_graph_case,
    run_dsv4_eagle_draft_extend_cuda_graph_runner_case,
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
        for case in self.COMPRESS_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                compress_ratio=case.compress_ratio,
            ):
                run_dsv4_compress_attention_case(self, case)

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
                run_dsv4_eagle_verify_cuda_graph_case(self, case, topk=1)

    # EAGLE DRAFT_EXTEND is SWA-only for DSV4 (see runner docstring).
    DRAFT_EXTEND_CASES = (
        DSV4AttentionCase(
            name="dsv4_swa_eagle_draft_extend",
            backend="dsv4",
            forward_mode=ForwardMode.DRAFT_EXTEND,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(64, 96),
            extend_lens=(2, 4),
        ),
    )

    def test_eagle_draft_extend_cases(self):
        for case in self.DRAFT_EXTEND_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsv4_draft_extend_attention_case(self, case)

    # CUDA-graph capture/replay for EAGLE DRAFT_EXTEND — SWA only
    # (init_forward_metadata_draft_extend uses need_compress=False; see
    # `Production-Unsupported` in dsv4/README.md). Uniform `extend_lens`
    # because DSV4 `forward(compress_ratio=0)` asserts
    # `swa_page_indices.shape[0] == q.shape[0]` and the graph metadata
    # builder uses uniform `num_tokens_per_bs = max_num_tokens // max_bs`
    # (see `deepseek_v4_backend.py:646-647`).
    EAGLE_DRAFT_EXTEND_CUDA_GRAPH_CASES = (
        DSV4AttentionCase(
            name="runner_cuda_graph_dsv4_swa_eagle_draft_extend",
            backend="dsv4",
            forward_mode=ForwardMode.DRAFT_EXTEND,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(64, 96),
            extend_lens=(4, 4),
        ),
    )

    def test_runner_mode_eagle_draft_extend_cuda_graph_cases(self):
        for case in self.EAGLE_DRAFT_EXTEND_CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsv4_eagle_draft_extend_cuda_graph_case(self, case)

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
                run_dsv4_eagle_draft_cuda_graph_runner_case(self, case)

    # Production EAGLE draft-extend graph runner (SWA only). Routes through
    # the prefill-side `DeepseekV4AttnBackend` (single backend, not
    # multi-step); `init_forward_metadata_draft_extend` forces
    # `need_compress=False` so C4/C128 is structurally unreachable for this
    # path.
    # Uniform `extend_lens` because the DSV4 graph contract requires
    # `q.shape[0] == swa_page_indices.shape[0]` and the
    # `init_forward_metadata_draft_extend` graph path uses
    # `num_tokens_per_bs = max_num_tokens // max_bs` (see
    # `deepseek_v4_backend.py:646-647`). Same constraint as the metadata-
    # style draft_extend CG case.
    PRODUCTION_EAGLE_DRAFT_EXTEND_RUNNER_CASES = (
        DSV4AttentionCase(
            name="runner_production_eagle_draft_extend_dsv4_swa",
            backend="dsv4",
            forward_mode=ForwardMode.DRAFT_EXTEND,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(64, 96),
            extend_lens=(4, 4),
        ),
    )

    def test_runner_mode_production_eagle_draft_extend_cuda_graph_runner_cases(self):
        for case in self.PRODUCTION_EAGLE_DRAFT_EXTEND_RUNNER_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsv4_eagle_draft_extend_cuda_graph_runner_case(self, case)


class TestDSV4BreakableCudaGraphMetadataContract(CustomTestCase):
    """CPU-only checks for the DSV4 BCG metadata replay contract."""

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
            c4_sparse_topk=128,
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
        metadata.c1_flashmla_metadata = object()
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
            "c1_flashmla_metadata",
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
        self.assertTrue(
            torch.equal(
                capture_metadata.core_attn_metadata.seq_lens_casual,
                replay_metadata.core_attn_metadata.seq_lens_casual,
            )
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
