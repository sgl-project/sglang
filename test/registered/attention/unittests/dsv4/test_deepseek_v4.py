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
