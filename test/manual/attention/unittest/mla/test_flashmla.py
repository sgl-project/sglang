import sys
import unittest
from pathlib import Path

import torch
import triton

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.mla_attention import (
    MLAAttentionCase,
    build_mla_attention_fixture,
    run_mla_attention_case,
)
from common.runner_modes.cuda_graph_decode_runner import (
    _init_cuda_graph_capture_metadata,
    _init_cuda_graph_replay_metadata,
    run_mla_cuda_graph_decode_case,
)
from common.runner_modes.eagle_draft_runner import (
    run_mla_eagle_draft_cuda_graph_runner_case,
)
from common.runner_modes.speculative_draft_extend_runner import (
    run_mla_eagle_draft_extend_case,
)
from common.runner_modes.speculative_target_verify_runner import (
    _make_eagle_verify_input,
    _prepare_target_verify_batch,
    run_mla_eagle_verify_case,
    run_mla_eagle_verify_cuda_graph_case,
)
from common.runner_modes.split_op_runner import run_mla_split_op_extend_case

MLA_SHAPE_KWARGS = dict(
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    hidden_size=1024,
    max_context_len=256,
)

# FlashMLA's KV cache is paginated with PAGE_SIZE=64
# (see `python/sglang/srt/layers/attention/flashmla_backend.py`).
FLASHMLA_PAGE_SIZE = 64


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestFlashMLAAttentionBackendCorrectness(CustomTestCase):
    CASES = (
        MLAAttentionCase(
            name="mla_extend_zero_prefix_exact_flashmla_page",
            backend="flashmla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(64,),
        ),
        MLAAttentionCase(
            name="mla_extend_cross_flashmla_page_boundary",
            backend="flashmla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(63,),
            extend_lens=(2,),
        ),
        MLAAttentionCase(
            name="mla_extend_ragged_flashmla_page_boundary",
            backend="flashmla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0, 32, 64),
            extend_lens=(63, 32, 1),
        ),
        MLAAttentionCase(
            name="mla_decode_flashmla_page_boundary",
            backend="flashmla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=64,
            prefix_lens=(61, 62, 63),
        ),
    )
    CUDA_GRAPH_CASES = (
        MLAAttentionCase(
            name="runner_cuda_graph_decode_flashmla_page_boundary",
            backend="flashmla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=64,
            prefix_lens=(61, 62, 63),
        ),
    )
    SPLIT_OP_CASES = (
        (
            MLAAttentionCase(
                name="runner_split_op_mla_flashmla_ragged_page_boundary",
                backend="flashmla",
                forward_mode=ForwardMode.EXTEND,
                num_heads=4,
                page_size=64,
                prefix_lens=(0, 32, 64),
                extend_lens=(63, 32, 1),
            ),
            96,
        ),
    )
    EAGLE_VERIFY_CASES = (
        (
            MLAAttentionCase(
                name="runner_eagle_verify_mla_flashmla_chain",
                backend="flashmla",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                page_size=64,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
        ),
    )
    EAGLE_VERIFY_CUDA_GRAPH_CASES = (
        (
            MLAAttentionCase(
                name="runner_cuda_graph_eagle_verify_mla_flashmla_chain",
                backend="flashmla",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                page_size=64,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
        ),
    )
    DRAFT_EXTEND_CASES = (
        MLAAttentionCase(
            name="runner_eagle_draft_extend_mla_flashmla_ragged_accept",
            backend="flashmla",
            forward_mode=ForwardMode.DRAFT_EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(5, 8),
            extend_lens=(2, 4),
        ),
    )
    EAGLE_DRAFT_RUNNER_CASES = (
        (
            MLAAttentionCase(
                name="runner_eagle_draft_decode_mla_flashmla_cuda_graph_chain",
                backend="flashmla",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                page_size=64,
                prefix_lens=(4, 7),
            ),
            1,
            3,
        ),
    )

    def test_tiny_deepseek_mla_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mla_attention_case(self, case, **MLA_SHAPE_KWARGS)

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mla_cuda_graph_decode_case(self, case, **MLA_SHAPE_KWARGS)

    def test_runner_mode_split_op_extend_cases(self):
        for case, static_num_tokens in self.SPLIT_OP_CASES:
            for breakable in (False, True):
                runner = "bcg" if breakable else "pcg"
                with self.subTest(
                    case=case.name,
                    backend=case.backend,
                    runner=runner,
                ):
                    run_mla_split_op_extend_case(
                        self,
                        case,
                        breakable=breakable,
                        static_num_tokens=static_num_tokens,
                        **MLA_SHAPE_KWARGS,
                    )

    def test_runner_mode_eagle_verify_cases(self):
        for case, topk in self.EAGLE_VERIFY_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_mla_eagle_verify_case(
                    self,
                    case,
                    topk=topk,
                    **MLA_SHAPE_KWARGS,
                )

    def test_runner_mode_eagle_verify_cuda_graph_cases(self):
        for case, topk in self.EAGLE_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_mla_eagle_verify_cuda_graph_case(
                    self,
                    case,
                    topk=topk,
                    **MLA_SHAPE_KWARGS,
                )

    def test_runner_mode_eagle_draft_extend_cases(self):
        for case in self.DRAFT_EXTEND_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mla_eagle_draft_extend_case(self, case, **MLA_SHAPE_KWARGS)

    def test_runner_mode_eagle_draft_cuda_graph_runner_cases(self):
        for case, topk, num_draft_tokens in self.EAGLE_DRAFT_RUNNER_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_mla_eagle_draft_cuda_graph_runner_case(
                    self,
                    case,
                    topk=topk,
                    speculative_num_draft_tokens=num_draft_tokens,
                    **MLA_SHAPE_KWARGS,
                )

    # ------------------------------------------------------------------
    # Direct FlashMLA target_verify metadata assertions for M14/M15/M16.
    #
    # The earlier eager + replay verify tests went green even when the
    # production code:
    #   - added `+1` to seq_lens (M14),
    #   - dropped `+ num_draft_tokens` in eager target_verify (M15), or
    #   - dropped `+ num_draft_tokens` in replay target_verify (M16).
    # The reason was twofold:
    #   1. For the existing `prefix_lens=(4, 7)` + `draft=3` case, the
    #      "correct" and "mutated" `seq_lens` both fit comfortably inside
    #      a single PAGE_SIZE=64 block, so the constructed
    #      `block_kv_indices` was identical.
    #   2. `forward_extend` recomputes its own `cache_seqlens =
    #      forward_batch.seq_lens + num_draft_tokens` from the
    #      *unmutated* batch fields and only reads the first
    #      `cdiv(cache_seqlens, PAGE_SIZE)` block entries, so any extra
    #      entries M14 might add are never read by the kernel.
    # The reliable signal is therefore the *shape* and per-row
    # population of `block_kv_indices`, which directly encodes the
    # mutated `seq_lens` regardless of what `forward_extend` later does.
    # We choose `prefix_lens=(61, 63)` with `draft=3` so:
    #   - correct: seq_lens+draft = (64, 66) -> per-row valid pages
    #     (1, 2), `max_seqlen_pad = 2`.
    #   - M14: seq_lens+draft+1 = (65, 67) -> (2, 2), pad = 2 (same
    #     overall pad, but row 0's valid count differs).
    #   - M15/M16: seq_lens = prefix = (61, 63) -> (1, 1), pad = 1
    #     (overall shape changes).
    # ------------------------------------------------------------------

    METADATA_VERIFY_CASE = MLAAttentionCase(
        name="metadata_eagle_verify_flashmla_page_boundary",
        backend="flashmla",
        forward_mode=ForwardMode.TARGET_VERIFY,
        num_heads=4,
        page_size=FLASHMLA_PAGE_SIZE,
        prefix_lens=(61, 63),
        extend_lens=(3, 3),
    )

    @staticmethod
    def _expected_block_kv_layout(
        prefix_lens: tuple[int, ...],
        num_draft_tokens: int,
    ) -> tuple[int, int, list[int]]:
        """Return (bs, expected_max_seqlen_pad, per_row_valid_pages)."""
        bs = len(prefix_lens)
        per_row_seq_lens = [p + num_draft_tokens for p in prefix_lens]
        max_seqlen_pad = triton.cdiv(max(per_row_seq_lens), FLASHMLA_PAGE_SIZE)
        per_row_valid = [
            triton.cdiv(s, FLASHMLA_PAGE_SIZE) for s in per_row_seq_lens
        ]
        return bs, max_seqlen_pad, per_row_valid

    def _build_target_verify_metadata_fixture(self, case):
        fixture = build_mla_attention_fixture(
            self,
            case,
            **MLA_SHAPE_KWARGS,
        )
        _prepare_target_verify_batch(fixture.forward_batch, case, fixture.runner.device)
        fixture.forward_batch.spec_info = _make_eagle_verify_input(
            case,
            fixture.forward_batch,
            topk=1,
            device=fixture.runner.device,
        )
        return fixture

    def test_eager_target_verify_block_kv_indices_metadata(self):
        case = self.METADATA_VERIFY_CASE
        num_draft_tokens = case.extend_lens[0]
        bs, expected_pad, expected_valid_pages = self._expected_block_kv_layout(
            case.prefix_lens, num_draft_tokens
        )

        fixture = self._build_target_verify_metadata_fixture(case)
        with torch.no_grad(), forward_context(
            ForwardContext(attn_backend=fixture.backend)
        ):
            fixture.backend.init_forward_metadata(fixture.forward_batch)

        block_kv_indices = fixture.backend.forward_metadata.block_kv_indices
        self.assertEqual(
            tuple(block_kv_indices.shape),
            (bs, expected_pad),
            "FlashMLA eager target_verify `block_kv_indices` shape must encode "
            "`max(seq_lens + num_draft_tokens)` rounded up to PAGE_SIZE. "
            "A `+1` mutation (M14) or a dropped `+ num_draft_tokens` "
            "(M15) will produce a different shape with the configured "
            "page-boundary prefix lens.",
        )
        valid_per_row = (block_kv_indices >= 0).sum(dim=1).cpu().tolist()
        self.assertEqual(
            valid_per_row,
            expected_valid_pages,
            "Per-request page-count populated in `block_kv_indices` must "
            "match `cdiv((prefix + num_draft_tokens) / PAGE_SIZE)`. "
            "M14 (+1) or M15 (drop num_draft_tokens) skews this count "
            "even when the overall shape happens to coincide.",
        )

    def test_replay_target_verify_block_kv_indices_metadata(self):
        # M16 only exists in `init_forward_metadata_replay_cuda_graph`,
        # so we drive the same metadata assertion through a capture+replay
        # cycle.
        case = self.METADATA_VERIFY_CASE
        num_draft_tokens = case.extend_lens[0]
        bs, expected_pad, _ = self._expected_block_kv_layout(
            case.prefix_lens, num_draft_tokens
        )

        # FlashMLA's verify replay path indexes into a pre-allocated 2D
        # `cuda_graph_kv_indices` shaped `(max_bs, ceil(max_ctx + PAGE) /
        # PAGE)` and slices it down to `[:bs, :max_seqlen_pad]`. The
        # *sliced* metadata is what the FlashMLA kernel sees, so we
        # assert its column count: that's `cdiv(max(seq_lens +
        # num_draft_tokens) / PAGE_SIZE)`. Dropping `+ num_draft_tokens`
        # (M16) shrinks this column count for the configured
        # page-boundary prefix lens (61+3=64 stays at 1 page, 63+3=66
        # crosses into 2 pages, so the correct column count is 2; M16
        # collapses to max(cdiv(61,64), cdiv(63,64))=1).
        # NOTE: This assertion intentionally uses *only* the slice shape
        # because the underlying `cuda_graph_kv_indices` buffer is
        # initialised to `1` (not `-1`), so "populated" entries cannot
        # be counted by sentinel comparison the way the eager path
        # allows.
        fixture = self._build_target_verify_metadata_fixture(case)
        backend = fixture.backend
        with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
            backend.init_cuda_graph_state(
                max_bs=bs,
                max_num_tokens=bs * num_draft_tokens,
            )
            _init_cuda_graph_capture_metadata(backend, bs, fixture.forward_batch)
            _init_cuda_graph_replay_metadata(backend, bs, fixture.forward_batch)

        block_kv_indices = backend.forward_metadata.block_kv_indices
        self.assertEqual(
            tuple(block_kv_indices.shape),
            (bs, expected_pad),
            "FlashMLA replay target_verify `block_kv_indices` slice must "
            "encode `max(seq_lens + num_draft_tokens)` rounded up to "
            "PAGE_SIZE. Dropping `+ num_draft_tokens` in the replay "
            "branch (M16) reduces the slice width below this expected "
            "value for the configured page-boundary prefix lens.",
        )


if __name__ == "__main__":
    unittest.main()
