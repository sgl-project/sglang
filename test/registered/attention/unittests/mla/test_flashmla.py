import sys
import unittest
from pathlib import Path

import torch
import triton

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.kits.attention_unittest.attention_methods.mla_attention import (
    MLAAttentionCase,
    build_mla_attention_fixture,
    run_mla_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    _init_cuda_graph_capture_metadata,
    _init_cuda_graph_replay_metadata,
    run_mla_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_extend_runner import (
    run_mla_draft_extend_cuda_graph_case,
    run_mla_eagle_draft_extend_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_runner import (
    run_mla_eagle_draft_cuda_graph_runner_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    _make_eagle_verify_input,
    _prepare_target_verify_batch,
    run_mla_eagle_verify_case,
    run_mla_eagle_verify_cuda_graph_case,
)
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_mla_split_op_extend_case,
)

MLA_SHAPE_KWARGS = dict(
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    hidden_size=1024,
    max_context_len=256,
)

# FlashMLA's KV cache is paginated with PAGE_SIZE=64
# (see `python/sglang/srt/layers/attention/flashmla_backend.py`).
FLASHMLA_PAGE_SIZE = 64

# FlashMLABackend.forward_decode and forward_target_verify require SM90a
# (Hopper architecture — H100/H200).  On Blackwell (SM10.x) those paths
# raise "Dense decode MLA is only supported on SM90a architecture".
# EXTEND falls through to the FlashInferMLAAttnBackend parent and works
# on any SM >= 9.
_DECODE_REQUIRES_SM90A = (
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] >= 10
)
_DECODE_SKIP_REASON = (
    "FlashMLA decode/target-verify requires SM90a (Hopper); "
    f"got SM{torch.cuda.get_device_capability()[0]}.x"
    if _DECODE_REQUIRES_SM90A and torch.cuda.is_available()
    else "CUDA unavailable"
)


from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-large")


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
        # Sequence length one below / exactly at / one above the page
        # boundary with zero prefix (Required input case: "Sequence length
        # one token below and one token above a page boundary").
        MLAAttentionCase(
            name="mla_extend_flashmla_input_page_edges",
            backend="flashmla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0, 0, 0),
            extend_lens=(63, 64, 65),
        ),
        # Prefix length exactly equal to one page (Required input case).
        MLAAttentionCase(
            name="mla_extend_prefix_exact_flashmla_page",
            backend="flashmla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(64,),
            extend_lens=(2,),
        ),
        # Prefix plus extend length exactly equal to one page (Required
        # input case).
        MLAAttentionCase(
            name="mla_extend_total_exact_flashmla_page",
            backend="flashmla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(32,),
            extend_lens=(32,),
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
        # Decode with nonzero prefix at batch-size 1 (Required input case).
        MLAAttentionCase(
            name="mla_decode_flashmla_bsz1_nonzero_prefix",
            backend="flashmla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=64,
            prefix_lens=(31,),
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
    DRAFT_EXTEND_CUDA_GRAPH_CASES = (
        MLAAttentionCase(
            name="runner_cuda_graph_eagle_draft_extend_mla_flashmla_ragged_accept",
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
                if case.forward_mode == ForwardMode.DECODE and _DECODE_REQUIRES_SM90A:
                    self.skipTest(_DECODE_SKIP_REASON)
                run_mla_attention_case(self, case, **MLA_SHAPE_KWARGS)

    # Layout-robustness. See dense/test_triton.py for the full
    # rationale. FlashMLA crashes on both EXTEND layouts (illegal
    # memory access) and on DECODE with interleaved_pages (shape
    # mismatch). Documented as LAYOUT_KNOWN_FAILURES.
    LAYOUT_ROBUSTNESS_CASES = (
        MLAAttentionCase(
            name="layout_mla_extend_prefix_exact_page",
            backend="flashmla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(64,),
            extend_lens=(2,),
        ),
        MLAAttentionCase(
            name="layout_mla_decode_page_boundary",
            backend="flashmla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=64,
            prefix_lens=(62, 63, 64),
        ),
    )
    LAYOUT_KNOWN_FAILURES = {
        ("layout_mla_extend_prefix_exact_page", "interleaved_pages"): (
            "FlashMLA extend path raises CUDA illegal memory access on "
            "interleaved-page layouts; the kernel assumes a tidy "
            "page-table layout."
        ),
        ("layout_mla_extend_prefix_exact_page", "non_monotonic_extend"): (
            "FlashMLA extend path raises CUDA illegal memory access on "
            "non-monotonic out_cache_loc within an extend."
        ),
        ("layout_mla_decode_page_boundary", "interleaved_pages"): (
            "FlashMLA decode path raises a shape mismatch "
            "(`shape '[-1, 64, 1, 32]' is invalid for input of size N`) "
            "on interleaved-page layouts."
        ),
    }

    def test_layout_robustness_cases(self):
        for case in self.LAYOUT_ROBUSTNESS_CASES:
            for layout in ("interleaved_pages", "non_monotonic_extend"):
                if layout == "non_monotonic_extend" and case.forward_mode.is_decode():
                    continue
                reason = self.LAYOUT_KNOWN_FAILURES.get((case.name, layout))
                if reason is not None:
                    print(
                        f"[layout-known-failure] {case.name} x {layout}: {reason}",
                        flush=True,
                    )
                    continue
                with self.subTest(case=case.name, layout=layout):
                    run_mla_attention_case(
                        self, case, loc_layout=layout, **MLA_SHAPE_KWARGS
                    )

    @unittest.skipIf(_DECODE_REQUIRES_SM90A, _DECODE_SKIP_REASON)
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

    @unittest.skipIf(_DECODE_REQUIRES_SM90A, _DECODE_SKIP_REASON)
    def test_runner_mode_eagle_verify_cases(self):
        for case, topk in self.EAGLE_VERIFY_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_mla_eagle_verify_case(
                    self,
                    case,
                    topk=topk,
                    **MLA_SHAPE_KWARGS,
                )

    @unittest.skipIf(_DECODE_REQUIRES_SM90A, _DECODE_SKIP_REASON)
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

    def test_runner_mode_eagle_draft_extend_cuda_graph_cases(self):
        for case in self.DRAFT_EXTEND_CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mla_draft_extend_cuda_graph_case(
                    self,
                    case,
                    **MLA_SHAPE_KWARGS,
                )

    @unittest.skipIf(_DECODE_REQUIRES_SM90A, _DECODE_SKIP_REASON)
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

    # `prefix_lens=(61, 63)` with `draft=3` straddles PAGE_SIZE=64 so the
    # constructed `block_kv_indices` shape/population differs between
    # correct, +1, and dropped-draft variants.
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
        per_row_valid = [triton.cdiv(s, FLASHMLA_PAGE_SIZE) for s in per_row_seq_lens]
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
        # The replay buffer is prefilled with 1s, so only the slice shape is stable.
        case = self.METADATA_VERIFY_CASE
        num_draft_tokens = case.extend_lens[0]
        bs, expected_pad, _ = self._expected_block_kv_layout(
            case.prefix_lens, num_draft_tokens
        )

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
