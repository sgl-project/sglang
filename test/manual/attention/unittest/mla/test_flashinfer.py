import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.mla_attention import (
    MLAAttentionCase,
    make_mla_cases,
    run_mla_attention_case,
)
from common.runner_modes.cuda_graph_decode_runner import run_mla_cuda_graph_decode_case
from common.runner_modes.eagle_draft_runner import (
    run_mla_eagle_draft_cuda_graph_runner_case,
    run_mla_eagle_draft_extend_cuda_graph_runner_case,
)
from common.runner_modes.speculative_draft_extend_runner import (
    run_mla_draft_extend_cuda_graph_case,
    run_mla_eagle_draft_extend_case,
)
from common.runner_modes.speculative_target_verify_runner import (
    run_mla_eagle_verify_case,
    run_mla_eagle_verify_cuda_graph_case,
)
from common.runner_modes.split_op_runner import run_mla_split_op_extend_case

MLA_SHAPE_KWARGS = dict(
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    hidden_size=1024,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestFlashInferMLAAttentionBackendCorrectness(CustomTestCase):
    CASES = make_mla_cases("flashinfer")
    CUDA_GRAPH_CASES = (
        MLAAttentionCase(
            name="runner_cuda_graph_decode_page_boundary",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )
    SPLIT_OP_CASES = (
        (
            MLAAttentionCase(
                name="runner_split_op_mla_extend_ragged_page_boundary",
                backend="flashinfer",
                forward_mode=ForwardMode.EXTEND,
                num_heads=4,
                page_size=16,
                prefix_lens=(0, 8, 16),
                extend_lens=(15, 8, 1),
            ),
            32,
        ),
    )
    EAGLE_VERIFY_CASES = (
        (
            MLAAttentionCase(
                name="runner_eagle_verify_mla_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
        ),
    )
    EAGLE_VERIFY_CUDA_GRAPH_CASES = (
        (
            MLAAttentionCase(
                name="runner_cuda_graph_eagle_verify_mla_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
        ),
    )
    DRAFT_EXTEND_CASES = (
        MLAAttentionCase(
            name="runner_eagle_draft_extend_mla_ragged_accept",
            backend="flashinfer",
            forward_mode=ForwardMode.DRAFT_EXTEND,
            num_heads=4,
            page_size=16,
            prefix_lens=(5, 8),
            extend_lens=(2, 4),
        ),
    )
    DRAFT_EXTEND_CUDA_GRAPH_CASES = (
        MLAAttentionCase(
            name="runner_cuda_graph_eagle_draft_extend_mla_ragged_accept",
            backend="flashinfer",
            forward_mode=ForwardMode.DRAFT_EXTEND,
            num_heads=4,
            page_size=16,
            prefix_lens=(5, 8),
            extend_lens=(2, 4),
        ),
    )
    EAGLE_DRAFT_EXTEND_RUNNER_CASES = (
        MLAAttentionCase(
            name="runner_eagle_draft_extend_mla_cuda_graph_runner_ragged_accept",
            backend="flashinfer",
            forward_mode=ForwardMode.DRAFT_EXTEND,
            num_heads=4,
            page_size=16,
            prefix_lens=(5, 8),
            extend_lens=(2, 4),
        ),
    )
    EAGLE_DRAFT_RUNNER_CASES = (
        (
            MLAAttentionCase(
                name="runner_eagle_draft_decode_mla_cuda_graph_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                page_size=16,
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

    def test_runner_mode_eagle_draft_extend_cuda_graph_cases(self):
        for case in self.DRAFT_EXTEND_CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mla_draft_extend_cuda_graph_case(
                    self,
                    case,
                    **MLA_SHAPE_KWARGS,
                )

    def test_runner_mode_eagle_draft_extend_cuda_graph_runner_cases(self):
        for case in self.EAGLE_DRAFT_EXTEND_RUNNER_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mla_eagle_draft_extend_cuda_graph_runner_case(
                    self,
                    case,
                    **MLA_SHAPE_KWARGS,
                )

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

    # M12 (mutation journal): mutating
    # `flashinfer_mla_backend.py:318` from `prefix_lens=None` to
    # `prefix_lens=forward_batch.seq_lens` in the eager target_verify
    # branch was MISSED. After tracing the call chain:
    #
    #   `FlashInferMLAAttnBackend.init_forward_metadata`
    #     -> `FlashInferMLAIndicesUpdaterPrefill.update`
    #     -> `FlashInferMLAIndicesUpdaterPrefill.call_begin_forward`
    #
    # `call_begin_forward` branches on `spec_info is None`. The
    # target_verify branch always supplies a non-None `spec_info`
    # (built by `_make_eagle_verify_input`), which means the
    # `spec_info` branch is taken and `paged_kernel_lens` /
    # `kv_indptr` are derived from `spec_info.generate_attn_arg_prefill`
    # alone. The `prefix_lens` parameter is *never* consumed in that
    # branch (see `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`
    # around lines 853-863), so the M12 mutation truly has no
    # observable effect.
    #
    # We deliberately do not add a test that "catches" M12: doing so
    # would require an artificial dependency on `prefix_lens` that the
    # production code does not actually have, which would make the
    # test fragile against legitimate refactors. Instead we record the
    # finding here and in `MUTATION_FIXES.md`.
    @unittest.skip(
        "M12 is a no-op in production: the `prefix_lens=None` argument at "
        "flashinfer_mla_backend.py:318 (eager target_verify) flows to "
        "FlashInferMLAIndicesUpdaterPrefill.call_begin_forward, which only "
        "uses prefix_lens in the `spec_info is None` branch. Target_verify "
        "always supplies a non-None spec_info, so the mutation is "
        "structurally undetectable."
    )
    def test_eager_target_verify_prefix_lens_is_noop(self):
        raise AssertionError("documentation-only skip; see docstring above")


if __name__ == "__main__":
    unittest.main()
