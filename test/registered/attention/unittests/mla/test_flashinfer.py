import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.kits.attention_unittest.attention_methods.mla_attention import (
    MLAAttentionCase,
    make_mla_cases,
    run_mla_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_mla_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_runner import (
    run_mla_eagle_draft_cuda_graph_runner_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_mla_eagle_verify_case,
    run_mla_eagle_verify_cuda_graph_case,
)
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_mla_split_op_extend_case,
)
from sglang.test.test_utils import CustomTestCase

MLA_SHAPE_KWARGS = dict(
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    hidden_size=1024,
)


from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-large")


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

    # Layout-robustness. See dense/test_triton.py for the full
    # rationale. FlashInfer MLA crashes with
    # `AcceleratorError: an illegal memory access was encountered`
    # on both EXTEND under interleaved_pages and non_monotonic_extend,
    # and crashes with `CUBLAS_STATUS_EXECUTION_FAILED` on DECODE under
    # interleaved_pages. The crashes happen inside FlashInfer's MLA
    # paged-prefill / paged-decode metadata; the kernel assumes a
    # tidy page-table layout that the non-tidy variants violate.
    # Documented as LAYOUT_KNOWN_FAILURES so the test method records
    # the production-side cause for future readers.
    LAYOUT_ROBUSTNESS_CASES = (
        MLAAttentionCase(
            name="layout_mla_extend_prefix_exact_page",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=16,
            prefix_lens=(16,),
            extend_lens=(2,),
        ),
        MLAAttentionCase(
            name="layout_mla_decode_page_boundary",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )
    LAYOUT_KNOWN_FAILURES = {
        ("layout_mla_extend_prefix_exact_page", "interleaved_pages"): (
            "FlashInfer MLA paged-prefill metadata assumes a tidy "
            "page-table layout; interleaved pages trip an illegal "
            "memory access inside the kernel."
        ),
        ("layout_mla_extend_prefix_exact_page", "non_monotonic_extend"): (
            "FlashInfer MLA paged-prefill metadata assumes monotonic "
            "out_cache_loc within an extend; scattered extend slots "
            "trip an illegal memory access."
        ),
        ("layout_mla_decode_page_boundary", "interleaved_pages"): (
            "FlashInfer MLA paged-decode metadata raises "
            "CUBLAS_STATUS_EXECUTION_FAILED on interleaved-page layouts."
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

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mla_cuda_graph_decode_case(self, case, **MLA_SHAPE_KWARGS)

    def test_cuda_graph_replay_updates_wrapper_kv_indptr_buffer(self):
        from sglang.srt.layers.attention import flashinfer_mla_backend

        class PassthroughTranslator:
            needs_read_translate = False

            def fill_packed_read_stream(self, **kwargs):
                return True

        updater = flashinfer_mla_backend.FlashInferMLAIndicesUpdaterDecode.__new__(
            flashinfer_mla_backend.FlashInferMLAIndicesUpdaterDecode
        )
        updater.scaling = 1.0
        updater.data_type = torch.float16
        updater.num_local_heads = 4
        updater.kv_lora_rank = 512
        updater.qk_rope_head_dim = 64
        updater.attn_backend = SimpleNamespace(
            kv_index_translator=PassthroughTranslator()
        )

        paged_kernel_lens = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
        eager_kv_indptr = torch.full((3,), -1, dtype=torch.int32, device="cuda")
        graph_kv_indptr = torch.tensor([0, -1, -1], dtype=torch.int32, device="cuda")
        wrapper = SimpleNamespace(plan=MagicMock())

        with patch.object(
            flashinfer_mla_backend,
            "get_parallel",
            return_value=SimpleNamespace(dcp_enabled=False),
        ):
            updater.call_begin_forward(
                wrapper,
                paged_kernel_lens,
                paged_kernel_lens_sum=5,
                q_indptr=torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"),
                kv_indptr=eager_kv_indptr,
                init_metadata_replay=True,
                spec_info=None,
                req_pool_indices=torch.tensor([0, 1], device="cuda"),
                qo_indptr_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
                kv_indptr_cpu=torch.tensor([0, 2, 5], dtype=torch.int32),
                kv_indptr_gpu=graph_kv_indptr,
                kv_len_arr_cpu=torch.tensor([2, 3], dtype=torch.int32),
                kv_indices=torch.empty(5, dtype=torch.int32, device="cuda"),
            )

        expected_graph_kv_indptr = torch.tensor(
            [0, 2, 5], dtype=torch.int32, device="cuda"
        )
        torch.testing.assert_close(graph_kv_indptr, expected_graph_kv_indptr)
        torch.testing.assert_close(
            eager_kv_indptr, torch.full_like(eager_kv_indptr, -1)
        )
        wrapper.plan.assert_called_once()

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

    def test_runner_mode_eagle_draft_cuda_graph_runner_cases(self):
        # Backend gate (KNOWN_FAILURES.md §3): FlashInfer MLA multi-step
        # draft CG capture/replay produces numerically wrong outputs on
        # Blackwell (SM10.x) — observed max abs diff ~22 vs reference on
        # GB300. Cause: the FlashInfer MLA decode kernel in the container
        # targets SM9x and falls back to a generic path on SM10.x that
        # does not restore metadata buffers correctly under graph replay.
        # The eager and DRAFT_EXTEND paths are unaffected; only this CG
        # decode runner regresses. Skip on SM10.x until FlashInfer ships
        # an SM10.x-compiled MLA multi-step decode kernel.
        major, minor = torch.cuda.get_device_capability()
        if major >= 10:
            self.skipTest(
                f"FlashInfer MLA EAGLE draft CG produces wrong outputs on "
                f"SM{major}.{minor} — FlashInfer MLA decode kernel falls back "
                f"to a generic path that breaks under graph replay. See "
                f"KNOWN_FAILURES.md §3. Update FlashInfer to a version that "
                f"ships an SM{major}.x-compiled MLA multi-step decode kernel."
            )
        for case, topk, num_draft_tokens in self.EAGLE_DRAFT_RUNNER_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_mla_eagle_draft_cuda_graph_runner_case(
                    self,
                    case,
                    topk=topk,
                    speculative_num_draft_tokens=num_draft_tokens,
                    **MLA_SHAPE_KWARGS,
                )


if __name__ == "__main__":
    unittest.main()
