import unittest

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.sparse_query import SparseQueryMetadata
from sglang.srt.utils import is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    build_dense_attention_fixture,
    expected_dense_fixture_output,
    make_dense_cases,
    run_dense_attention_case,
    run_dense_fixture_eager,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_dense_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_runner import (
    run_dense_eagle_draft_cuda_graph_runner_case,
    run_dense_frozen_kv_mtp_cuda_graph_runner_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_dense_spec_verify_case,
    run_dense_spec_verify_cuda_graph_case,
)
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_dense_split_op_extend_case,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=18, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    "CUDA + flashinfer are required",
)
class TestFlashInferDenseAttentionBackendCorrectness(CustomTestCase):
    # FlashInfer SM90 prefill kernels require value head dim in {64, 128, 256}.
    HEAD_DIM = 64
    HIDDEN_SIZE = 256

    CASES = make_dense_cases("flashinfer")
    CUDA_GRAPH_CASES = (
        DenseAttentionCase(
            name="runner_cuda_graph_decode_page_boundary",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
        DenseAttentionCase(
            name="runner_cuda_graph_gqa_decode_page_boundary",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
        DenseAttentionCase(
            name="runner_cuda_graph_mqa_decode_bsz1",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=1,
            page_size=16,
            prefix_lens=(7,),
        ),
    )
    SPLIT_OP_CASES = (
        (
            DenseAttentionCase(
                name="runner_split_op_mha_extend_ragged_page_boundary",
                backend="flashinfer",
                forward_mode=ForwardMode.EXTEND,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(0, 8, 16),
                extend_lens=(15, 8, 1),
            ),
            32,
        ),
        (
            DenseAttentionCase(
                name="runner_split_op_gqa_extend_cross_page_boundary",
                backend="flashinfer",
                forward_mode=ForwardMode.EXTEND,
                num_heads=4,
                num_kv_heads=2,
                page_size=16,
                prefix_lens=(15,),
                extend_lens=(2,),
            ),
            4,
        ),
    )
    SPEC_VERIFY_CASES = (
        (
            DenseAttentionCase(
                name="runner_eagle_verify_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            DenseAttentionCase(
                name="runner_eagle_verify_tree",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
        (
            DenseAttentionCase(
                name="runner_frozen_kv_mtp_verify_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            DenseAttentionCase(
                name="runner_dflash_verify_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            DenseAttentionCase(
                name="runner_ngram_verify_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
        ),
    )
    SPEC_VERIFY_CUDA_GRAPH_CASES = (
        (
            DenseAttentionCase(
                name="runner_cuda_graph_eagle_verify_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            DenseAttentionCase(
                name="runner_cuda_graph_eagle_verify_tree",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
        (
            DenseAttentionCase(
                name="runner_cuda_graph_frozen_kv_mtp_verify_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            DenseAttentionCase(
                name="runner_cuda_graph_dflash_verify_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            DenseAttentionCase(
                name="runner_cuda_graph_ngram_verify_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
        ),
    )
    EAGLE_DRAFT_RUNNER_CASES = (
        (
            DenseAttentionCase(
                name="runner_eagle_draft_decode_cuda_graph_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
            ),
            1,
            3,
        ),
        (
            DenseAttentionCase(
                name="runner_eagle_draft_decode_cuda_graph_tree",
                backend="flashinfer",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                num_kv_heads=4,
                page_size=1,
                prefix_lens=(4, 7),
            ),
            2,
            4,
        ),
    )
    FROZEN_KV_MTP_RUNNER_CASES = (
        DenseAttentionCase(
            name="runner_frozen_kv_mtp_decode_cuda_graph_chain",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(4, 7),
        ),
    )

    def test_sparse_queries_preserve_positions_and_unselected_kv(self) -> None:
        """Compacting queries must preserve causal positions and physical writes."""
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                case = DenseAttentionCase(
                    name="sparse_gqa_across_requests_and_pages",
                    backend="flashinfer",
                    forward_mode=ForwardMode.EXTEND,
                    num_heads=4,
                    num_kv_heads=2,
                    page_size=16,
                    prefix_lens=(0,) * batch_size,
                    extend_lens=(17, 21)[:batch_size],
                )
                fixture = build_dense_attention_fixture(
                    self, case, head_dim=self.HEAD_DIM, hidden_size=self.HIDDEN_SIZE
                )
                with torch.no_grad():
                    run_dense_fixture_eager(fixture)
                    batch = fixture.forward_batch
                    selected = torch.tensor(
                        [0, 7, 16, 18, 22, 37][: 3 * batch_size], device="cuda"
                    )
                    fixture.input_hidden[selected] += 0.5
                    expected = expected_dense_fixture_output(fixture)[selected]
                    storage = fixture.runner.token_to_kv_pool.layer_router.read(0)
                    full_slots = batch.out_cache_loc.clone()
                    old_k = storage.key()[full_slots].clone()
                    old_v = storage.value()[full_slots].clone()

                    fixture.input_hidden = fixture.input_hidden[selected]
                    batch.input_ids = batch.input_ids[selected]
                    batch.positions = batch.positions[selected]
                    batch.iteration_ids.bind(
                        batch, fixture.runner.runner_ids, cols=selected
                    )
                    batch.extend_num_tokens = len(selected)
                    batch.extend_seq_lens = torch.tensor(
                        [3] * batch_size, dtype=torch.int32, device="cuda"
                    )
                    batch.extend_seq_lens_cpu = [3] * batch_size
                    batch.extend_start_loc = torch.tensor(
                        [3 * i for i in range(batch_size)],
                        dtype=torch.int32,
                        device="cuda",
                    )
                    batch.sparse_query = SparseQueryMetadata(
                        (3,) * batch_size, batch.positions
                    )

                    actual = run_dense_fixture_eager(fixture)
                    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
                    _, new_k, new_v = fixture.actual_module.project_qkv(
                        fixture.input_hidden
                    )
                    for cached, previous, updated in (
                        (storage.key(), old_k, new_k),
                        (storage.value(), old_v, new_v),
                    ):
                        torch.testing.assert_close(
                            cached[batch.out_cache_loc],
                            updated.view(-1, case.num_kv_heads, self.HEAD_DIM),
                            atol=0,
                            rtol=0,
                        )
                        untouched = torch.ones(
                            len(full_slots), dtype=torch.bool, device=cached.device
                        )
                        untouched[selected] = False
                        torch.testing.assert_close(
                            cached[full_slots[untouched]],
                            previous[untouched],
                            atol=0,
                            rtol=0,
                        )

    def test_projected_dense_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    # Layout-robustness: see dense/test_triton.py for full rationale.
    # Re-runs a representative extend + decode under non-tidy
    # (req_to_token, out_cache_loc) mappings to catch backend bugs in
    # page-table derivation that the default contiguous layout hides.
    LAYOUT_ROBUSTNESS_CASES = (
        DenseAttentionCase(
            name="layout_extend_two_request_ragged",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_heads=12,
            num_kv_heads=12,
            page_size=16,
            prefix_lens=(8, 16),
            extend_lens=(8, 16),
        ),
        DenseAttentionCase(
            name="layout_decode_page_boundary",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=12,
            num_kv_heads=12,
            page_size=16,
            prefix_lens=(15, 16, 17),
        ),
    )

    def test_layout_robustness_cases(self):
        for case in self.LAYOUT_ROBUSTNESS_CASES:
            # shuffled_pages is the default and already covered.
            for layout in (
                "interleaved_pages",
                "non_monotonic_extend",
            ):
                if layout == "non_monotonic_extend" and case.forward_mode.is_decode():
                    continue
                with self.subTest(case=case.name, layout=layout):
                    run_dense_attention_case(
                        self,
                        case,
                        head_dim=self.HEAD_DIM,
                        hidden_size=self.HIDDEN_SIZE,
                        loc_layout=layout,
                    )

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_cuda_graph_decode_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_runner_mode_split_op_extend_cases(self):
        for case, static_num_tokens in self.SPLIT_OP_CASES:
            for breakable in (False, True):
                runner = "bcg" if breakable else "pcg"
                with self.subTest(
                    case=case.name,
                    backend=case.backend,
                    runner=runner,
                ):
                    run_dense_split_op_extend_case(
                        self,
                        case,
                        breakable=breakable,
                        static_num_tokens=static_num_tokens,
                        head_dim=self.HEAD_DIM,
                        hidden_size=self.HIDDEN_SIZE,
                    )

    def test_runner_mode_spec_verify_cases(self):
        for case, topk, spec_kind in self.SPEC_VERIFY_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_dense_spec_verify_case(
                    self,
                    case,
                    topk=topk,
                    spec_kind=spec_kind,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_runner_mode_spec_verify_cuda_graph_cases(self):
        for case, topk, spec_kind in self.SPEC_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_dense_spec_verify_cuda_graph_case(
                    self,
                    case,
                    topk=topk,
                    spec_kind=spec_kind,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_runner_mode_eagle_draft_cuda_graph_runner_cases(self):
        for case, topk, num_draft_tokens in self.EAGLE_DRAFT_RUNNER_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_dense_eagle_draft_cuda_graph_runner_case(
                    self,
                    case,
                    topk=topk,
                    speculative_num_draft_tokens=num_draft_tokens,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_runner_mode_frozen_kv_mtp_cuda_graph_runner_cases(self):
        for case in self.FROZEN_KV_MTP_RUNNER_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_frozen_kv_mtp_cuda_graph_runner_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )


if __name__ == "__main__":
    unittest.main()
