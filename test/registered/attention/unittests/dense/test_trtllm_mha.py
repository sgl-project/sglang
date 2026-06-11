import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import (
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
)
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    run_dense_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_dense_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_runner import (
    run_dense_eagle_draft_cuda_graph_runner_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_dense_spec_verify_case,
    run_dense_spec_verify_cuda_graph_case,
)

register_cuda_ci(est_time=20, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(
    not torch.cuda.is_available()
    or not is_flashinfer_available()
    or not (is_sm90_supported() or is_sm100_supported() or is_sm120_supported()),
    "CUDA + FlashInfer TRT-LLM MHA decode support are required",
)
class TestTRTLLMMHADenseAttentionBackendCorrectness(CustomTestCase):
    HEAD_DIM = 64
    HIDDEN_SIZE = 256

    DECODE_CASES = (
        DenseAttentionCase(
            name="trtllm_mha_decode_page_boundary",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
        DenseAttentionCase(
            name="trtllm_mha_gqa_decode_page_boundary",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
        DenseAttentionCase(
            name="trtllm_mha_mqa_decode_bsz1",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=1,
            page_size=16,
            prefix_lens=(7,),
        ),
        DenseAttentionCase(
            name="trtllm_mha_decode_page32_boundary",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=32,
            prefix_lens=(31, 32),
        ),
    )

    # CG decode replay across MHA/GQA/MQA layouts and a page-32 case.
    # Previously documented as "currently mismatches on replay"; the
    # FlashInfer TRT-LLM Gen FMHA decode backend has since stabilized
    # the capture/replay metadata path and all four shapes match the
    # HF-style dense reference.
    CUDA_GRAPH_DECODE_CASES = (
        DenseAttentionCase(
            name="runner_cuda_graph_trtllm_mha_decode_page_boundary",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
        DenseAttentionCase(
            name="runner_cuda_graph_trtllm_mha_gqa_decode_page_boundary",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
        DenseAttentionCase(
            name="runner_cuda_graph_trtllm_mha_mqa_decode_bsz1",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=1,
            page_size=16,
            prefix_lens=(7,),
        ),
        DenseAttentionCase(
            name="runner_cuda_graph_trtllm_mha_decode_page32_boundary",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=32,
            prefix_lens=(31, 32),
        ),
    )

    # EAGLE draft CG runner — chain (topk=1) plus tree (topk>1) cases. Tree
    # draft decode expands the batch to bs*topk rows with per-branch page
    # tables over the spec-v2 page-aligned branch layout
    # (`eagle_info_v2.prepare_for_v2_draft`). Tuples are
    # (case, topk, num_draft_tokens, max_context_len); prefix_lens straddle
    # page boundaries so branch regions start mid-page (last_page > 0) and
    # exactly on a page boundary (last_page == 0).
    EAGLE_DRAFT_RUNNER_CASES = (
        (
            DenseAttentionCase(
                name="runner_eagle_draft_decode_trtllm_mha_cuda_graph_chain",
                backend="trtllm_mha",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
            ),
            1,  # topk
            3,  # num_draft_tokens
            64,  # max_context_len
        ),
        (
            DenseAttentionCase(
                name="runner_eagle_draft_decode_trtllm_mha_cuda_graph_tree_page16",
                backend="trtllm_mha",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(14, 15, 16),
            ),
            2,  # topk
            4,  # num_draft_tokens
            64,  # max_context_len
        ),
        (
            DenseAttentionCase(
                name="runner_eagle_draft_decode_trtllm_mha_cuda_graph_tree_gqa_topk4",
                backend="trtllm_mha",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                num_kv_heads=2,
                page_size=16,
                prefix_lens=(14, 15, 16),
            ),
            4,  # topk
            6,  # num_draft_tokens
            128,  # max_context_len
        ),
        (
            DenseAttentionCase(
                name="runner_eagle_draft_decode_trtllm_mha_cuda_graph_tree_page32",
                backend="trtllm_mha",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                num_kv_heads=4,
                page_size=32,
                prefix_lens=(31, 32),
            ),
            2,  # topk
            4,  # num_draft_tokens
            128,  # max_context_len
        ),
    )

    def test_projected_dense_decode_cases(self):
        for case in self.DECODE_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_DECODE_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_cuda_graph_decode_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    # EAGLE verify: chain (topk=1) goes through the causal q_len_per_req
    # path; tree (topk=2) passes the QLEN_ONLY tree mask to the trtllm-gen
    # custom-mask kernels. The kit hardcodes draft_token_num=3 with parent
    # indices (-1, 0, 0) for topk>1. prefix_lens straddle page boundaries.
    SPEC_VERIFY_CASES = (
        (
            DenseAttentionCase(
                name="runner_trtllm_mha_eagle_verify_chain",
                backend="trtllm_mha",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,  # topk
        ),
        (
            DenseAttentionCase(
                name="runner_trtllm_mha_eagle_verify_tree",
                backend="trtllm_mha",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(14, 15, 16),
                extend_lens=(3, 3, 3),
                speculative_eagle_topk=2,
            ),
            2,  # topk
        ),
        (
            DenseAttentionCase(
                name="runner_trtllm_mha_eagle_verify_tree_gqa_page32",
                backend="trtllm_mha",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=2,
                page_size=32,
                prefix_lens=(30, 31),
                extend_lens=(3, 3),
                speculative_eagle_topk=2,
            ),
            2,  # topk
        ),
    )

    SPEC_VERIFY_CUDA_GRAPH_CASES = (
        (
            DenseAttentionCase(
                name="runner_cuda_graph_trtllm_mha_eagle_verify_tree",
                backend="trtllm_mha",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(14, 15, 16),
                extend_lens=(3, 3, 3),
                speculative_eagle_topk=2,
            ),
            2,  # topk
        ),
    )

    def test_runner_mode_spec_verify_cases(self):
        for case, topk in self.SPEC_VERIFY_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_dense_spec_verify_case(
                    self,
                    case,
                    topk=topk,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_runner_mode_spec_verify_cuda_graph_cases(self):
        for case, topk in self.SPEC_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_dense_spec_verify_cuda_graph_case(
                    self,
                    case,
                    topk=topk,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_runner_mode_eagle_draft_cuda_graph_runner_cases(self):
        for (
            case,
            topk,
            num_draft_tokens,
            max_context_len,
        ) in self.EAGLE_DRAFT_RUNNER_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_dense_eagle_draft_cuda_graph_runner_case(
                    self,
                    case,
                    topk=topk,
                    speculative_num_draft_tokens=num_draft_tokens,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                    max_context_len=max_context_len,
                )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTRTLLMMHADraftBranchPageTables(CustomTestCase):
    """Validate build_draft_branch_page_tables against (a) a direct python
    re-derivation of the spec-v2 branch layout and (b) the production
    generate_draft_decode_kv_indices triton kernel (independent code path):
    every token that kernel emits must live in the page the table points at,
    at the matching in-page offset."""

    CASES = (
        # (page_size, topk, num_steps, prefix_lens)
        (16, 2, 3, (14, 15, 16)),
        (16, 4, 3, (1, 14, 31, 32)),
        (32, 2, 4, (31, 32, 33)),
        (64, 8, 5, (63, 64, 130)),
        (1, 2, 3, (5, 9)),
    )

    def test_matches_layout_and_kv_indices(self):
        from sglang.srt.layers.attention.trtllm_mha_backend import (
            build_draft_branch_page_tables,
            draft_branch_num_pages,
        )
        from sglang.srt.speculative.triton_ops.cache_locs import (
            generate_draft_decode_kv_indices,
        )
        from sglang.srt.utils import next_power_of_2

        device = "cuda"
        # Row width; multiple of every tested page size and large enough for
        # the widest branch footprint (page 64, topk 8: 128 + 8*64 = 640).
        ctx = 1024
        for page_size, topk, num_steps, prefix_lens in self.CASES:
            with self.subTest(page_size=page_size, topk=topk, steps=num_steps):
                bs = len(prefix_lens)
                pos = torch.arange(ctx, dtype=torch.int32, device=device)
                req_to_token = (
                    page_size
                    + torch.arange(bs, dtype=torch.int32, device=device).view(-1, 1)
                    * ctx
                    + pos.view(1, -1)
                )
                req_pool_indices = torch.arange(bs, dtype=torch.int32, device=device)
                seq_lens = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
                num_pages = draft_branch_num_pages(seq_lens.cpu(), num_steps, page_size)
                page_tables = build_draft_branch_page_tables(
                    req_to_token,
                    req_pool_indices,
                    seq_lens,
                    num_pages,
                    topk,
                    num_steps,
                    page_size,
                ).cpu()

                # (a) direct python re-derivation of the branch layout.
                rows_cpu = req_to_token.cpu()
                for req_idx, prefix_len in enumerate(prefix_lens):
                    last_page = prefix_len % page_size
                    prefix_base = prefix_len - last_page
                    num_prefix_pages = prefix_base // page_size
                    if topk == 1 or page_size == 1:
                        branch_stride = num_steps
                    else:
                        branch_stride = (
                            (last_page + num_steps + page_size - 1) // page_size
                        ) * page_size
                    branch_pages = (last_page + num_steps + page_size - 1) // page_size
                    for branch in range(topk):
                        row = page_tables[req_idx * topk + branch]
                        for col in range(num_prefix_pages):
                            self.assertEqual(
                                int(row[col]),
                                int(rows_cpu[req_idx, col * page_size]) // page_size,
                            )
                        for j in range(branch_pages):
                            position = (
                                prefix_base + branch * branch_stride + j * page_size
                            )
                            self.assertEqual(
                                int(row[num_prefix_pages + j]),
                                int(rows_cpu[req_idx, position]) // page_size,
                            )

                # (b) cross-check against the production kv-indices kernel.
                width = (sum(prefix_lens) + bs * num_steps) * topk + 64
                kv_indices = torch.zeros(
                    (num_steps, width), dtype=torch.int32, device=device
                )
                kv_indptr = torch.zeros(
                    (num_steps, bs * topk + 1), dtype=torch.int32, device=device
                )
                positions = (
                    seq_lens.repeat_interleave(topk).to(torch.int64).contiguous()
                )
                generate_draft_decode_kv_indices[(num_steps, bs, topk)](
                    req_pool_indices,
                    req_to_token,
                    seq_lens,
                    kv_indices,
                    kv_indptr,
                    positions,
                    ctx,
                    width,
                    kv_indptr.shape[1],
                    next_power_of_2(bs),
                    next_power_of_2(num_steps),
                    next_power_of_2(bs * topk),
                    page_size,
                )
                kv_indices_cpu = kv_indices.cpu()
                seq_lens_cpu = [int(s) for s in prefix_lens]
                for step in range(num_steps):
                    iters = step + 1
                    for req_idx, prefix_len in enumerate(prefix_lens):
                        cum_seq_len = sum(seq_lens_cpu[:req_idx])
                        last_page = prefix_len % page_size
                        prefix_base = prefix_len - last_page
                        for branch in range(topk):
                            offset = (
                                cum_seq_len * topk
                                + req_idx * iters * topk
                                + branch * (prefix_len + iters)
                            )
                            branch_kv = kv_indices_cpu[
                                step, offset : offset + prefix_len + iters
                            ]
                            row = page_tables[req_idx * topk + branch]
                            # Full prefix pages must match token-by-token.
                            for j in range(prefix_base):
                                self.assertEqual(
                                    int(branch_kv[j]) // page_size,
                                    int(row[j // page_size]),
                                )
                            # Draft tokens land in the branch's private pages
                            # at the matching in-page offsets.
                            for s in range(iters):
                                token = int(branch_kv[prefix_len + s])
                                page_col = (prefix_base + last_page + s) // page_size
                                self.assertEqual(token // page_size, int(row[page_col]))
                                self.assertEqual(
                                    token % page_size,
                                    (last_page + s) % page_size,
                                )


if __name__ == "__main__":
    unittest.main()
