import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.server_args import set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.gdn_attention import (
    GDNAttentionCase,
    build_gdn_attention_fixture,
    make_gdn_cases,
    run_gdn_attention_case,
    run_gdn_fixture_eager,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_gdn_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_gdn_eagle_verify_case,
    run_gdn_eagle_verify_cuda_graph_case,
)
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_gdn_split_op_extend_case,
)

register_cuda_ci(est_time=20, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

_cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
_sm_major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
_supports_flashinfer_linear_gdn = _sm_major == 9 or (
    _sm_major == 10 and _cuda_major >= 13
)


@unittest.skipIf(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    "CUDA + flashinfer are required",
)
class TestFlashInferFullAttentionWithTritonGDNCorrectness(CustomTestCase):
    # FlashInfer SM90 prefill kernels require value head dim in {64, 128, 256}.
    HEAD_K_DIM = 64
    HEAD_V_DIM = 64

    CASES = make_gdn_cases("flashinfer")
    CUDA_GRAPH_CASES = (
        GDNAttentionCase(
            name="runner_cuda_graph_gdn_decode_page_boundary",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_k_heads=2,
            num_v_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )
    SPLIT_OP_CASES = (
        (
            GDNAttentionCase(
                name="runner_split_op_gdn_extend_ragged_page_boundary",
                backend="flashinfer",
                forward_mode=ForwardMode.EXTEND,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(0, 8, 16),
                extend_lens=(15, 8, 1),
            ),
            32,
        ),
    )
    EAGLE_VERIFY_CASES = (
        (
            GDNAttentionCase(
                name="runner_eagle_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_eagle_verify_gdn_tree",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_frozen_kv_mtp_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            GDNAttentionCase(
                name="runner_dflash_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            GDNAttentionCase(
                name="runner_ngram_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
        ),
    )
    EAGLE_VERIFY_CUDA_GRAPH_CASES = (
        (
            GDNAttentionCase(
                name="runner_cuda_graph_eagle_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_eagle_verify_gdn_tree",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_frozen_kv_mtp_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_dflash_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_ngram_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
        ),
    )

    def test_projected_gdn_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_gdn_attention_case(
                    self,
                    case,
                    head_k_dim=self.HEAD_K_DIM,
                    head_v_dim=self.HEAD_V_DIM,
                )

    # Layout-robustness. See dense/test_triton.py for the rationale.
    LAYOUT_ROBUSTNESS_CASES = (
        GDNAttentionCase(
            name="layout_gdn_extend_two_request",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_k_heads=4,
            num_v_heads=4,
            page_size=16,
            prefix_lens=(0, 0),
            extend_lens=(16, 16),
        ),
        GDNAttentionCase(
            name="layout_gdn_decode_page_boundary",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_k_heads=4,
            num_v_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )

    def test_layout_robustness_cases(self):
        for case in self.LAYOUT_ROBUSTNESS_CASES:
            for layout in ("interleaved_pages", "non_monotonic_extend"):
                if layout == "non_monotonic_extend" and case.forward_mode.is_decode():
                    continue
                with self.subTest(case=case.name, layout=layout):
                    run_gdn_attention_case(
                        self,
                        case,
                        head_k_dim=self.HEAD_K_DIM,
                        head_v_dim=self.HEAD_V_DIM,
                        loc_layout=layout,
                    )

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_gdn_cuda_graph_decode_case(
                    self,
                    case,
                    head_k_dim=self.HEAD_K_DIM,
                    head_v_dim=self.HEAD_V_DIM,
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
                    run_gdn_split_op_extend_case(
                        self,
                        case,
                        breakable=breakable,
                        static_num_tokens=static_num_tokens,
                        head_k_dim=self.HEAD_K_DIM,
                        head_v_dim=self.HEAD_V_DIM,
                    )

    def test_runner_mode_eagle_verify_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_gdn_eagle_verify_case(
                    self,
                    case,
                    topk=topk,
                    spec_kind=spec_kind,
                    head_k_dim=self.HEAD_K_DIM,
                    head_v_dim=self.HEAD_V_DIM,
                )

    def test_runner_mode_eagle_verify_cuda_graph_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_gdn_eagle_verify_cuda_graph_case(
                    self,
                    case,
                    topk=topk,
                    spec_kind=spec_kind,
                    head_k_dim=self.HEAD_K_DIM,
                    head_v_dim=self.HEAD_V_DIM,
                )


@unittest.skipUnless(
    torch.cuda.is_available()
    and is_flashinfer_available()
    and _supports_flashinfer_linear_gdn,
    "FlashInfer linear GDN requires SM90 or SM100/SM103 with CUDA 13+",
)
class TestFlashInferLinearGDNBackendCorrectness(CustomTestCase):
    # FlashInfer's SM100 GDN prefill kernel requires head size 128. SM90 supports 64.
    HEAD_DIM = 128 if _sm_major == 10 else 64
    PREFILL_CASE = GDNAttentionCase(
        name="flashinfer_gdn_prefill_ragged",
        backend="triton",
        linear_attn_prefill_backend="flashinfer",
        forward_mode=ForwardMode.EXTEND,
        num_k_heads=2,
        num_v_heads=4,
        page_size=16,
        prefix_lens=(3, 7),
        extend_lens=(65, 17),
    )
    CHECKPOINT_CASE = GDNAttentionCase(
        name="flashinfer_gdn_prefill_state_checkpoints",
        backend="triton",
        linear_attn_prefill_backend="flashinfer",
        forward_mode=ForwardMode.EXTEND,
        num_k_heads=2,
        num_v_heads=4,
        page_size=16,
        prefix_lens=(0, 64, 128),
        extend_lens=(64, 65, 129),
    )
    EAGLE_VERIFY_CASES = (
        (
            GDNAttentionCase(
                name="flashinfer_linear_gdn_verify_chain",
                backend="triton",
                linear_attn_prefill_backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
        ),
        (
            GDNAttentionCase(
                name="flashinfer_linear_gdn_verify_tree_triton_fallback",
                backend="triton",
                linear_attn_prefill_backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=4,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
        ),
    )

    def test_prefill_output_and_final_state(self):
        run_gdn_attention_case(
            self,
            self.PREFILL_CASE,
            head_k_dim=self.HEAD_DIM,
            head_v_dim=self.HEAD_DIM,
            max_context_len=128,
        )

    def test_prefill_tracked_state_checkpoints(self):
        fixture = build_gdn_attention_fixture(
            self,
            self.CHECKPOINT_CASE,
            head_k_dim=self.HEAD_DIM,
            head_v_dim=self.HEAD_DIM,
            max_context_len=320,
            runner_batch_size=6,
        )
        set_global_server_args_for_scheduler(fixture.runner.server_args)
        batch = fixture.forward_batch
        # Simulate the tracking metadata produced by the extra-buffer scheduler.
        # This test covers checkpoint mapping and state copies, not scheduler setup.
        batch.mamba_track_mask = torch.ones(3, dtype=torch.bool, device="cuda")
        batch.mamba_track_indices = torch.tensor(
            [4, 5, 6], dtype=torch.int64, device="cuda"
        )
        batch.mamba_track_seqlens = torch.tensor(
            # The final entry selects the second checkpoint at absolute S256.
            [64, 129, 257],
            dtype=torch.int64,
            device="cuda",
        )

        cache = fixture.runner.req_to_token_pool.mamba2_layer_cache(0)
        initial_conv = cache.conv[0].clone()
        initial_ssm = cache.temporal.clone()
        flashinfer_output = run_gdn_fixture_eager(fixture)
        flashinfer_tracked = cache.temporal[batch.mamba_track_indices].clone()

        cache.conv[0].copy_(initial_conv)
        cache.temporal.copy_(initial_ssm)
        fixture.backend.linear_attn_backend.kernel_dispatcher.extend_kernel = (
            TritonGDNKernel()
        )
        triton_output = run_gdn_fixture_eager(fixture)
        triton_tracked = cache.temporal[batch.mamba_track_indices]

        torch.testing.assert_close(
            flashinfer_output, triton_output, atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(
            flashinfer_tracked, triton_tracked, atol=3e-2, rtol=3e-2
        )

    def test_verify_chain_and_tree_fallback(self):
        for case, topk in self.EAGLE_VERIFY_CASES:
            with self.subTest(case=case.name, topk=topk):
                run_gdn_eagle_verify_case(
                    self,
                    case,
                    topk=topk,
                    head_k_dim=self.HEAD_DIM,
                    head_v_dim=self.HEAD_DIM,
                )


if __name__ == "__main__":
    unittest.main()
