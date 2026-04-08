"""Unit tests for torch-profiler analysis helper scripts."""

import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_sglang_llm_torch_profile as llm  # noqa: E402
import analyze_sglang_profiler_overlap as overlap  # noqa: E402
import analyze_sglang_torch_profile as triage  # noqa: E402

CUTLASS_FP8_GEMM = (
    "_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalIN4cute5tupleIJiiiiEEENS1_10collective"
    "13CollectiveMmaINS1_35MainloopSm100TmaUmmaWarpSpecializedILi13ELi2ELi4ENS5_IJNS4_1CILi1EEE"
    "NSA_ILi4EEESB_EEEEENS5_IJNSA_ILi64EEESF_NSA_ILi128EEEEEENS_12float_e4m3_tENS5_IJlSB_lEEE"
    "SI_SJ_NS4_8TiledMMAINS4_8MMA_AtomIJNS4_10MMA_TraitsINS4_19SM100_MMA_F8F6F4_SSEJSI_SI_fSF_SF_"
    "NS4_17integral_constantINS4_4UMMA5MajorELSQ_0EEESR_NSO_INSP_7ScaleInELSS_0EEEST_EEEEEENS4_6Lay"
    "outINS5_IJSB_SB_SB_EEENS5_IJNSA_ILi0EEESY_SY_EEEEENS5_IJNS4_10UnderscoreES11_S11_EEEEENS4_23S"
    "M90_TMA_LOAD_MULTICASTENS4_14ComposedLayoutINS4_7SwizzleILi3ELi4ELi3EEENS4_18smem_ptr_flag_bi"
    "tsILi8EEENSW_INS5_IJNSA_ILi8EEESG_EEENS5_IJSG_SB_EEEEEEEvNS4_8identityENS4_13SM90_TMA_LOADES1"
    "E_vS1F_EENS_8epilogue10collective18CollectiveEpilogueINS1I_23Sm100TmaWarpSpecializedILi1ELi1E"
    "Li32ELb0ELb1EEEJSH_NS5_IJNSW_ISF_SB_EES1N_EEEvSJ_NS_6half_tESJ_NS1I_6fusion15Sm90TreeVisitorIN"
    "S1Q_11Sm90ComputeINS_10multipliesES1P_fLNS_15FloatRoundStyleE2EvEEJNS1Q_16Sm90ColBroadcastILi0"
    "ESH_ffNS5_IJSB_SY_SY_EEELi4ELb1EEENS1R_INS1S_IS1T_ffLS1U_2EvEEJNS1Q_16Sm90RowBroadcastILi0ESH_f"
    "fNS5_IJSY_SB_SY_EEELi4ELb1EEENS1Q_12Sm90AccFetchEEEEEEENS4_5SM1004TMEM4LOAD26SM100_TMEM_LOAD_1"
    "6dp256b8xES1G_NS15_IS17_NS18_ILi16EEENSW_INS5_IJS1A_SF_EEENS5_IJSF_SB_EEEEEEENS4_17SM75_U32x4_"
    "LDSM_NENS4_14SM90_TMA_STOREES2E_NS4_17SM90_U32x4_STSM_NENS4_39AutoVectorizingCopyWithAssumedAl"
    "ignmentILi128EEEEEEvvEEEEvNT_6ParamsE"
)

FLOOR_ELEMENTWISE = (
    "void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, "
    "at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::"
    "operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*"
    ", 2ul> >(int, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor"
    "_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operat"
    "or()() const::{lambda(int, int)#1}>, std::array<char*, 2ul>)"
)

COPY_KERNEL = (
    "void at::native::vectorized_elementwise_kernel<8, at::native::float16_copy_kernel>"
)

FLASHINFER_ACT = "void flashinfer::activation::act_and_mul_kernel<__half, &"

FLASHINFER_NORM = "void flashinfer::norm::FusedAddRMSNormKernel<8u, __half>"

GPU_USER_ANNOTATION = {
    "ph": "X",
    "cat": "gpu_user_annotation",
    "name": "## Call CompiledFxGraph deadbeef ##",
    "ts": 1.0,
    "dur": 2.0,
    "args": {"External id": 7},
}

CORRELATION_EXTERNAL_TRACE = {
    "traceEvents": [
        {
            "ph": "X",
            "cat": "python_function",
            "name": "/tmp/worktrees/sglang/python/sglang/srt/layers/quantization/fp8_utils.py(1341): apply_fp8_linear",
            "pid": "11",
            "tid": "11",
            "ts": 0.0,
            "dur": 100.0,
            "args": {"Python id": 1, "Python parent id": None},
        },
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "sgl_kernel::fp8_scaled_mm",
            "pid": "11",
            "tid": "11",
            "ts": 48.0,
            "dur": 14.0,
            "args": {"External id": 7},
        },
        {
            "ph": "X",
            "cat": "cuda_runtime",
            "name": "cudaLaunchKernelExC",
            "pid": "11",
            "tid": "11",
            "ts": 50.0,
            "dur": 3.0,
            "args": {"External id": 7, "correlation": 42},
        },
        {
            "ph": "X",
            "cat": "kernel",
            "name": "_static_quant_fp8",
            "pid": "0",
            "tid": "13",
            "ts": 60.0,
            "dur": 2.0,
            "args": {"correlation": 42, "stream": 13},
        },
    ]
}

LAUNCH_FALLBACK_TRACE = {
    "traceEvents": [
        {
            "ph": "X",
            "cat": "python_function",
            "name": "/tmp/worktrees/sglang/python/sglang/srt/layers/attention/trtllm_mha_backend.py(695): forward_decode",
            "pid": "21",
            "tid": "21",
            "ts": 10.0,
            "dur": 60.0,
            "args": {"Python id": 1, "Python parent id": None},
        },
        {
            "ph": "X",
            "cat": "cuda_driver",
            "name": "cuLaunchKernelEx",
            "pid": "21",
            "tid": "21",
            "ts": 40.0,
            "dur": 4.0,
            "args": {"correlation": 99},
        },
        {
            "ph": "X",
            "cat": "kernel",
            "name": "fmhaSm100fKernel_QkvFp16OFp16H128PagedKvCausalP64MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen",
            "pid": "0",
            "tid": "13",
            "ts": 55.0,
            "dur": 8.0,
            "args": {"correlation": 99, "stream": 13},
        },
    ]
}


class TestKernelClassification(unittest.TestCase):
    def test_llm_breakdown_classifies_cutlass_fp8_linear_as_gemm(self):
        self.assertEqual(llm.classify_kernel(CUTLASS_FP8_GEMM), "gemm")

    def test_llm_breakdown_classifies_floor_kernel_as_elementwise(self):
        self.assertEqual(llm.classify_kernel(FLOOR_ELEMENTWISE), "elementwise")

    def test_llm_breakdown_classifies_copy_kernel_as_memory(self):
        self.assertEqual(llm.classify_kernel(COPY_KERNEL), "memory")

    def test_llm_breakdown_preserves_flashinfer_activation_and_norm(self):
        self.assertEqual(llm.classify_kernel(FLASHINFER_ACT), "activation")
        self.assertEqual(llm.classify_kernel(FLASHINFER_NORM), "norm")

    def test_overlap_classifies_cutlass_fp8_linear_as_compute(self):
        self.assertEqual(overlap.classify_kernel(CUTLASS_FP8_GEMM), "compute")

    def test_overlap_classifies_floor_kernel_as_elementwise(self):
        self.assertEqual(overlap.classify_kernel(FLOOR_ELEMENTWISE), "elementwise")

    def test_llm_breakdown_ignores_gpu_user_annotation(self):
        self.assertFalse(llm.is_gpu_kernel_event(GPU_USER_ANNOTATION))

    def test_overlap_ignores_gpu_user_annotation(self):
        self.assertFalse(overlap.is_kernel_event(GPU_USER_ANNOTATION))

    def test_breakdown_recovers_external_id_from_correlation(self):
        kernels, cpu_ops, python_frames, launch_events, _, _ = llm.extract_trace_data(
            CORRELATION_EXTERNAL_TRACE
        )
        self.assertEqual(kernels[0].external_id, 7)
        site_stats = llm.aggregate_kernel_sites(
            kernels,
            llm.build_cpu_op_index(cpu_ops),
            python_frames,
            launches_by_correlation=llm.build_launch_index(launch_events),
        )
        sites = site_stats[kernels[0].canonical_name]
        self.assertIn(
            "python/sglang/srt/layers/quantization/fp8_utils.py:1341 apply_fp8_linear",
            sites,
        )
        self.assertEqual(
            sites[
                "python/sglang/srt/layers/quantization/fp8_utils.py:1341 apply_fp8_linear"
            ].cpu_ops.most_common(1)[0][0],
            "sgl_kernel::fp8_scaled_mm",
        )

    def test_breakdown_falls_back_to_launch_scope_when_cpu_op_is_missing(self):
        kernels, cpu_ops, python_frames, launch_events, _, _ = llm.extract_trace_data(
            LAUNCH_FALLBACK_TRACE
        )
        self.assertIsNone(kernels[0].external_id)
        site_stats = llm.aggregate_kernel_sites(
            kernels,
            llm.build_cpu_op_index(cpu_ops),
            python_frames,
            launches_by_correlation=llm.build_launch_index(launch_events),
        )
        sites = site_stats[kernels[0].canonical_name]
        self.assertIn(
            "python/sglang/srt/layers/attention/trtllm_mha_backend.py:695 forward_decode",
            sites,
        )
        self.assertEqual(
            sites[
                "python/sglang/srt/layers/attention/trtllm_mha_backend.py:695 forward_decode"
            ].cpu_ops.most_common(1)[0][0],
            "cuLaunchKernelEx",
        )

    def test_best_site_summary_labels_within_kernel_site_share(self):
        location, cpu_op = llm.best_site_summary(
            {
                "sites": [
                    {
                        "location": (
                            "python/sglang/srt/layers/quantization/fp8_utils.py:1341 "
                            "apply_fp8_linear"
                        ),
                        "display_location": (
                            "python/sglang/srt/layers/quantization/fp8_utils.py:1341 "
                            "apply_fp8_linear"
                        ),
                        "share_pct_within_kernel": 63.2,
                        "top_cpu_op": "sgl_kernel::fp8_scaled_mm",
                    },
                    {
                        "location": "python/sglang/jit_kernel/rope.py:179 apply_rope",
                        "display_location": (
                            "python/sglang/jit_kernel/rope.py:179 apply_rope"
                        ),
                        "share_pct_within_kernel": 36.8,
                        "top_cpu_op": "sglang::apply_rope_inplace",
                    },
                ]
            }
        )
        self.assertEqual(
            location,
            "python/sglang/srt/layers/quantization/fp8_utils.py:1341 apply_fp8_linear "
            "(site share 63%)<br>python/sglang/jit_kernel/rope.py:179 apply_rope "
            "(site share 37%)",
        )
        self.assertEqual(
            cpu_op,
            "sgl_kernel::fp8_scaled_mm<br>sglang::apply_rope_inplace",
        )

    def test_breakdown_normalizes_arbitrary_repo_prefix_from_source_locations(self):
        self.assertEqual(
            llm.normalize_source_location(
                "/mnt/random/worktrees/feature-branch/sglang/python/sglang/srt/models/qwen3_5.py(766): _apply_qk_norm"
            ),
            "python/sglang/srt/models/qwen3_5.py:766 _apply_qk_norm",
        )

    def test_overlap_normalizes_arbitrary_repo_prefix_from_python_scope(self):
        self.assertEqual(
            overlap.canonicalize_python_scope_name(
                "/tmp/custom-root/another-worktree/sglang/python/sglang/srt/layers/layernorm.py(89): _forward_with_allreduce_fusion"
            ),
            "python/sglang/srt/layers/layernorm.py(89): _forward_with_allreduce_fusion",
        )

    def test_breakdown_keeps_anonymous_namespace_kernel_name(self):
        self.assertEqual(
            llm.canonicalize_name(
                "void (anonymous namespace)::copy_blocks_kernel(int, int)"
            ),
            "void (anonymous namespace)::copy_blocks_kernel",
        )

    def test_breakdown_preserves_full_template_kernel_name(self):
        kernel_name = (
            "void at::native::elementwise_kernel<128, 4, "
            "at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<"
            "c10::BFloat16, c10::BFloat16, c10::BFloat16, "
            "at::native::binary_internal::MulFunctor<c10::BFloat16>>, "
            "at::detail::Array<char*, 3>>>(at::TensorIteratorBase&, int)"
        )
        canonical = llm.canonicalize_name(kernel_name)
        self.assertNotIn("<...>", canonical)
        self.assertIn("binary_internal::MulFunctor<c10::BFloat16>", canonical)

    def test_overlap_preserves_full_template_kernel_name(self):
        kernel_name = (
            "void (anonymous namespace)::store_kvcache<2048l, 4, true, long>(long*, "
            "long*, int)"
        )
        canonical = overlap.canonicalize_name(kernel_name)
        self.assertEqual(
            canonical,
            "void (anonymous namespace)::store_kvcache<2048l, 4, true, long>",
        )

    def test_fuse_location_summary_formats_function_first(self):
        self.assertEqual(
            llm.summarize_locations(
                [
                    "python/sglang/srt/models/qwen3_5.py:766 _apply_qk_norm",
                    "python/sglang/srt/mem_cache/memory_pool.py:86 _set_kv_buffer_impl",
                ]
            ),
            "_apply_qk_norm @ python/sglang/srt/models/qwen3_5.py:766"
            "<br>_set_kv_buffer_impl @ python/sglang/srt/mem_cache/memory_pool.py:86",
        )

    def test_fuse_evidence_preserves_full_kernel_names(self):
        kernel_name = (
            "void at::native::elementwise_kernel<128, 4, "
            "at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<"
            "c10::BFloat16, c10::BFloat16, c10::BFloat16, "
            "at::native::binary_internal::MulFunctor<c10::BFloat16>>>>"
        )
        summary = llm.summarize_evidence(
            [
                llm.KernelRow(
                    name=kernel_name,
                    category="elementwise",
                    aggregate=llm.Aggregate(total_us=297.7, count=4),
                    location="python/sglang/srt/models/qwen3_5.py:766 _apply_qk_norm",
                    cpu_op="aten::mul",
                    entry=None,
                )
            ],
            total_us=1000.0,
        )
        self.assertIn(kernel_name, summary)
        self.assertNotIn("<...>", summary)

    def test_triage_tables_preserve_full_kernel_and_scope_text(self):
        long_kernel = (
            "void extremely_long_kernel_name_with_many_template_arguments_and_suffixes_"
            "that_should_remain_fully_visible_in_markdown_tables_for_review"
        )
        long_scope = (
            "python/sglang/srt/models/qwen3_really_long_module_name.py:1234 "
            "forward_decode_with_detailed_scope_information_and_callsite_context"
        )

        kernel_lines = triage.render_kernel_table(
            [
                {
                    "stage": "decode",
                    "kernel": long_kernel,
                    "category": "gemm",
                    "total_us": 297.7,
                    "share_pct": 9.6,
                    "launches": 4,
                    "location": f"{long_scope} (site share 100%)",
                    "cpu_op": "sgl_kernel::fp8_scaled_mm",
                }
            ]
        )
        self.assertIn(long_kernel, kernel_lines[2])
        self.assertIn(long_scope, kernel_lines[2])

        overlap_lines = triage.render_overlap_table(
            [
                {
                    "stage": "decode",
                    "priority": "P1",
                    "verdict": "actionable",
                    "kernel": long_kernel,
                    "python_scope": long_scope,
                    "total_us": 297.7,
                    "share_pct": 9.6,
                    "exclusive_ratio": 1.0,
                    "hidden_ratio": 0.0,
                    "dependency_signal": "adjacency unclear",
                    "recommendation": "inspect code path",
                }
            ]
        )
        self.assertIn(long_kernel, overlap_lines[2])
        self.assertIn(long_scope, overlap_lines[2])

    def test_overlap_action_table_preserves_full_kernel_and_scope_text(self):
        row = overlap.ActionRow(
            priority="P1",
            verdict="actionable",
            kernel=(
                "void another_extremely_long_kernel_name_for_overlap_action_table_"
                "that_should_not_be_shortened_when_rendered"
            ),
            category="compute",
            total_us=297.7,
            share_pct=9.6,
            exclusive_ratio=1.0,
            hidden_ratio=0.0,
            python_scope=(
                "python/sglang/srt/layers/some_deeply_nested_module.py:567 "
                "launch_kernel_from_a_scope_that_should_stay_complete"
            ),
            launch_op="cudaLaunchKernelExC",
            mapping_ratio=1.0,
            dependency_signal="adjacency unclear",
            prev_neighbor="unmapped",
            next_neighbor="unmapped",
            recommendation="inspect overlap window",
            suggestion="investigate",
            representative_idx=0,
        )
        lines = overlap.render_action_table([row])
        self.assertIn(row.kernel, lines[2])
        self.assertIn(row.python_scope, lines[2])


if __name__ == "__main__":
    unittest.main()
