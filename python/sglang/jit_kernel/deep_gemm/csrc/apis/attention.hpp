#pragma once

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
#include "../jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/smxx_fp8_mqa_logits.hpp"
#include "../jit_kernels/impls/smxx_fp8_paged_mqa_logits.hpp"
#include "../jit_kernels/impls/smxx_clean_logits.hpp"
#endif

#include "layout.hpp"

namespace deep_gemm::attention {

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
static void fp8_gemm_nt_skip_head_mid(const std::pair<torch::Tensor, torch::Tensor>& a,
                                      const std::pair<torch::Tensor, torch::Tensor>& b,
                                      const torch::Tensor& d,
                                      const std::tuple<int, int, int>& head_splits,
                                      std::optional<std::tuple<int, int, int>> recipe,
                                      const std::string& compiled_dims,
                                      const bool& disable_ue8m0_cast) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto& major_a = get_major_type_ab(a.first);
    const auto& major_b = get_major_type_ab(b.first);
    if (fp8_requires_k_major()) {
        DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    }

    // D must be N-major
    check_major_type_cd(d);

    // Type and shape checks
    const auto& [m , k ] = get_shape<2>(a.first);
    const auto& [n , k_] = get_shape<2>(b.first);
    const auto& [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Check head splits and N
    const auto& [left, mid, right] = head_splits;
    DG_HOST_ASSERT(n % (left + right) == 0 and n_ == n + n / (left + right) * mid);

    // Do nothing if the problem is empty
    if (m == 0)
        return;

    // Transform SFA and SFB into compute-required layout
    const auto& [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, disable_ue8m0_cast);
    DG_HOST_ASSERT(gran_k_a == 128 and gran_k_b == 128);

    // Dispatch into different implements
    const auto& arch_major = device_runtime->get_arch_major();
    const auto& epilogue_type = fmt::format("EpilogueHeadSplits<{}, {}, {}>", left, mid, right);
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat and std::get<1>(recipe.value()) != 1) {
        const auto& major_sfb = get_major_type_ab(sfb);
        sm90_fp8_gemm_1d2d(a.first, sfa, b.first, sfb, std::nullopt, d, m, n, k, major_a, major_b, major_sfb, compiled_dims, epilogue_type);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        // NOTES: Only granularity 128 and FP8 are exposed in the API
        sm100_fp8_fp4_gemm_1d1d(a.first, sfa, b.first, sfb, std::nullopt, d, m, n, k,
                                128, 128, major_a, major_b, compiled_dims, epilogue_type);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static torch::Tensor fp8_mqa_logits(const torch::Tensor& q,
                                    const std::pair<torch::Tensor, torch::Tensor>& kv,
                                    const torch::Tensor& weights,
                                    const torch::Tensor& cu_seq_len_k_start,
                                    const torch::Tensor& cu_seq_len_k_end,
                                    const bool& clean_logits,
                                    const int& max_seqlen_k) {
    const auto& [seq_len, num_heads, head_dim] = get_shape<3>(q);
    const auto& [seq_len_kv, head_dim_] = get_shape<2>(kv.first);
    const auto& [seq_len_, num_heads_] = get_shape<2>(weights);
    const auto& [seq_len_kv_] = get_shape<1>(kv.second);

    DG_HOST_ASSERT(seq_len == seq_len_);
    DG_HOST_ASSERT(num_heads == num_heads_ and head_dim == head_dim_);
    DG_HOST_ASSERT(seq_len_kv == seq_len_kv_);
    DG_HOST_ASSERT(cu_seq_len_k_start.size(0) == seq_len);
    DG_HOST_ASSERT(cu_seq_len_k_end.size(0) == seq_len);

    DG_HOST_ASSERT(q.is_contiguous() and kv.first.is_contiguous());
    DG_HOST_ASSERT(kv.second.is_contiguous());
    DG_HOST_ASSERT(weights.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_start.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_end.is_contiguous());

    DG_HOST_ASSERT(q.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(kv.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(kv.second.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(cu_seq_len_k_end.scalar_type() == torch::kInt);

    constexpr int seq_len_alignment = 4;
    constexpr int block_kv = 256;
    const auto aligned_seq_len = align(seq_len, seq_len_alignment);
    
    torch::Tensor logits;
    int stride_logits;
    if (max_seqlen_k == 0) {
        stride_logits = align(seq_len_kv + block_kv, 4);
        logits = torch::empty({aligned_seq_len, stride_logits}, q.options().dtype(torch::kFloat));
        logits = logits.index({torch::indexing::Slice(0, seq_len), torch::indexing::Slice(0, seq_len_kv)});
    } else {
        stride_logits = align(max_seqlen_k, block_kv);
        logits = torch::empty({aligned_seq_len, stride_logits}, q.options().dtype(torch::kFloat));
        logits = logits.index({torch::indexing::Slice(0, seq_len), torch::indexing::Slice(0, max_seqlen_k)});
        DG_HOST_ASSERT(not clean_logits);
    }

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 or arch_major == 10) {
        smxx_fp8_mqa_logits(q, kv.first, kv.second, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits,
                            seq_len, seq_len_kv, max_seqlen_k, stride_logits, num_heads, head_dim, seq_len_alignment);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Clean unfilled logits
    if (clean_logits)
        smxx_clean_logits(logits, cu_seq_len_k_start, cu_seq_len_k_end, 1, seq_len, seq_len_kv, stride_logits);
    return logits;
}

static torch::Tensor get_paged_mqa_logits_metadata(const torch::Tensor& context_lens, int block_kv, int num_sms) {
    const bool is_context_lens_2d = context_lens.dim() == 2;
    int batch_size = 0, next_n = 0;
    if (is_context_lens_2d) {
        batch_size = context_lens.size(0);
        next_n = context_lens.size(1);
    } else {
        DG_HOST_ASSERT(context_lens.dim() == 1);
        batch_size = context_lens.size(0);
    }
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(context_lens.is_contiguous());

    auto schedule_metadata = torch::empty({num_sms + 1, 2}, context_lens.options());

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 or arch_major == 10) {
        smxx_paged_mqa_logits_metadata(context_lens, schedule_metadata, batch_size, next_n, block_kv, num_sms, is_context_lens_2d);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    return schedule_metadata;
}

static torch::Tensor fp8_paged_mqa_logits(const torch::Tensor& q,
                                          const torch::Tensor& fused_kv_cache,
                                          const torch::Tensor& weights,
                                          const torch::Tensor& context_lens,
                                          const torch::Tensor& block_table,
                                          const torch::Tensor& schedule_meta,
                                          const int& max_context_len,
                                          const bool& clean_logits) {
    const auto& [batch_size, next_n, num_heads, head_dim] = get_shape<4>(q);
    const auto& [num_kv_blocks, block_kv, num_heads_kv, head_dim_with_sf] = get_shape<4>(fused_kv_cache);
    const auto& [batch_size_next_n, num_heads_] = get_shape<2>(weights);
    const auto& [batch_size_, max_block_len] = get_shape<2>(block_table);
    const auto& [schedule_meta_size, meta_info_size] = get_shape<2>(schedule_meta);
    const auto& num_sms = device_runtime->get_num_sms();
    const auto& kv_cache_stride_bytes = fused_kv_cache.stride(0);
    const auto& block_table_stride = block_table.stride(0);

    const bool is_context_lens_2d = context_lens.dim() == 2;
    if (is_context_lens_2d) {
        const auto& [batch_size__, next_n_] = get_shape<2>(context_lens);
        DG_HOST_ASSERT(batch_size == batch_size__ and next_n == next_n_);
    } else {
        DG_HOST_ASSERT(context_lens.dim() == 1);
        const auto& [batch_size__] = get_shape<1>(context_lens);
        DG_HOST_ASSERT(batch_size == batch_size__);
    }

    DG_HOST_ASSERT(batch_size == batch_size_);
    DG_HOST_ASSERT(batch_size_next_n == batch_size * next_n);
    DG_HOST_ASSERT(num_heads == num_heads_ and num_heads_kv == 1);
    DG_HOST_ASSERT(head_dim_with_sf == head_dim + static_cast<int>(sizeof(float)));
    DG_HOST_ASSERT(schedule_meta_size == num_sms + 1 and meta_info_size == 2);

    DG_HOST_ASSERT(next_n == 1 or next_n == 2);
    DG_HOST_ASSERT(block_kv == 64);

    DG_HOST_ASSERT(q.is_contiguous());
    DG_HOST_ASSERT(kv_cache_stride_bytes % sizeof(float) == 0);
    DG_HOST_ASSERT(fused_kv_cache.stride(1) == head_dim_with_sf);
    DG_HOST_ASSERT(fused_kv_cache.stride(2) == head_dim_with_sf);
    DG_HOST_ASSERT(fused_kv_cache.stride(3) == 1);
    DG_HOST_ASSERT(weights.is_contiguous());
    DG_HOST_ASSERT(context_lens.is_contiguous());
    DG_HOST_ASSERT(block_table.stride(1) == 1);
    DG_HOST_ASSERT(schedule_meta.is_contiguous());

    DG_HOST_ASSERT(q.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::kByte);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(schedule_meta.scalar_type() == torch::kInt);

    // Derive FP8 values and SF tensor from KV cache
    const auto& kv_cache = torch::from_blob(
        fused_kv_cache.data_ptr(),
        {num_kv_blocks, block_kv, head_dim},
        {kv_cache_stride_bytes, head_dim, 1},
        torch::TensorOptions().dtype(torch::kFloat8_e4m3fn)
    );
    const auto& kv_cache_scales = torch::from_blob(
        fused_kv_cache.data_ptr<uint8_t>() + block_kv * head_dim,
        {num_kv_blocks, block_kv},
        {kv_cache_stride_bytes / static_cast<int>(sizeof(float)), 1},
        torch::TensorOptions().dtype(torch::kFloat32)
    );

    // Allocate output
    constexpr int split_kv = 256;
    const auto& aligned_max_context_len = align(max_context_len, split_kv);
    auto logits = torch::empty({batch_size * next_n, aligned_max_context_len}, q.options().dtype(torch::kFloat));
    logits = logits.slice(-1, 0, max_context_len);

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 or arch_major == 10) {
        smxx_fp8_paged_mqa_logits(q, kv_cache, kv_cache_scales, weights, context_lens, logits, block_table, schedule_meta,
                                  batch_size, next_n, num_heads, head_dim, num_kv_blocks, block_kv, is_context_lens_2d,
                                  kv_cache_stride_bytes, aligned_max_context_len, block_table_stride, num_sms, split_kv);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Clean unfilled logits
    if (clean_logits) {
        DG_HOST_ASSERT(not is_context_lens_2d);
        smxx_clean_logits(logits, std::nullopt, context_lens, next_n, batch_size * next_n, max_context_len, aligned_max_context_len);
    }
    return logits;
}

#endif

static void register_apis(pybind11::module_& m) {
#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    m.def("fp8_gemm_nt_skip_head_mid", &fp8_gemm_nt_skip_head_mid,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("head_splits"),
          py::arg("recipe") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false);
    m.def("fp8_mqa_logits", &fp8_mqa_logits,
          py::arg("q"), py::arg("kv"), py::arg("weights"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"),
          py::arg("clean_logits") = true,
          py::arg("max_seqlen_k") = 0);
    m.def("get_paged_mqa_logits_metadata", &get_paged_mqa_logits_metadata,
          py::arg("context_lens"), py::arg("block_kv"), py::arg("num_sms"));
    m.def("fp8_paged_mqa_logits", &fp8_paged_mqa_logits,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
          py::arg("max_context_len"), py::arg("clean_logits") = false);
#endif
}

} // namespace deep_gemm::attention
