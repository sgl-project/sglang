// Torch binding for the MLA decode stage-1 CUDA/HIP kernels, used ONLY by
// bench_stage1_triton_vs_cuda.py (JIT-built via torch.utils.cpp_extension.load).
//
// Two kernels, two sources (they can't share a TU — each uses its own anonymous
// namespace for BLOCK_H / f32x4 / ...):
//   - bf16 KV: decode_grouped_attention_mla_stage1.cu, #included here.
//   - fp8  KV: decode_grouped_attention_mla_stage1_fp8.cu, compiled separately
//              and exposed via the forward-declared launcher below. Add it to the
//              load() sources list.

#include <c10/cuda/CUDAStream.h>
#include <hip/hip_fp8.h>
#include <torch/extension.h>

#include "../../../sgl-kernel/csrc/attention/decode_grouped_attention_mla_stage1.cu"

// Defined in decode_grouped_attention_mla_stage1_fp8.cu (separate TU).
void launch_fwd_grouped_stage1_mla_fp8(
    const __hip_bfloat16* Q, const __hip_fp8_e4m3* K_Buffer, float sm_scale_withk,
    const int* kv_indptr, const int* kv_indices, float* Att_Out, float* Att_Lse,
    const int* num_kv_splits, long stride_qbs, long stride_qh, long stride_buf_kbs,
    long stride_buf_kh, long stride_mid_ob, long stride_mid_oh, long stride_mid_os,
    int batch, int q_head_num, int kv_group_num, int max_kv_splits, int Lv,
    hipStream_t stream);

static void common_checks(torch::Tensor q, torch::Tensor att_out,
                          torch::Tensor att_lse) {
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bf16");
  TORCH_CHECK(att_out.scalar_type() == at::kFloat, "att_out must be fp32");
  TORCH_CHECK(att_lse.scalar_type() == at::kFloat, "att_lse must be fp32");
  TORCH_CHECK(att_lse.is_contiguous(), "att_lse must be contiguous");
}

void stage1_mla(
    torch::Tensor q, torch::Tensor k_buffer, torch::Tensor att_out,
    torch::Tensor att_lse, torch::Tensor kv_indptr, torch::Tensor kv_indices,
    torch::Tensor num_kv_splits, double sm_scale, int64_t max_kv_splits) {
  common_checks(q, att_out, att_lse);
  TORCH_CHECK(k_buffer.scalar_type() == at::kBFloat16, "k_buffer must be bf16");
  const int q_head_num = q.size(1);
  const int kv_group_num = q_head_num / k_buffer.size(1);
  launch_fwd_grouped_stage1_mla(
      reinterpret_cast<const __hip_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __hip_bfloat16*>(k_buffer.data_ptr()),
      static_cast<float>(sm_scale), kv_indptr.data_ptr<int>(),
      kv_indices.data_ptr<int>(), att_out.data_ptr<float>(),
      att_lse.data_ptr<float>(), num_kv_splits.data_ptr<int>(), q.stride(0),
      q.stride(1), k_buffer.stride(0), k_buffer.stride(1), att_out.stride(0),
      att_out.stride(1), att_out.stride(2), q.size(0), q_head_num, kv_group_num,
      static_cast<int>(max_kv_splits), att_out.size(3),
      at::cuda::getCurrentCUDAStream().stream());
}

// fp8 (e4m3) KV cache. sm_scale must already include k_scale (Triton folds it as
// sm_scale * k_scale); v_scale is applied in stage 2. Q is quantized to fp8.
void stage1_mla_fp8(
    torch::Tensor q, torch::Tensor k_buffer, torch::Tensor att_out,
    torch::Tensor att_lse, torch::Tensor kv_indptr, torch::Tensor kv_indices,
    torch::Tensor num_kv_splits, double sm_scale, int64_t max_kv_splits) {
  common_checks(q, att_out, att_lse);
  TORCH_CHECK(k_buffer.scalar_type() == at::kFloat8_e4m3fn,
              "k_buffer must be float8_e4m3fn");
  const int q_head_num = q.size(1);
  const int kv_group_num = q_head_num / k_buffer.size(1);
  launch_fwd_grouped_stage1_mla_fp8(
      reinterpret_cast<const __hip_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __hip_fp8_e4m3*>(k_buffer.data_ptr()),
      static_cast<float>(sm_scale), kv_indptr.data_ptr<int>(),
      kv_indices.data_ptr<int>(), att_out.data_ptr<float>(),
      att_lse.data_ptr<float>(), num_kv_splits.data_ptr<int>(), q.stride(0),
      q.stride(1), k_buffer.stride(0), k_buffer.stride(1), att_out.stride(0),
      att_out.stride(1), att_out.stride(2), q.size(0), q_head_num, kv_group_num,
      static_cast<int>(max_kv_splits), att_out.size(3),
      at::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("stage1_mla", &stage1_mla, "MLA decode stage-1 (bf16 KV)");
  m.def("stage1_mla_fp8", &stage1_mla_fp8, "MLA decode stage-1 (fp8 e4m3 KV, native fp8 MFMA)");
}
