/* SPDX-License-Identifier: Apache-2.0
 *
 * cuBLASLt NVFP4 GEMM, JIT-compiled.
 *
 * Mirrors TRT-LLM's `CublasLtFP4GemmRunner`
 * (TRT-LLM/cpp/tensorrt_llm/thop/cublasFp4ScaledMM.cpp).  At runtime cuBLASLt
 * fires the NVIDIA-tuned `nvjet_sm100_*_Avec16UE4M3_Bvec16UE4M3` family.
 *
 * CUDA-graph safety:
 *   * `cublasLtHandle_t` and `cublasLtMatmulPreference_t` are per-device
 *     globals, created on first call (warmup), reused forever.
 *   * Per-(m, n, k, out_dtype) algo cache populated by heuristic search on
 *     the first call (warmup); capture-time calls reuse the cached algo.
 *   * Workspace is a `torch::Tensor` passed in from Python — same allocator
 *     domain as the rest of the captured graph.
 *   * `cublasLtMatmul` itself is the only API issued during capture.
 *
 * Inputs (row-major PyTorch tensors):
 *   D[m, n]   bf16 / fp16 / fp32 — output
 *   A[m, k/2] uint8  (FP4 packed, 2 elems per byte)
 *   B[n, k/2] uint8
 *   A_sf      float8_e4m3fn block scales (16-vec swizzled)
 *   B_sf      float8_e4m3fn block scales
 *   alpha     float32 scalar (device pointer)
 *   workspace uint8 tensor, 32 MiB
 */

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

using namespace host;

namespace {

inline void cublas_check(cublasStatus_t status, const char* expr) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuBLASLt error ") + std::to_string(status) + " at: " + expr);
  }
}
#define CUBLAS_CHECK(expr) cublas_check((expr), #expr)

inline cudaDataType_t out_cuda_dtype(DLDataType dtype) {
  if (host::is_type<bf16_t>(dtype)) return CUDA_R_16BF;
  if (host::is_type<fp16_t>(dtype)) return CUDA_R_16F;
  if (host::is_type<float>(dtype)) return CUDA_R_32F;
  throw std::runtime_error("Unsupported output dtype for cublaslt_fp4_gemm");
}

// Per-device cuBLASLt handle. Created on first call.
cublasLtHandle_t get_cublas_handle(int device) {
  static std::mutex mtx;
  static std::unordered_map<int, cublasLtHandle_t> handles;
  std::lock_guard<std::mutex> guard(mtx);
  auto it = handles.find(device);
  if (it == handles.end()) {
    cublasLtHandle_t handle{};
    CUBLAS_CHECK(cublasLtCreate(&handle));
    it = handles.emplace(device, handle).first;
  }
  return it->second;
}

// Per-device zero-valued beta scalar living on the device. cuBLASLt's
// `CUBLASLT_POINTER_MODE_DEVICE` requires BOTH alpha and beta to be device
// pointers; passing a host stack address for beta deferences invalid GPU
// memory inside the kernel and shows up as `cudaErrorLaunchFailure` only
// under async execution (LAUNCH_BLOCKING masks it because some algos
// short-circuit when *beta == 0 on host before the kernel reads device).
// Mirrors TRT-LLM's `getBetaDevicePointer()`.
float const* get_beta_device_ptr(int device) {
  static std::mutex mtx;
  static std::unordered_map<int, float*> betas;
  std::lock_guard<std::mutex> guard(mtx);
  auto it = betas.find(device);
  if (it == betas.end()) {
    int prev = -1;
    cudaGetDevice(&prev);
    cudaSetDevice(device);
    float* d_beta = nullptr;
    if (cudaMalloc(&d_beta, sizeof(float)) != cudaSuccess) {
      cudaSetDevice(prev);
      throw std::runtime_error("cudaMalloc for beta scalar failed");
    }
    cudaMemset(d_beta, 0, sizeof(float));
    cudaSetDevice(prev);
    it = betas.emplace(device, d_beta).first;
  }
  return it->second;
}

// Per-(device, m, n, k, out_dtype) algo cache. The heuristic call is NOT
// stream-capture safe; SGL warmup populates the cache before capture, and
// capture-time calls hit it.
struct AlgoKey {
  int device;
  int m;
  int n;
  int k;
  int out_dtype;
  bool operator==(const AlgoKey& o) const noexcept {
    return device == o.device && m == o.m && n == o.n && k == o.k && out_dtype == o.out_dtype;
  }
};
struct AlgoKeyHash {
  size_t operator()(const AlgoKey& k) const noexcept {
    size_t h = static_cast<size_t>(k.device);
    h = h * 1315423911ull ^ static_cast<size_t>(k.m);
    h = h * 1315423911ull ^ static_cast<size_t>(k.n);
    h = h * 1315423911ull ^ static_cast<size_t>(k.k);
    h = h * 1315423911ull ^ static_cast<size_t>(k.out_dtype);
    return h;
  }
};

// Cached descriptors + algo — kept alive forever so async kernel execution
// can never reference a destroyed cuBLASLt object. cuBLASLt's internal worker
// holds raw pointers to the operation/layout descriptors during its async
// kernel execution; destroying them between cublasLtMatmul return and actual
// kernel completion can corrupt CUDA state on multi-stream workloads (mfest
// as a downstream `cudaErrorLaunchFailure` only without LAUNCH_BLOCKING).
struct ShapeBundle {
  cublasLtMatmulDesc_t op_desc{};
  cublasLtMatrixLayout_t a_layout{};
  cublasLtMatrixLayout_t b_layout{};
  cublasLtMatrixLayout_t c_layout{};
  cublasLtMatrixLayout_t d_layout{};
  cublasLtMatmulAlgo_t algo{};
  bool valid = false;
};

auto& shape_cache_mutex() {
  static std::mutex m;
  return m;
}
auto& shape_cache_map() {
  static std::unordered_map<AlgoKey, ShapeBundle, AlgoKeyHash> c;
  return c;
}

}  // namespace

// ---------------------------------------------------------------------------
// Kernel entry.
// ---------------------------------------------------------------------------
void cublaslt_fp4_gemm_sm100a(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    tvm::ffi::TensorView workspace) {
  RuntimeCheck(A.device().device_type == kDLCUDA, "A must be a CUDA tensor");
  RuntimeCheck(B.device().device_type == kDLCUDA, "B must be a CUDA tensor");

  auto same_device = [](DLDevice x, DLDevice y) {
    return x.device_type == y.device_type && x.device_id == y.device_id;
  };
  RuntimeCheck(same_device(A.device(), B.device()), "A,B same device");
  RuntimeCheck(same_device(A.device(), D.device()), "A,D same device");
  RuntimeCheck(same_device(A.device(), A_sf.device()), "A,A_sf same device");
  RuntimeCheck(same_device(A.device(), B_sf.device()), "A,B_sf same device");
  RuntimeCheck(same_device(A.device(), alpha.device()), "A,alpha same device");
  RuntimeCheck(same_device(A.device(), workspace.device()), "A,workspace same device");

  RuntimeCheck(A.is_contiguous(), "A must be contiguous");
  RuntimeCheck(B.is_contiguous(), "B must be contiguous");
  RuntimeCheck(A_sf.is_contiguous(), "A_sf must be contiguous");
  RuntimeCheck(B_sf.is_contiguous(), "B_sf must be contiguous");
  RuntimeCheck(D.is_contiguous(), "D must be contiguous");

  RuntimeCheck(host::is_type<uint8_t>(A.dtype()), "A must be uint8");
  RuntimeCheck(host::is_type<uint8_t>(B.dtype()), "B must be uint8");
  RuntimeCheck(host::is_type<fp8_e4m3_t>(A_sf.dtype()), "A_sf must be float8_e4m3fn");
  RuntimeCheck(host::is_type<fp8_e4m3_t>(B_sf.dtype()), "B_sf must be float8_e4m3fn");
  RuntimeCheck(host::is_type<float>(alpha.dtype()), "alpha must be float32");
  RuntimeCheck(alpha.numel() == 1, "alpha must be 1 element");
  RuntimeCheck(host::is_type<uint8_t>(workspace.dtype()), "workspace uint8");

  RuntimeCheck(A.dim() == 2 && B.dim() == 2 && D.dim() == 2, "A/B/D 2D");
  RuntimeCheck(A.size(1) == B.size(1), "A.shape[1] == B.shape[1]");

  const int m = static_cast<int>(A.size(0));
  const int n = static_cast<int>(B.size(0));
  const int k = static_cast<int>(A.size(1) * 2);
  RuntimeCheck(D.size(0) == m, "D.shape[0] == m");
  RuntimeCheck(D.size(1) == n, "D.shape[1] == n");

  const cudaDataType_t out_type = out_cuda_dtype(D.dtype());
  cublasLtHandle_t handle = get_cublas_handle(A.device().device_id);
  cudaStream_t stream = static_cast<cudaStream_t>(::TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

  // ---------------------------------------------------------------------
  // Look up cached descriptors + algo for this shape; on miss (first call,
  // must be during SGL warmup BEFORE CUDA-graph capture) build them and run
  // heuristic. The descriptors are kept alive forever so the async
  // cublasLtMatmul kernel never sees a destroyed cuBLASLt object — which
  // we observed manifests as `cudaErrorLaunchFailure` during capture.
  // ---------------------------------------------------------------------
  const AlgoKey key{A.device().device_id, m, n, k, static_cast<int>(out_type)};
  ShapeBundle* bundle = nullptr;
  {
    std::lock_guard<std::mutex> guard(shape_cache_mutex());
    auto it = shape_cache_map().find(key);
    if (it != shape_cache_map().end()) {
      bundle = &it->second;
    }
  }

  if (bundle == nullptr) {
    ShapeBundle b;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&b.op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t trans_a = CUBLAS_OP_T;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(b.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(b.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

    int8_t fast_accum = 0;
    CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(b.op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum)));

    cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        b.op_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));

    cublasLtMatmulMatrixScale_t vec16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulMatrixScale_t scalar = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(b.op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vec16, sizeof(vec16)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(b.op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vec16, sizeof(vec16)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(b.op_desc, CUBLASLT_MATMUL_DESC_C_SCALE_MODE, &scalar, sizeof(scalar)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(b.op_desc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &scalar, sizeof(scalar)));
    CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(b.op_desc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &scalar, sizeof(scalar)));
    void const* null_p = nullptr;
    CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(b.op_desc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &null_p, sizeof(null_p)));
    CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(b.op_desc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &null_p, sizeof(null_p)));
    CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(b.op_desc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &null_p, sizeof(null_p)));

    // For the heuristic we need *some* scale pointer set so cuBLASLt picks an
    // algo compatible with block scales. Use this call's actual scale ptrs;
    // they will be overwritten on every call below before cublasLtMatmul.
    void* a_sf0 = const_cast<void*>(B_sf.data_ptr());
    void* b_sf0 = const_cast<void*>(A_sf.data_ptr());
    CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(b.op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_sf0, sizeof(void*)));
    CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(b.op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_sf0, sizeof(void*)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b.a_layout, CUDA_R_4F_E2M1, k, n, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b.b_layout, CUDA_R_4F_E2M1, k, m, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b.c_layout, out_type, n, m, n));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b.d_layout, out_type, n, m, n));

    cublasLtMatmulPreference_t pref{};
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    uint64_t ws_size = static_cast<uint64_t>(workspace.numel());
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size)));
    uint32_t reduction_mask = CUBLASLT_REDUCTION_SCHEME_MASK;
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction_mask, sizeof(reduction_mask)));

    constexpr int kMaxHeuristics = 32;
    cublasLtMatmulHeuristicResult_t h[kMaxHeuristics];
    int returned = 0;
    cublasStatus_t hstat = cublasLtMatmulAlgoGetHeuristic(
        handle, b.op_desc, b.a_layout, b.b_layout, b.c_layout, b.d_layout, pref, kMaxHeuristics, h, &returned);
    cublasLtMatmulPreferenceDestroy(pref);

    if (hstat != CUBLAS_STATUS_SUCCESS || returned == 0) {
      throw std::runtime_error(
          "cublasLtMatmulAlgoGetHeuristic returned no algo for FP4 GEMM (m=" + std::to_string(m) +
          ", n=" + std::to_string(n) + ", k=" + std::to_string(k) + ")");
    }
    // Prefer smallest workspace among valid algos.
    size_t best_ws = static_cast<size_t>(workspace.numel()) + 1;
    for (int i = 0; i < returned; ++i) {
      if (h[i].state == CUBLAS_STATUS_SUCCESS && h[i].workspaceSize <= static_cast<size_t>(workspace.numel()) &&
          h[i].workspaceSize < best_ws) {
        best_ws = h[i].workspaceSize;
        b.algo = h[i].algo;
        b.valid = true;
        if (best_ws == 0) break;
      }
    }
    if (!b.valid) {
      throw std::runtime_error("No FP4 GEMM algo fit the workspace");
    }

    {
      std::lock_guard<std::mutex> guard(shape_cache_mutex());
      // Re-check (another thread might have populated already)
      auto it = shape_cache_map().find(key);
      if (it == shape_cache_map().end()) {
        bundle = &shape_cache_map().emplace(key, b).first->second;
      } else {
        // Discard our build; another thread won the race
        cublasLtMatmulDescDestroy(b.op_desc);
        cublasLtMatrixLayoutDestroy(b.a_layout);
        cublasLtMatrixLayoutDestroy(b.b_layout);
        cublasLtMatrixLayoutDestroy(b.c_layout);
        cublasLtMatrixLayoutDestroy(b.d_layout);
        bundle = &it->second;
      }
    }
  }

  // Per-call scale pointer update — A_sf/B_sf live with the activation tensor
  // which is freshly allocated each forward, so the pointer changes.
  void* a_sf_ptr_swapped = const_cast<void*>(B_sf.data_ptr());
  void* b_sf_ptr_swapped = const_cast<void*>(A_sf.data_ptr());
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      bundle->op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_sf_ptr_swapped, sizeof(void*)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      bundle->op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_sf_ptr_swapped, sizeof(void*)));

  // beta MUST be a device pointer because `CUBLASLT_POINTER_MODE_DEVICE` is
  // set above for alpha — cuBLASLt applies the same pointer mode to beta.
  float const* beta_device = get_beta_device_ptr(A.device().device_id);
  CUBLAS_CHECK(cublasLtMatmul(
      handle,
      bundle->op_desc,
      const_cast<void*>(alpha.data_ptr()),
      const_cast<void*>(B.data_ptr()),
      bundle->a_layout,
      const_cast<void*>(A.data_ptr()),
      bundle->b_layout,
      beta_device,
      D.data_ptr(),
      bundle->c_layout,
      D.data_ptr(),
      bundle->d_layout,
      &bundle->algo,
      const_cast<void*>(workspace.data_ptr()),
      static_cast<size_t>(workspace.numel()),
      stream));
}

void cublaslt_fp4_gemm(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    tvm::ffi::TensorView workspace) {
  cublaslt_fp4_gemm_sm100a(D, A, B, A_sf, B_sf, alpha, workspace);
}
