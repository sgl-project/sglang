#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <cublasLt.h>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace sglang {

// BF16 out[M, N] = x[M, K] @ w[N, K]^T (the F.linear layout) through cuBLASLt with a
// caller-chosen algorithm: the opaque cublasLtMatmulAlgo_t (8 x uint64) recorded by
// the offline tuner on the same cuBLASLt build and GPU. fp32 compute and scale, like
// torch.mm on BF16.
struct CublasLtPinnedGemm {
  static constexpr int64_t kAlgoWords = 8;
  // enumerate() row: the algo words, then its workspace bytes and split-K count.
  static constexpr int64_t kRowWords = kAlgoWords + 2;

  static_assert(sizeof(cublasLtMatmulAlgo_t) == kAlgoWords * sizeof(uint64_t));

  static void check(cublasStatus_t status, const char* what) {
    host::RuntimeCheck(status == CUBLAS_STATUS_SUCCESS, "cuBLASLt ", what, " failed with status ", int(status));
  }

  static cublasLtHandle_t handle() {
    static cublasLtHandle_t h = [] {
      cublasLtHandle_t created = nullptr;
      check(cublasLtCreate(&created), "cublasLtCreate");
      return created;
    }();
    return h;
  }

  // Column-major view of the row-major problem: out^T[N, M] = w^T[N, K] * x^T[K, M].
  struct Problem {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;

    Problem(int64_t m, int64_t n, int64_t k, int64_t ldw, int64_t ldx, int64_t ldo) {
      check(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "MatmulDescCreate");
      const cublasOperation_t trans_a = CUBLAS_OP_T, trans_b = CUBLAS_OP_N;
      check(
          cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)),
          "MatmulDescSetAttribute");
      check(
          cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)),
          "MatmulDescSetAttribute");
      check(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, k, n, ldw), "MatrixLayoutCreate");
      check(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, m, ldx), "MatrixLayoutCreate");
      check(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, n, m, ldo), "MatrixLayoutCreate");
    }
    ~Problem() {
      cublasLtMatrixLayoutDestroy(a);
      cublasLtMatrixLayoutDestroy(b);
      cublasLtMatrixLayoutDestroy(c);
      cublasLtMatmulDescDestroy(op);
    }
    Problem(const Problem&) = delete;
    Problem& operator=(const Problem&) = delete;
  };

  // Descriptors depend only on the shape and leading dimensions, not on the device or
  // the data, so one set per (m, n, k, ldw, ldx, ldo) serves every call; the table
  // has a few dozen shapes. Created once instead of 4 creates + 4 destroys per GEMM.
  static const Problem& cached_problem(int64_t m, int64_t n, int64_t k, int64_t ldw, int64_t ldx, int64_t ldo) {
    // Never destroyed: process exit may already have unloaded cuBLASLt.
    static auto* mutex = new std::mutex();
    static auto* cache = new std::map<std::array<int64_t, 6>, std::unique_ptr<Problem>>();
    const std::array<int64_t, 6> key{m, n, k, ldw, ldx, ldo};
    std::lock_guard<std::mutex> lock(*mutex);
    auto& slot = (*cache)[key];
    if (!slot) {
      slot = std::make_unique<Problem>(m, n, k, ldw, ldx, ldo);
    }
    return *slot;
  }

  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView algo,
      const tvm::ffi::TensorView workspace) {
    using namespace host;

    auto M = SymbolicSize{"num_tokens"};
    auto N = SymbolicSize{"out_features"};
    auto K = SymbolicSize{"in_features"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(w);
    TensorMatcher({M, N}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(out);
    TensorMatcher({kAlgoWords}).with_dtype<int64_t>().with_device<kDLCPU>().verify(algo);
    TensorMatcher({-1}).with_dtype<uint8_t>().with_device(device).verify(workspace);

    cublasLtMatmulAlgo_t pinned;
    std::memcpy(&pinned, algo.data_ptr(), sizeof(pinned));
    const Problem& problem =
        cached_problem(M.unwrap(), N.unwrap(), K.unwrap(), w.stride(0), x.stride(0), out.stride(0));
    const float alpha = 1.0f, beta = 0.0f;
    const auto ws_bytes = static_cast<size_t>(workspace.size(0));
    check(
        cublasLtMatmul(
            handle(),
            problem.op,
            &alpha,
            w.data_ptr(),
            problem.a,
            x.data_ptr(),
            problem.b,
            &beta,
            out.data_ptr(),
            problem.c,
            out.data_ptr(),
            problem.c,
            &pinned,
            ws_bytes > 0 ? workspace.data_ptr() : nullptr,
            ws_bytes,
            LaunchKernel::resolve_device(device.unwrap())),
        "Matmul");
  }

  static int64_t version() {
    return static_cast<int64_t>(cublasLtGetVersion());
  }

  // Every algorithm cuBLASLt accepts for the packed (m, n, k) problem with split-K
  // off, over all tile, stage, swizzle and custom-option values, plus the
  // heuristic's own top choices (which may use split-K), within `max_ws_bytes` of
  // workspace. Writes up to rows.size(0) rows of kRowWords int64 to the CPU tensor
  // and returns the number of candidates found (which can exceed the rows written).
  static int64_t enumerate(int64_t m, int64_t n, int64_t k, int64_t max_ws_bytes, const tvm::ffi::TensorView rows) {
    using namespace host;

    auto capacity = SymbolicSize{"capacity"};
    TensorMatcher({capacity, kRowWords}).with_dtype<int64_t>().with_device<kDLCPU>().verify(rows);
    auto* dst = static_cast<int64_t*>(rows.data_ptr());
    int64_t found = 0;
    auto emit = [&](const cublasLtMatmulAlgo_t& candidate, size_t ws, int32_t splits) {
      if (found < capacity.unwrap()) {
        int64_t* row = dst + found * kRowWords;
        std::memcpy(row, &candidate, sizeof(candidate));
        row[kAlgoWords] = static_cast<int64_t>(ws);
        row[kAlgoWords + 1] = splits;
      }
      ++found;
    };

    Problem problem(m, n, k, k, k, n);
    constexpr int kMaxIds = 256;
    int ids[kMaxIds];
    int num_ids = 0;
    check(
        cublasLtMatmulAlgoGetIds(
            handle(),
            CUBLAS_COMPUTE_32F,
            CUDA_R_32F,
            CUDA_R_16BF,
            CUDA_R_16BF,
            CUDA_R_16BF,
            CUDA_R_16BF,
            kMaxIds,
            ids,
            &num_ids),
        "MatmulAlgoGetIds");
    for (int i = 0; i < num_ids; ++i) {
      cublasLtMatmulAlgo_t base;
      if (cublasLtMatmulAlgoInit(
              handle(),
              CUBLAS_COMPUTE_32F,
              CUDA_R_32F,
              CUDA_R_16BF,
              CUDA_R_16BF,
              CUDA_R_16BF,
              CUDA_R_16BF,
              ids[i],
              &base) != CUBLAS_STATUS_SUCCESS) {
        continue;
      }
      const auto tiles = cap_list(base, CUBLASLT_ALGO_CAP_TILE_IDS);
      const auto stages = cap_list(base, CUBLASLT_ALGO_CAP_STAGES_IDS);
      size_t written = 0;
      int32_t swizzle_support = 0, custom_max = 0;
      cublasLtMatmulAlgoCapGetAttribute(
          &base, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzle_support, sizeof(swizzle_support), &written);
      cublasLtMatmulAlgoCapGetAttribute(
          &base, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &custom_max, sizeof(custom_max), &written);
      for (uint32_t tile : tiles) {
        for (uint32_t stage : stages) {
          for (uint32_t swizzle = 0; swizzle <= (swizzle_support ? 1u : 0u); ++swizzle) {
            for (int32_t custom = 0; custom <= custom_max; ++custom) {
              cublasLtMatmulAlgo_t candidate = base;
              const int32_t no_split = 1;
              set_config(candidate, CUBLASLT_ALGO_CONFIG_TILE_ID, tile);
              set_config(candidate, CUBLASLT_ALGO_CONFIG_STAGES_ID, stage);
              set_config(candidate, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, no_split);
              set_config(candidate, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, swizzle);
              set_config(candidate, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, custom);
              cublasLtMatmulHeuristicResult_t result{};
              if (cublasLtMatmulAlgoCheck(
                      handle(), problem.op, problem.a, problem.b, problem.c, problem.c, &candidate, &result) !=
                  CUBLAS_STATUS_SUCCESS) {
                continue;
              }
              if (result.workspaceSize <= static_cast<size_t>(max_ws_bytes)) {
                emit(candidate, result.workspaceSize, no_split);
              }
            }
          }
        }
      }
    }

    cublasLtMatmulPreference_t pref = nullptr;
    check(cublasLtMatmulPreferenceCreate(&pref), "MatmulPreferenceCreate");
    const auto max_ws = static_cast<size_t>(max_ws_bytes);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws, sizeof(max_ws));
    constexpr int kHeuristicResults = 16;
    cublasLtMatmulHeuristicResult_t results[kHeuristicResults];
    int num_results = 0;
    const auto status = cublasLtMatmulAlgoGetHeuristic(
        handle(),
        problem.op,
        problem.a,
        problem.b,
        problem.c,
        problem.c,
        pref,
        kHeuristicResults,
        results,
        &num_results);
    cublasLtMatmulPreferenceDestroy(pref);
    if (status == CUBLAS_STATUS_SUCCESS) {
      for (int i = 0; i < num_results; ++i) {
        int32_t splits = 1;
        size_t written = 0;
        cublasLtMatmulAlgoConfigGetAttribute(
            &results[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splits, sizeof(splits), &written);
        emit(results[i].algo, results[i].workspaceSize, splits);
      }
    }
    return found;
  }

 private:
  static std::vector<uint32_t> cap_list(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoCapAttributes_t attr) {
    size_t bytes = 0;
    cublasLtMatmulAlgoCapGetAttribute(&algo, attr, nullptr, 0, &bytes);
    std::vector<uint32_t> values(bytes / sizeof(uint32_t));
    if (values.empty()) {
      values.push_back(0);
    } else {
      cublasLtMatmulAlgoCapGetAttribute(&algo, attr, values.data(), bytes, &bytes);
    }
    return values;
  }

  template <typename T>
  static void set_config(cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr, T value) {
    cublasLtMatmulAlgoConfigSetAttribute(&algo, attr, &value, sizeof(value));
  }
};

}  // namespace sglang
