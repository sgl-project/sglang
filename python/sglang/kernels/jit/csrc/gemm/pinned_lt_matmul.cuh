// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SGLang project
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

// out[M, N] = x[M, K] @ w[N, K]^T in BF16 (FP32 compute) through cuBLASLt with an explicit algorithm, so a
// caller can run the algorithm torch's F.linear picked for one shape on a column block or a concatenation of w.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cublasLt.h>

#include <cstdint>
#include <cstring>

namespace sglang {

struct PinnedLtMatmul {
  static cublasLtHandle_t handle() {
    static cublasLtHandle_t h = [] {
      cublasLtHandle_t v = nullptr;
      host::RuntimeCheck(cublasLtCreate(&v) == CUBLAS_STATUS_SUCCESS, "cublasLtCreate failed");
      return v;
    }();
    return h;
  }

  struct Desc {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    // torch's F.linear layout in column-major terms: C[N, M] = op_T(W[K, N]) * X[K, M].
    Desc(int64_t M, int64_t N, int64_t K, int64_t ldw, int64_t ldx, int64_t ldo) {
      check(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
      const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
      check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
      check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
      check(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, K, N, ldw));
      check(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, K, M, ldx));
      check(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, N, M, ldo));
    }
    ~Desc() {
      if (op) cublasLtMatmulDescDestroy(op);
      if (a) cublasLtMatrixLayoutDestroy(a);
      if (b) cublasLtMatrixLayoutDestroy(b);
      if (c) cublasLtMatrixLayoutDestroy(c);
    }
    static void check(cublasStatus_t s) {
      host::RuntimeCheck(s == CUBLAS_STATUS_SUCCESS, "cuBLASLt call failed: ", static_cast<int>(s));
    }
  };

  // Writes up to out.size(0) results as rows [8 algo words, workspace bytes]; returns how many.
  static int64_t heuristics(
      int64_t M, int64_t N, int64_t K, int64_t ldw, int64_t ws_bytes, tvm::ffi::TensorView out) {
    const int64_t n_req = out.size(0);
    host::RuntimeCheck(out.size(1) == 9 && n_req >= 1, "heuristics: out must be [n, 9] int64");
    Desc d(M, N, K, ldw, K, N);
    cublasLtMatmulPreference_t pref;
    Desc::check(cublasLtMatmulPreferenceCreate(&pref));
    const uint64_t ws = static_cast<uint64_t>(ws_bytes);
    Desc::check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    cublasLtMatmulHeuristicResult_t res[32];
    int got = 0;
    const int want = static_cast<int>(n_req < 32 ? n_req : 32);
    const auto st = cublasLtMatmulAlgoGetHeuristic(handle(), d.op, d.a, d.b, d.c, d.c, pref, want, res, &got);
    cublasLtMatmulPreferenceDestroy(pref);
    Desc::check(st);
    auto* o = static_cast<int64_t*>(out.data_ptr());
    for (int i = 0; i < got; ++i) {
      std::memcpy(o + i * 9, res[i].algo.data, sizeof(res[i].algo.data));
      o[i * 9 + 8] = static_cast<int64_t>(res[i].workspaceSize);
    }
    return got;
  }

  static void
  matmul(tvm::ffi::TensorView x,
         tvm::ffi::TensorView w,
         tvm::ffi::TensorView out,
         tvm::ffi::TensorView algo,
         tvm::ffi::TensorView workspace) {
    const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
    host::RuntimeCheck(w.size(1) == K && out.size(0) == M && out.size(1) == N, "matmul: shape mismatch");
    host::RuntimeCheck(x.stride(1) == 1 && w.stride(1) == 1 && out.stride(1) == 1, "matmul: rows must be dense");
    Desc d(M, N, K, w.stride(0), x.stride(0), out.stride(0));
    cublasLtMatmulAlgo_t a;
    std::memcpy(a.data, algo.data_ptr(), sizeof(a.data));
    const float alpha = 1.f, beta = 0.f;
    const auto device = x.device();
    const auto stream = static_cast<cudaStream_t>(::TVMFFIEnvGetStream(device.device_type, device.device_id));
    Desc::check(cublasLtMatmul(
        handle(), d.op, &alpha, w.data_ptr(), d.a, x.data_ptr(), d.b, &beta, out.data_ptr(), d.c, out.data_ptr(), d.c,
        &a, workspace.data_ptr(), static_cast<size_t>(workspace.size(0)), stream));
  }
};

}  // namespace sglang
