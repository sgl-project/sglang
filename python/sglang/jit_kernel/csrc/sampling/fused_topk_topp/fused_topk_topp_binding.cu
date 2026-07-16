// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED.

#include "fused_topk_topp.h"
#include "tvm_ffi_utils.h"

void fused_topk_topp_renorm(
    TensorView probs,
    TensorView top_ks,
    TensorView top_ps,
    TensorView out,
    TensorView workspace,
    int64_t side_stream_handle) {
  CHECK_INPUT(probs);
  CHECK_INPUT(top_ks);
  CHECK_INPUT(top_ps);
  CHECK_INPUT(out);
  CHECK_INPUT(workspace);
  CHECK_DEVICE(probs, out);
  CHECK_DEVICE(probs, top_ks);
  CHECK_DEVICE(probs, top_ps);
  CHECK_DEVICE(probs, workspace);
  CHECK_DIM(2, probs);
  CHECK_DIM(2, out);
  CHECK_DIM(1, top_ks);
  CHECK_DIM(1, top_ps);
  CHECK_DIM(1, workspace);
  CHECK_INPUT_TYPE(probs, dl_float32);
  CHECK_INPUT_TYPE(top_ks, dl_int32);
  CHECK_INPUT_TYPE(top_ps, dl_float32);
  CHECK_INPUT_TYPE(out, dl_float32);
  CHECK_INPUT_TYPE(workspace, dl_uint8);

  int bs = static_cast<int>(probs.size(0));
  int V = static_cast<int>(probs.size(1));
  TVM_FFI_ICHECK_EQ(top_ks.size(0), bs);
  TVM_FFI_ICHECK_EQ(top_ps.size(0), bs);
  TVM_FFI_ICHECK_EQ(out.size(0), bs);
  TVM_FFI_ICHECK_EQ(out.size(1), V);

  size_t needed = fused_topk_topp::getWorkspaceSize(bs, V);
  TVM_FFI_ICHECK_GE(static_cast<size_t>(workspace.size(0)), needed)
      << "fused_topk_topp_renorm workspace too small: have " << workspace.size(0) << " bytes, need " << needed;

  cudaStream_t mainStream = get_stream(probs.device());
  cudaStream_t sideStream = reinterpret_cast<cudaStream_t>(side_stream_handle);
  // Treat self-handle (== mainStream) and null as "no side stream"; invokeFused
  // checks that internally too, but normalize here so the bench fast-paths.
  if (sideStream == mainStream) {
    sideStream = nullptr;
  }

  // The host caller is responsible for keeping `sideStream` alive across this
  // call (typically a long-lived per-device stream cached in Python). Gating
  // events are created inline below; they're captured into any in-flight CUDA
  // graph as fork/join nodes and replayed each iteration. Stream creation
  // itself is illegal during capture, which is why we accept the handle from
  // Python rather than lazy-init it here.
  fused_topk_topp::invokeFusedTopKTopP(
      static_cast<float const*>(probs.data_ptr()),
      static_cast<int32_t const*>(top_ks.data_ptr()),
      static_cast<float const*>(top_ps.data_ptr()),
      static_cast<float*>(out.data_ptr()),
      workspace.data_ptr(),
      bs,
      V,
      mainStream,
      sideStream);
}

int64_t fused_topk_topp_workspace_size(int64_t bs, int64_t V) {
  return static_cast<int64_t>(fused_topk_topp::getWorkspaceSize(static_cast<int32_t>(bs), static_cast<int32_t>(V)));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_topk_topp_renorm, fused_topk_topp_renorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_topk_topp_workspace_size, fused_topk_topp_workspace_size);
