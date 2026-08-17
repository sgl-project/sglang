// Copyright 2026 SGLang Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// TVM-FFI entry point for the GLM-4.5 FP8 fused MoE kernel. The CUDA body
// targets H200 and specializes the TP=8 layout captured in production.

#include "../tvm_ffi_utils.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <utility>

extern "C" void launch_fused_moe(
    void* hidden,
    const void* w1map,
    const void* w2map,
    const void* iqmap,
    const void* w2,
    const float* tkw,
    const int* tki,
    const float* w1s,
    const float* w2s,
    int M,
    int E,
    int maxBlocks,
    uint8_t* Aq,
    float* a1s,
    int* cnt,
    int* fill,
    int* rowOff,
    int* ebk,
    int* nbt,
    int* sorted,
    float* a2s,
    uint8_t* interq,
    void* C3,
    int* flags,
    int* tokcnt,
    cudaStream_t stream);

namespace sglang {
namespace {

constexpr int64_t kHidden = 5120;
constexpr int64_t kGateUp = 384;
constexpr int64_t kIntermediate = 192;
constexpr int64_t kTopK = 9;
constexpr int64_t kBlockM = 64;
constexpr int64_t kMaxTokens = 8192;

using CacheKey = std::pair<int, uintptr_t>;

int64_t align256(int64_t value) {
  return (value + 255) & ~int64_t(255);
}

int64_t max_blocks_for(int64_t tokens, int64_t experts) {
  const int64_t pairs = tokens * kTopK;
  const int64_t nonzero_experts = std::min(experts, pairs);
  return (pairs + nonzero_experts * (kBlockM - 1) + kBlockM - 1) / kBlockM;
}

int64_t workspace_size_for(int64_t tokens, int64_t experts) {
  const int64_t pairs = tokens * kTopK;
  const int64_t max_blocks = max_blocks_for(tokens, experts);
  const int64_t max_rows = max_blocks * kBlockM;
  const int64_t o_c3 = 0;
  const int64_t o_interq = align256(o_c3 + pairs * kHidden * 2);
  const int64_t o_aq = align256(o_interq + max_rows * kIntermediate);
  const int64_t o_sorted = align256(o_aq + tokens * kHidden);
  const int64_t o_a2s = align256(o_sorted + max_rows * 4);
  const int64_t o_a1s = align256(o_a2s + max_rows * 4);
  const int64_t o_ebk = align256(o_a1s + tokens * 4);
  const int64_t o_row_off = align256(o_ebk + max_blocks * 4);
  const int64_t o_cnt = align256(o_row_off + (experts + 1) * 4);
  const int64_t o_nbt = align256(o_cnt + experts * 8);
  const int64_t o_flags = align256(o_nbt + 16);
  const int64_t o_tokcnt = align256(o_flags + max_blocks * 4);
  return align256(o_tokcnt + tokens * 4);
}

void check_driver(CUresult result, const char* what) {
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS) << what;
}

CUtensorMap build_w1_map(const void* pointer, int64_t experts) {
  CUtensorMap map;
  const uint64_t dimensions[2] = {
      static_cast<uint64_t>(kHidden),
      static_cast<uint64_t>(experts * kGateUp),
  };
  const uint64_t strides[1] = {static_cast<uint64_t>(kHidden)};
  const uint32_t box[2] = {128, 192};
  const uint32_t element_strides[2] = {1, 1};
  check_driver(
      cuTensorMapEncodeTiled(
          &map,
          CU_TENSOR_MAP_DATA_TYPE_UINT8,
          2,
          const_cast<void*>(pointer),
          dimensions,
          strides,
          box,
          element_strides,
          CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B,
          CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled(w1) failed");
  return map;
}

CUtensorMap build_w2_like_map(const void* pointer, int64_t rows, uint32_t box_rows, uint32_t box_inner) {
  CUtensorMap map;
  const uint64_t dimensions[2] = {
      static_cast<uint64_t>(kIntermediate),
      static_cast<uint64_t>(rows),
  };
  const uint64_t strides[1] = {static_cast<uint64_t>(kIntermediate)};
  const uint32_t box[2] = {box_inner, box_rows};
  const uint32_t element_strides[2] = {1, 1};
  check_driver(
      cuTensorMapEncodeTiled(
          &map,
          CU_TENSOR_MAP_DATA_TYPE_UINT8,
          2,
          const_cast<void*>(pointer),
          dimensions,
          strides,
          box,
          element_strides,
          CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B,
          CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled(w2-like) failed");
  return map;
}

std::mutex g_map_mutex;
std::map<CacheKey, CUtensorMap> g_w1_maps;
std::map<CacheKey, std::array<CUtensorMap, 2>> g_w2_maps;
std::map<CacheKey, std::array<CUtensorMap, 2>> g_intermediate_maps;

CUtensorMap get_w1_map(int device, const void* pointer, int64_t experts) {
  const CacheKey key{device, reinterpret_cast<uintptr_t>(pointer)};
  std::lock_guard<std::mutex> guard(g_map_mutex);
  auto [it, inserted] = g_w1_maps.try_emplace(key);
  if (inserted) {
    it->second = build_w1_map(pointer, experts);
  }
  return it->second;
}

std::array<CUtensorMap, 2> get_w2_maps(int device, const void* pointer, int64_t experts) {
  const CacheKey key{device, reinterpret_cast<uintptr_t>(pointer)};
  std::lock_guard<std::mutex> guard(g_map_mutex);
  auto [it, inserted] = g_w2_maps.try_emplace(key);
  if (inserted) {
    const int64_t rows = experts * kHidden;
    it->second[0] = build_w2_like_map(pointer, rows, 256, 128);
    it->second[1] = build_w2_like_map(pointer, rows, 256, 64);
  }
  return it->second;
}

std::array<CUtensorMap, 2> get_intermediate_maps(int device, const void* pointer, int64_t rows) {
  const CacheKey key{device, reinterpret_cast<uintptr_t>(pointer)};
  std::lock_guard<std::mutex> guard(g_map_mutex);
  auto [it, inserted] = g_intermediate_maps.try_emplace(key);
  if (inserted) {
    it->second[0] = build_w2_like_map(pointer, rows, kBlockM, 128);
    it->second[1] = build_w2_like_map(pointer, rows, kBlockM, 64);
  }
  return it->second;
}

}  // namespace

int64_t glm45_fused_moe_workspace_size() {
  return workspace_size_for(kMaxTokens, 161);
}

void glm45_fused_moe(
    TensorView hidden,
    TensorView w1,
    TensorView w2,
    TensorView topk_weights,
    TensorView topk_ids,
    TensorView w1_scale,
    TensorView w2_scale,
    TensorView workspace) {
  CHECK_INPUT_AND_TYPE(hidden, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(w1, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(w2, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(topk_weights, dl_float32);
  CHECK_INPUT_AND_TYPE(topk_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(w1_scale, dl_float32);
  CHECK_INPUT_AND_TYPE(w2_scale, dl_float32);
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  CHECK_DEVICE(hidden, w1);
  CHECK_DEVICE(hidden, w2);
  CHECK_DEVICE(hidden, topk_weights);
  CHECK_DEVICE(hidden, topk_ids);
  CHECK_DEVICE(hidden, w1_scale);
  CHECK_DEVICE(hidden, w2_scale);
  CHECK_DEVICE(hidden, workspace);
  CHECK_DIM(2, hidden);
  CHECK_DIM(3, w1);
  CHECK_DIM(3, w2);
  CHECK_DIM(2, topk_weights);
  CHECK_DIM(2, topk_ids);
  CHECK_DIM(1, workspace);

  const int64_t tokens = hidden.size(0);
  const int64_t experts = w1.size(0);
  TVM_FFI_ICHECK_GE(tokens, 1);
  TVM_FFI_ICHECK_LE(tokens, kMaxTokens);
  TVM_FFI_ICHECK_EQ(hidden.size(1), kHidden);
  TVM_FFI_ICHECK_EQ(experts, 161);
  TVM_FFI_ICHECK_EQ(w1.size(1), kGateUp);
  TVM_FFI_ICHECK_EQ(w1.size(2), kHidden);
  TVM_FFI_ICHECK_EQ(w2.size(0), experts);
  TVM_FFI_ICHECK_EQ(w2.size(1), kHidden);
  TVM_FFI_ICHECK_EQ(w2.size(2), kIntermediate);
  TVM_FFI_ICHECK_EQ(topk_weights.size(0), tokens);
  TVM_FFI_ICHECK_EQ(topk_weights.size(1), kTopK);
  TVM_FFI_ICHECK_EQ(topk_ids.size(0), tokens);
  TVM_FFI_ICHECK_EQ(topk_ids.size(1), kTopK);
  TVM_FFI_ICHECK_EQ(w1_scale.numel(), experts * kGateUp);
  TVM_FFI_ICHECK_EQ(w2_scale.numel(), experts * kHidden);

  const int64_t required_workspace = workspace_size_for(tokens, experts);
  TVM_FFI_ICHECK_GE(workspace.numel(), glm45_fused_moe_workspace_size());
  TVM_FFI_ICHECK_GE(workspace.numel(), required_workspace);

  const int64_t pairs = tokens * kTopK;
  const int64_t max_blocks = max_blocks_for(tokens, experts);
  const int64_t max_rows = max_blocks * kBlockM;
  const int64_t max_capacity_rows = max_blocks_for(kMaxTokens, experts) * kBlockM;

  const int64_t o_c3 = 0;
  const int64_t o_interq = align256(o_c3 + pairs * kHidden * 2);
  const int64_t o_aq = align256(o_interq + max_rows * kIntermediate);
  const int64_t o_sorted = align256(o_aq + tokens * kHidden);
  const int64_t o_a2s = align256(o_sorted + max_rows * 4);
  const int64_t o_a1s = align256(o_a2s + max_rows * 4);
  const int64_t o_ebk = align256(o_a1s + tokens * 4);
  const int64_t o_row_off = align256(o_ebk + max_blocks * 4);
  const int64_t o_cnt = align256(o_row_off + (experts + 1) * 4);
  const int64_t o_nbt = align256(o_cnt + experts * 8);
  const int64_t o_flags = align256(o_nbt + 16);
  const int64_t o_tokcnt = align256(o_flags + max_blocks * 4);

  cudaSetDevice(hidden.device().device_id);
  auto* base = static_cast<uint8_t*>(workspace.data_ptr());
  const auto w1_map = get_w1_map(hidden.device().device_id, w1.data_ptr(), experts);
  const auto w2_maps = get_w2_maps(hidden.device().device_id, w2.data_ptr(), experts);
  const auto intermediate_maps = get_intermediate_maps(hidden.device().device_id, base + o_interq, max_capacity_rows);

  launch_fused_moe(
      hidden.data_ptr(),
      &w1_map,
      w2_maps.data(),
      intermediate_maps.data(),
      w2.data_ptr(),
      static_cast<const float*>(topk_weights.data_ptr()),
      static_cast<const int*>(topk_ids.data_ptr()),
      static_cast<const float*>(w1_scale.data_ptr()),
      static_cast<const float*>(w2_scale.data_ptr()),
      static_cast<int>(tokens),
      static_cast<int>(experts),
      static_cast<int>(max_blocks),
      base + o_aq,
      reinterpret_cast<float*>(base + o_a1s),
      reinterpret_cast<int*>(base + o_cnt),
      reinterpret_cast<int*>(base + o_cnt + experts * 4),
      reinterpret_cast<int*>(base + o_row_off),
      reinterpret_cast<int*>(base + o_ebk),
      reinterpret_cast<int*>(base + o_nbt),
      reinterpret_cast<int*>(base + o_sorted),
      reinterpret_cast<float*>(base + o_a2s),
      base + o_interq,
      base + o_c3,
      reinterpret_cast<int*>(base + o_flags),
      reinterpret_cast<int*>(base + o_tokcnt),
      get_stream(hidden.device()));

  const cudaError_t error = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(error, cudaSuccess) << "GLM-4.5 fused MoE launch failed: " << cudaGetErrorString(error);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(workspace_size, glm45_fused_moe_workspace_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, glm45_fused_moe);

}  // namespace sglang
