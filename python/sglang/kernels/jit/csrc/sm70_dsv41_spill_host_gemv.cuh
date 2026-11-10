// SPDX-License-Identifier: Apache-2.0
// In-graph spill_request / spill_join against the host mailbox.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cstring>

#include "sm70_dsv41_spill_host_gemv_common.h"

namespace sglang::sm70_dsv41 {

SpillHostMailbox* dsv41_spill_mbox_host = nullptr;
SpillHostMailbox* dsv41_spill_mbox_dev = nullptr;

inline constexpr int32_t kJoinSpinLimit = 80000000;

__global__ void spill_request_kernel(int32_t* __restrict__ topk_ids,
                                     const float* __restrict__ topk_weights,
                                     const fp16_t* __restrict__ hidden,
                                     const int32_t* __restrict__ map_table,
                                     const int32_t* __restrict__ host_map,
                                     const int64_t* __restrict__ bases,
                                     SpillHostMailbox* __restrict__ mbox,
                                     int32_t n_tok,
                                     int32_t k,
                                     int32_t n_logical,
                                     int32_t hidden_ld) {
  SpillHostMailbox* m = mbox;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int32_t n_hits = 0;
    const int32_t n = n_tok * k;
    for (int32_t i = 0; i < n; ++i) {
      const int32_t id = topk_ids[i];
      if (id < 0 || id >= n_logical) {
        continue;
      }
      const int32_t phys = map_table[id];
      if (phys >= 0) {
        topk_ids[i] = phys;
        continue;
      }
      const int32_t hs = host_map[id];
      topk_ids[i] = -1;
      if (hs < 0 || n_hits >= kMaxHits) {
        continue;
      }
      m->host_rows[n_hits] = hs;
      m->weights[n_hits] = topk_weights[i];
      m->tok_of[n_hits] = i / k;
      ++n_hits;
    }
    m->n_tok = n_tok;
    m->n_hits = n_hits;
    m->w13_ptr = bases[0];
    m->s13_ptr = bases[1];
    m->w2_ptr = bases[2];
    m->s2_ptr = bases[3];
    m->n_host = static_cast<int32_t>(bases[4]);
    m->error = 0;
  }
  __syncthreads();
  for (int32_t t = 0; t < n_tok; ++t) {
    const uint16_t* src = reinterpret_cast<const uint16_t*>(hidden + t * hidden_ld);
    uint16_t* dst = m->x[t];
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < kHidden;
         i += static_cast<int32_t>(blockDim.x)) {
      dst[i] = src[i];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    __threadfence_system();
    uint32_t s = m->seq + 1u;
    if (s == 0u) {
      s = 1u;
    }
    m->seq = s;
    __threadfence_system();
  }
}

__global__ void spill_join_kernel(fp16_t* __restrict__ output,
                                  SpillHostMailbox* __restrict__ mbox,
                                  int32_t n_tok,
                                  int32_t out_ld) {
  SpillHostMailbox* m = mbox;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const uint32_t seq = m->seq;
    int32_t spins = 0;
    while (m->done != seq) {
      if (++spins >= kJoinSpinLimit) {
        m->error = 1;
        break;
      }
    }
    __threadfence_system();
  }
  __syncthreads();
  if (m->error != 0) {
    return;
  }
  const int32_t nt = n_tok > kMaxTok ? kMaxTok : n_tok;
  for (int32_t t = 0; t < nt; ++t) {
    fp16_t* dst = output + t * out_ld;
    const uint16_t* src = m->y[t];
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < kHidden;
         i += static_cast<int32_t>(blockDim.x)) {
      const float a = static_cast<float>(dst[i]);
      const fp16_t b = *reinterpret_cast<const fp16_t*>(&src[i]);
      dst[i] = static_cast<fp16_t>(a + static_cast<float>(b));
    }
  }
}

inline void spill_request(tvm::ffi::TensorView topk_ids,
                          tvm::ffi::TensorView topk_weights,
                          tvm::ffi::TensorView hidden,
                          tvm::ffi::TensorView map_table,
                          tvm::ffi::TensorView host_map,
                          tvm::ffi::TensorView bases) {
  using namespace host;
  CHECK_HOST(dsv41_spill_mbox_dev != nullptr) << "sm70_dsv41 spill_request: host_gemv_start first";
  SymbolicSize n_tok = {"n_tok"};
  SymbolicSize k = {"k"};
  SymbolicSize n_logical = {"n_logical"};
  SymbolicSize five = {"five"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({n_tok, k})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(topk_ids);
  TensorMatcher({n_tok, k})  //
      .with_dtype<float>()
      .with_device<kDLCUDA>(device_)
      .verify(topk_weights);
  TensorMatcher({n_tok, kHidden})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(hidden);
  TensorMatcher({n_logical})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(map_table)
      .verify(host_map);
  TensorMatcher({five})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(bases);
  CHECK_HOST(five.unwrap() == 5) << "sm70_dsv41 spill_request: bases [5]";
  const int32_t t = static_cast<int32_t>(n_tok.unwrap());
  const int32_t kk = static_cast<int32_t>(k.unwrap());
  CHECK_HOST(t >= 1 && t <= kMaxTok) << "sm70_dsv41 spill_request: n_tok " << t;
  CHECK_HOST(hidden.is_contiguous() && topk_ids.is_contiguous() && topk_weights.is_contiguous())
      << "sm70_dsv41 spill_request: not contiguous";
  const DLDevice dev = device_.unwrap();
  LaunchKernel(1, 128, dev)(
      spill_request_kernel,
      static_cast<int32_t*>(topk_ids.data_ptr()),
      static_cast<const float*>(topk_weights.data_ptr()),
      static_cast<const fp16_t*>(hidden.data_ptr()),
      static_cast<const int32_t*>(map_table.data_ptr()),
      static_cast<const int32_t*>(host_map.data_ptr()),
      static_cast<const int64_t*>(bases.data_ptr()),
      dsv41_spill_mbox_dev,
      t,
      kk,
      static_cast<int32_t>(n_logical.unwrap()),
      kHidden);
}

inline void spill_join(tvm::ffi::TensorView output) {
  using namespace host;
  CHECK_HOST(dsv41_spill_mbox_dev != nullptr) << "sm70_dsv41 spill_join: host_gemv_start first";
  SymbolicSize n_tok = {"n_tok"};
  SymbolicSize hidden = {"hidden"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({n_tok, hidden})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(output);
  const int32_t t = static_cast<int32_t>(n_tok.unwrap());
  const int32_t h = static_cast<int32_t>(hidden.unwrap());
  CHECK_HOST(t >= 1 && t <= kMaxTok) << "sm70_dsv41 spill_join: n_tok " << t;
  CHECK_HOST(h >= kHidden) << "sm70_dsv41 spill_join: hidden " << h;
  CHECK_HOST(output.is_contiguous()) << "sm70_dsv41 spill_join: not contiguous";
  const DLDevice dev = device_.unwrap();
  LaunchKernel(1, 128, dev)(
      spill_join_kernel, static_cast<fp16_t*>(output.data_ptr()), dsv41_spill_mbox_dev, t, h);
}

inline void host_gemv_start(tvm::ffi::TensorView n_threads) {
  using namespace host;
  SymbolicSize one = {"one"};
  SymbolicDevice cpu;
  cpu.set_options<kDLCPU>();
  TensorMatcher({one})  //
      .with_dtype<int32_t>()
      .with_device<kDLCPU>(cpu)
      .verify(n_threads);
  CHECK_HOST(one.unwrap() == 1) << "sm70_dsv41 host_gemv_start: n_threads rank-1";
  if (dsv41_spill_mbox_host != nullptr) {
    return;
  }
  int n = *static_cast<const int32_t*>(n_threads.data_ptr());
  if (n < 1) {
    n = 1;
  }
  if (n > kMaxHostThreads) {
    n = kMaxHostThreads;
  }
  SpillHostMailbox* host = nullptr;
  CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host), sizeof(SpillHostMailbox),
                           cudaHostAllocMapped | cudaHostAllocPortable))
      << "sm70_dsv41 host_gemv_start: cudaHostAlloc mailbox";
  std::memset(host, 0, sizeof(*host));
  SpillHostMailbox* dev = nullptr;
  const cudaError_t gp = cudaHostGetDevicePointer(reinterpret_cast<void**>(&dev), host, 0);
  if (gp != cudaSuccess || dev == nullptr) {
    dev = host;
  }
  dsv41_spill_mbox_host = host;
  dsv41_spill_mbox_dev = dev;
  dsv41_host_workers_start(n);
}

inline void host_gemv_stop(tvm::ffi::TensorView unused) {
  using namespace host;
  SymbolicSize one = {"one"};
  SymbolicDevice cpu;
  cpu.set_options<kDLCPU>();
  TensorMatcher({one})  //
      .with_dtype<int32_t>()
      .with_device<kDLCPU>(cpu)
      .verify(unused);
  if (dsv41_spill_mbox_host == nullptr) {
    return;
  }
  dsv41_host_workers_stop();
  CHECK_CUDA(cudaFreeHost(dsv41_spill_mbox_host)) << "sm70_dsv41 host_gemv_stop: cudaFreeHost";
  dsv41_spill_mbox_host = nullptr;
  dsv41_spill_mbox_dev = nullptr;
}

inline void host_mxfp4_moe_expert(tvm::ffi::TensorView x,
                                  tvm::ffi::TensorView w13,
                                  tvm::ffi::TensorView s13,
                                  tvm::ffi::TensorView w2,
                                  tvm::ffi::TensorView s2,
                                  tvm::ffi::TensorView y,
                                  tvm::ffi::TensorView weight) {
  using namespace host;
  SymbolicDevice cpu;
  cpu.set_options<kDLCPU>();
  TensorMatcher({kHidden})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCPU>(cpu)
      .verify(x)
      .verify(y);
  TensorMatcher({kW13KTiles, kGateUp * 2})  //
      .with_dtype<int32_t>()
      .with_device<kDLCPU>(cpu)
      .verify(w13);
  TensorMatcher({kW2KTiles, kHidden * 2})  //
      .with_dtype<int32_t>()
      .with_device<kDLCPU>(cpu)
      .verify(w2);
  TensorMatcher({kW13Groups, kGateUp})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCPU>(cpu)
      .verify(s13);
  TensorMatcher({kW2Groups, kHidden})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCPU>(cpu)
      .verify(s2);
  SymbolicSize one = {"one"};
  TensorMatcher({one})  //
      .with_dtype<float>()
      .with_device<kDLCPU>(cpu)
      .verify(weight);
  CHECK_HOST(x.is_contiguous() && y.is_contiguous() && w13.is_contiguous() && w2.is_contiguous() &&
             s13.is_contiguous() && s2.is_contiguous())
      << "sm70_dsv41 host_mxfp4_moe_expert: tensors must be contiguous";
  dsv41_mxfp4_moe_expert_serial(reinterpret_cast<const uint16_t*>(x.data_ptr()),
                                static_cast<const uint32_t*>(w13.data_ptr()),
                                static_cast<const uint8_t*>(s13.data_ptr()),
                                static_cast<const uint32_t*>(w2.data_ptr()),
                                static_cast<const uint8_t*>(s2.data_ptr()),
                                *static_cast<const float*>(weight.data_ptr()),
                                reinterpret_cast<uint16_t*>(y.data_ptr()));
}

}  // namespace sglang::sm70_dsv41
