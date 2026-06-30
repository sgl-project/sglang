// Native CUDA fast path for Cosmos3 VAE causal-Conv3D cat/pad copy.
//
// The op writes the output of:
//   pad(cat(cache_x, x, dim=T), (Wl, Wr, Ht, Hb, Dl - cache_t, Dr))
// for 5D NCTHW tensors. It is a memory-bound copy/zero-fill kernel and is only
// entered for contiguous CUDA tensors; unsupported cases fall back to Triton in
// the Python caller.
//
// Developed with MIT HAN Lab Kernel Design Agents:
// https://github.com/mit-han-lab/kernel-design-agents

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/vec.cuh>    // For device::AlignedVector

#include <cstdint>

namespace sglang_causal_conv3d_cat_pad {

namespace {

constexpr int kBlockSize = 256;

template <typename ET, int kVec>
__global__ void __launch_bounds__(kBlockSize) cat_pad_flat_kernel(
    const ET* __restrict__ x,
    const ET* __restrict__ cache,
    ET* __restrict__ out,
    int64_t total_vecs,
    int64_t channels,
    int64_t t_size,
    int64_t h_size,
    int64_t w_size,
    int64_t cache_t,
    int64_t out_t,
    int64_t out_h,
    int64_t out_w,
    int64_t pad_d_left,
    int64_t pad_h_top,
    int64_t pad_w_left) {
  using Pack = device::AlignedVector<ET, kVec>;

  const int64_t nthreads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t vid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; vid < total_vecs; vid += nthreads) {
    int64_t base = vid * kVec;
    int64_t ow = base % out_w;
    int64_t tmp = base / out_w;
    int64_t oh = tmp % out_h;
    tmp /= out_h;
    int64_t od = tmp % out_t;
    tmp /= out_t;
    int64_t oc = tmp % channels;
    int64_t ob = tmp / channels;

    int64_t ih = oh - pad_h_top;
    int64_t src_t = od - pad_d_left;
    bool interior = ih >= 0 && ih < h_size && src_t >= 0 && src_t < cache_t + t_size;

    const ET* src = nullptr;
    if (interior) {
      if (src_t < cache_t) {
        src = cache + (((ob * channels + oc) * cache_t + src_t) * h_size + ih) * w_size;
      } else {
        src = x + (((ob * channels + oc) * t_size + (src_t - cache_t)) * h_size + ih) * w_size;
      }
    }

    Pack pack;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      ET value = ET(0);
      if (interior) {
        const int64_t iw = ow - pad_w_left;
        if (iw >= 0 && iw < w_size) {
          value = SGLANG_LDG(src + iw);
        }
      }
      pack[i] = value;

      if (++ow == out_w) {
        ow = 0;
        if (++oh == out_h) {
          oh = 0;
          if (++od == out_t) {
            od = 0;
            if (++oc == channels) {
              oc = 0;
              ++ob;
            }
          }
        }
        ih = oh - pad_h_top;
        src_t = od - pad_d_left;
        interior = ih >= 0 && ih < h_size && src_t >= 0 && src_t < cache_t + t_size;
        if (interior) {
          if (src_t < cache_t) {
            src = cache + (((ob * channels + oc) * cache_t + src_t) * h_size + ih) * w_size;
          } else {
            src = x + (((ob * channels + oc) * t_size + (src_t - cache_t)) * h_size + ih) * w_size;
          }
        } else {
          src = nullptr;
        }
      }
    }

    pack.store(out, vid);
  }
}

template <typename ET, int kVec>
void launch_cat_pad_flat(
    const void* x,
    const void* cache,
    void* out,
    int64_t total,
    int64_t channels,
    int64_t t_size,
    int64_t h_size,
    int64_t w_size,
    int64_t cache_t,
    int64_t out_t,
    int64_t out_h,
    int64_t out_w,
    int64_t depth_left,
    int64_t pad_h_top,
    int64_t pad_w_left,
    DLDevice device) {
  const int64_t total_vecs = total / kVec;
  const uint32_t grid = static_cast<uint32_t>(host::div_ceil(total_vecs, static_cast<int64_t>(kBlockSize)));
  host::LaunchKernel(grid, kBlockSize, device)(
      cat_pad_flat_kernel<ET, kVec>,
      static_cast<const ET*>(x),
      static_cast<const ET*>(cache),
      static_cast<ET*>(out),
      total_vecs,
      channels,
      t_size,
      h_size,
      w_size,
      cache_t,
      out_t,
      out_h,
      out_w,
      depth_left,
      pad_h_top,
      pad_w_left);
}

}  // namespace

template <typename T>
struct CausalConv3dCatPadKernel {
  static void
  run(tvm::ffi::TensorView out,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView cache,
      int64_t pad_w_left,
      int64_t pad_w_right,
      int64_t pad_h_top,
      int64_t pad_h_bottom,
      int64_t pad_d_left,
      int64_t pad_d_right) {
    using namespace host;

    auto bsz = SymbolicSize{"batch"};
    auto channels = SymbolicSize{"channels"};
    auto t_size = SymbolicSize{"t_size"};
    auto h_size = SymbolicSize{"h_size"};
    auto w_size = SymbolicSize{"w_size"};
    auto cache_t = SymbolicSize{"cache_t"};
    auto out_t = SymbolicSize{"out_t"};
    auto out_h = SymbolicSize{"out_h"};
    auto out_w = SymbolicSize{"out_w"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({bsz, channels, t_size, h_size, w_size})
        .with_dtype<T>()
        .template with_device<kDLCUDA>(device)
        .verify(x);
    TensorMatcher({bsz, channels, cache_t, h_size, w_size})
        .with_dtype<T>()
        .template with_device<kDLCUDA>(device)
        .verify(cache);
    TensorMatcher({bsz, channels, out_t, out_h, out_w})
        .with_dtype<T>()
        .template with_device<kDLCUDA>(device)
        .verify(out);

    const int64_t depth_left = pad_d_left - cache_t.unwrap();
    RuntimeCheck(depth_left >= 0, "pad_d_left must be >= cache_t");
    RuntimeCheck(pad_d_right == 0, "pad_d_right must be 0");
    RuntimeCheck(pad_w_left == pad_w_right, "width padding must be symmetric");
    RuntimeCheck(pad_h_top == pad_h_bottom, "height padding must be symmetric");
    RuntimeCheck(out_t.unwrap() == t_size.unwrap() + cache_t.unwrap() + depth_left + pad_d_right, "out_t mismatch");
    RuntimeCheck(out_h.unwrap() == h_size.unwrap() + pad_h_top + pad_h_bottom, "out_h mismatch");
    RuntimeCheck(out_w.unwrap() == w_size.unwrap() + pad_w_left + pad_w_right, "out_w mismatch");

    const int64_t total = bsz.unwrap() * channels.unwrap() * out_t.unwrap() * out_h.unwrap() * out_w.unwrap();
    if (total == 0) {
      return;
    }

    constexpr int kVec = 16 / sizeof(T);
    RuntimeCheck(total % kVec == 0, "output element count must be divisible by vector width");
    RuntimeCheck(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "output pointer must be 16-byte aligned");

    if constexpr (sizeof(T) == 2) {
      launch_cat_pad_flat<uint16_t, kVec>(
          x.data_ptr(),
          cache.data_ptr(),
          out.data_ptr(),
          total,
          channels.unwrap(),
          t_size.unwrap(),
          h_size.unwrap(),
          w_size.unwrap(),
          cache_t.unwrap(),
          out_t.unwrap(),
          out_h.unwrap(),
          out_w.unwrap(),
          depth_left,
          pad_h_top,
          pad_w_left,
          device.unwrap());
    } else {
      launch_cat_pad_flat<uint32_t, kVec>(
          x.data_ptr(),
          cache.data_ptr(),
          out.data_ptr(),
          total,
          channels.unwrap(),
          t_size.unwrap(),
          h_size.unwrap(),
          w_size.unwrap(),
          cache_t.unwrap(),
          out_t.unwrap(),
          out_h.unwrap(),
          out_w.unwrap(),
          depth_left,
          pad_h_top,
          pad_w_left,
          device.unwrap());
    }
  }
};

}  // namespace sglang_causal_conv3d_cat_pad
