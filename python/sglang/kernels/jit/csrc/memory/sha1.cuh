// Device-side SHA-1 over (optional_prefix || data) byte stream.
// Digest matches Python: hashlib.sha1(prefix + data_bytes).digest()
//
// Used by the presharded weight loader so multi-GB CUDA tensors can be
// fingerprinted without a full device-to-host copy.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For CHECK_HOST

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

SGL_DEVICE uint32_t sha1_rotl(uint32_t x, uint32_t n) {
  return (x << n) | (x >> (32u - n));
}

SGL_DEVICE uint32_t sha1_load_be(const uint8_t* p) {
  return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

SGL_DEVICE void sha1_store_be(uint8_t* p, uint32_t v) {
  p[0] = uint8_t((v >> 24) & 0xffu);
  p[1] = uint8_t((v >> 16) & 0xffu);
  p[2] = uint8_t((v >> 8) & 0xffu);
  p[3] = uint8_t(v & 0xffu);
}

SGL_DEVICE void sha1_compress(uint32_t state[5], const uint8_t block[64]) {
  uint32_t w[80];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    w[i] = sha1_load_be(block + 4 * i);
  }
#pragma unroll
  for (int i = 16; i < 80; ++i) {
    w[i] = sha1_rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1u);
  }

  uint32_t a = state[0];
  uint32_t b = state[1];
  uint32_t c = state[2];
  uint32_t d = state[3];
  uint32_t e = state[4];

#pragma unroll
  for (int i = 0; i < 80; ++i) {
    uint32_t f, k;
    if (i < 20) {
      f = (b & c) | ((~b) & d);
      k = 0x5A827999u;
    } else if (i < 40) {
      f = b ^ c ^ d;
      k = 0x6ED9EBA1u;
    } else if (i < 60) {
      f = (b & c) | (b & d) | (c & d);
      k = 0x8F1BBCDCu;
    } else {
      f = b ^ c ^ d;
      k = 0xCA62C1D6u;
    }
    const uint32_t temp = sha1_rotl(a, 5u) + f + e + k + w[i];
    e = d;
    d = c;
    c = sha1_rotl(b, 30u);
    b = a;
    a = temp;
  }

  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
}

// Logical stream = prefix[0..prefix_len) || data[0..data_len)
// Thread 0 drives compress; the CTA fills each 64-byte block cooperatively.
__global__ void sha1_prefix_data_kernel(
    const uint8_t* __restrict__ prefix,
    uint64_t prefix_len,
    const uint8_t* __restrict__ data,
    uint64_t data_len,
    uint8_t* __restrict__ digest_out) {
  __shared__ uint8_t smem_block[64];
  __shared__ uint32_t smem_state[5];

  if (threadIdx.x == 0) {
    smem_state[0] = 0x67452301u;
    smem_state[1] = 0xEFCDAB89u;
    smem_state[2] = 0x98BADCFEu;
    smem_state[3] = 0x10325476u;
    smem_state[4] = 0xC3D2E1F0u;
  }
  __syncthreads();

  const uint64_t n_bytes = prefix_len + data_len;
  const uint64_t bit_len = n_bytes * 8ull;
  const uint64_t pad_len = ((n_bytes + 8ull) / 64ull + 1ull) * 64ull;
  const uint64_t n_blocks = pad_len / 64ull;

  for (uint64_t bi = 0; bi < n_blocks; ++bi) {
    for (int t = static_cast<int>(threadIdx.x); t < 64; t += static_cast<int>(blockDim.x)) {
      const uint64_t global_off = bi * 64ull + static_cast<uint64_t>(t);
      uint8_t byte;
      if (global_off < n_bytes) {
        if (global_off < prefix_len) {
          byte = prefix[global_off];
        } else {
          byte = data[global_off - prefix_len];
        }
      } else if (global_off == n_bytes) {
        byte = 0x80u;
      } else if (global_off < pad_len - 8ull) {
        byte = 0x00u;
      } else {
        const int shift = static_cast<int>((pad_len - 1ull - global_off) * 8ull);
        byte = static_cast<uint8_t>((bit_len >> shift) & 0xffull);
      }
      smem_block[t] = byte;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      sha1_compress(smem_state, smem_block);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      sha1_store_be(digest_out + 4 * i, smem_state[i]);
    }
  }
}

// digest: uint8[20] on CUDA
// data:   uint8[N] on CUDA (may be empty)
// prefix: uint8[P] on CUDA (may be empty) — typically shape/dtype ASCII
void sha1_prefix_data(tvm::ffi::TensorView digest, tvm::ffi::TensorView data, tvm::ffi::TensorView prefix) {
  using namespace host;

  SymbolicSize N = {"num_data_bytes"};
  SymbolicSize P = {"num_prefix_bytes"};
  SymbolicSize D = {"digest_len"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({N})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(data);

  TensorMatcher({P})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(prefix);

  TensorMatcher({D})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(digest);

  const size_t n_data = static_cast<size_t>(N.unwrap());
  const size_t n_prefix = static_cast<size_t>(P.unwrap());
  const size_t digest_len = static_cast<size_t>(D.unwrap());
  CHECK_HOST(digest_len == 20) << "sha1_prefix_data: digest must be 20 bytes, got " << digest_len;

  const DLDevice device = device_.unwrap();
  constexpr uint32_t kBlockSize = 128;
  LaunchKernel(/*grid=*/1, kBlockSize, device)(
      sha1_prefix_data_kernel,
      static_cast<const uint8_t*>(prefix.data_ptr()),
      static_cast<uint64_t>(n_prefix),
      static_cast<const uint8_t*>(data.data_ptr()),
      static_cast<uint64_t>(n_data),
      static_cast<uint8_t*>(digest.data_ptr()));
}

// Convenience: data-only (empty prefix).
void sha1_bytes(tvm::ffi::TensorView digest, tvm::ffi::TensorView data) {
  // Allocate a 0-length prefix on the same device via a temporary empty view
  // is awkward from C++; require Python to pass empty prefix through
  // sha1_prefix_data instead. Keep a dedicated data-only path with null prefix.
  using namespace host;

  SymbolicSize N = {"num_bytes"};
  SymbolicSize D = {"digest_len"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({N})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(data);
  TensorMatcher({D})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(digest);

  const size_t n_bytes = static_cast<size_t>(N.unwrap());
  CHECK_HOST(static_cast<size_t>(D.unwrap()) == 20) << "sha1_bytes: digest must be 20 bytes";

  const DLDevice device = device_.unwrap();
  constexpr uint32_t kBlockSize = 128;
  LaunchKernel(/*grid=*/1, kBlockSize, device)(
      sha1_prefix_data_kernel,
      static_cast<const uint8_t*>(nullptr),
      static_cast<uint64_t>(0),
      static_cast<const uint8_t*>(data.data_ptr()),
      static_cast<uint64_t>(n_bytes),
      static_cast<uint8_t*>(digest.data_ptr()));
}

}  // namespace
