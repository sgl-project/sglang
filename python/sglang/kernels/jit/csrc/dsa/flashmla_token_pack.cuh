// Token packing for FlashMLA's V3.2 sparse decode kernel (heads_per_group > 0): up to 5 tokens share
// one q row of the kernel, and each packed index carries in bits 26..30 which of them attend to it.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cub/block/block_scan.cuh>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace flashmla_token_pack {

inline constexpr int32_t kTable = 16384;
inline constexpr int32_t kThreads = 256;
inline constexpr int32_t kPerThread = kTable / kThreads;
inline constexpr int32_t kShift = 26;
inline constexpr int32_t kSlotMask = (1 << kShift) - 1;
/// 16-byte vectors per 576-wide BF16 q row, and the first of them that holds RoPE.
inline constexpr int32_t kVecsPerRow = 72;
inline constexpr int32_t kNopeVecs = 64;

struct Params {
  const uint4* __restrict__ q;
  const uint4* __restrict__ q_rope;  // nullptr: q already holds all 576 dims
  int64_t q_row_stride;              // in 16-byte vectors
  int64_t q_head_stride;
  int64_t rope_row_stride;
  int64_t rope_head_stride;
  uint4* __restrict__ packed_q;  // nullptr: packed_q is a view of q, nothing to copy
  const int32_t* __restrict__ rows;
  int64_t row_stride;
  int32_t tokens;
  int32_t heads;
  int32_t topk;
  int32_t group;
  int32_t out_stride;
  int32_t* __restrict__ out;
  int32_t* __restrict__ lengths;
};

/**
 * \brief One CTA per group of `group` tokens: copy their q rows into the packed q and write the union of
 *        their top-k slots, each tagged with the tokens that attend to it.
 *
 * A group holds at most 5 * 2048 entries, so the 16384-entry hash table cannot fill. Top-k slots must be
 * unique within each input row. The union order is not significant.
 */
__global__ void __launch_bounds__(kThreads) prepare_kernel(const Params params) {
  extern __shared__ int32_t table[];
  using Scan = cub::BlockScan<int32_t, kThreads>;
  __shared__ typename Scan::TempStorage scan_tmp;
  const int32_t first_token = blockIdx.x * params.group;
  int32_t* dst = params.out + int64_t(blockIdx.x) * params.out_stride;
  for (int32_t i = threadIdx.x; i < params.out_stride; i += kThreads)
    dst[i] = -1;

  // Concatenate split Q directly into the packed allocation, zeroing only the final group's missing tokens.
  if (params.packed_q != nullptr) {
    const int32_t row_vecs = params.heads * kVecsPerRow;
    for (int32_t i = threadIdx.x; i < params.group * row_vecs; i += kThreads) {
      const int32_t token = first_token + i / row_vecs;
      const int32_t head = (i / kVecsPerRow) % params.heads;
      const int32_t vec = i % kVecsPerRow;
      uint4 value = make_uint4(0, 0, 0, 0);
      if (token < params.tokens) {
        value = params.q_rope != nullptr && vec >= kNopeVecs
                    ? params.q_rope
                          [int64_t(token) * params.rope_row_stride + head * params.rope_head_stride + vec - kNopeVecs]
                    : params.q[int64_t(token) * params.q_row_stride + head * params.q_head_stride + vec];
      }
      params.packed_q[int64_t(blockIdx.x) * params.group * row_vecs + i] = value;
    }
  }
  for (int32_t i = threadIdx.x; i < kTable; i += kThreads)
    table[i] = -1;
  __syncthreads();
  for (int32_t e = threadIdx.x; e < params.group * params.topk; e += kThreads) {
    const int32_t token = first_token + e / params.topk;
    if (token >= params.tokens) continue;
    const int32_t slot = params.rows[int64_t(token) * params.row_stride + e % params.topk];
    if (slot == -1) continue;
    if (slot < 0 || slot > kSlotMask) __trap();  // the slot bits would collide with the token bits
    const int32_t bit = 1 << (kShift + e / params.topk);
    uint32_t h = (static_cast<uint32_t>(slot) * 2654435761u) & (kTable - 1);
    while (true) {
      const int32_t prev = atomicCAS(&table[h], -1, slot | bit);
      if (prev == -1) break;
      if ((prev & kSlotMask) == slot) {
        atomicOr(&table[h], bit);
        break;
      }
      h = (h + 1) & (kTable - 1);
    }
  }
  __syncthreads();
  int32_t values[kPerThread], count = 0;
#pragma unroll
  for (int32_t j = 0; j < kPerThread; ++j) {
    // Consecutive lanes read different banks; a blocked per-thread layout creates a 32-way conflict here.
    values[j] = table[j * kThreads + threadIdx.x];
    count += values[j] != -1;
  }
  int32_t offset, total;
  Scan(scan_tmp).ExclusiveSum(count, offset, total);
#pragma unroll
  for (int32_t j = 0; j < kPerThread; ++j) {
    if (values[j] != -1) dst[offset++] = values[j];
  }
  if (threadIdx.x == 0) params.lengths[blockIdx.x] = total;
}

/// Switches to `device_id` for its lifetime: the smem attribute and the launch apply to the current device.
class DeviceGuard {
 public:
  explicit DeviceGuard(const int32_t device_id) {
    CHECK_CUDA(cudaGetDevice(&m_prev));
    if (m_prev != device_id) CHECK_CUDA(cudaSetDevice(device_id));
  }
  ~DeviceGuard() {
    int32_t current = m_prev;
    cudaGetDevice(&current);
    if (current != m_prev) cudaSetDevice(m_prev);
  }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int32_t m_prev = 0;
};

}  // namespace flashmla_token_pack

/**
 * \brief Pack `tokens` query rows and their top-k indices for FlashMLA's token-packed sparse decode.
 *
 * \param q              [tokens, heads, 576] BF16, or [tokens, heads, 512] when `q_rope` is given
 * \param q_rope         Optional [tokens, heads, 64] BF16 RoPE part
 * \param indices        [>= tokens, topk] int32 top-k slots, inner stride 1
 * \param packed_q       Optional [groups, 1, group * heads, 576] BF16 output; omit when it is a view of `q`
 * \param packed_indices [groups, width] int32 output, width a multiple of 64 covering `group * topk`
 * \param lengths        [groups] int32 output, the union size of each group
 * \param group          Tokens per group (at most 5)
 */
inline void flashmla_token_pack_prepare(
    const tvm::ffi::TensorView q,
    const tvm::ffi::Optional<tvm::ffi::TensorView> q_rope,
    const tvm::ffi::TensorView indices,
    const tvm::ffi::Optional<tvm::ffi::TensorView> packed_q,
    const tvm::ffi::TensorView packed_indices,
    const tvm::ffi::TensorView lengths,
    const int64_t group) {
  using namespace host;
  using namespace flashmla_token_pack;

  auto T = SymbolicSize{"tokens"};
  auto H = SymbolicSize{"heads"};
  auto R = SymbolicSize{"index_rows"};
  auto K = SymbolicSize{"topk"};
  auto G = SymbolicSize{"groups"};
  auto W = SymbolicSize{"width"};
  auto SQ0 = SymbolicSize{"q_row_stride"};
  auto SQ1 = SymbolicSize{"q_head_stride"};
  auto SI = SymbolicSize{"index_row_stride"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  const int64_t q_dim = q_rope.has_value() ? 512 : 576;
  TensorMatcher({T, H, q_dim})  //
      .with_strides({SQ0, SQ1, 1})
      .with_dtype<bf16_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(q);
  TensorMatcher({R, K}).with_strides({SI, 1}).with_dtype<int32_t>().with_device(device).verify(indices);
  TensorMatcher({G, W}).with_dtype<int32_t>().with_device(device).verify(packed_indices);
  TensorMatcher({G}).with_dtype<int32_t>().with_device(device).verify(lengths);

  const int64_t tokens = T.unwrap();
  const int64_t heads = H.unwrap();
  const int64_t topk = K.unwrap();
  const int64_t groups = G.unwrap();
  const int64_t width = W.unwrap();
  CHECK_HOST(R.unwrap() >= tokens) << "indices has " << R.unwrap() << " rows for " << tokens << " tokens";
  CHECK_HOST(1 <= topk && topk <= 2048) << "top-k must be in [1, 2048], got " << topk;
  CHECK_HOST(1 <= heads && heads <= 32) << "head count must be in [1, 32], got " << heads;
  CHECK_HOST(1 <= group && group <= 5 && group * heads <= 64)
      << "invalid group " << group << " for " << heads << " heads";
  CHECK_HOST(groups == div_ceil(tokens, group)) << groups << " groups for " << tokens << " tokens of group " << group;
  CHECK_HOST(width % 64 == 0 && width >= group * topk) << "packed index width " << width << " is too small";

  int64_t rope_row_stride = 0, rope_head_stride = 0;
  const void* rope_ptr = nullptr;
  if (q_rope.has_value()) {
    auto SR0 = SymbolicSize{"rope_row_stride"};
    auto SR1 = SymbolicSize{"rope_head_stride"};
    TensorMatcher({T, H, 64})  //
        .with_strides({SR0, SR1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(q_rope.value());
    rope_row_stride = SR0.unwrap() / 8;
    rope_head_stride = SR1.unwrap() / 8;
    rope_ptr = q_rope.value().data_ptr();
  }
  void* packed_q_ptr = nullptr;
  if (packed_q.has_value()) {
    TensorMatcher({G, 1, group * heads, 576}).with_dtype<bf16_t>().with_device(device).verify(packed_q.value());
    packed_q_ptr = packed_q.value().data_ptr();
  } else {
    CHECK_HOST(!q_rope.has_value() && tokens % group == 0) << "packed_q may be omitted only when it is a view of q";
  }
  if (tokens == 0) return;

  const auto params = Params{
      .q = static_cast<const uint4*>(q.data_ptr()),
      .q_rope = static_cast<const uint4*>(rope_ptr),
      .q_row_stride = SQ0.unwrap() / 8,
      .q_head_stride = SQ1.unwrap() / 8,
      .rope_row_stride = rope_row_stride,
      .rope_head_stride = rope_head_stride,
      .packed_q = static_cast<uint4*>(packed_q_ptr),
      .rows = static_cast<const int32_t*>(indices.data_ptr()),
      .row_stride = SI.unwrap(),
      .tokens = static_cast<int32_t>(tokens),
      .heads = static_cast<int32_t>(heads),
      .topk = static_cast<int32_t>(topk),
      .group = static_cast<int32_t>(group),
      .out_stride = static_cast<int32_t>(width),
      .out = static_cast<int32_t*>(packed_indices.data_ptr()),
      .lengths = static_cast<int32_t*>(lengths.data_ptr()),
  };
  constexpr uint32_t kSmemBytes = kTable * sizeof(int32_t);
  const DLDevice dl_device = device.unwrap();
  const DeviceGuard device_guard{dl_device.device_id};
  // Set on every call: the attribute is per device, and a launch may target another device than the last.
  CHECK_CUDA(cudaFuncSetAttribute(prepare_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  LaunchKernel(static_cast<uint32_t>(groups), kThreads, dl_device, kSmemBytes)(prepare_kernel, params);
}

}  // namespace sglang
