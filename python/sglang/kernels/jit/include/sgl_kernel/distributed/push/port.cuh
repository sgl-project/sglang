#pragma once

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/fusion.cuh>
#include <sgl_kernel/distributed/ptx.cuh>

#include <cstdint>
#include <type_traits>

namespace sglang::device::distributed::push {

/// Pushes this rank's vector `vid` into the slot at byte `offset` of the workspaces `dst[0, kNumDst)`, its +0 words
/// remapped to -0 so that no word looks empty.
template <typename T, uint32_t kNumDst>
struct PushPort {
 public:
  using Vec = VecStorage<T>;
  using Lamport = VecLamport<T>;

  SGL_DEVICE PushPort(uint8_t* const* dst, uint32_t offset) : m_dst(dst), m_offset(offset) {}

  SGL_DEVICE void store(uint32_t vid, const Vec& vec) const {
    auto v = vec;
    Lamport::clear_pos_zero(v.data());
#pragma unroll
    for (uint32_t i = 0; i < kNumDst; ++i) {
      this->put(i, vid, v);
    }
  }

  SGL_DEVICE void store(uint32_t dst, uint32_t vid, const Vec& vec) const {
    auto v = vec;
    Lamport::clear_pos_zero(v.data());
    this->put(dst, vid, v);
  }

  SGL_DEVICE void store_except(uint32_t skip, uint32_t vid, const Vec& vec) const {
    auto v = vec;
    Lamport::clear_pos_zero(v.data());
#pragma unroll
    for (uint32_t i = 0; i < kNumDst; ++i) {
      if (i != skip) this->put(i, vid, v);
    }
  }

 private:
  SGL_DEVICE void put(uint32_t dst, uint32_t vid, const Vec& v) const {
    ptx::st_relaxed_16B(v, m_dst[dst] + m_offset, vid);
  }

  uint8_t* const* m_dst;
  uint32_t m_offset;
};

/// Pushes this rank's vector `vid` into the slot at byte `offset` of every rank's workspace with one multicast store
/// to the workspaces' multicast VA `mc`, its +0 words remapped to -0 as `PushPort`'s.
template <typename T>
struct MulticastPushPort {
 public:
  using Vec = VecStorage<T>;
  using Lamport = VecLamport<T>;

  SGL_DEVICE MulticastPushPort(uint8_t* mc, uint32_t offset) : m_dst(mc + offset) {}

  SGL_DEVICE void store(uint32_t vid, const Vec& vec) const {
    auto v = vec;
    Lamport::clear_pos_zero(v.data());
    ptx::st_multimem_16B(v, m_dst, vid);
  }

 private:
  uint8_t* m_dst;
};

/// This rank's `kWorldSize` slots of one round: rank `src`'s vector `vid` lands in slot `src` at `round + src * slot`.
template <typename T, uint32_t kWorldSize>
struct PollPort {
 public:
  using Vec = VecStorage<T>;
  using Lamport = VecLamport<T>;

  SGL_DEVICE PollPort(uint8_t* round, uint32_t slot_bytes) : m_round(round), m_slot_bytes(slot_bytes) {}

  /// Spins until vector `vid` of every rank has arrived, then sums them in rank order in fp32 and rounds to T (bf16).
  SGL_DEVICE Vec wait_reduce(uint32_t vid) const {
    static_assert(std::is_same_v<T, bf16_t>, "the sum uses bf16 x bf16 -> fp32 FMAs");
    Vec vec[kWorldSize];
    bool pending;
    do {
      pending = false;
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        ptx::ld_relaxed_16B(vec[i], m_round + i * m_slot_bytes, vid);
        pending |= Lamport::has_pos_zero(vec[i].data());
      }
    } while (pending);
    // x * 1 + acc is exact up to the one rounding of the add, so the mixed-precision FMA gives the bits of an
    // fp32 add without converting each bf16 value first
    Vec out;
#pragma unroll
    for (uint32_t j = 0; j < kVecPairs<T>; ++j) {
      auto acc = cast<fp32x2_t>(vec[0][j]);
#pragma unroll
      for (uint32_t i = 1; i < kWorldSize; ++i) {
        acc.x = math::fma_f32_bf16(vec[i][j].x, bf16_t{1.0f}, acc.x);
        acc.y = math::fma_f32_bf16(vec[i][j].y, bf16_t{1.0f}, acc.y);
      }
      out[j] = cast<packed_t<T>>(acc);
    }
    return out;
  }

  /// Spins until vector `vid` of rank `src` has arrived in its slot (2-shot: the owner's reduced vector).
  SGL_DEVICE Vec wait(uint32_t src, uint32_t vid) const {
    Vec vec;
    do {
      ptx::ld_relaxed_16B(vec, m_round + src * m_slot_bytes, vid);
    } while (Lamport::has_pos_zero(vec.data()));
    return vec;
  }

  /// Restores the empty marker of vector `vid` in slot `src`.
  SGL_DEVICE void reset(uint32_t src, uint32_t vid) const {
    Vec zero;
    Lamport::fill_pos_zero(zero.data());
    ptx::st_global_16B(zero, m_round + src * m_slot_bytes, vid);
  }

  /// Restores the empty marker of vector `vid` in every slot, for the next round on this phase.
  SGL_DEVICE void reset(uint32_t vid) const {
    Vec zero;
    Lamport::fill_pos_zero(zero.data());
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::st_global_16B(zero, m_round + i * m_slot_bytes, vid);
    }
  }

 private:
  uint8_t* m_round;
  uint32_t m_slot_bytes;
};

}  // namespace sglang::device::distributed::push
