// JIT copy of sgl-kernel's kernels/aot/csrc/allreduce/quick_all_reduce.cuh restructured for latency, cut to the
// INT4 and INT8 codecs at TP4 on bf16 data reduced in fp16; same buffer sizes, flags and arithmetic, so its outputs
// are bit-identical.
#pragma once

#include <hip/hip_runtime.h>

#include "quick_all_reduce_base.h"

namespace quickreduce {

struct CodecBase {
  const int thread;
  const int rank;
  __quickreduce_device_inline__ CodecBase(int thread, int rank) : thread(thread), rank(rank) {
    set_fp16_ovfl(true);
  }
};

// Int4 symmetric quantization codec at TP4.
// We quantize the FP16 data to block-scaled Int4 in blocks of 4 *
// kThreadGroupSize.
struct CodecQ4 : public CodecBase {
  static constexpr int kWorldSize = 4;
  // Atoms (16B) per thread in a block's tile, and the tile's bytes.
  static constexpr int kAtoms = 8;
  static constexpr int kTileSize = kBlockSize * kAtoms * sizeof(int32x4_t);

  // Codec tile size process by this workgroup.
  // Each threads processes a fragment of fp16x8_t (16B),
  // into a int4x8_t (4B) and a fp16 scale shared among 32 values.
  static constexpr int kRankAtoms = kAtoms / kWorldSize;
  static constexpr int kRankTileStride = 1152;
  static constexpr int kRankTransmittedTileSize = kRankTileStride * kRankAtoms;
  static_assert(kRankTransmittedTileSize % 16 == 0, "kRankTransmittedTileSize must be 16B aligned.");

  // Total tile size for the collective communication.
  static constexpr int kTransmittedTileSize = kRankTransmittedTileSize * kWorldSize;

  // Constants configuration

  // {-1/8.0h, -1/8.0h}, f16x2_t
  static constexpr int kScaleFactor = 0xB000B000;

  // {1e-7, 1e-7}, f16x2_t
  static constexpr int kScaleEpsilon = 0x00010001;

  // {-8, -8}, f16x2_t
  static constexpr int kRangeMin = 0xC800C800;

  // {+7, +7}, f16x2_t
  static constexpr int kRangeMax = 0x47004700;

  // {+8, +8}, int16x2_t
  static constexpr int kRangeBias = 0x00080008;

  __quickreduce_device_inline__ CodecQ4(int thread, int rank) : CodecBase(thread, rank) {}

  struct Packed {
    int32_t qw;
    int scale;
  };

  __quickreduce_device_inline__ void encode(int32x4_t const* __restrict__ data, Packed* __restrict__ packed) {
    for (int k = 0; k < kRankAtoms; k++) {
      int32x4_t const atom = data[k];

      // Compute the absolute maximum of the atom in the thread group
      // In 2 blocks of values, upper/lower halves of the f16x2_t
      int wblockmax = group_abs_max<half>(atom);

      // Derive scales
      int decoding_scale;
      int encoding_scale;
      decoding_scale = packed_mul<half>(wblockmax, kScaleFactor);
      encoding_scale = packed_add<half>(decoding_scale, kScaleEpsilon);
      encoding_scale = packed_rcp<half>(encoding_scale);

      // Apply scales to get quantized values
      int32x4_t w;
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<half>(atom[i], encoding_scale);
        w[i] = packed_max<half>(w[i], kRangeMin);
        w[i] = packed_min<half>(w[i], kRangeMax);
      }

      // Convert from f16x2_t to uint16x2_t
      int32x4_t q;
      {
        int16_t* qi = reinterpret_cast<int16_t*>(&q);
        half* wh = reinterpret_cast<half*>(&w);
        for (int i = 0; i < 8; i++)
          qi[i] = (int16_t)rintf(T2float_cast(wh[i]));

        for (int i = 0; i < 4; i++) {
          q[i] = packed_add<int16_t>(q[i], kRangeBias);
        }
      }

      // Pack 8 x q4 into int32_t
      int qw = q[0] | (q[1] << 4) | (q[2] << 8) | (q[3] << 12);

      packed[k].qw = qw;
      packed[k].scale = decoding_scale;
    }
  }

  // Wave-major wire layout: per wave, [64 lanes x kRankAtoms qw][8 groups x kRankAtoms scales], so a
  // lane moves its atoms in one access and a wave's scales fill whole lines.
  static constexpr int kWaveSegment = kRankAtoms * (kWavefront * 4 + 8 * 4);
  static_assert(kWaveSegment * (kBlockSize / kWavefront) == kRankTransmittedTileSize, "layout covers the rank tile");

  __quickreduce_device_inline__ void store(uint8_t* __restrict__ send_buffer, Packed const* __restrict__ packed) {
    int wave = thread / kWavefront, lane = thread % kWavefront;
    uint8_t* wave_buffer = send_buffer + wave * kWaveSegment;
    static_assert(kRankAtoms == 2, "a lane moves its two atoms in one 8B access");
    int32x2_t qw = {packed[0].qw, packed[1].qw};
    __builtin_nontemporal_store(qw, QR_GLOBAL(int32x2_t, wave_buffer) + lane);
    // Scale slot s holds (atom s / 8, group s % 8) of this wave; every lane writes one, and the
    // lanes that share a slot write the same value.
    int slot = lane % (8 * kRankAtoms);
    int src_lane = (slot % 8) * kThreadGroupSize;
    int scale = 0;
    for (int k = 0; k < kRankAtoms; k++) {
      int v = __shfl(packed[k].scale, src_lane);
      scale = (slot / 8 == k) ? v : scale;
    }
    __builtin_nontemporal_store(scale, QR_GLOBAL(int, wave_buffer + kWavefront * 4 * kRankAtoms) + slot);
  }

  __quickreduce_device_inline__ void load(uint8_t* __restrict__ recv_buffer, Packed* __restrict__ packed) {
    int wave = thread / kWavefront, lane = thread % kWavefront;
    uint8_t* wave_buffer = recv_buffer + wave * kWaveSegment;
    int32x2_t qw = __builtin_nontemporal_load(QR_GLOBAL(int32x2_t, wave_buffer) + lane);
    packed[0].qw = qw[0];
    packed[1].qw = qw[1];
    int group = lane / kThreadGroupSize;
    for (int k = 0; k < kRankAtoms; k++) {
      packed[k].scale =
          __builtin_nontemporal_load(QR_GLOBAL(int, wave_buffer + kWavefront * 4 * kRankAtoms) + k * 8 + group);
    }
  }

  __quickreduce_device_inline__ void decode(Packed const* __restrict__ packed, int32x4_t* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      int32_t qw = packed[k].qw;
      int qs = packed[k].scale;

      // Unpack q4 into f16x8_t
      int32x4_t w;
      {
        static constexpr uint kMask000F = 0x000F000F;
        static constexpr uint kHalf2_1024 = 0x64006400;  // {1024.0, 1024.0}, fp16x2_t
        static uint constexpr kHalf2_1032 = 0xE408E408;  // {-1032.0, -1032.0}, fp16x2_t

        for (int i = 0; i < 4; i++) {
          int32_t q4 = ((qw >> (i * 4)) & kMask000F) | kHalf2_1024;
          w[i] = packed_add<half>(q4, kHalf2_1032);
        }
      }

      // Apply decoding scales
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<half>(w[i], qs);
      }

      data[k] = w;
    }
  }
};

// Int8 symmetric quantization codec at TP4.
// We quantize the FP16 data to block-scaled Int8 in blocks of 4 *
// kThreadGroupSize.
struct CodecQ8 : public CodecBase {
  static constexpr int kWorldSize = 4;
  // Atoms (16B) per thread in a block's tile, and the tile's bytes.
  static constexpr int kAtoms = 8;
  static constexpr int kTileSize = kBlockSize * kAtoms * sizeof(int32x4_t);

  // Codec tile size process by this workgroup.
  // Each threads processes a fragment of f16x8_t (16B),
  // into a int8x8_t (8B) and a f16 scale shared among 32 values.
  static constexpr int kRankAtoms = kAtoms / kWorldSize;
  static constexpr int kRankTileStride = 2176;
  static constexpr int kRankTransmittedTileSize = kRankTileStride * kRankAtoms;
  static_assert(kRankTransmittedTileSize % 16 == 0, "kRankTransmittedTileSize must be 16B aligned.");

  // Total tile size for the collective communication.
  static constexpr int kTransmittedTileSize = kRankTransmittedTileSize * kWorldSize;

  // Constants configuration

  // {-1/128.0h, -1/128.0h}, f16x2_t
  static constexpr int kScaleFactor = 0xA000A000;

  // {1e-7, 1e-7}, f16x2_t
  static constexpr int kScaleEpsilon = 0x00010001;

  // {-128, -128}, f16x2_t
  static constexpr int kRangeMin = 0xD800D800;

  // {+127, +127}, f16x2_t
  static constexpr int kRangeMax = 0x57F057F0;

  // {+128, +128}, int16x2_t
  static constexpr int kRangeBias = 0x00800080;

  __quickreduce_device_inline__ CodecQ8(int thread, int rank) : CodecBase(thread, rank) {}

  struct Packed {
    int32x2_t qw;
    int scale;
  };

  __quickreduce_device_inline__ void encode(int32x4_t const* __restrict__ data, Packed* __restrict__ packed) {
    for (int k = 0; k < kRankAtoms; k++) {
      int32x4_t const atom = data[k];

      // Compute the absolute maximum of the atom in the thread group
      // In 2 blocks of values, upper/lower halves of the f16x2_t
      int wblockmax = group_abs_max<half>(atom);

      // Derive scales
      int decoding_scale;
      int encoding_scale;
      decoding_scale = packed_mul<half>(wblockmax, kScaleFactor);
      encoding_scale = packed_add<half>(decoding_scale, kScaleEpsilon);
      encoding_scale = packed_rcp<half>(encoding_scale);

      // Apply scales to get quantized values
      int32x4_t w;
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<half>(atom[i], encoding_scale);
        w[i] = packed_max<half>(w[i], kRangeMin);
        w[i] = packed_min<half>(w[i], kRangeMax);
      }

      // Convert from f16x2_t to uint16x2_t
      int32x4_t q;
      {
        int16_t* qi = reinterpret_cast<int16_t*>(&q);
        half* wh = reinterpret_cast<half*>(&w);
        for (int i = 0; i < 8; i++)
          qi[i] = (int16_t)rintf(T2float_cast(wh[i]));

        for (int i = 0; i < 4; i++) {
          q[i] = packed_add<int16_t>(q[i], kRangeBias);
        }
      }

      // Pack 8 x q8 into int32x2_t
      int32x2_t qw;
      qw[0] = q[0] | (q[1] << 8);
      qw[1] = q[2] | (q[3] << 8);

      packed[k].qw = qw;
      packed[k].scale = decoding_scale;
    }
  }

  // Wave-major wire layout, as CodecQ4's: per wave [64 lanes x kRankAtoms qw][8 groups x kRankAtoms scales].
  static constexpr int kWaveSegment = kRankAtoms * (kWavefront * 8 + 8 * 4);
  static_assert(kWaveSegment * (kBlockSize / kWavefront) == kRankTransmittedTileSize, "layout covers the rank tile");

  __quickreduce_device_inline__ void store(uint8_t* __restrict__ send_buffer, Packed const* __restrict__ packed) {
    int wave = thread / kWavefront, lane = thread % kWavefront;
    uint8_t* wave_buffer = send_buffer + wave * kWaveSegment;
    static_assert(kRankAtoms == 2, "a lane moves its two atoms in one 16B access");
    int32x4_t qw = {packed[0].qw[0], packed[0].qw[1], packed[1].qw[0], packed[1].qw[1]};
    __builtin_nontemporal_store(qw, QR_GLOBAL(int32x4_t, wave_buffer) + lane);
    // Scale slot s holds (atom s / 8, group s % 8) of this wave; every lane writes one, and the
    // lanes that share a slot write the same value.
    int slot = lane % (8 * kRankAtoms);
    int src_lane = (slot % 8) * kThreadGroupSize;
    int scale = 0;
    for (int k = 0; k < kRankAtoms; k++) {
      int v = __shfl(packed[k].scale, src_lane);
      scale = (slot / 8 == k) ? v : scale;
    }
    __builtin_nontemporal_store(scale, QR_GLOBAL(int, wave_buffer + kWavefront * 8 * kRankAtoms) + slot);
  }

  __quickreduce_device_inline__ void load(uint8_t* __restrict__ recv_buffer, Packed* __restrict__ packed) {
    int wave = thread / kWavefront, lane = thread % kWavefront;
    uint8_t* wave_buffer = recv_buffer + wave * kWaveSegment;
    int32x4_t qw = __builtin_nontemporal_load(QR_GLOBAL(int32x4_t, wave_buffer) + lane);
    packed[0].qw[0] = qw[0];
    packed[0].qw[1] = qw[1];
    packed[1].qw[0] = qw[2];
    packed[1].qw[1] = qw[3];
    int group = lane / kThreadGroupSize;
    for (int k = 0; k < kRankAtoms; k++) {
      packed[k].scale =
          __builtin_nontemporal_load(QR_GLOBAL(int, wave_buffer + kWavefront * 8 * kRankAtoms) + k * 8 + group);
    }
  }

  __quickreduce_device_inline__ void decode(Packed const* __restrict__ packed, int32x4_t* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      int32x2_t qw = packed[k].qw;
      int qs = packed[k].scale;

      // Unpack q8 into f16x8_t
      int32x4_t w;
      {
        static uint constexpr kMask00FF = 0x00FF00FF;
        static uint constexpr kHalf2_1024 = 0x64006400;  // {1024.0, 1024.0}, fp16x2_t
        static uint constexpr kHalf2_1152 = 0xE480E480;  // {-1152.0, -1152.0}, fp16x2_t

        for (int i = 0; i < 4; i++) {
          int32_t q8 = ((qw[i / 2] >> ((i % 2) * 8)) & kMask00FF) | kHalf2_1024;
          w[i] = packed_add<half>(q8, kHalf2_1152);
        }
      }

      // Apply decoding scales
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<half>(w[i], qs);
      }

      data[k] = w;
    }
  }
};

// Twoshot All Reduce of bf16 data in the fp16 kernels: converted on load and store. The block scales
// normalize every 32 values and MODE.FP16_OVFL clamps an overflow, so the cast needs no range scale.
template <class Codec>
struct AllReduceTwoshot {
  static constexpr int kWorldSize = Codec::kWorldSize;
  static constexpr int kAtoms = Codec::kAtoms;
  static constexpr int kTileSize = Codec::kTileSize;

  // Lane i's peer buffer, selected from registers: a dynamic index would put the array in scratch.
  __quickreduce_device_inline__ static uint8_t* lane_peer(uint8_t* const* peer, int i) {
    uint8_t* p = peer[0];
#pragma unroll
    for (int r = 1; r < kWorldSize; r++) {
      p = (i == r) ? peer[r] : p;
    }
    return p;
  }

  // Waits for every rank's flag at once (lane r polls rank r), then puts all ranks' loads in flight.
  __quickreduce_device_inline__ static void wait_and_load(
      Codec& codec,
      uint8_t* recv_buffer,
      uint32_t* flag_ptr,
      uint32_t flag_color,
      typename Codec::Packed (&packed)[kWorldSize][Codec::kRankAtoms]) {
    int thread = threadIdx.x + threadIdx.y * kWavefront;
    if (thread < kWorldSize) {
      wait_sync_flag(&flag_ptr[thread], flag_color);
    }
    __syncthreads();
    for (int r = 0; r < kWorldSize; r++) {
      codec.load(recv_buffer + r * Codec::kRankTransmittedTileSize, packed[r]);
    }
  }

  __quickreduce_device_inline__ static void
  raise_flags(uint8_t* const* peer, uint32_t flags_offset, int rank, uint32_t flag_color) {
    int thread = threadIdx.x + threadIdx.y * kWavefront;
    wait_own_stores();
    __syncthreads();
    if (thread < kWorldSize) {
      set_sync_flag(
          reinterpret_cast<uint32_t*>(lane_peer(peer, thread) + flags_offset + rank * sizeof(uint32_t)), flag_color);
    }
  }

  __device__ static void
  run(half const* __restrict__ input,
      half* __restrict__ output,
      uint32_t const N,            // number of elements
      int const block,             // block index
      int const rank,              // rank index
      uint8_t* const* peer,        // [kWorldSize] communication buffers, held in registers
      uint32_t const data_offset,  // offset to start of the data buffer
      uint32_t flag_color,
      int64_t data_size_per_phase) {
    int thread = threadIdx.x + threadIdx.y * kWavefront;
    uint8_t* rank_buffer = lane_peer(peer, rank);
    Codec codec(thread, rank);
    int block_id = blockIdx.x;
    // --------------------------------------------------------
    // Read input into registers
    int32x4_t tA[kAtoms];

    BufferResource src_buffer(const_cast<half*>(input), N * sizeof(half));
    uint32_t src_offset = block * kTileSize + thread * sizeof(int32x4_t);
    for (int i = 0; i < kAtoms; i++) {
      tA[i] = buffer_load_dwordx4(src_buffer.descriptor, src_offset, 0, 0);
      src_offset += kAtomStride * sizeof(int32x4_t);
    }
    // Every input load in flight before the first conversion: left to itself, the scheduler
    // interleaves them with the conversions and waits for each load in turn.
    __builtin_amdgcn_sched_barrier(0);
    for (int i = 0; i < kAtoms; i++) {
      const nv_bfloat162* bf_buf = reinterpret_cast<const nv_bfloat162*>(&tA[i]);
      half2 half_buf[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        half_buf[j] = __float22half2_rn(__bfloat1622float2(bf_buf[j]));
      }
      tA[i] = *reinterpret_cast<const int32x4_t*>(half_buf);
    }

    uint32_t comm_data0_offset = data_offset + block_id * Codec::kTransmittedTileSize;
    uint32_t comm_data1_offset = data_size_per_phase + comm_data0_offset;

    uint32_t comm_flags0_offset = block_id * (kWorldSize * sizeof(uint32_t));
    uint32_t comm_flags1_offset = (data_offset / 2) + comm_flags0_offset;

    // --------------------------------------------------------
    // Phase-1A: Write segment data into the communication buffer of the target
    // rank responsible for this segment.
    for (int r = 0; r < kWorldSize; r++) {
      typename Codec::Packed packed[Codec::kRankAtoms];
      codec.encode(&tA[r * Codec::kRankAtoms], packed);
      codec.store(peer[r] + comm_data0_offset + rank * Codec::kRankTransmittedTileSize, packed);
    }
    raise_flags(peer, comm_flags0_offset, rank, flag_color);

    // --------------------------------------------------------
    // Phase-1B: Reduce the segment data from the communication buffers, in rank order.
    int32x4_t tR[Codec::kRankAtoms] = {};
    {
      typename Codec::Packed packed[kWorldSize][Codec::kRankAtoms];
      wait_and_load(
          codec,
          rank_buffer + comm_data0_offset,
          reinterpret_cast<uint32_t*>(rank_buffer + comm_flags0_offset),
          flag_color,
          packed);
      for (int r = 0; r < kWorldSize; r++) {
        // note: we reuse tA as temp buffer here
        codec.decode(packed[r], tA);
        for (int i = 0; i < Codec::kRankAtoms; i++) {
          packed_assign_add<half>(&tR[i], &tA[i]);
        }
      }
    }

    // Phase-2: Write the reduced segment to every other rank, encoded once.
    {
      typename Codec::Packed packed[Codec::kRankAtoms];
      codec.encode(tR, packed);
      for (int r = 0; r < kWorldSize; r++) {
        codec.store(peer[r] + comm_data1_offset + rank * Codec::kRankTransmittedTileSize, packed);
      }
    }
    raise_flags(peer, comm_flags1_offset, rank, flag_color);

    // Phase-2: Read the gather segments from the rank's communication buffer.
    {
      typename Codec::Packed packed[kWorldSize][Codec::kRankAtoms];
      wait_and_load(
          codec,
          rank_buffer + comm_data1_offset,
          reinterpret_cast<uint32_t*>(rank_buffer + comm_flags1_offset),
          flag_color,
          packed);
      for (int r = 0; r < kWorldSize; r++) {
        // Gather all reduced and final rank segments into tA.
        codec.decode(packed[r], &tA[r * Codec::kRankAtoms]);
      }
    }

    // --------------------------------------------------------
    // Write the result to output.
    BufferResource dst_buffer(output, N * sizeof(half));
    uint32_t dst_offset = block * kTileSize + thread * sizeof(int32x4_t);

    for (int i = 0; i < kAtoms; i++) {
      const half2* half_buf = reinterpret_cast<const half2*>(&tA[i]);
      nv_bfloat162 bf16_buf[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        bf16_buf[j] = __float22bfloat162_rn(__half22float2(half_buf[j]));
      }
      buffer_store_dwordx4(*reinterpret_cast<const int32x4_t*>(bf16_buf), dst_buffer.descriptor, dst_offset, 0, 0);
      dst_offset += kAtomStride * sizeof(int32x4_t);
    }
  }
};

}  // namespace quickreduce
