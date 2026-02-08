/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Utilities for mixed input data type kernels.
*/

#pragma once

#include <cuda.h>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/arch/mma_sm90.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cute/util/type_traits.hpp"

namespace cutlass {

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

template <
  class QuantizedElement,
  class DequantizedElement,
  class OperandLayout,
  class ElementScale,
  class ElementZero,
  class ScaleBroadCastLayout,
  class ThrLayout>
__global__ void dequantize_kernel(DequantizedElement* dq_buffer,
                                  QuantizedElement const* q_buffer,
                                  OperandLayout const operand_layout,
                                  ElementScale const* scale_buffer,
                                  ElementZero const* zero_buffer,
                                  ScaleBroadCastLayout const broadcasted_scale_layout,
                                  ThrLayout thr_layout) {
  using namespace cute;

  // Represent the full tensors to gmem elements.
  // These are expected to have shape [MN, K, L]
  cute::Tensor gmem_op_dq = cute::make_tensor(cute::make_gmem_ptr(dq_buffer), operand_layout);
  cute::Tensor gmem_op_q  = cute::make_tensor(cute::make_gmem_ptr<QuantizedElement const>(q_buffer), operand_layout);
  // While the scales are expected to have shape [MN, G, L] but with a stride to allow broadcasting
  // It is expected that K % G == 0
  cute::Tensor gmem_scale_broadcasted = cute::make_tensor(make_gmem_ptr(scale_buffer), broadcasted_scale_layout);
  cute::Tensor gmem_zero_broadcasted = cute::make_tensor(make_gmem_ptr(zero_buffer), broadcasted_scale_layout);

  // Assign 1 thread per element in the thread block
  auto blk_shape = cute::make_shape(size<0>(thr_layout), _1{}, _1{}); //
  auto blk_coord = cute::make_coord(_, blockIdx.x, blockIdx.y);  // (MN, K, L)

  // Tile across the block
  auto gOp_dq = cute::local_tile(gmem_op_dq, blk_shape, blk_coord);
  auto gScale = cute::local_tile(gmem_scale_broadcasted, blk_shape, blk_coord);
  auto gZero  = cute::local_tile(gmem_zero_broadcasted,  blk_shape, blk_coord);
  auto gOp_q  = cute::local_tile(gmem_op_q, blk_shape, blk_coord);

  auto tOpDq_gOpDq = cute::local_partition(gOp_dq, thr_layout, threadIdx.x);
  auto tScale_gScale = cute::local_partition(gScale, thr_layout, threadIdx.x);
  auto tZero_gZero = cute::local_partition(gZero, thr_layout, threadIdx.x);
  auto tOpQ_gOpQ = cute::local_partition(gOp_q, thr_layout, threadIdx.x);

  // Make a fragment of registers to hold gmem loads
  cute::Tensor rmem_op_q = cute::make_fragment_like(tOpQ_gOpQ(_, _, _, 0));
  cute::Tensor rmem_scale = cute::make_fragment_like(tScale_gScale(_, _, _, 0));
  cute::Tensor rmem_zero = cute::make_fragment_like(tZero_gZero(_, _, _, 0));
  cute::Tensor rmem_op_dq = cute::make_fragment_like(tOpDq_gOpDq(_, _, _, 0));
  cute::Tensor rmem_op_scaled = cute::make_fragment_like<ElementScale>(rmem_op_dq);
  cute::Tensor rmem_zero_buf = cute::make_fragment_like<ElementScale>(rmem_zero);

  cute::Tensor pred_id = cute::make_identity_tensor(shape(operand_layout));
  auto pred_blk_tile = cute::local_tile(pred_id, blk_shape, blk_coord);
  auto pred_thr_partition = cute::local_partition(pred_blk_tile, thr_layout, threadIdx.x);

  const auto num_iters = cute::size<3>(tOpDq_gOpDq);

  for (int ii = 0; ii < num_iters; ++ii) {
    const auto thread_offset = cute::get<0>(pred_thr_partition(0, 0, 0, ii));
    if (thread_offset < cute::size<0>(operand_layout)) {
      cute::copy(tOpQ_gOpQ(_, _, _, ii), rmem_op_q);
      cute::copy(tScale_gScale(_, _, _, ii), rmem_scale);
      cute::copy(tZero_gZero(_, _, _, ii), rmem_zero);
      cute::transform(rmem_op_q, rmem_op_scaled, [] (const QuantizedElement& elt) { return ElementScale(elt); } );
      cute::transform(rmem_zero, rmem_zero_buf, [] (const ElementZero& elt) { return ElementScale(elt); } );
      cute::transform(rmem_op_scaled, rmem_scale, rmem_op_scaled, cute::multiplies{});
      cute::transform(rmem_op_scaled, rmem_zero_buf, rmem_op_scaled, cute::plus{});
      cute::transform(rmem_op_scaled, rmem_op_dq, [] (const ElementScale& elt) { return DequantizedElement(elt); } );
      cute::copy(rmem_op_dq, tOpDq_gOpDq(_, _, _, ii));
    }
  }
}

template <
  class QuantizedElement,
  class DequantizedElement,
  class OperandLayout,
  class ElementScale,
  class ElementZero,
  class ScaleLayout>
static void dequantize(DequantizedElement* dq_buffer,
                       QuantizedElement const* q_buffer,
                       OperandLayout const operand_layout,
                       ElementScale const* scale_buffer,
                       ElementZero const* zero_buffer,
                       ScaleLayout const scale_layout,
                       int const group_size,
                       cudaStream_t &stream) {
  using namespace cute;

  constexpr int tpb = 128;
  auto thr_layout = make_layout(make_shape(Int<tpb>{}));

  const auto num_rows = get<0>(shape(operand_layout));
  const auto gemm_k = get<1>(shape(operand_layout));   // [MN, K, L]
  const auto batches = get<2>(shape(operand_layout));  // [MN, K, L]
  const auto scale_k = get<1>(shape(scale_layout));    // [MN, Scale_K, L]

  if (num_rows != size<0>(scale_layout)) {
    std::cerr << "Invalid first dimension for scales. Must match first dim for weights."
              << " But got shapes " << shape(operand_layout) << " " << shape(scale_layout)
              << std::endl;
    exit(-1);
  }

  const auto scale_stride0 = get<0>(stride(scale_layout));
  const auto scale_stride1 = get<1>(stride(scale_layout));
  const auto scale_stride2 = get<2>(stride(scale_layout));

  auto scale_shape_bcast = make_shape(num_rows, make_shape(group_size, scale_k), batches);
  auto scale_stride_bcast = make_stride(scale_stride0, make_stride(0, scale_stride1), scale_stride2);
  auto scale_layout_bcast = make_layout(scale_shape_bcast, scale_stride_bcast);

  const auto blocks_x = gemm_k;
  const auto blocks_y = batches;

  dim3 blocks(blocks_x, blocks_y, 1);
  dequantize_kernel<<<blocks, tpb, 0, stream>>>(dq_buffer, q_buffer, operand_layout, scale_buffer, zero_buffer, scale_layout_bcast, thr_layout);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename T>
class packed_scale_t {
public:
  static_assert(cute::is_same_v<T, cutlass::int8_t> ||
                cute::is_same_v<T, cutlass::uint8_t> ||
                cute::is_same_v<T, cutlass::float_e4m3_t> ||
                cute::is_same_v<T, cutlass::float_e5m2_t>,
                "only 8 bit arithmetic types are supported.");
  CUTLASS_HOST_DEVICE
  explicit packed_scale_t(T val) {
    if constexpr (!cute::is_unsigned_v<T>) {
      // Only pack negative values. The positive values are generated in flight in the mainloop.
      storage[0] = pack4(T(float(val) * -8.f), T(float(val) * -7.f), T(float(val) * -6.f), T(float(val) * -5.f));
      storage[1] = pack4(T(float(val) * -4.f), T(float(val) * -3.f), T(float(val) * -2.f), -val);
    }
    else {
      storage[0] = pack4(T(float(val) * 8.f), T(float(val) * 7.f), T(float(val) * 6.f), T(float(val) * 5.f));
      storage[1] = pack4(T(float(val) * 4.f), T(float(val) * 3.f), T(float(val) * 2.f), val);
    }
  }
  CUTLASS_HOST_DEVICE
  packed_scale_t() = default;
  CUTLASS_HOST_DEVICE
  explicit operator float() const {
    return float(get());
  }
  CUTLASS_HOST_DEVICE
  bool operator==(packed_scale_t const& rhs) const {
    return storage[0] == rhs.storage[0] && storage[1] == rhs.storage[1];
  }
  CUTLASS_HOST_DEVICE
  bool operator!=(packed_scale_t const& rhs) const {
    return !(*this == rhs);
  }
  CUTLASS_HOST_DEVICE
  friend packed_scale_t operator+(packed_scale_t const& lhs, packed_scale_t const& rhs) {
    return packed_scale_t(lhs.get() + rhs.get());
  }
  CUTLASS_HOST_DEVICE
  friend packed_scale_t operator-(packed_scale_t const& lhs, packed_scale_t const& rhs) {
    return packed_scale_t(lhs.get() - rhs.get());
  }
  CUTLASS_HOST_DEVICE
  friend packed_scale_t operator*(packed_scale_t const& lhs, packed_scale_t const& rhs) {
    return packed_scale_t(lhs.get() * rhs.get());
  }
  CUTLASS_HOST_DEVICE
  friend packed_scale_t operator/(packed_scale_t const& lhs, packed_scale_t const& rhs) {
    return packed_scale_t(lhs.get() / rhs.get());
  }

private:
  using Storage = uint32_t;
  using Stage = uint8_t;

  Storage storage[2] {};

  CUTLASS_HOST_DEVICE
  static Storage pack4(T c1, T c2, T c3, T c4) {
    Storage result = 0;
    result |= (static_cast<Storage>(reinterpret_cast<Stage const&>(c4)) << 24);
    result |= (static_cast<Storage>(reinterpret_cast<Stage const&>(c3)) << 16);
    result |= (static_cast<Storage>(reinterpret_cast<Stage const&>(c2)) << 8);
    result |= static_cast<Storage>(reinterpret_cast<Stage const&>(c1));
    return result;
  }
  CUTLASS_HOST_DEVICE
  T get() const {
    auto stage = static_cast<Stage>(storage[0] >> 8);
    #if defined(__CUDA_ARCH__)
    return reinterpret_cast<T const&>(stage);
    #else
    T tmp;
    std::memcpy(&tmp, &stage, sizeof(Stage));
    return tmp;
    #endif
  }
  CUTLASS_HOST_DEVICE
  T get(int idx) const {
    Stage stage;
    if (idx < 4) stage = static_cast<Stage>(storage[0] >> (8 * idx));
    else         stage = static_cast<Stage>(storage[1] >> (8 * idx - 32));
    #if defined(__CUDA_ARCH__)
    return reinterpret_cast<T const&>(stage);
    #else
    T tmp;
    std::memcpy(&tmp, &stage, sizeof(Stage));
    return tmp;
    #endif
  }
};

// In the mainloop, PRMT selects 1 byte from only 8 bytes so the sign bit is handled in an extra PRMT.
// Here the encodings of positive values and negative values are unified (except for the sign bit).
// For instance, 1 becomes 0b0111, which is the same encoding as -1 (0b1111).
static bool unified_encode_int4b(cutlass::int4b_t const *block_in, cutlass::int4b_t *block_out, const size_t block_size) {

  using StorageType = cutlass::int4b_t::Storage;
  constexpr int pack = cute::sizeof_bits_v<StorageType> / 4;
  const size_t host_buf_size = block_size / pack;
  std::vector<StorageType> host_buf(host_buf_size);
  cutlass::device_memory::copy_to_host(host_buf.data(), (StorageType *) block_in, host_buf_size);

  for (auto&& d : host_buf) {
    StorageType out = 0;
    StorageType mask = 0x0f;
    for (int i = 0; i < pack; i++) {
      cutlass::int4b_t curr;
      curr.storage = (d >> (i * 4)) & 0x0f;
      switch (curr) {
        case 1: curr.storage = StorageType(0b0111); break; // 2's complement
        case 2: curr.storage = StorageType(0b0110); break; // 2's complement
        case 3: curr.storage = StorageType(0b0101); break; // 2's complement
        case 4: curr.storage = StorageType(0b0100); break; // 2's complement
        case 5: curr.storage = StorageType(0b0011); break; // 2's complement
        case 6: curr.storage = StorageType(0b0010); break; // 2's complement
        case 7: curr.storage = StorageType(0b0001); break; // 2's complement
        default: break;
      }
      out |= (curr.storage << (4 * i)) & mask;
      mask <<= 4;
    }
    d = out;
  }

  cutlass::device_memory::copy_to_device((StorageType*) block_out, host_buf.data(), host_buf_size);
  return true;
}

template <class ElementScale>
static bool pack_scale_fp8(ElementScale const *block_in, cutlass::Array<ElementScale, 8> *block_out, const size_t block_size) {
  std::vector<ElementScale> data_in(block_size);
  std::vector<cutlass::Array<ElementScale, 8>> data_out(block_size);

  try {
    cutlass::device_memory::copy_to_host(data_in.data(), block_in, block_size);
  }
  catch (cutlass::cuda_exception const& e) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(e.cudaError()) << std::endl;
    return false;
  }

  for (size_t i = 0; i < block_size; i++) {
    cutlass::packed_scale_t<ElementScale> tmp(data_in[i]);
    data_out[i] = reinterpret_cast<cutlass::Array<ElementScale, 8> const&>(tmp);
  }

  try {
    cutlass::device_memory::copy_to_device(block_out, data_out.data(), block_size);
  }
  catch (cutlass::cuda_exception const& e) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(e.cudaError()) << std::endl;
    return false;
  }
  return true;
}

template <class T, class = void>
struct UnderlyingElement {
  using type = T;
};

template <class T>
struct UnderlyingElement<T, cute::void_t<typename T::Element>> {
  using type = typename T::Element;
};

// Given a type of MMA instruction, compute a memory reordering atom that places all values
// owned by each thread in contiguous memory locations. This improves smem load vectorization,
// particularly for mixed dtype GEMMs where a narrow type is loaded in the thread/value order
// of the wider type and may result in inefficient sub-bank (8-bit or 16-bit) accesses.
// In addition, we can reorder the values across several MMA instructions to get even wider
// vectorization (AtomLayout parameter) and permute the values within each instruction to get
// more optimal conversion instruction sequences (ValLayout parameter).
template <class ElementMma,
         class AtomLayout = cute::Layout<cute::_1>,
         class ValLayout  = cute::Layout<cute::_1>>
constexpr auto compute_memory_reordering_atom(AtomLayout atom_layout = {}, ValLayout val_layout = {})
{
  using namespace cute;

  static_assert(is_static_v<ValLayout>, "ValLayout must be static");
  static_assert(is_static_v<AtomLayout>, "AtomLayout must be static");

  // 1. Choose an MMA atom to access TV layout and MN shape
  // Note: parameters like GMMA Major, TileShape, ElementC don't affect TV layout of A, use arbitrary
  using MmaAtom = decltype(SM90::GMMA::rs_op_selector<ElementMma, ElementMma, float, Shape<_64,_16,_32>>());
  using MmaTraits = MMA_Traits<MmaAtom>;
  auto mk_shape_mma = select<0,2>(typename MmaTraits::Shape_MNK{});
  auto tv_layout_mma = typename MmaTraits::ALayout{};
  static_assert(size<1>(tv_layout_mma) % size(val_layout) == 0, "Value layout must evenly divide the MMA value layout");

  // 2. Create a single warp's TV layout from that of the whole MMA and invert to get (m,k -> thr,val)
  // Note: this assumes A is partitioned between warps along M mode
  auto tv_tiler_warp = make_shape(Int<32>{}, size<1>(tv_layout_mma));
  auto mk_shape_warp = shape_div(mk_shape_mma, size(typename MmaTraits::ThrID{}) / Int<32>{});
  auto tv_layout_mma_warp = make_layout_like(composition(tv_layout_mma, tv_tiler_warp));
  auto mk_layout_mma_warp = right_inverse(tv_layout_mma_warp).with_shape(mk_shape_warp);

  // 3. Repeat the warp layout NumAtoms times along K mode to get wider vectorization
  auto mk_layout_mma_trgt = blocked_product(mk_layout_mma_warp, atom_layout);

  // 4. Compose with a contiguous layout of values in each thread (required for smem vectorization)
  auto val_to_offset = logical_product(val_layout, size<1>(tv_layout_mma) / size(val_layout) * size(atom_layout));
  auto thr_to_offset = make_layout(size<0>(tv_layout_mma_warp));
  auto tv_to_offset = select<1,0>(logical_product(val_to_offset, thr_to_offset));
  auto layout_atom = composition(tv_to_offset, mk_layout_mma_trgt);

  return layout_atom;
}

template <class TileShape, class EngineSrc, class LayoutSrc, class EngineDst, class LayoutDst, class TiledCopy>
__global__ void reorder_tensor_kernel(
  cute::Tensor<EngineSrc, LayoutSrc> S,
  cute::Tensor<EngineDst, LayoutDst> D,
  TiledCopy tiled_copy)
{
  using namespace cute;

  using T = typename EngineDst::value_type;

  Tensor gS = local_tile(S, TileShape{}, make_coord(blockIdx.x, _, blockIdx.z));
  Tensor gD = local_tile(D, TileShape{}, make_coord(blockIdx.x, _, blockIdx.z));

  auto thread_copy = tiled_copy.get_slice(threadIdx.x);
  Tensor tS = thread_copy.partition_S(gS);
  Tensor tD = thread_copy.partition_D(gD);

  copy(tiled_copy, tS, tD);
}

template <class EngineSrc, class LayoutSrc, class EngineDst, class LayoutDst>
void reorder_tensor(
  cute::Tensor<EngineSrc, LayoutSrc> S,
  cute::Tensor<EngineDst, LayoutDst> D)
{
  using namespace cute;

  using T = typename EngineDst::value_type;
  static_assert(is_same_v<remove_const_t<typename EngineSrc::value_type>, T>, "Type mismatch");

  // Construct a value layout that assigns at least 8 bits of contiguous elements in destination tensor to a thread
  // This avoids a race condition when writing out subbyte types (e.g. int4b_t).
  auto has_major_mode = [](auto s) {
    return any_of(flatten(s), [](auto a){ return is_constant<1, decltype(a)>{}; });
  };
  static_assert(has_major_mode(stride<0>(LayoutDst{})) ^ has_major_mode(stride<1>(LayoutDst{})),
                "Could not find stride-1 mode in destination layout");
  constexpr int N = shape_div(Int<8>{}, Int<sizeof_bits_v<T>>{});
  auto val_layout = conditional_return<has_major_mode(stride<0>(LayoutDst{}))>(
    make_layout(make_shape(Int<N>{}, Int<1>{}), GenColMajor{}),
    make_layout(make_shape(Int<1>{}, Int<N>{}), GenRowMajor{}));

  // Make a tiled copy with a simple row-major thread order and above layout
  int constexpr NumThreads = 128;
  auto const thr_layout = make_layout(make_shape(Int<1>{}, Int<NumThreads>{}));
  auto tiled_copy = make_tiled_copy(Copy_Atom<DefaultCopy, T>{}, thr_layout, val_layout);

  // Assign a group of 16 rows to a threadblock; this matches the shuffle atom size for Hopper
  using TileShape = Shape<_16>;
  auto tiled_D = group_modes<3,rank_v<LayoutDst>>(tiled_divide(D, TileShape{}));
  dim3 blocks{unsigned(size<1>(tiled_D)), 1u, unsigned(size<3>(tiled_D))};

  reorder_tensor_kernel<TileShape><<<blocks, NumThreads>>>(S, D, tiled_copy);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// In-place version
template <class T, class LayoutSrc, class LayoutDst>
void reorder_tensor(
  T const* src,
  LayoutSrc const& layout_src,
  T * dst,
  LayoutDst const& layout_dst)
{
  using namespace cute;
  reorder_tensor(make_tensor(make_gmem_ptr<T>(src), layout_src),
                 make_tensor(make_gmem_ptr<T>(dst), layout_dst));
}

// In-place version
template <class T, class LayoutSrc, class LayoutDst>
void reorder_tensor(
  T * data,
  LayoutSrc const& layout_src,
  LayoutDst const& layout_dst)
{
  using namespace cute;
  cutlass::DeviceAllocation<T> temp(size(layout_src));
  reorder_tensor(data, layout_src, temp.get(), layout_dst);
  cutlass::device_memory::copy_device_to_device(data, temp.get(), static_cast<size_t>(size(layout_src)));
}

#undef CUDA_CHECK

}  // namespace cutlass
