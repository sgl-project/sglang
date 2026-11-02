// SPDX-License-Identifier: Apache-2.0

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

/**
 * \brief Restore BHWC window order, remove padding, and add the shortcut.
 *
 * \tparam kHidden Hidden width, divisible by eight BF16 values.
 * \tparam kUsePDL Whether to emit the PDL wait/trigger pair.
 */
template <int64_t kHidden, bool kUsePDL>
__global__ void window_unpartition_add_kernel(
    bf16_t* __restrict__ output,
    const bf16_t* __restrict__ windows,
    const bf16_t* __restrict__ shortcut,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t window_size,
    uint32_t windows_per_row,
    int64_t stride_batch,
    int64_t stride_row,
    int64_t stride_col,
    int64_t stride_hidden,
    bool width_major) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr uint32_t kElementsPerImage = kHidden;
  const uint32_t elements_per_image = height * width * kElementsPerImage;
  const uint32_t num_elements = batch_size * elements_per_image;
  if (index >= num_elements) {
    return;
  }

  const uint32_t batch = index / elements_per_image;
  const uint32_t local = index % elements_per_image;
  uint32_t row;
  uint32_t col;
  uint32_t hidden;
  if (width_major) {
    col = local % width;
    const uint32_t rest = local / width;
    row = rest % height;
    hidden = rest / height;
  } else {
    hidden = local % kHidden;
    const uint32_t spatial = local / kHidden;
    col = spatial % width;
    row = spatial / width;
  }

  const uint32_t windows_per_image = ((height + window_size - 1) / window_size) * windows_per_row;
  const uint32_t window = batch * windows_per_image + (row / window_size) * windows_per_row + col / window_size;
  const uint32_t inner = (row % window_size) * window_size + col % window_size;
  const int64_t source_offset = (static_cast<int64_t>(window) * window_size * window_size + inner) * kHidden + hidden;
  const int64_t output_offset = static_cast<int64_t>(batch) * stride_batch + static_cast<int64_t>(row) * stride_row +
                                static_cast<int64_t>(col) * stride_col + static_cast<int64_t>(hidden) * stride_hidden;

  device::PDLWaitPrimary<kUsePDL>();
  output[output_offset] = shortcut[output_offset] + windows[source_offset];
  device::PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief Validate tensors and launch window_unpartition_add_kernel.
 */
template <int64_t kHidden, bool kUsePDL>
void window_unpartition_add(
    tvm::ffi::TensorView output, tvm::ffi::TensorView windows, tvm::ffi::TensorView shortcut, int64_t window_size) {
  using namespace host;
  static_assert(kHidden % 8 == 0, "Hidden width must be divisible by eight");

  auto B = SymbolicSize{"batch"};
  auto H = SymbolicSize{"height"};
  auto W = SymbolicSize{"width"};
  auto NW = SymbolicSize{"num_windows"};
  auto WS = SymbolicSize{"window_size"};
  auto C = SymbolicSize{"hidden"};
  auto SB = SymbolicSize{"batch_stride"};
  auto SH = SymbolicSize{"height_stride"};
  auto SW = SymbolicSize{"width_stride"};
  auto SC = SymbolicSize{"hidden_stride"};
  auto device = SymbolicDevice{};
  WS.set_value(window_size);
  C.set_value(kHidden);
  device.set_options<kDLCUDA>();

  TensorMatcher({B, H, W, C})
      .with_strides({SB, SH, SW, SC})
      .with_dtype<bf16_t>()
      .with_device(device)
      .verify(output)
      .verify(shortcut);
  TensorMatcher({NW, WS, WS, C}).with_dtype<bf16_t>().with_device(device).verify(windows);

  const auto batch = static_cast<uint32_t>(B.unwrap());
  const auto height = static_cast<uint32_t>(H.unwrap());
  const auto width = static_cast<uint32_t>(W.unwrap());
  const auto ws = static_cast<uint32_t>(WS.unwrap());
  const auto windows_per_row = div_ceil(width, ws);
  const auto windows_per_image = div_ceil(height, ws) * windows_per_row;
  CHECK_HOST(NW.unwrap() == static_cast<int64_t>(batch) * windows_per_image)
      << "Unexpected number of windows: " << NW.unwrap();

  const bool width_major = SW.unwrap() == 1 && SH.unwrap() == static_cast<int64_t>(width);
  CHECK_HOST(
      width_major ||
      (SC.unwrap() == 1 && SW.unwrap() == kHidden && SH.unwrap() == static_cast<int64_t>(width) * kHidden))
      << "Unsupported shortcut layout";

  constexpr uint32_t kBlockSize = 256;
  const uint32_t num_elements = batch * height * width * kHidden;
  const uint32_t num_blocks = div_ceil(num_elements, kBlockSize);
  LaunchKernel(num_blocks, kBlockSize, device.unwrap())
      .enable_pdl(kUsePDL)(
          window_unpartition_add_kernel<kHidden, kUsePDL>,
          static_cast<bf16_t*>(output.data_ptr()),
          static_cast<const bf16_t*>(windows.data_ptr()),
          static_cast<const bf16_t*>(shortcut.data_ptr()),
          batch,
          height,
          width,
          ws,
          windows_per_row,
          SB.unwrap(),
          SH.unwrap(),
          SW.unwrap(),
          SC.unwrap(),
          width_major);
}

}  // namespace sglang
