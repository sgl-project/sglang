// Exact byte transpose for changing weight layout once per request phase.
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
namespace sglang {
namespace nvfp4_weight_transpose {
template <unsigned Rows, unsigned Cols, unsigned Tile>
__global__ __launch_bounds__(256) void kernel(const uint8_t* __restrict__ x, uint8_t* __restrict__ y) {
  __shared__ uint8_t data[Tile][Tile + 4];
  unsigned tx = threadIdx.x % 32, ty = threadIdx.x / 32;
  unsigned rbase = blockIdx.y * Tile, cbase = blockIdx.x * Tile;
  size_t expert = static_cast<size_t>(blockIdx.z) * Rows * Cols;
#pragma unroll
  for (unsigned j = 0; j < Tile / 8; ++j) {
    unsigned row = rbase + ty + j * 8, col = cbase + tx * 4;
    unsigned v = 0;
    if (row < Rows && col < Cols) v = *reinterpret_cast<const unsigned*>(x + expert + row * Cols + col);
    *reinterpret_cast<unsigned*>(&data[ty + j * 8][tx * 4]) = v;
  }
  __syncthreads();
#pragma unroll
  for (unsigned j = 0; j < Tile / 8; ++j) {
    unsigned row = cbase + ty + j * 8, col = rbase + tx * 4;
    if (row < Cols && col < Rows) {
      unsigned v = static_cast<unsigned>(data[tx * 4 + 0][ty + j * 8]) |
                   (static_cast<unsigned>(data[tx * 4 + 1][ty + j * 8]) << 8) |
                   (static_cast<unsigned>(data[tx * 4 + 2][ty + j * 8]) << 16) |
                   (static_cast<unsigned>(data[tx * 4 + 3][ty + j * 8]) << 24);
      *reinterpret_cast<unsigned*>(y + expert + row * Rows + col) = v;
    }
  }
}
}  // namespace nvfp4_weight_transpose
template <unsigned Rows, unsigned Cols>
struct Nvfp4WeightTranspose {
  static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
    using namespace host;
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({512, Rows, Cols}).with_dtype<uint8_t>().with_device(dev).verify(x);
    TensorMatcher({512, Cols, Rows}).with_dtype<uint8_t>().with_device(dev).verify(y);
    LaunchKernel({(Cols + 127) / 128, (Rows + 127) / 128, 512}, 256, dev.unwrap())(
        nvfp4_weight_transpose::kernel<Rows, Cols, 128>,
        static_cast<const uint8_t*>(x.data_ptr()),
        static_cast<uint8_t*>(y.data_ptr()));
  }
};
}  // namespace sglang
