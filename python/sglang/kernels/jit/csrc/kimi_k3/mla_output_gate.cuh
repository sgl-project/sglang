// K3 MLA output gate: out = bf16(bf16(x) * bf16(sigmoid(gate))), replacing
// the torch.sigmoid + mul elementwise pair (two launches, two memory passes)
// with one kernel. sigmoid is computed in fp32 and rounded to bf16 before the
// multiply, reproducing the unfused pair's double rounding bit-for-bit.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/activation.cuh>
#include <sgl_kernel/type.cuh>   // For bf16_t, fp32_t, device::cast
#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

struct MlaOutputGateParams {
  const bf16_t* __restrict__ x;     // [N] contiguous (flattened [T, H])
  const bf16_t* __restrict__ gate;  // [N] contiguous
  bf16_t* __restrict__ out;         // [N] contiguous
  uint32_t n_vecs;
};

struct MlaOutputGateStridedParams {
  const bf16_t* __restrict__ x;     // [T, H] contiguous
  const bf16_t* __restrict__ gate;  // [T, H] row-strided (stride(1) == 1)
  bf16_t* __restrict__ out;         // [T, H] contiguous
  int64_t gate_stride_m;            // row stride of gate in bf16 elements
  uint32_t row_vecs;                // H / 4
  uint32_t n_vecs;                  // T * (H / 4)
};

template <int kThreads, bool kUsePDL>
__global__ void mla_output_gate_kernel(const MlaOutputGateParams __grid_constant__ params) {
  using namespace device;

  constexpr int kVecN = 4;
  using vec_bf16_t = AlignedVector<bf16_t, kVecN>;

  const uint32_t v = blockIdx.x * kThreads + threadIdx.x;
  if (v >= params.n_vecs) return;

  PDLWaitPrimary<kUsePDL>();

  vec_bf16_t xv, gv, ov;
  xv.load(params.x, v);
  gv.load(params.gate, v);
#pragma unroll
  for (int i = 0; i < kVecN; ++i) {
    // Match torch.sigmoid(bf16): fp32 sigmoid, round to bf16, then the bf16
    // multiply upcasts both operands to fp32 and rounds once more.
    const float g = cast<fp32_t>(gv[i]);
    const bf16_t s = cast<bf16_t>(activation::sigmoid<false>(g));
    ov[i] = cast<bf16_t>(cast<fp32_t>(xv[i]) * cast<fp32_t>(s));
  }
  ov.store(params.out, v);

  PDLTriggerSecondary<kUsePDL>();
}

template <int kThreads, bool kUsePDL>
__global__ void mla_output_gate_strided_kernel(const MlaOutputGateStridedParams __grid_constant__ params) {
  using namespace device;

  constexpr int kVecN = 4;
  using vec_bf16_t = AlignedVector<bf16_t, kVecN>;

  const uint32_t v = blockIdx.x * kThreads + threadIdx.x;
  if (v >= params.n_vecs) return;

  const uint32_t row = v / params.row_vecs;
  const uint32_t col_vec = v - row * params.row_vecs;
  const bf16_t* gate_row = params.gate + static_cast<int64_t>(row) * params.gate_stride_m;

  PDLWaitPrimary<kUsePDL>();

  vec_bf16_t xv, gv, ov;
  xv.load(params.x, v);
  gv.load(gate_row, col_vec);
#pragma unroll
  for (int i = 0; i < kVecN; ++i) {
    const float g = cast<fp32_t>(gv[i]);
    const bf16_t s = cast<bf16_t>(activation::sigmoid<false>(g));
    ov[i] = cast<bf16_t>(cast<fp32_t>(xv[i]) * cast<fp32_t>(s));
  }
  ov.store(params.out, v);

  PDLTriggerSecondary<kUsePDL>();
}

template <int kThreads, bool kUsePDL>
struct MlaOutputGateKernel {
  static constexpr auto kernel = mla_output_gate_kernel<kThreads, kUsePDL>;
  static constexpr auto strided_kernel = mla_output_gate_strided_kernel<kThreads, kUsePDL>;

  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView gate, const tvm::ffi::TensorView out) {
    using namespace host;

    auto N_ = SymbolicSize{"numel"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    TensorMatcher({N_}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N_}).with_dtype<bf16_t>().with_device(device).verify(gate);
    TensorMatcher({N_}).with_dtype<bf16_t>().with_device(device).verify(out);

    const auto N = static_cast<uint32_t>(N_.unwrap());
    RuntimeCheck(N % 8 == 0, "numel must be divisible by 8");
    if (N == 0) return;

    const auto params = MlaOutputGateParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .gate = static_cast<const bf16_t*>(gate.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .n_vecs = N / 4,
    };
    const uint32_t n_blocks = (params.n_vecs + kThreads - 1) / kThreads;
    LaunchKernel(n_blocks, kThreads, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }

  static void
  run_strided(const tvm::ffi::TensorView x, const tvm::ffi::TensorView gate, const tvm::ffi::TensorView out) {
    using namespace host;

    auto T_ = SymbolicSize{"num_tokens"};
    auto H_ = SymbolicSize{"hidden_size"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({T_, H_}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(gate);
    TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device(device).verify(out);

    const auto T = static_cast<int64_t>(T_.unwrap());
    const auto H = static_cast<int64_t>(H_.unwrap());
    const auto gate_stride_m = gate.stride(0);
    RuntimeCheck(H % 8 == 0, "hidden_size must be divisible by 8");
    RuntimeCheck(
        gate_stride_m >= H && gate_stride_m % 8 == 0,
        "gate row stride must be >= hidden_size and divisible by 8");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(gate.data_ptr()) % 8 == 0, "gate base pointer must be 8-byte aligned");
    if (T == 0 || H == 0) return;

    const int64_t total_vecs = T * (H / 4);
    RuntimeCheck(total_vecs <= UINT32_MAX, "mla_output_gate_strided: total vectors exceed uint32");
    const auto params = MlaOutputGateStridedParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .gate = static_cast<const bf16_t*>(gate.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .gate_stride_m = gate_stride_m,
        .row_vecs = static_cast<uint32_t>(H / 4),
        .n_vecs = static_cast<uint32_t>(total_vecs),
    };
    const uint32_t n_blocks = (params.n_vecs + kThreads - 1) / kThreads;
    LaunchKernel(n_blocks, kThreads, device.unwrap()).enable_pdl(kUsePDL)(strided_kernel, params);
  }
};

}  // namespace sglang
