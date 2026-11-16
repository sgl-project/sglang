#pragma once

#include <sgl_kernel/quantization/fp8.cuh>

// Forwards the FP8 / UE8M0 helpers, which moved to sgl_kernel/quantization/fp8.cuh,
// to their old names; remove once every caller uses that header directly.

namespace sglang {

namespace deepseek_v4::fp8 {

using namespace device::fp8;

}  // namespace deepseek_v4::fp8

}  // namespace sglang
