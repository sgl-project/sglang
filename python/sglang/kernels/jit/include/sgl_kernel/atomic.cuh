/// \file atomic.cuh
/// \brief Device-side atomic operations.

#pragma once
#include <sgl_kernel/utils.cuh>

namespace device::atomic {

/**
 * \brief Atomically computes the maximum of `*addr` and `value`, storing the
 *        result in `*addr`.
 * \param addr Pointer to the value in global/shared memory to be updated.
 * \param value The value to compare against.
 * \return The old value at `*addr` before the update.
 * \note On CUDA, this uses `atomicMax`/`atomicMin` on the reinterpreted
 *       integer representation. On ROCm, a CAS loop is used as a fallback.
 */
SGL_DEVICE float max(float* addr, float value) {
#ifndef USE_ROCM
  float old;
  old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
                     : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
  return old;
#else
  int* addr_as_i = (int*)addr;
  int old = *addr_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr_as_i, assumed, __float_as_int(fmaxf(value, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
#endif
}

}  // namespace device::atomic
