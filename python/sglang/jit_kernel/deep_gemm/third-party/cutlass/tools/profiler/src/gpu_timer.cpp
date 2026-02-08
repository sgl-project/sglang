/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/* \file
   \brief Defines a math function
*/

#include <stdexcept>
#include <cstring>

#include "cutlass/profiler/gpu_timer.h"


namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

GpuTimer::GpuTimer() {
  cudaError_t result;

  for (auto & event : events) {
    result = cudaEventCreate(&event);
    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to create CUDA event");
    }
  }
}

GpuTimer::GpuTimer(GpuTimer&& gpu_timer) noexcept {
  memcpy(events, gpu_timer.events, sizeof(events));
  memset(gpu_timer.events, 0, sizeof(gpu_timer.events));
}

GpuTimer::~GpuTimer() {
  for (const auto & event : events) {
    if (event != nullptr) {
      cudaEventDestroy(event);
    }
  }
}

/// Records a start event in the stream, the flag is for cudaEventRecordWithFlags
void GpuTimer::start(cudaStream_t stream, const unsigned int flag) {
  cudaError_t result = cudaEventRecordWithFlags(events[0], stream, flag);
  if (result != cudaSuccess) {
    throw std::runtime_error("Failed to record start event.");
  }
}

/// Records a stop event in the stream, the flag is for cudaEventRecordWithFlags
void GpuTimer::stop(cudaStream_t stream, const unsigned int flag) {
cudaError_t result = cudaEventRecordWithFlags(events[1], stream, flag);
  if (result != cudaSuccess) {
    throw std::runtime_error("Failed to record stop event.");
  }
}

/// Records a stop event in the stream and synchronizes on the stream, the flag is for cudaEventRecordWithFlags
void GpuTimer::stop_and_wait(cudaStream_t stream, const unsigned int flag) {

  stop(stream, flag);

  cudaError_t result;
  if (stream) {
    result = cudaStreamSynchronize(stream);
    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to synchronize with non-null CUDA stream.");
    }
  }
  else {
    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to synchronize with CUDA device.");
    }
  }
}

/// Returns the duration in milliseconds
double GpuTimer::duration(int iterations) const {

  float avg_ms;

  cudaError_t result = cudaEventElapsedTime(&avg_ms, events[0], events[1]);
  if (result != cudaSuccess) {
    throw std::runtime_error("Failed to query elapsed time from CUDA events.");
  }

  return double(avg_ms) / double(iterations);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass
