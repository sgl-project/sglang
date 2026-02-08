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

//

//

#pragma once

#include <cute/config.hpp>


namespace cute {


//
// Cluster launch utility
//
CUTE_HOST
bool
initialize_preferred_cluster_launch(void const* const kernel_function,
                                   dim3 const& grid_dims,
                                   dim3 const& cluster_dims_preferred,
                                   dim3 const& cluster_dims_fallback)
{
  //
  // Validate cluster_dims
  //

  // Total number of cluster cannot be greater than 32 (hardware requirement)
  if (cluster_dims_preferred.x * cluster_dims_preferred.y * cluster_dims_preferred.z <= 0 ||
      cluster_dims_preferred.x * cluster_dims_preferred.y * cluster_dims_preferred.z > 32) {
    std::cout << "Invalid preferred cluster dimensions: Attempting to init preferred cluster (" << cluster_dims_preferred.x << "," << cluster_dims_preferred.y << "," << cluster_dims_preferred.z
              << ") [" << (cluster_dims_preferred.x * cluster_dims_preferred.y * cluster_dims_preferred.z) << "] which must be within (0,32]." << std::endl;
    return false;
  }

  // Total number of cluster cannot be greater than 32 (hardware requirement)
  if (cluster_dims_fallback.x * cluster_dims_fallback.y * cluster_dims_fallback.z <= 0 ||
      cluster_dims_fallback.x * cluster_dims_fallback.y * cluster_dims_fallback.z > 32) {
    std::cout << "Invalid cluster dimensions: Attempting to init fallback cluster (" << cluster_dims_fallback.x << "," << cluster_dims_fallback.y << "," << cluster_dims_fallback.z
              << ") [" << (cluster_dims_fallback.x * cluster_dims_fallback.y * cluster_dims_fallback.z) << "] which must be within (0,32]." << std::endl;
    return false;
  }

  // Total grid dimensions must be within (2^32, 2^16, 2^16)
  if (grid_dims.y > (1 << 16) || grid_dims.z > (1 << 16)) {
    std::cout << "Invalid grid dimensions: Attempting to init grid dimensions (" << grid_dims.x << "," << grid_dims.y << "," << grid_dims.z
              << ") which must be within (2^32, 2^16, 2^16)." << std::endl;
    return false;
  }

  // grid_dims should be divisible by cluster_dims_preferred
  if (grid_dims.x % cluster_dims_preferred.x != 0 ||
      grid_dims.y % cluster_dims_preferred.y != 0 ||
      grid_dims.z % cluster_dims_preferred.z != 0) {
    std::cout << "Invalid grid dimensions: Preferred cluster (" << cluster_dims_preferred.x << "," << cluster_dims_preferred.y << "," << cluster_dims_preferred.z
              << ") does not divide Grid (" << grid_dims.x << "," << grid_dims.y << "," << grid_dims.z << ")." << std::endl;
    return false;
  }

  // cluster_dims_preferred should be divisible by cluster_dims_fallback
  if (cluster_dims_preferred.x % cluster_dims_fallback.x != 0 ||
      cluster_dims_preferred.y % cluster_dims_fallback.y != 0 ||
      cluster_dims_preferred.z % cluster_dims_fallback.z != 0) {
    std::cout << "Invalid cluster dimensions: Fallback cluster (" << cluster_dims_fallback.x << "," << cluster_dims_fallback.y << "," << cluster_dims_fallback.z
              << ") does not divide Preferred cluster (" << cluster_dims_preferred.x << "," << cluster_dims_preferred.y << "," << cluster_dims_preferred.z << ")." << std::endl;
    return false;
  }

  // Both cluster dimenions should have the same depth
  if (cluster_dims_preferred.z != cluster_dims_fallback.z) {
    std::cout << "Invalid cluster dimensions: Fallback cluster (" << cluster_dims_fallback.x << "," << cluster_dims_fallback.y << "," << cluster_dims_fallback.z
              << ") and Preferred cluster (" << cluster_dims_preferred.x << "," << cluster_dims_preferred.y << "," << cluster_dims_preferred.z << ") does not have the same depth." << std::endl;
    return false;
  }

  return true;
}
} // end namespace cute
