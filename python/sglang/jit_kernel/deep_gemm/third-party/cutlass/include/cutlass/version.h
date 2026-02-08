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

#pragma once

#include <cstdint>
#include <string>

#define CUTLASS_MAJOR 4
#define CUTLASS_MINOR 2
#define CUTLASS_PATCH 1

#ifdef CUTLASS_VERSIONS_GENERATED
#include "cutlass/version_extended.h"
#else
#define CUTLASS_BUILD 0
#define CUTLASS_REVISION ""
#endif

#define CUTLASS_VERSION ((CUTLASS_MAJOR)*100 + (CUTLASS_MINOR)*10 + CUTLASS_PATCH)

namespace cutlass {

  inline constexpr uint32_t getVersion() {
    return CUTLASS_VERSION;
  }
  inline constexpr uint32_t getVersionMajor() {
    return CUTLASS_MAJOR;
  }
  inline constexpr uint32_t getVersionMinor() {
    return CUTLASS_MINOR;
  }
  inline constexpr uint32_t getVersionPatch() {
    return CUTLASS_PATCH;
  }
  inline constexpr uint32_t getVersionBuild() {
    return CUTLASS_BUILD + 0;
  }

  inline std::string getVersionString() {
    std::string version = "@CUTLASS_VERSION@";
    if (getVersionBuild()) {
      version += "." + std::to_string(getVersionBuild());
    }
    return version;
  }
  
  inline std::string getGitRevision() {
    return "@CUTLASS_REVISION@";
  }

} // namespace cutlass
