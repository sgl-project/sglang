// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

// Forward declaration of the MockTensor struct for testing only
struct MockTensor {
  void *ptr;
  struct {
    int32_t shape[3];
  } shape;

  struct {
    int32_t strides[3];
  } strides;
};

NB_MODULE(tensor, m) {
  // create a tensor for testing
  m.def("make_tensor", [](int64_t ptr, std::vector<int32_t> shape,
                          std::vector<int32_t> strides) {
    auto *tensor = new MockTensor();
    tensor->ptr = reinterpret_cast<void *>(ptr);

    assert(shape.size() == 3 && "shape must have 3 elements");
    assert(strides.size() == 3 && "strides must have 3 elements");

    for (size_t i = 0; i < shape.size(); i++) {
      tensor->shape.shape[i] = shape[i];
      tensor->strides.strides[i] = strides[i];
    }

    return nb::steal(PyCapsule_New(tensor, "tensor", [](PyObject *capsule) {
      auto n = PyCapsule_GetName(capsule);
      if (void *p = PyCapsule_GetPointer(capsule, n)) {
        delete reinterpret_cast<MockTensor *>(p);
      }
    }));
  });

  m.def(
      "pycapsule_get_pointer",
      [](nb::object &capsule) {
        void *ptr = PyCapsule_GetPointer(capsule.ptr(), "tensor");
        if (!ptr) {
          throw std::runtime_error("Invalid tensor capsule");
        }
        return reinterpret_cast<uintptr_t>(ptr);
      },
      "Get pointer from PyCapsule");
}
