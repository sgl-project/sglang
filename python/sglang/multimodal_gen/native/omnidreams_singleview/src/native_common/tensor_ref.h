/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <type_traits>

#include "native_common/macros.h"

namespace omnidreams_native {

template <typename T, int Rank>
struct TensorRef {
  static_assert(Rank >= 1 && Rank <= 8, "TensorRef supports ranks 1..8");

  T* ptr;
  int64_t shape[Rank];
  int64_t strides[Rank];

  template <typename... Dims>
  static OMNIDREAMS_NATIVE_HOST_DEVICE TensorRef contiguous(T* data, Dims... dims) {
    static_assert(sizeof...(dims) == Rank, "Number of dimensions must match Rank");
    TensorRef ref{};
    ref.ptr = data;
    int64_t d[] = {static_cast<int64_t>(dims)...};
    for (int i = 0; i < Rank; ++i) {
      ref.shape[i] = d[i];
    }
    ref.strides[Rank - 1] = 1;
    for (int i = Rank - 2; i >= 0; --i) {
      ref.strides[i] = ref.strides[i + 1] * ref.shape[i + 1];
    }
    return ref;
  }

  static OMNIDREAMS_NATIVE_HOST_DEVICE TensorRef strided(
      T* data,
      const int64_t (&shp)[Rank],
      const int64_t (&str)[Rank]) {
    TensorRef ref{};
    ref.ptr = data;
    for (int i = 0; i < Rank; ++i) {
      ref.shape[i] = shp[i];
      ref.strides[i] = str[i];
    }
    return ref;
  }

  static OMNIDREAMS_NATIVE_HOST_DEVICE TensorRef strided(
      T* data,
      const int64_t* shp,
      const int64_t* str) {
    TensorRef ref{};
    ref.ptr = data;
    for (int i = 0; i < Rank; ++i) {
      ref.shape[i] = shp[i];
      ref.strides[i] = str[i];
    }
    return ref;
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE T* data_ptr() {
    return ptr;
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE const T* data_ptr() const {
    return ptr;
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE int rank() const {
    return Rank;
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE int64_t dim(int i) const {
    assert(i >= 0 && i < Rank);
    return shape[i];
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE int64_t stride(int i) const {
    assert(i >= 0 && i < Rank);
    return strides[i];
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE int64_t numel() const {
    int64_t n = 1;
    for (int i = 0; i < Rank; ++i) {
      n *= shape[i];
    }
    return n;
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE int64_t nbytes() const {
    return numel() * static_cast<int64_t>(sizeof(T));
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE bool is_contiguous() const {
    int64_t expected = 1;
    for (int i = Rank - 1; i >= 0; --i) {
      if (strides[i] != expected) {
        return false;
      }
      expected *= shape[i];
    }
    return true;
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE bool empty() const {
    return ptr == nullptr || numel() == 0;
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE T& operator()(int64_t i0) const {
    static_assert(Rank == 1, "operator()(i0) requires Rank==1");
    return ptr[i0 * strides[0]];
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE T& operator()(int64_t i0, int64_t i1) const {
    static_assert(Rank == 2, "operator()(i0,i1) requires Rank==2");
    return ptr[i0 * strides[0] + i1 * strides[1]];
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE T& operator()(
      int64_t i0,
      int64_t i1,
      int64_t i2) const {
    static_assert(Rank == 3, "operator()(i0,i1,i2) requires Rank==3");
    return ptr[i0 * strides[0] + i1 * strides[1] + i2 * strides[2]];
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE T& operator()(
      int64_t i0,
      int64_t i1,
      int64_t i2,
      int64_t i3) const {
    static_assert(Rank == 4, "operator()(i0..i3) requires Rank==4");
    return ptr[i0 * strides[0] + i1 * strides[1] + i2 * strides[2] +
        i3 * strides[3]];
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE T& operator()(
      int64_t i0,
      int64_t i1,
      int64_t i2,
      int64_t i3,
      int64_t i4) const {
    static_assert(Rank == 5, "operator()(i0..i4) requires Rank==5");
    return ptr[i0 * strides[0] + i1 * strides[1] + i2 * strides[2] +
        i3 * strides[3] + i4 * strides[4]];
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE int64_t linear_offset(
      const int64_t (&idx)[Rank]) const {
    int64_t off = 0;
    for (int i = 0; i < Rank; ++i) {
      off += idx[i] * strides[i];
    }
    return off;
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE T& at(const int64_t (&idx)[Rank]) const {
    return ptr[linear_offset(idx)];
  }

  template <int R = Rank, typename std::enable_if<(R >= 2), int>::type = 0>
  OMNIDREAMS_NATIVE_HOST_DEVICE TensorRef<T, R - 1> slice(int64_t idx0) const {
    TensorRef<T, R - 1> sub{};
    sub.ptr = ptr + idx0 * strides[0];
    for (int i = 0; i < R - 1; ++i) {
      sub.shape[i] = shape[i + 1];
      sub.strides[i] = strides[i + 1];
    }
    return sub;
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE TensorRef narrow(
      int d,
      int64_t start,
      int64_t length) const {
    assert(d >= 0 && d < Rank);
    assert(start >= 0 && start + length <= shape[d]);
    TensorRef out = *this;
    out.ptr = ptr + start * strides[d];
    out.shape[d] = length;
    return out;
  }

  template <typename U>
  OMNIDREAMS_NATIVE_HOST_DEVICE TensorRef<U, Rank> reinterpret_as() const {
    static_assert(sizeof(T) == sizeof(U), "reinterpret_as requires same element size");
    TensorRef<U, Rank> out{};
    out.ptr = reinterpret_cast<U*>(ptr);
    for (int i = 0; i < Rank; ++i) {
      out.shape[i] = shape[i];
      out.strides[i] = strides[i];
    }
    return out;
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE
  TensorRef<typename std::remove_const<T>::type, Rank> as_mutable() const {
    TensorRef<typename std::remove_const<T>::type, Rank> out{};
    out.ptr = const_cast<typename std::remove_const<T>::type*>(ptr);
    for (int i = 0; i < Rank; ++i) {
      out.shape[i] = shape[i];
      out.strides[i] = strides[i];
    }
    return out;
  }

  template <int NewRank, typename... NewDims>
  OMNIDREAMS_NATIVE_HOST_DEVICE TensorRef<T, NewRank> reshape(
      NewDims... new_dims) const {
    static_assert(sizeof...(new_dims) == NewRank, "Number of dims must match NewRank");
    assert(is_contiguous() && "reshape requires contiguous tensor");
    auto out = TensorRef<T, NewRank>::contiguous(ptr, new_dims...);
    assert(out.numel() == numel() && "reshape must preserve element count");
    return out;
  }

  OMNIDREAMS_NATIVE_HOST_DEVICE TensorRef<T, 1> flatten() const {
    assert(is_contiguous() && "flatten requires contiguous tensor");
    return TensorRef<T, 1>::contiguous(ptr, numel());
  }
};

template <typename T>
using TensorRef1D = TensorRef<T, 1>;
template <typename T>
using TensorRef2D = TensorRef<T, 2>;
template <typename T>
using TensorRef3D = TensorRef<T, 3>;
template <typename T>
using TensorRef4D = TensorRef<T, 4>;
template <typename T>
using TensorRef5D = TensorRef<T, 5>;

template <typename T, typename... Dims>
OMNIDREAMS_NATIVE_HOST_DEVICE auto make_tensor_ref(T* data, Dims... dims) {
  return TensorRef<T, sizeof...(Dims)>::contiguous(data, dims...);
}

template <typename T, int Rank>
OMNIDREAMS_NATIVE_HOST_DEVICE TensorRef<T, Rank> null_tensor_ref() {
  TensorRef<T, Rank> ref{};
  ref.ptr = nullptr;
  for (int i = 0; i < Rank; ++i) {
    ref.shape[i] = 0;
    ref.strides[i] = 0;
  }
  return ref;
}

}  // namespace omnidreams_native
