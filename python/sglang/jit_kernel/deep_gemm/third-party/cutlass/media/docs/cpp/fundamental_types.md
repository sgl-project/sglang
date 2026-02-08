![ALT](../../images/gemm-hierarchy-with-epilogue-no-labels.png "CUTLASS")

# Fundamental Types

CUTLASS defies several fundamental numeric and container classes upon which computations and
algorithms algorithms for linear algebra computations are implemented. 

Where possible, CUTLASS fundamental types mirror the C++ Standard Library. However, there are circumstances that necessitate divergence from the Standard Library's specification. In such cases, the CUTLASS implementation adopts unique capitalization to distinguish that standard vocabulary types may not be safely substituted in all cases.

Most types in CUTLASS are usable in both host code and device code. Moreover, they are functional regardless of compute capability, but they may only be efficient when hardware support is present.

## Numeric Types

CUTLASS defines classes for the following numeric data types.

* `half_t`: IEEE half-precision floating point (exponent: 5b, mantissa: 10b; literal suffix `_hf`)
* `bfloat16_t`: BFloat16 data type (exponent: 8b, mantissa: 7b; literal suffix `_bf16`)
* `tfloat32_t`: Tensor Float 32 data type (exponent: 8b, mantissa: 10b; literal suffix `_tf32`)
* `int4_t`, `uint4_t`: 4b signed and unsigned integer (literal suffx `_s4`, `_u4`)
* `bin1_t`: 1b binary numeric type (literal suffix `_b1`)
* `float_e5m2_t`: 8bits signed float (exponent: 5 bits, mantissa: 2 bits)
* `float_e4m3_t`: 8bits signed float (exponent: 4 bits, mantissa: 3 bits)
* `float_ue4m3_t`: 8bits unsigned float (exponent: 4 bits, mantissa: 3 bits)
* `float_ue8m0_t`: 8bits unsigned float (exponent: 8 bits, mantissa: 0 bits)
* `float_e3m2_t`: 6bits signed float (exponent: 3 bits, mantissa: 2 bits)
* `float_e2m3_t`: 6bits signed float (exponent: 2 bits, mantissa: 3 bits)
* `float_e2m1_t`: 4bits signed float (exponent: 2 bits, mantissa: 1 bits)
* `type_erased_dynamic_float8_t`: Type agnostic 8 bits signed float allowing the user to provide a specific datatype as runtime argument.
* `type_erased_dynamic_float6_t`: Type agnostic 6 bits signed float allowing the user to provide a specific datatype as runtime argument.
* `type_erased_dynamic_float4_t`: Type agnostic 4 bits signed float allowing the user to provide a specific datatype as runtime argument.
* `mx_float8_t<float_e5m2_t>` or `mx_float8_t<float_e4m3_t>` : Block scaled data type with fp8 element type and float_ue8m0_t scale factor and vector size of 32.
* `mx_float6_t<float_e3m2_t>` or `mx_float6_t<float_e2m3_t>` : Block scaled data type with fp6 element type and float_ue8m0_t scale factor and vector size of 32.
* `mx_float4_t<float_e2m1_t>` : Block scaled data type with signed e2m1 element type and float_ue8m0_t scale factor and vector size of 32.
* `nv_float4_t<float_e2m1_t>` : Block scaled data type with signed e2m1 element type and float_ue4m3_t scale factor and vector size of 16.
* `complex<T>`: defines complex-valued data type based on the supplied real-valued numeric type

Numeric types in CUTLASS may be used in both host and device code and are intended to function
like any other plain-old-data type. 

If CUTLASS is compiled with `CUTLASS_F16C_ENABLED`, then hardware conversion is used for 
half-precision types in host code. Regardless, `cutlass::half_t` uses the most efficient 
NVIDIA GPU hardware instructions available in device code.

Example:
```c++
#include <iostream>
#include <cutlass/numeric_types.h>

__global__ void kernel(cutlass::half_t x) {
  printf("Device: %f\n", float(x * 2.0_hf));
}

int main() {

  cutlass::half_t x = 0.5_hf;

  std::cin >> x;

  std::cout << "Host: " << 2.0_hf * x << std::endl;

  kernel<<< dim3(1,1), dim3(1,1,1) >>>(x);

  return 0;
}
```

## Containers

CUTLASS uses the following containers extensively for implementing efficient CUDA kernels.

### Array

```c++
template <
  typename T,       // element type
  int N             // number of elements
>
struct Array;
```

`Array<class T, int N>` defines a statically sized array of elements of type _T_ and size _N_. This class is similar to 
[`std::array<>`](https://en.cppreference.com/w/cpp/container/array) in the Standard Library with one notable exception:
partial specializations exist to pack or unpack elements smaller than one byte.

`Array<>` is intended to be a convenient and uniform container class to store arrays of numeric elements regardless of data type or vector length. The storage needed is expected to be the minimum necessary given the logical size of each numeric type in bits (numeric types smaller than one byte are densely packed). Nevertheless, the size reported by `sizeof(Array<T, N>)` is always an integer multiple of bytes.

Storing numeric elements in a C++ STL-style container class enables useful modern C++ mechanisms such as range-based for loops. For example, to print the elements of `Array<>`, the following range-based for loop syntax is always valid regardless of numeric data type, compute capability, or context in host or device code.

Example:
```c++
int const kN;
Array<T, kN> elements;

CUTLASS_PRAGMA_UNROLL                        // required to ensure array remains in registers
for (auto x : elements) {
  printf("%d, %f", int64_t(x), double(x));   // explictly convert to int64_t or double
}
```

When copying `Array<>` objects or passing them as arguments to methods, it is best to avoid accessing individual elements. This enables the use of vector instructions to perform the operation more efficiently. For example, setting all elements to zero is best performed by calling the `clear()` method. Copies should be performed by assigning the entire object.

Example:
```c++
#include <cutlass/array.h>

int const kN;
Array<T, kN> source;
Array<T, kN> destination;

source.clear();         // set all elements to value of zero

destination = source;   // copy to `destination`
```

`Array<>` may be used to store elements smaller than one byte such as 4b integers.
```c++
Array<int4b_t, 2> packed_integers;

static_assert(
  sizeof(packed_integers) == 1,
 "Packed storage of sub-byte data types is compact.");

// Access array elements using usual indirection and assignment operators
packed_integers[0] = 2_s4;
packed_integers[1] = 3_s4;

CUTLASS_PRAGMA_UNROLL
for (auto x : elements) {
  printf("%d", int(x));       // access elements normally
}

```

### AlignedArray

```c++
template <
  typename T,          // element type
  int N,               // number of elements
  int Alignment        // alignment requirement in bytes
>
class AlignedArray;
```

`AlignedArray` is derived from `Array<T, N>` and supports an optional alignment field. Pointers to objects of type `AlignedArray<>` reliably yield vectorized memory accesses when dereferenced.

Example:
```c++
int const kN = 8;
ArrayAligned<half_t, kN> source;
ArrayAligned<half_t, kN> const *ptr = ...;

source = *ptr;          // 128b aligned memory access
```

### AlignedBuffer

```c++
template <
  typename T,          // element type
  int N,               // number of elements
  int Alignment        // alignment requirement in bytes
>
class AlignedBuffer;
```

`AlignedBuffer` provides a uniform way to define aligned memory allocations for all data types. This is particularly
useful in defining allocations within shared memory with guaranteed memory alignment needed for vectorized access. 
Note, constructors of the elements within AlignedBuffer<> are not called, and so the elements are initially in an
undefined state.

Use `AlignedBuffer<>::data()` to obtain a pointer to the first element of the buffer.

**Example:** Guaranteed aligned shared memory allocation. Note, shared memory contents are uninitialized.
```c++
int const kN = 32;
int const kAlignment = 16;                  // alignment in bytes

// Define a shared memory allocation in device code
__shared__ AlignedBuffer<complex<half_t>, kN, kAlignment> matrix_tile;

complex<half_t> *ptr = matrix_tile.data();  // ptr is guaranteed to have 128b (16 Byte) alignment
```

Note, `AlignedBuffer<>` only guarantees that its internal memory allocation is aligned, obtained by `AlignedBuffer<>::data()`. There is no guarantee that the `AlignedBuffer<>` object itself satisfies alignment constraints or that its internal memory allocation is contiguous. Device code performing vectorized memory accesses should use the `AlignedArray<>` type.

**_Example_:** Vectorized memory access to shared memory allocations.
```c++
int const kN = 1024;

__shared__ AlignedBuffer<half_t, kN> smem_buffer;

AlignedArray<half_t, 8> *ptr = reinterpret_cast<AlignedArray<half_t, 8> *>(smem_buffer.data());

AlignedArray<half_t, 8> x = ptr[threadIdx.x];     // 128b shared memory load
```

### Numeric Conversion

CUTLASS defines procedures for performing numeric conversion between data types in `cutlass/numeric_conversion.h`. 
Where possible, these target hardware acceleration on the target architecture and support multiple rounding modes.

```c++
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

NumericConverter<half_t, float>     convert_f32_to_f16;
NumericConverter<tfloat32_t, float> convert_f32_to_tf32;

half_t     x = convert_f32_to_f16(3.14159f);
tfloat32_t y = convert_f32_to_tf32(3.14159f);
```

Recent GPU architectures such as NVIDIA Turing and Ampere combine numeric conversion with efficient packing
into bit vectors. Consequently, CUTLASS defines conversion on both scalars and `Array<>` objects to implement 
the optimal code sequence on all architectures.

```c++
//
// Example: convert and pack 32b signed integers to a vector of packed signed 8-bit integers.
//
int const kN = 16;
Array<int8_t, kN> destination;
Array<int,    kN> source;

NumericConverter<descltype(destination), decltype(source)> convert;

destination = convert(source);
```

### Coord

```c++
template <
  int Rank,
  typename Index = int
>
class Coord;
```

`Coord<Rank, class T = int>` is a container used explicitly for defining logical coordinates in tensors of known rank. Traditional vector operators are defined such as `+`, `-`, and scalar multiplication `*` to simplify the creation of vector-valued expressions on tensor coordinates.

**Example:** Vector operations on coordinates.
```c++
Coord<2> compute_offset(Coord<2> const & base) {
  
  Coord<2> stride = make_Coord(1, kM);

  return base + stride * make_Coord(threadIdx.x, threadIdx.y); 
}
```

Instances of `Coord<>` are used throughout CUTLASS to compute indices into tensors. Frequently, the dimensions of tensors of known layouts may be given names such as "rows" or "columns". To clarify the code, we have implemented several classes derived from `Coord<>` with accessors for each coordinate member.

Such classes include:
```c++
struct MatrixCoord : public Coord<2> {
  Index & row();
  Index & column();
};
```
and

```c++
struct Tensor4DCoord : public Coord<4> {
  Index & n();
  Index & h();
  Index & w();
  Index & c();
};
```

### PredicateVector<int Bits>

`PredicateVector<int Bits>` contains a statically sized array of hardware predicates packed into registers to enable efficient access within unrolled loops. 

This container is optimized for sequential access through iterators, though these are only efficient when used within fully unrolled loops.

Moreover, instances of `PredicateVector<>` are not guaranteed to be updated until any non-const iterator objects have gone out of scope. This is because iterators are effectively caches that update the `PredicateVector<>` instance's internal storage as a batch.

**Example:** Managing an array of predicates.
```c++

unsigned mask;
PredicateVector<kBits> predicates;

// Nested scope to update predicates via an iterator
{
  auto pred_it = predicates.begin();

  CUTLASS_PRAGMA_UNROLL
  for (int bit = 0; bit < kBits; ++bit, ++pred_it) {
    bool guard = (mask & (1u << bit));
    pred_it.set(guard);
  }
}

// Efficient use of predicates to guard memory instructions
T *ptr;
Array<T, kAccesses> fragment;

auto pred_it = predicates.const_begin();
for (int access = 0; access < kAccesses; ++access, ++pred_it) {
  if (*pred_it) {
    fragment[access] = ptr[access];
  }
}

```

Note: `PredicateVector<>` is not efficient when accessed via dynamic random access. If an array of bits is needed with dynamic random access (in contrast with access via _constexpr_ indices), then `Array<bin1_t, N>` should be used instead.

## Functional

CUTLASS defines function objects corresponding to basic arithmetic operations modeled after C++ Standard Library's `<functional>` header.

CUTLASS extends this by defining `multiply_add<T>` which computes `d = a * b + c`. The partial specialization `multiply_add<complex<T>>` computes complex-valued multiplication and addition using four real-valued multiply-add operations; these may correspond to native hardware instructions.

Example:
```c++
complex<float> a;
complex<float> b;
complex<float> c;
complex<float> d;

multiply_add<complex<float>> mad_op;

d = mad_op(a, b, c);    // four single-precision multiply-add instructions
```

CUTLASS defines partial specializations for type `Array<T, N>`, performing elementwise operations on each element. A further partial specialization for `Array<half_t, N>` targets may target native SIMD instructions for compute capability SM60 and beyond.

**Example:** Fused multiply-add of arrays of half-precision elements.
```c++
static int const kN = 8;

Array<half_t, kN> a;
Array<half_t, kN> b;
Array<half_t, kN> c;
Array<half_t, kN> d;

multiply_add<Array<half_t, kN>> mad_op;

d = mad_op(a, b, c);   // efficient multiply-add for Array of half-precision elements
```

## Numeric Conversion

Operators are define to convert between numeric types in `numeric_conversion.h`. Conversion operators are defined in
terms of individual numeric elements and on arrays which enable the possibility of efficient hardware
support on current and future NVIDIA GPUs.

**Example:** Converting between 32-b and 8-b integers.
```c++

```

### Copyright

Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
