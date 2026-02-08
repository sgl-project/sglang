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
/*! \file
    \brief Provides a mechanism for packing and unpacking elements smaller than one byte
*/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/integer_subbyte.h"
#include "cutlass/fast_math.h"

namespace cutlass {

namespace detail {
// This is an implementation detail of cutlass::SubbyteReference and.
// cutlass::HostTensor.  For a given logical element type Element,
// and its corresponding storage (physical) element type StorageUnit,
// it computes quantities that help with managing allocations.
//
// CUTLASS uses a hidden "ContainerUnitType" or StorageUnit type to support
// packed arrays of subbyte types such as int4.  Element is the "logical" type
// for computations, while CUTLASS uses StorageUnit as the element type
// of a packed array of Element.  If Element is not a subbyte type,
// then the corresponding StorageUnit type is just Element itself.
//
// The ContainerType is always calculated as an array StorageUnit type (the StorageUnit
// is always a byte for subbyte types),
// and its number of bits is the lcm of the subbyte type's number of bits and 8.
// Below are some examples for different subbyte types.
//
// * Subbyte Type=int2, ContainerType=StorageUnit[1] (StorageUnit=uint8_t)
// * Subbyte Type=int4, ContainerType=StorageUnit[1] (StorageUnit=uint8_t)
template<class Element, class StorageUnit>
struct StorageContainerCalculator {
  // kContainerTypeNumBits: The number of bits needed for ContainerType
  static constexpr int kContainerTypeNumBits   = (sizeof_bits<Element>::value < 8) ? cutlass::lcm_cxx11(sizeof_bits<Element>::value, sizeof_bits<StorageUnit>::value) : sizeof_bits<Element>::value;
  static_assert(kContainerTypeNumBits % sizeof_bits<Element>::value == 0, "The bits of ContainerType should be divisible by the element's number of bits");
  // kContainerTypeNumLogicalElements: The number of logical Element instance(s) that can be stored per ContainerType instance
  static constexpr int kContainerTypeNumLogicalElements = kContainerTypeNumBits / sizeof_bits<Element>::value;
  /// 3. kContainerTypeNumBytes: The number of bytes per ContainerType instance
  static constexpr int kContainerTypeNumBytes = kContainerTypeNumBits / 8;
  /// 4. kContainerTypeNumBytes: The number of base StorageUnit in the ContainerType
  static constexpr int kContainerTypeNumStorageUnit = kContainerTypeNumBits / sizeof_bits<StorageUnit>::value;

  static_assert(kContainerTypeNumBits != 0, "kContainerTypeNumBits can not be zero");
  static_assert(kContainerTypeNumLogicalElements != 0, "kContainerTypeNumLogicalElements can not be zero");
  static_assert(kContainerTypeNumBytes != 0, "kContainerTypeNumBytes can not be zero");
};
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This class provides a mechanism for packing and unpacking elements smaller than one byte. It
/// assumes these sub-byte elements are packed in a traditional C++ numeric type.
///
/// The intended application is to provide a mechanism to indirectly reference elements in
/// memory or Array<> objects whose addresses cannot otherwise be taken since they are smaller
/// than one byte.
/// 
/// Supports basic pointer arithmetic:
///
/// Example:
///
///   int4b_t *ptr = ...;
///
///   SubbyteReference<int4b_t> ref = ptr;
///   ref += 15;
///
///   int4b_t x = ref;      // load an int4b_t
///   ref = x + 2_s4;      // perform arithmetic on int4b_t and then store
///
template <
  typename Element_,              /// CUTLASS numeric element type.
  typename Storage_ = uint8_t,    /// Underlying storage type. Must be able to hold an integer 
                                  ///   number of objects of type Element.
  class = void
>
class ConstSubbyteReference {
public:

  using Element = Element_;
  using Storage = Storage_;
  using StoragePointer = Storage const *;

  static_assert(sizeof_bits<Element>::value <= sizeof_bits<Storage>::value,
    "Size of Element must not be greater than Storage.");

  static_assert(!(sizeof_bits<Storage>::value % sizeof_bits<Element>::value),
    "Storage must be divisible by Element");

private:

  ///! Number of elements per storage vector
  int const kElementsPerVector = sizeof_bits<Storage>::value / sizeof_bits<Element>::value;

  ///! Bit mask 
  Storage const kMask = 
    ((sizeof_bits<Element>::value < sizeof_bits<Storage>::value) ? 
      (Storage(1) << sizeof_bits<Element>::value) - Storage(1) :
      ~Storage(0));

private:

  /// Pointer to array containing element
  StoragePointer ptr_;

  /// Offset (in units of elements) from pointer.
  ///
  /// Invariant: must always be in range [0, kElementsPerVector)
  int offset_;

public:

  CUTLASS_HOST_DEVICE
  ConstSubbyteReference(): ptr_(nullptr), offset_(0) { }

  /// Constructor
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference(
    Element const *ptr,           /// pointer to memory
    int64_t offset          /// logical offset in units of Element
  ): 
    ptr_(reinterpret_cast<StoragePointer>(ptr)),
    offset_(0) {

    int64_t offset_in_vectors = offset / kElementsPerVector;
    int64_t offset_in_elements = offset % kElementsPerVector;

    ptr_ += offset_in_vectors;
    offset_ = int(offset_in_elements);
  }

  /// Constructor
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference(
    Element *ptr = nullptr
  ): ConstSubbyteReference(ptr, 0) { }

  /// Gets storage pointer
  CUTLASS_HOST_DEVICE
  StoragePointer storage_pointer() const {
    return ptr_;
  }

  /// Gets element offset within storage vector
  CUTLASS_HOST_DEVICE
  int element_offset() const {
    return offset_;
  }

  /// Unpacks an element from memory
  CUTLASS_HOST_DEVICE
  Element get() const {
    Storage item = Storage((*ptr_ >> (offset_ * sizeof_bits<Element>::value)) & kMask);
    return reinterpret_cast<Element const &>(item);
  }

  /// Unpacks an element from memory
  CUTLASS_HOST_DEVICE
  operator Element() const {
    return get();
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference &operator+=(int offset) {

    offset += offset_;
    
    int offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = offset % kElementsPerVector;

    ptr_ += offset_in_vectors;
    offset_ = offset_in_elements;

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference &operator+=(long long offset) {

    offset += offset_;
    
    long long offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = int(offset % kElementsPerVector);

    ptr_ += offset_in_vectors;
    offset_ = offset_in_elements;

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference &operator-=(int offset) {
    
    int offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = offset % kElementsPerVector;

    ptr_ -= offset_in_vectors;
    offset_ -= offset_in_elements;

    if (offset_ < 0) {
      offset_ += kElementsPerVector;
      --ptr_;
    }

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference &operator-=(long long offset) {
    
    long long offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = int(offset % kElementsPerVector);

    ptr_ -= offset_in_vectors;
    offset_ -= offset_in_elements;

    if (offset_ < 0) {
      offset_ += kElementsPerVector;
      --ptr_;
    }

    return *this;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference operator+(int offset) const {

    ConstSubbyteReference ref(ptr_, offset_);
    ref += offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference operator+(long long offset) const {
    
    ConstSubbyteReference ref(ptr_, offset_);
    ref += offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference operator-(int offset) const {

    ConstSubbyteReference ref(ptr_, offset_);
    ref -= offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference operator-=(long long offset) const {

    ConstSubbyteReference ref(ptr_, offset_);
    ref -= offset;

    return ref;
  }

  /// Computes the difference in elements between references
  CUTLASS_HOST_DEVICE
  ptrdiff_t operator-(ConstSubbyteReference ref) const {
    return (ptr_ - ref.ptr_) * kElementsPerVector + (offset_ - ref.offset_);
  }

  /// Explicit cast to int
  CUTLASS_HOST_DEVICE
  explicit operator int() const {
    return int(get());
  }

  /// Explicit cast to signed 64-bit integer
  CUTLASS_HOST_DEVICE
  explicit operator int64_t() const {
    return int64_t(get());
  }

  /// Explicit cast to unsigned 64-bit integer
  CUTLASS_HOST_DEVICE
  explicit operator uint64_t() const {
    return uint64_t(get());
  }

  /// Explicit cast to float
  CUTLASS_HOST_DEVICE
  explicit operator float() const {
    return float(get());
  }

  /// Explicit cast to double
  CUTLASS_HOST_DEVICE
  explicit operator double() const {
    return double(get());
  }
};

template <
  typename Element_,              /// CUTLASS numeric element type.
  typename Storage_ =             /// Underlying storage type. Must be able to hold an integer
                                  ///   number of objects of type Element.

#if defined(__CUDA_ARCH__)        /// Default size depends on width of atomicCas() overloads.
  #if (__CUDA_ARCH__ >= 700)      ///
  uint16_t
  #else
  uint32_t
  #endif
#else
  uint8_t
#endif
  ,
  class = void
>
class SubbyteReference {
public:

  using Element = Element_;
  using Storage = Storage_;
  using StoragePointer = Storage *;

  static_assert(sizeof_bits<Element>::value <= sizeof_bits<Storage>::value,
    "Size of Element must not be greater than Storage.");

  static_assert(!(sizeof_bits<Storage>::value % sizeof_bits<Element>::value),
    "Storage must be divisible by Element");

private:

  ///! Number of elements per storage vector
  int const kElementsPerVector = sizeof_bits<Storage>::value / sizeof_bits<Element>::value;

  ///! Bit mask 
  Storage const kMask = 
    ((sizeof_bits<Element>::value < sizeof_bits<Storage>::value) ? 
      (Storage(1) << sizeof_bits<Element>::value) - Storage(1) :
      ~Storage(0));

private:

  /// Pointer to array containing element
  StoragePointer ptr_;

  /// Offset (in units of elements) from pointer.
  ///
  /// Invariant: must always be in range [0, kElementsPerVector)
  int offset_;

public:

  CUTLASS_HOST_DEVICE
  SubbyteReference(): ptr_(nullptr), offset_(0) { }

  /// Constructor
  CUTLASS_HOST_DEVICE
  SubbyteReference(
    Element *ptr,           /// pointer to memory
    int64_t offset          /// logical offset in units of Element
  ): 
    ptr_(reinterpret_cast<StoragePointer>(ptr)),
    offset_(0) {

    int64_t offset_in_vectors = offset / kElementsPerVector;
    int64_t offset_in_elements = offset % kElementsPerVector;

    ptr_ += offset_in_vectors;
    offset_ = int(offset_in_elements);
  }

  /// Constructor
  CUTLASS_HOST_DEVICE
  SubbyteReference(
    Element *ptr = nullptr
  ): SubbyteReference(ptr, 0) { }

  /// Gets storage pointer
  CUTLASS_HOST_DEVICE
  StoragePointer storage_pointer() const {
    return ptr_;
  }

  /// Gets storage pointer
  CUTLASS_HOST_DEVICE
  Element * operator&() const {
    return reinterpret_cast<Element *>(ptr_);
  }

  /// Gets element offset within storage vector
  CUTLASS_HOST_DEVICE
  int element_offset() const {
    return offset_;
  }

  /// Unpacks an element from memory
  CUTLASS_HOST_DEVICE
  Element get() const {
    uint8_t const* byte_ptr = reinterpret_cast<uint8_t const*>(ptr_);
    // Convert offset in elements to offset in bytes
    constexpr int elements_per_byte = cutlass::sizeof_bits<uint8_t>::value / cutlass::sizeof_bits<Element>::value;
    byte_ptr += offset_ / elements_per_byte;
    // Offset of element within a byte
    int byte_offset = offset_ % elements_per_byte;
    uint8_t item = uint8_t((*byte_ptr >> (byte_offset * cutlass::sizeof_bits<Element>::value)) & kMask);
    return reinterpret_cast<Element const &>(item);
  }

  /// Stores an element to memory
  CUTLASS_HOST_DEVICE
  SubbyteReference & set(Element const &x) {

    Storage item        = (reinterpret_cast<Storage const &>(x) & kMask);
    Storage kUpdateMask = Storage(~(kMask << (offset_ * cutlass::sizeof_bits<Element>::value)));
    Storage new_bits    = Storage(item << (offset_ * cutlass::sizeof_bits<Element>::value));

#if defined(__CUDA_ARCH__)

    //
    // Homebrew read-modify-write
    //
    Storage original;
    Storage updated;

    do {

      original = (*ptr_);

      updated  = Storage((original & kUpdateMask) | new_bits);

      original = atomicCAS(ptr_, original, updated);

    } while (updated != original);

#else

    Storage original = (*ptr_);
    Storage updated  = Storage((original & kUpdateMask) | new_bits);
    *ptr_ = updated;

#endif

    return *this;
  }

  ////

  /// Unpacks an element from memory
  CUTLASS_HOST_DEVICE
  operator Element() const {
    return get();
  }

  /// Stores an element to memory
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator=(Element const & x) {
    return set(x);
  }

  /// Stores an element to memory
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator=(SubbyteReference const & x) {
    return set(x.get());
  }

  /// Stores an element to memory
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator=(
      ConstSubbyteReference<Element, Storage> const &x) {
    return set(x.get());
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator+=(int offset) {

    offset += offset_;
    
    int offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = offset % kElementsPerVector;

    ptr_ += offset_in_vectors;
    offset_ = offset_in_elements;

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator+=(long long offset) {

    offset += offset_;
    
    long long offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = int(offset % kElementsPerVector);

    ptr_ += offset_in_vectors;
    offset_ = offset_in_elements;

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator-=(int offset) {
    
    int offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = offset % kElementsPerVector;

    ptr_ -= offset_in_vectors;
    offset_ -= offset_in_elements;

    if (offset_ < 0) {
      offset_ += kElementsPerVector;
      --ptr_;
    }

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator-=(long long offset) {
    
    long long offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = int(offset % kElementsPerVector);

    ptr_ -= offset_in_vectors;
    offset_ -= offset_in_elements;

    if (offset_ < 0) {
      offset_ += kElementsPerVector;
      --ptr_;
    }

    return *this;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  SubbyteReference operator+(int offset) const {

    SubbyteReference ref(ptr_, offset_);
    ref += offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  SubbyteReference operator+(long long offset) const {
    
    SubbyteReference ref(ptr_, offset_);
    ref += offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  SubbyteReference operator-(int offset) const {

    SubbyteReference ref(ptr_, offset_);
    ref -= offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  SubbyteReference operator-=(long long offset) const {

    SubbyteReference ref(ptr_, offset_);
    ref -= offset;

    return ref;
  }

  /// Computes the difference in elements between references
  CUTLASS_HOST_DEVICE
  ptrdiff_t operator-(SubbyteReference ref) const {
    return (ptr_ - ref.ptr_) * kElementsPerVector + (offset_ - ref.offset_);
  }

  /// Explicit cast to int
  CUTLASS_HOST_DEVICE
  explicit operator int() const {
    return int(get());
  }

  /// Explicit cast to signed 64-bit integer
  CUTLASS_HOST_DEVICE
  explicit operator int64_t() const {
    return int64_t(get());
  }

  /// Explicit cast to unsigned 64-bit integer
  CUTLASS_HOST_DEVICE
  explicit operator uint64_t() const {
    return uint64_t(get());
  }

  /// Explicit cast to float
  CUTLASS_HOST_DEVICE
  explicit operator float() const {
    return float(get());
  }

  /// Explicit cast to double
  CUTLASS_HOST_DEVICE
  explicit operator double() const {
    return double(get());
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T> using _war = T;
template <
  typename Element_,              /// CUTLASS numeric element type.
  typename Storage_               /// Underlying basic storage type.
>
class SubbyteReference<Element_, Storage_, 
    typename platform::enable_if<sizeof_bits<Storage_>::value % sizeof_bits<Element_>::value != 0>::type> {
public:

  using Element = Element_;
  /// Note: It's possible that StorageUnit is not divisible by Element.
  /// For example, an Element instance might be stored across 2 StorageUnit instances.
  /// Thus, CUTLASS needs a storage vector to hold an integer number of Element instances.

  using StorageUnit = Storage_;
private:
  using StorageContainerCalculator = cutlass::detail::StorageContainerCalculator<Element, StorageUnit>;
public:
  static int const kBitsStoredVec = StorageContainerCalculator::kContainerTypeNumBits; 
  static int const kNumStorageUnitPerStoredVec = StorageContainerCalculator::kContainerTypeNumStorageUnit;

  using StorageVec = StorageUnit[kNumStorageUnitPerStoredVec];
  using StorageVecPointer = StorageVec *;
  
  using CudaAtomicType = typename platform::conditional<
      sizeof_bits<StorageUnit>::value == 16,
      uint32_t,
      uint64_t
    >::type;

  static_assert(sizeof_bits<Element>::value <= sizeof_bits<StorageVec>::value,
    "Size of Element must not be greater than StorageVec.");

  static_assert(!(sizeof_bits<StorageVec>::value % sizeof_bits<Element>::value),
    "StorageVec must be divisible by Element");

private:

  ///! Number of elements per storage vector
  int const kElementsPerVector = sizeof_bits<StorageVec>::value / sizeof_bits<Element>::value;

  ///! Bit mask for storage unit.
  StorageUnit const kMask = (StorageUnit(1) << sizeof_bits<Element>::value) - StorageUnit(1);

  /// Pointer to array containing element
  _war<StorageVecPointer> ptr_;

  /// Offset (in units of elements) from pointer.
  ///
  /// Invariant: must always be in range [0, kElementsPerVector)
  int offset_;

  /// Element may be stored across 2 storage unit.
  ///   Low storage unit index in StorageVec
  ///   High storage unit index in StorageVec
  int low_storage_unit_idx_;
  int high_storage_unit_idx_;

  /// Full Mask to extract the entire element
  uint64_t full_element_mask_;

  /// Mask to extract the Element from Low storage unit and High storage unit.
  StorageUnit low_storage_mask_;
  StorageUnit high_storage_mask_;

  /// Start bit index inside the storage unit.
  int start_bit_idx_;

private:

  CUTLASS_HOST_DEVICE
  void update_element_status() {
    int num_bits = offset_ * sizeof_bits<Element>::value;

    start_bit_idx_ = num_bits % sizeof_bits<StorageUnit>::value;
    
    low_storage_unit_idx_ = num_bits / sizeof_bits<StorageUnit>::value;
    high_storage_unit_idx_ = sizeof_bits<StorageUnit>::value - (start_bit_idx_) < sizeof_bits<Element>::value 
                              ? low_storage_unit_idx_ + 1 : low_storage_unit_idx_;
    
    full_element_mask_ = uint64_t(kMask) << start_bit_idx_;
    low_storage_mask_ = StorageUnit(full_element_mask_ & ~StorageUnit(0));
    high_storage_mask_ = StorageUnit((full_element_mask_ >> sizeof_bits<StorageUnit>::value) & ~StorageUnit(0));
  }

public:

  CUTLASS_HOST_DEVICE
  SubbyteReference(): ptr_(nullptr), offset_(0) { }

  /// Constructor
  CUTLASS_HOST_DEVICE
  SubbyteReference(
    Element *ptr,           /// pointer to memory
    int64_t offset          /// logical offset in units of Element
  ): 
    ptr_(reinterpret_cast<StorageVecPointer>(ptr)),
    offset_(0) {
    int64_t offset_in_vectors = offset / kElementsPerVector;
    int64_t offset_in_elements = offset % kElementsPerVector;

    ptr_ += offset_in_vectors;
    offset_ = int(offset_in_elements);

    update_element_status();
  }

  /// Constructor
  CUTLASS_HOST_DEVICE
  SubbyteReference(
    Element *ptr = nullptr
  ): SubbyteReference(ptr, 0) { }

  /// Gets StorageVec pointer
  CUTLASS_HOST_DEVICE
  StorageVecPointer storage_pointer() const {
    return ptr_;
  }

  /// Gets StorageVec pointer
  CUTLASS_HOST_DEVICE
  Element * operator&() const {
    return reinterpret_cast<Element *>(ptr_);
  }

  /// Gets element offset within StorageVec vector
  CUTLASS_HOST_DEVICE
  int element_offset() const {
    return offset_;
  }

  /// Unpacks an element from memory
  CUTLASS_HOST_DEVICE
  Element get() const {
    StorageUnit low_bits = (*ptr_)[low_storage_unit_idx_] & low_storage_mask_;
    StorageUnit high_bits = low_storage_unit_idx_ != high_storage_unit_idx_ ? (*ptr_)[high_storage_unit_idx_] & high_storage_mask_ : 0;

    uint64_t full_item = ((uint64_t)high_bits << sizeof_bits<StorageUnit>::value) | low_bits;
    uint8_t result = uint8_t(full_item >> start_bit_idx_);

    return reinterpret_cast<Element const &>(result);
  }

  /// Stores an element to memory
  CUTLASS_HOST_DEVICE
  SubbyteReference & set(Element const &x) {

    uint64_t item = static_cast<uint64_t>((reinterpret_cast<uint8_t const &>(x) & kMask)) << start_bit_idx_;
    
    StorageUnit low_new_bits  = StorageUnit(item & ~StorageUnit(0));
    StorageUnit high_new_bits = StorageUnit(item >> sizeof_bits<StorageUnit>::value);

    StorageUnit const kLowUpdateMask  = StorageUnit((~full_element_mask_) & (~StorageUnit(0)));
    StorageUnit const kHighUpdateMask = StorageUnit(((~full_element_mask_) >> sizeof_bits<StorageUnit>::value) & (~StorageUnit(0)));

#if defined(__CUDA_ARCH__)
    //
    // Homebrew read-modify-write
    //
    if(high_storage_unit_idx_ != low_storage_unit_idx_){
      /// Only need update 2 storage unit at once.
      /// consider misaligned address issue, we need to do atomicCAS twice 
      StorageUnit original_low_bits, original_high_bits, update_low_bits, update_high_bits;
      do {
        original_low_bits  = ((*ptr_)[low_storage_unit_idx_]);
        update_low_bits  = (original_low_bits & kLowUpdateMask) | low_new_bits;
        original_low_bits = atomicCAS(&((*ptr_)[low_storage_unit_idx_]), original_low_bits, update_low_bits);
      } while (update_low_bits != original_low_bits);
      do {
        original_high_bits = ((*ptr_)[high_storage_unit_idx_]);
        update_high_bits  = (original_high_bits & kHighUpdateMask) | high_new_bits;
        original_high_bits = atomicCAS(&((*ptr_)[high_storage_unit_idx_]), original_high_bits, update_high_bits);
      } while (update_high_bits != original_high_bits);
    }
    else {
      /// Only need update 1 storage unit.
      StorageUnit original, updated;
      do {
        original = ((*ptr_)[low_storage_unit_idx_]);

        updated = (original & kLowUpdateMask) | low_new_bits;

        original = atomicCAS(&((*ptr_)[low_storage_unit_idx_]), original, updated);

      } while (updated != original);
    }
#else


    StorageUnit update_low_bits  = ((*ptr_)[low_storage_unit_idx_] & kLowUpdateMask) | low_new_bits;
    StorageUnit update_high_bits = ((*ptr_)[high_storage_unit_idx_] & kHighUpdateMask) | high_new_bits;

    (*ptr_)[low_storage_unit_idx_] = update_low_bits;

    if(low_storage_unit_idx_ != high_storage_unit_idx_)
      (*ptr_)[high_storage_unit_idx_] = update_high_bits;
#endif

    return *this;
  }

  ////

  /// Unpacks an element from memory
  CUTLASS_HOST_DEVICE
  operator Element() const {
    return get();
  }

  /// Stores an element to memory
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator=(Element const & x) {
    return set(x);
  }

  /// Stores an element to memory
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator=(SubbyteReference const & x) {
    return set(x.get());
  }

  /// Stores an element to memory
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator=(
      ConstSubbyteReference<Element, StorageVec> const &x) {
    return set(x.get());
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator+=(int offset) {

    offset += offset_;
    
    int offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = offset % kElementsPerVector;

    ptr_ += offset_in_vectors;
    offset_ = offset_in_elements;

    update_element_status();

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator+=(long long offset) {

    offset += offset_;
    
    long long offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = int(offset % kElementsPerVector);

    ptr_ += offset_in_vectors;
    offset_ = offset_in_elements;

    update_element_status();

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator-=(int offset) {
    
    int offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = offset % kElementsPerVector;

    ptr_ -= offset_in_vectors;
    offset_ -= offset_in_elements;

    if (offset_ < 0) {
      offset_ += kElementsPerVector;
      --ptr_;
    }

    update_element_status();
    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator-=(long long offset) {
    
    long long offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = int(offset % kElementsPerVector);

    ptr_ -= offset_in_vectors;
    offset_ -= offset_in_elements;

    if (offset_ < 0) {
      offset_ += kElementsPerVector;
      --ptr_;
    }

    update_element_status();
    return *this;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  SubbyteReference operator+(int offset) const {

    SubbyteReference ref(ptr_, offset_);
    ref += offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  SubbyteReference operator+(long long offset) const {
    
    SubbyteReference ref(ptr_, offset_);
    ref += offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  SubbyteReference operator-(int offset) const {

    SubbyteReference ref(ptr_, offset_);
    ref -= offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  SubbyteReference operator-=(long long offset) const {

    SubbyteReference ref(ptr_, offset_);
    ref -= offset;

    return ref;
  }

  /// Computes the difference in elements between references
  CUTLASS_HOST_DEVICE
  ptrdiff_t operator-(SubbyteReference ref) const {
    return (ptr_ - ref.ptr_) * kElementsPerVector + (offset_ - ref.offset_);
  }

  /// Explicit cast to int
  CUTLASS_HOST_DEVICE
  explicit operator int() const {
    return int(get());
  }

  /// Explicit cast to signed 64-bit integer
  CUTLASS_HOST_DEVICE
  explicit operator int64_t() const {
    return int64_t(get());
  }

  /// Explicit cast to unsigned 64-bit integer
  CUTLASS_HOST_DEVICE
  explicit operator uint64_t() const {
    return uint64_t(get());
  }

  /// Explicit cast to float
  CUTLASS_HOST_DEVICE
  explicit operator float() const {
    return float(get());
  }

  /// Explicit cast to double
  CUTLASS_HOST_DEVICE
  explicit operator double() const {
    return double(get());
  }
};

template<typename T> using _war = T;
template <
  typename Element_,              /// CUTLASS numeric element type.
  typename Storage_               /// Underlying storage type. Must be able to hold an integer 
>
class ConstSubbyteReference<Element_, Storage_, 
    typename platform::enable_if<sizeof_bits<Storage_>::value % sizeof_bits<Element_>::value != 0>::type> {
public:

  using Element = Element_;
  ///! Note: Storage unit could not be divisibale by Element,   
  ///   Type element may be stored across 2 storage units, so need a storage vector to hold integer
  ///   number of objects of type Element.
  using StorageUnit = Storage_;
  static int const kBitsStoredVec = cutlass::lcm_cxx11(sizeof_bits<Element>::value, sizeof_bits<StorageUnit>::value); 
  static int const kNumStorageUnitPerStoredVec = kBitsStoredVec / sizeof_bits<StorageUnit>::value;

  using StorageVec = StorageUnit[kNumStorageUnitPerStoredVec];
  using StorageVecPointer = StorageVec const *;
  
  using CudaAtomicType = typename platform::conditional<
      sizeof_bits<StorageUnit>::value == 16,
      uint32_t,
      uint64_t
    >::type;

  static_assert(sizeof_bits<Element>::value <= sizeof_bits<StorageVec>::value,
    "Size of Element must not be greater than StorageVec.");

  static_assert(!(sizeof_bits<StorageVec>::value % sizeof_bits<Element>::value),
    "StorageVec must be divisible by Element");

private:

  ///! Number of elements per storage vector
  int const kElementsPerVector = sizeof_bits<StorageVec>::value / sizeof_bits<Element>::value;

  ///! Bit mask for storage unit.
  StorageUnit const kMask = (StorageUnit(1) << sizeof_bits<Element>::value) - StorageUnit(1);

  /// Pointer to array containing element
  _war<StorageVecPointer> ptr_;

  /// Offset (in units of elements) from pointer.
  ///
  /// Invariant: must always be in range [0, kElementsPerVector)
  int offset_;

  /// Element may be stored across 2 storage unit.
  ///   Low storage unit index in StorageVec
  ///   High storage unit index in StorageVec
  int low_storage_unit_idx_;
  int high_storage_unit_idx_;

  /// Full Mask to extract the entire element
  uint64_t full_element_mask_;

  /// Mask to extract the Element from Low storage unit and High storage unit.
  StorageUnit low_storage_mask_;
  StorageUnit high_storage_mask_;

  /// Start bit index inside the storage unit.
  int start_bit_idx_;

private:

  CUTLASS_HOST_DEVICE
  void update_element_status() {
    int num_bits = offset_ * sizeof_bits<Element>::value;

    start_bit_idx_ = num_bits % sizeof_bits<StorageUnit>::value;
    
    low_storage_unit_idx_ = num_bits / sizeof_bits<StorageUnit>::value;
    high_storage_unit_idx_ = sizeof_bits<StorageUnit>::value - (start_bit_idx_) < sizeof_bits<Element>::value 
                              ? low_storage_unit_idx_ + 1 : low_storage_unit_idx_;
    
    full_element_mask_ = uint64_t(kMask) << start_bit_idx_;
    low_storage_mask_ = StorageUnit(full_element_mask_ & ~StorageUnit(0));
    high_storage_mask_ = StorageUnit((full_element_mask_ >> sizeof_bits<StorageUnit>::value) & ~StorageUnit(0));
  }

public:

  CUTLASS_HOST_DEVICE
  ConstSubbyteReference(): ptr_(nullptr), offset_(0) { }

  /// Constructor
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference(
    Element const *ptr,           /// pointer to memory
    int64_t offset          /// logical offset in units of Element
  ): 
    ptr_(reinterpret_cast<StorageVecPointer>(ptr)),
    offset_(0) {

    int64_t offset_in_vectors = offset / kElementsPerVector;
    int64_t offset_in_elements = offset % kElementsPerVector;

    ptr_ += offset_in_vectors;
    offset_ = int(offset_in_elements);

    update_element_status();
  }

  /// Constructor
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference(
    Element *ptr = nullptr
  ): ConstSubbyteReference(ptr, 0) { }

  /// Gets storage pointer
  CUTLASS_HOST_DEVICE
  StorageVecPointer storage_pointer() const {
    return ptr_;
  }

  /// Gets element offset within storage vector
  CUTLASS_HOST_DEVICE
  int element_offset() const {
    return offset_;
  }

  /// Unpacks an element from memory
  CUTLASS_HOST_DEVICE
  Element get() const {
    StorageUnit low_bits = (*ptr_)[low_storage_unit_idx_] & low_storage_mask_;
    StorageUnit high_bits = low_storage_unit_idx_ != high_storage_unit_idx_ ? (*ptr_)[high_storage_unit_idx_] & high_storage_mask_ : 0;

    uint64_t full_item = ((uint64_t)high_bits << sizeof_bits<StorageUnit>::value) | low_bits;
    uint8_t result = uint8_t(full_item >> start_bit_idx_);

    return reinterpret_cast<Element const &>(result);
  }

  /// Unpacks an element from memory
  CUTLASS_HOST_DEVICE
  operator Element() const {
    return get();
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference &operator+=(int offset) {

    offset += offset_;
    
    int offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = offset % kElementsPerVector;

    ptr_ += offset_in_vectors;
    offset_ = offset_in_elements;

    update_element_status();

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference &operator+=(long long offset) {

    offset += offset_;
    
    long long offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = int(offset % kElementsPerVector);

    ptr_ += offset_in_vectors;
    offset_ = offset_in_elements;

    update_element_status();

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference &operator-=(int offset) {
    
    int offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = offset % kElementsPerVector;

    ptr_ -= offset_in_vectors;
    offset_ -= offset_in_elements;

    if (offset_ < 0) {
      offset_ += kElementsPerVector;
      --ptr_;
    }

    update_element_status();

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference &operator-=(long long offset) {
    
    long long offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = int(offset % kElementsPerVector);

    ptr_ -= offset_in_vectors;
    offset_ -= offset_in_elements;

    if (offset_ < 0) {
      offset_ += kElementsPerVector;
      --ptr_;
    }

    update_element_status();

    return *this;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference operator+(int offset) const {

    ConstSubbyteReference ref(ptr_, offset_);
    ref += offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference operator+(long long offset) const {
    
    ConstSubbyteReference ref(ptr_, offset_);
    ref += offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference operator-(int offset) const {

    ConstSubbyteReference ref(ptr_, offset_);
    ref -= offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference operator-=(long long offset) const {

    ConstSubbyteReference ref(ptr_, offset_);
    ref -= offset;

    return ref;
  }

  /// Computes the difference in elements between references
  CUTLASS_HOST_DEVICE
  ptrdiff_t operator-(ConstSubbyteReference ref) const {
    return (ptr_ - ref.ptr_) * kElementsPerVector + (offset_ - ref.offset_);
  }

  /// Explicit cast to int
  CUTLASS_HOST_DEVICE
  explicit operator int() const {
    return int(get());
  }

  /// Explicit cast to signed 64-bit integer
  CUTLASS_HOST_DEVICE
  explicit operator int64_t() const {
    return int64_t(get());
  }

  /// Explicit cast to unsigned 64-bit integer
  CUTLASS_HOST_DEVICE
  explicit operator uint64_t() const {
    return uint64_t(get());
  }

  /// Explicit cast to float
  CUTLASS_HOST_DEVICE
  explicit operator float() const {
    return float(get());
  }

  /// Explicit cast to double
  CUTLASS_HOST_DEVICE
  explicit operator double() const {
    return double(get());
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element, bool subbyte = (sizeof_bits<Element>::value < 8)>
struct ReferenceFactory;

template <typename Element>
struct ReferenceFactory<Element, false> {

  ///! Number of elements per storage vector
  static int const kElementsPerVector = 1;

  CUTLASS_HOST_DEVICE
  static Element &get(Element *ptr, int64_t offset) {
    return ptr[offset];
  }

  CUTLASS_HOST_DEVICE
  static Element const &get(Element const *ptr, int64_t offset) {
    return ptr[offset];
  }

  CUTLASS_HOST_DEVICE
  static Element *add_pointer_offset(Element *ptr, int64_t offset) {
    return ptr + offset;
  }

  CUTLASS_HOST_DEVICE
  static Element const *add_pointer_offset(Element const *ptr, int64_t offset) {
    return ptr + offset;
  }
};

template <typename Element>
struct ReferenceFactory<Element, true> {

  //
  // Static methods
  //

  CUTLASS_HOST_DEVICE
  static SubbyteReference<Element> get(Element *ptr, int64_t offset) {
    return SubbyteReference<Element>(ptr, offset);
  }

  CUTLASS_HOST_DEVICE
  static ConstSubbyteReference<Element> get(Element const *ptr,
                                             int64_t offset) {
    return ConstSubbyteReference<Element>(ptr, offset);
  }

  /// Helper to add an offset in number of elements, assuming this offset is divisible
  /// by the vector size.
  CUTLASS_HOST_DEVICE
  static Element *add_pointer_offset(Element *ptr, int64_t offset_in_elements) {
    return &SubbyteReference<Element>(ptr, offset_in_elements);
  }

  /// Helper to add an offset in number of elements, assuming this offset is divisible
  /// by the vector size.
  CUTLASS_HOST_DEVICE
  static Element const *add_pointer_offset(Element const *ptr, int64_t offset_in_elements) {
    return &ConstSubbyteReference<Element>(ptr, offset_in_elements);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
