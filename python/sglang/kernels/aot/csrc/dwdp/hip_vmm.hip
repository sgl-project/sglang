/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <hip/hip_runtime_api.h>
#include <hip/hip_version.h>
#include <unistd.h>

#include <cstdint>
#include <limits>

#include "dwdp/hip_vmm.h"

namespace {

void check_hip(hipError_t status, const char* operation) {
  if (status == hipSuccess) {
    return;
  }
  const char* name = hipGetErrorName(status);
  const char* detail = hipGetErrorString(status);
  TORCH_CHECK(
      false,
      operation,
      " failed with HIP error ",
      static_cast<int>(status),
      " (",
      name == nullptr ? "unknown" : name,
      "): ",
      detail == nullptr ? "no error description" : detail);
}

int validate_device(int64_t device_id) {
  TORCH_CHECK(device_id >= 0, "device_id must be non-negative, got ", device_id);
  TORCH_CHECK(device_id <= std::numeric_limits<int>::max(), "device_id is too large for HIP: ", device_id);

  int device_count = 0;
  check_hip(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
  TORCH_CHECK(
      device_id < device_count,
      "device_id ",
      device_id,
      " is out of range for ",
      device_count,
      " visible HIP device(s)");
  return static_cast<int>(device_id);
}

size_t validate_size(int64_t value, const char* name) {
  TORCH_CHECK(value > 0, name, " must be positive, got ", value);
  return static_cast<size_t>(value);
}

size_t validate_offset(int64_t value) {
  TORCH_CHECK(value >= 0, "offset must be non-negative, got ", value);
  return static_cast<size_t>(value);
}

void* validate_address(int64_t address, const char* name) {
  TORCH_CHECK(address > 0, name, " must be a nonzero address, got ", address);
  return reinterpret_cast<void*>(static_cast<uintptr_t>(address));
}

hipMemGenericAllocationHandle_t validate_handle(int64_t handle) {
  TORCH_CHECK(handle > 0, "handle must be nonzero, got ", handle);
  return reinterpret_cast<hipMemGenericAllocationHandle_t>(static_cast<uintptr_t>(handle));
}

template <typename Pointer>
int64_t pointer_to_int(Pointer pointer, const char* operation) {
  const auto raw = reinterpret_cast<uintptr_t>(pointer);
  TORCH_CHECK(raw != 0, operation, " returned a null value");
  TORCH_CHECK(
      raw <= static_cast<uintptr_t>(std::numeric_limits<int64_t>::max()),
      operation,
      " returned an address outside the signed 64-bit range");
  return static_cast<int64_t>(raw);
}

hipMemAllocationProp make_allocation_prop(int device_id, bool shareable) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = shareable ? hipMemHandleTypePosixFileDescriptor : hipMemHandleTypeNone;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device_id;
  return prop;
}

}  // namespace

bool hip_vmm_is_supported(int64_t device_id) {
  const int hip_device_id = validate_device(device_id);
  int supported = 0;
  const hipError_t status =
      hipDeviceGetAttribute(&supported, hipDeviceAttributeVirtualMemoryManagementSupported, hip_device_id);
  if (status == hipErrorNotSupported) {
    return false;
  }
  check_hip(status, "hipDeviceGetAttribute(VirtualMemoryManagementSupported)");
  return supported != 0;
}

int64_t hip_vmm_get_allocation_granularity(int64_t device_id, bool shareable, bool recommended) {
  const int hip_device_id = validate_device(device_id);
  const hipMemAllocationProp prop = make_allocation_prop(hip_device_id, shareable);
  const auto option = recommended ? hipMemAllocationGranularityRecommended : hipMemAllocationGranularityMinimum;
  size_t granularity = 0;
  check_hip(hipMemGetAllocationGranularity(&granularity, &prop, option), "hipMemGetAllocationGranularity");
  TORCH_CHECK(granularity > 0, "hipMemGetAllocationGranularity returned zero");
  TORCH_CHECK(
      granularity <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
      "HIP allocation granularity is outside the signed 64-bit range");
  return static_cast<int64_t>(granularity);
}

int64_t hip_vmm_create(int64_t size, int64_t device_id, bool shareable) {
  const size_t allocation_size = validate_size(size, "size");
  const int hip_device_id = validate_device(device_id);
  const hipMemAllocationProp prop = make_allocation_prop(hip_device_id, shareable);

  hipMemGenericAllocationHandle_t handle = nullptr;
  const hipError_t status = hipMemCreate(&handle, allocation_size, &prop, 0);
  if (status != hipSuccess && handle != nullptr) {
    (void)hipMemRelease(handle);
    handle = nullptr;
  }
  check_hip(status, "hipMemCreate");
  return pointer_to_int(handle, "hipMemCreate");
}

void hip_vmm_release(int64_t handle) {
  check_hip(hipMemRelease(validate_handle(handle)), "hipMemRelease");
}

int64_t hip_vmm_address_reserve(int64_t size, int64_t alignment, int64_t requested_address) {
  const size_t reservation_size = validate_size(size, "size");
  TORCH_CHECK(alignment >= 0, "alignment must be non-negative, got ", alignment);
  TORCH_CHECK(
      alignment == 0 || (alignment & (alignment - 1)) == 0,
      "alignment must be zero or a power of two, got ",
      alignment);
  TORCH_CHECK(requested_address >= 0, "requested_address must be zero or a positive address, got ", requested_address);

  void* address = nullptr;
  const hipError_t status = hipMemAddressReserve(
      &address,
      reservation_size,
      static_cast<size_t>(alignment),
      reinterpret_cast<void*>(static_cast<uintptr_t>(requested_address)),
      0);
  if (status != hipSuccess && address != nullptr) {
    (void)hipMemAddressFree(address, reservation_size);
    address = nullptr;
  }
  check_hip(status, "hipMemAddressReserve");
  return pointer_to_int(address, "hipMemAddressReserve");
}

void hip_vmm_address_free(int64_t address, int64_t size) {
  check_hip(hipMemAddressFree(validate_address(address, "address"), validate_size(size, "size")), "hipMemAddressFree");
}

void hip_vmm_map(int64_t address, int64_t size, int64_t handle, int64_t offset) {
  check_hip(
      hipMemMap(
          validate_address(address, "address"),
          validate_size(size, "size"),
          validate_offset(offset),
          validate_handle(handle),
          0),
      "hipMemMap");
}

void hip_vmm_unmap(int64_t address, int64_t size) {
  check_hip(hipMemUnmap(validate_address(address, "address"), validate_size(size, "size")), "hipMemUnmap");
}

void hip_vmm_set_access(int64_t address, int64_t size, int64_t device_id) {
  hipMemAccessDesc access{};
  access.location.type = hipMemLocationTypeDevice;
  access.location.id = validate_device(device_id);
  access.flags = hipMemAccessFlagsProtReadWrite;
  check_hip(
      hipMemSetAccess(validate_address(address, "address"), validate_size(size, "size"), &access, 1),
      "hipMemSetAccess");
}

int64_t hip_vmm_export_fd(int64_t handle) {
  int fd = -1;
  const hipError_t status =
      hipMemExportToShareableHandle(&fd, validate_handle(handle), hipMemHandleTypePosixFileDescriptor, 0);
  if (status != hipSuccess && fd >= 0) {
    (void)::close(fd);
    fd = -1;
  }
  check_hip(status, "hipMemExportToShareableHandle(POSIX_FD)");
  TORCH_CHECK(fd >= 0, "hipMemExportToShareableHandle returned an invalid fd");
  return static_cast<int64_t>(fd);
}

int64_t hip_vmm_import_fd(int64_t fd) {
  TORCH_CHECK(fd >= 0, "fd must be non-negative, got ", fd);
  TORCH_CHECK(fd <= std::numeric_limits<int>::max(), "fd is too large: ", fd);
  int native_fd = static_cast<int>(fd);

  // HIP changed this ABI in ROCm 7.1. Older runtimes expect an int*, while
  // ROCm 7.1+ expects the descriptor value cast to void*.
#if HIP_VERSION >= 70100000
  void* os_handle = reinterpret_cast<void*>(static_cast<uintptr_t>(native_fd));
#else
  void* os_handle = &native_fd;
#endif

  hipMemGenericAllocationHandle_t handle = nullptr;
  const hipError_t status = hipMemImportFromShareableHandle(&handle, os_handle, hipMemHandleTypePosixFileDescriptor);
  if (status != hipSuccess && handle != nullptr) {
    (void)hipMemRelease(handle);
    handle = nullptr;
  }
  check_hip(status, "hipMemImportFromShareableHandle(POSIX_FD)");
  return pointer_to_int(handle, "hipMemImportFromShareableHandle");
}

at::Tensor hip_vmm_tensor_from_address(
    int64_t address, const std::vector<int64_t>& shape, at::ScalarType dtype, int64_t device_id) {
  void* data = validate_address(address, "address");
  const int hip_device_id = validate_device(device_id);
  TORCH_CHECK(dtype != at::ScalarType::Undefined, "dtype must be explicitly specified");

  int64_t numel = 1;
  for (const int64_t dimension : shape) {
    TORCH_CHECK(dimension >= 0, "shape dimensions must be non-negative, got dimension ", dimension);
    if (dimension == 0) {
      numel = 0;
      continue;
    }
    if (numel != 0) {
      TORCH_CHECK(numel <= std::numeric_limits<int64_t>::max() / dimension, "shape element count overflows int64");
      numel *= dimension;
    }
  }
  (void)numel;

  const auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, static_cast<c10::DeviceIndex>(hip_device_id));
  // VMM ownership stays with the caller. The Tensor must not outlive the
  // mapping; deleting it never releases or unmaps the external allocation.
  return at::from_blob(data, shape, [](void*) {}, options);
}
