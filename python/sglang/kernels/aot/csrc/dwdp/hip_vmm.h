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

#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <vector>

bool hip_vmm_is_supported(int64_t device_id);
int64_t hip_vmm_get_allocation_granularity(int64_t device_id, bool shareable, bool recommended);
int64_t hip_vmm_create(int64_t size, int64_t device_id, bool shareable);
void hip_vmm_release(int64_t handle);
int64_t hip_vmm_address_reserve(int64_t size, int64_t alignment, int64_t requested_address);
void hip_vmm_address_free(int64_t address, int64_t size);
void hip_vmm_map(int64_t address, int64_t size, int64_t handle, int64_t offset);
void hip_vmm_unmap(int64_t address, int64_t size);
void hip_vmm_set_access(int64_t address, int64_t size, int64_t device_id);
// The returned fd is caller-owned and must be closed exactly once.
int64_t hip_vmm_export_fd(int64_t handle);
// Borrows fd for the duration of the call; it never consumes or closes fd.
int64_t hip_vmm_import_fd(int64_t fd);
at::Tensor hip_vmm_tensor_from_address(
    int64_t address, const std::vector<int64_t>& shape, at::ScalarType dtype, int64_t device_id);
