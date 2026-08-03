/* Copyright 2026 SGLang Team. All Rights Reserved. */

#pragma once

#include <ATen/ATen.h>

#include <cstdint>

bool dwdp_hsa_copy_is_available();
int64_t dwdp_hsa_copy_engine_for_devices(int64_t destination_device, int64_t source_device);
int64_t dwdp_hsa_copy_async(
    const at::Tensor& destination, const at::Tensor& source, int64_t destination_device, int64_t source_device);
int64_t dwdp_hsa_copy_wait(int64_t ticket);
void dwdp_hsa_copy_destroy(int64_t ticket);
