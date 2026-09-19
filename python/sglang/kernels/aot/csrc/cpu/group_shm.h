#pragma once

#include <ATen/ATen.h>

#include <cstddef>
#include <cstdint>
#include <string>

int64_t shm_group_initialize(const std::string& group_name, int64_t group_size, int64_t group_rank);
void group_all_gather(int64_t handle, char* output_ptr, char* input_ptr, size_t data_size);
void group_all_to_all(int64_t handle, char* output_ptr, char* input_ptr, size_t data_size);
void group_all_reduce(int64_t handle, char* data_ptr, c10::ScalarType scalar_type, size_t data_size, size_t numel);