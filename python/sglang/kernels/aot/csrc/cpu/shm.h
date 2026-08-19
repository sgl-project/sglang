#include <torch/all.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#ifndef __SHM_COLLECTIVES__
#define __SHM_COLLECTIVES__

constexpr int STATE_GROUP_SYMMETRIC_ALLREDUCE = 0;
constexpr int STATE_GROUP_DISTRIBUTED_ALLREDUCE = 1;
constexpr int STATE_GROUP_ALL_GATHER = 2;
constexpr int STATE_GROUP_ALL_GATHER_INTO_TENSOR = 3;
constexpr int STATE_GROUP_REDUCE_SCATTER = 4;

void shm_initialize(int size, int rank, const char* addr_string, const char* port_string);
void all_reduce_outer_loop(torch::Tensor& data, size_t numel, int data_size);
template <int STATE_GROUP>
torch::Tensor& all_gather(torch::Tensor& result, torch::Tensor& data, int dim, size_t numel, int data_size);
void reduce_scatter_outer_loop(torch::Tensor& output, torch::Tensor& data, size_t numel, int data_size);
int64_t shm_group_initialize(const std::string& group_name, int64_t group_size, int64_t group_rank);
void group_all_gather(int64_t handle, char* output_ptr, char* input_ptr, size_t data_size);
void group_all_to_all(int64_t handle, char* output_ptr, char* input_ptr, size_t data_size);
void group_all_reduce(int64_t handle, char* data_ptr, c10::ScalarType scalar_type, size_t data_size, size_t numel);
#endif
