#include <torch/all.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#ifndef __SHM_COLLECTIVES__
#define __SHM_COLLECTIVES__
#define VECTOR_LENGTH_IN_BYTES 32

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
#endif
