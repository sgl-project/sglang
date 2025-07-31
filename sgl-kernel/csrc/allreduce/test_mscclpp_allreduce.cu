/*
 * this file is used to test mscclpp_allreduce.cu using mpirun
 * this file is adapted from https://github.com/flashinfer-ai/flashinfer/blob/v0.2.5/src/test_sum_all_reduce.cu
usage:
cd PATH-TO-THIS-FILE
export MPI_HOME=/usr/local/mpi
# export MPI_HOME=/opt/hpcx/ompi/
export MSCCLPP_HOME=/workspace/test/mscclpp
nvcc -O2 -arch=native -std=c++17 test_mscclpp_allreduce.cu \
  -o test_mscclpp_allreduce -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I${MSCCLPP_HOME}/include -L${MSCCLPP_HOME}/build -lmscclpp \
  -lnccl -I${MPI_HOME}/include -L${MPI_HOME}/lib -lmpi

/opt/hpcx/ompi/bin/
mpirun --allow-run-as-root -H 127.0.0.1:8 -np 8 \
  --map-by ppr:8:node \
  --mca btl_openib_warn_no_device_params_found 0 \
  --mca btl_tcp_if_include bond0 \
  --allow-run-as-root -np 8 \
  -x NCCL_RUNTIME_CONNECT=0 -x NCCL_IB_GID_INDEX=3 -x NCCL_DEBUG=WARN \
  -x LD_PRELOAD=${MSCCLPP_HOME}/build/libmscclpp.so ./test_mscclpp_allreduce
 */
#include <mpi.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#ifndef CHECK_CUDA_SUCCESS
#define CHECK_CUDA_SUCCESS(cmd)                                                             \
  do {                                                                                      \
    cudaError_t e = cmd;                                                                    \
    if (e != cudaSuccess) {                                                                 \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)
#endif

#include <cstdint>

#include "mscclpp_allreduce.cuh"

template <typename T>
bool isclose(T a, T b, float rtol = 1e-5, float atol = 1e-8) {
  return fabs(a - b) <= (atol + rtol * fabs(b));
}

int main(int argc, char* argv[]) {
  // init mpi
  MPI_Init(&argc, &argv);
  printf("MPI Initialized.\n");
  int nranks, rank;

  // get work size and rank id
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cudaSetDevice(rank);
  printf("nranks: %d, rank: %d\n", nranks, rank);

  // init host and device buffers
  using T = float;
  using ReduceT = float;
  const size_t num_elems = 2 * 1024 * 1024;
  std::vector<T> host_buf(num_elems);
  for (uint32_t i = 0; i < num_elems; ++i) {
    host_buf[i] = T(i + rank);
  }
  thrust::device_vector<T> device_buf(host_buf);
  const size_t buf_size_in_bytes = num_elems * sizeof(T);
  std::vector<T> host_result_buf(num_elems);
  thrust::device_vector<T> device_result_buf(host_result_buf);

  std::vector<T> host_scratch_buf(num_elems * 8);
  for (uint32_t i = 0; i < num_elems; ++i) {
    host_scratch_buf[i] = 1;
  }
  thrust::device_vector<T> device_scratch_buf(host_scratch_buf);
  std::vector<T> host_put_buf(num_elems);
  thrust::device_vector<T> device_put_buf(host_put_buf);

  mscclpp::UniqueId unique_id;
  if (rank == 0) unique_id = mscclpp::TcpBootstrap::createUniqueId();
  MPI_Bcast(&unique_id, sizeof(unique_id), MPI_BYTE, 0, MPI_COMM_WORLD);

  std::vector<int64_t> rank_to_node(nranks);
  std::vector<int64_t> rank_to_ib(nranks);
  for (int i = 0; i < nranks; i++) {
    rank_to_node[i] = i / 8;
    rank_to_ib[i] = i % 8;
  }

  cudaStream_t s;
  CHECK_CUDA_SUCCESS(cudaStreamCreate(&s));
  CHECK_CUDA_SUCCESS(cudaStreamSynchronize(s));
  if (nranks == 8) {
    auto context = std::make_shared<sglang::Msccl1NodeLLcontext>(
        unique_id,
        rank,
        nranks,
        thrust::raw_pointer_cast(device_scratch_buf.data()),
        buf_size_in_bytes * 8,
        rank_to_node,
        rank_to_ib);
    printf("rank: %d, Msccl1NodeLLcontext setup.\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    context->allreduce<T>(
        s,
        thrust::raw_pointer_cast(device_buf.data()),
        thrust::raw_pointer_cast(device_result_buf.data()),
        device_buf.size());
  } else if (nranks == 16) {
    // TODO: this branch is untested since there is something wrong with mpirun in my test machince
    auto context = std::make_shared<sglang::Msccl2NodeLLcontext>(
        unique_id,
        rank,
        nranks,
        thrust::raw_pointer_cast(device_scratch_buf.data()),
        buf_size_in_bytes * 8,
        thrust::raw_pointer_cast(device_put_buf.data()),
        buf_size_in_bytes,
        rank_to_node,
        rank_to_ib);
    printf("rank: %d, Msccl2NodeLLcontext setup.\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    context->allreduce<T>(
        s,
        thrust::raw_pointer_cast(device_buf.data()),
        thrust::raw_pointer_cast(device_result_buf.data()),
        device_buf.size());
  }

  // check result correctness
  thrust::host_vector<T> host_buf_result = device_result_buf;
  size_t num_results_error_atol_1e_3_rtol_1e_3 = 0;
  bool nan_detected = false;

  for (uint32_t i = 0; i < num_elems; ++i) {
    T expected = T(i * nranks + (nranks - 1) * nranks / 2);
    if (std::isnan(float(host_buf_result[i]))) {
      nan_detected = true;
    }
    if (!isclose(float(host_buf_result[i]), float(expected), 1e-3, 1e-3)) {
      num_results_error_atol_1e_3_rtol_1e_3++;
    }
  }
  float result_accuracy = 1. - float(num_results_error_atol_1e_3_rtol_1e_3) / float(num_elems);

  printf("rank: %d, nan_detected: %d accuracy: %f\n", rank, nan_detected, result_accuracy);

  CHECK_CUDA_SUCCESS(cudaStreamDestroy(s));
  MPI_Finalize();
  return 0;
}
