// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once
#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_fp16.h>
#else
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/nvls_device.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>

// comment this for test_mscclpp_allreduce.cu
#include "utils.h"

namespace sglang {

__device__ mscclpp::DeviceSyncer deviceSyncer;
__device__ mscclpp::DeviceSyncer allGatherDeviceSyncer;
__device__ mscclpp::DeviceSyncer reduceScatterDeviceSyncer;
__device__ mscclpp::DeviceSyncer ibDeviceSyncer;

template <typename To, typename From>
__forceinline__ __device__ To bit_cast(const From& src) {
  static_assert(sizeof(To) == sizeof(From), "Size mismatch for bit_cast");

  union {
    From f;
    To t;
  } u;
  u.f = src;
  return u.t;
}

template <typename T>
__forceinline__ __device__ T add_elements(T a, T b) {
  return a + b;
}

template <>
__forceinline__ __device__ __half2 add_elements(__half2 a, __half2 b) {
  return __hadd2(a, b);
}

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
template <>
__forceinline__ __device__ __nv_bfloat162 add_elements(__nv_bfloat162 a, __nv_bfloat162 b) {
  return __hadd2(a, b);
}
#endif

template <typename T>
__forceinline__ __device__ int4 add_vectors_helper(int4 a, int4 b) {
  int4 ret;
  ret.w = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.w), bit_cast<T, int>(b.w)));
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  ret.z = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.z), bit_cast<T, int>(b.z)));
  return ret;
}

template <typename T>
__forceinline__ __device__ int4 add_vectors(int4 a, int4 b) {
  return add_vectors_helper<T>(a, b);
}

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
template <>
__forceinline__ __device__ int4 add_vectors<__nv_bfloat16>(int4 a, int4 b) {
  return add_vectors_helper<__nv_bfloat162>(a, b);
}
#endif

template <>
__forceinline__ __device__ int4 add_vectors<__half>(int4 a, int4 b) {
  return add_vectors_helper<__half2>(a, b);
}

template <typename T>
__forceinline__ __device__ uint2 add_vectors_helper(uint2 a, uint2 b) {
  uint2 ret;
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  return ret;
}

template <typename T>
__forceinline__ __device__ uint2 add_vectors(uint2 a, uint2 b) {
  return add_vectors_helper<T>(a, b);
}

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
template <>
__forceinline__ __device__ uint2 add_vectors<__nv_bfloat16>(uint2 a, uint2 b) {
  return add_vectors_helper<__nv_bfloat162>(a, b);
}
#endif

template <>
__forceinline__ __device__ uint2 add_vectors<__half>(uint2 a, uint2 b) {
  return add_vectors_helper<__half2>(a, b);
}

template <typename T>
__forceinline__ __device__ int add_vectors_helper(int a, int b) {
  return bit_cast<int, T>(add_elements(bit_cast<T, int>(a), bit_cast<T, int>(b)));
}

template <typename T>
__forceinline__ __device__ int add_vectors(int a, int b) {
  return add_vectors_helper<T>(a, b);
}

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
template <>
__forceinline__ __device__ int add_vectors<__nv_bfloat16>(int a, int b) {
  return add_vectors_helper<__nv_bfloat162>(a, b);
}
#endif

template <>
__forceinline__ __device__ int add_vectors<__half>(int a, int b) {
  return add_vectors_helper<__half2>(a, b);
}

// -------------------------------------------------------
// allreduce_LL_1node using LLPacket, origin allreduce2
// -------------------------------------------------------

__device__ uint64_t globalFlag = 1;

template <typename TYPE>
__global__ void __launch_bounds__(1024, 1) allreduce_LL_1node(
    mscclpp::MemoryChannelDeviceHandle* memChans,
    TYPE* buff,
    TYPE* scratch,
    void* resultBuff,
    int rank,
    int worldSize,
    size_t nelems) {
  nelems = nelems / (sizeof(int) / sizeof(TYPE));
  // This version of allreduce only works for single nodes
  const int nPeers = worldSize - 1;
  const size_t nPkts = nelems / 2;
  const int nelemsPerRank = nelems / worldSize;
  const int nPktsPerRank = nelemsPerRank / 2;
  // flag for packets. Initially 1
  const uint32_t flag = (uint32_t)globalFlag;
  // thread block & channel info
  const int nBlocksPerPeer = gridDim.x / nPeers;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  const int remoteRank = peerIdx < rank ? peerIdx : peerIdx + 1;
  mscclpp::MemoryChannelDeviceHandle memChan = memChans[peerIdx];
  const int tid = threadIdx.x + localBlockIdx * blockDim.x;
  // double buffering
  size_t scratchBaseOffset = (flag & 1) ? 0 : nPkts * sizeof(mscclpp::LLPacket);
  void* scratchBuff = (void*)((char*)scratch + scratchBaseOffset);
  size_t scratchOffset = scratchBaseOffset + rank * nPktsPerRank * sizeof(mscclpp::LLPacket);
  size_t scratchResultOffset =
      (flag & 1) ? 2 * nPkts * sizeof(mscclpp::LLPacket) : 3 * nPkts * sizeof(mscclpp::LLPacket);
  size_t srcOffset = remoteRank * nelemsPerRank * sizeof(int);
  uint2* src = (uint2*)((char*)buff + rank * nelemsPerRank * sizeof(int));
  uint2* dst = (uint2*)((char*)resultBuff + rank * nelemsPerRank * sizeof(int));

  // step 1: write to scratch buffer
  memChan.putPackets(scratchOffset, srcOffset, nelemsPerRank * sizeof(int), tid, blockDim.x * nBlocksPerPeer, flag);
  // step 2: get data from scratch buffer, reduce data and write result to remote scratch buffer
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * gridDim.x) {
    uint2 data = make_uint2(0, 0);
    for (int index = 0; index < nPeers; index++) {
      const int remoteRank = index < rank ? index : index + 1;
      mscclpp::LLPacket* dstPkt = (mscclpp::LLPacket*)scratchBuff + remoteRank * nPktsPerRank;
      uint2 val = dstPkt[idx].read(flag);
      data = add_vectors<TYPE>(val, data);
    }
    data = add_vectors<TYPE>(data, src[idx]);
    dst[idx] = data;

    mscclpp::LLPacket packet;
    packet.data1 = data.x;
    packet.flag1 = flag;
    packet.data2 = data.y;
    packet.flag2 = flag;
    size_t offset = scratchResultOffset / sizeof(mscclpp::LLPacket) + (idx + rank * nPktsPerRank);
    for (int index = 0; index < nPeers; index++) {
      memChans[index].write(offset, packet);
    }
  }
  // step 3: get data result from scratch buffer
  mscclpp::LLPacket* dstPkt = (mscclpp::LLPacket*)((char*)scratch + scratchResultOffset);
  const int dstOffset = remoteRank * nPktsPerRank;
  uint2* result = (uint2*)((char*)resultBuff + remoteRank * nelemsPerRank * sizeof(int));
  for (int idx = threadIdx.x + localBlockIdx * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * nBlocksPerPeer) {
    uint2 data = dstPkt[idx + dstOffset].read(flag);
    result[idx].x = data.x;
    result[idx].y = data.y;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    globalFlag += 1;
  }
}

// -------------------------------------------------------
// allreduce_LL_2node using LLPacket, origin allreduce5
// -------------------------------------------------------

template <typename TYPE>
__global__ void __launch_bounds__(1024, 1) allreduce_LL_2node(
    mscclpp::MemoryChannelDeviceHandle* memChans,
    mscclpp::PortChannelDeviceHandle* portChans,
    TYPE* buff,
    TYPE* scratch,
    TYPE* putBuff,
    TYPE* resultBuff,
    int rank,
    int nRanksPerNode,
    int worldSize,
    size_t nelems) {
  nelems = nelems / (sizeof(int) / sizeof(TYPE));
  // This version of allreduce only works for single nodes
  const int nPeersInNode = nRanksPerNode - 1;
  const int nPkts = nelems / 2;
  const int nelemsPerLocalRank = nelems / nRanksPerNode;
  const int nPktsPerLocalRank = nelemsPerLocalRank / 2;
  const int localRankId = rank % nRanksPerNode;
  // flag for packets. Initially 1
  const uint32_t flag = (uint32_t)globalFlag;
  // thread block & channel info
  const int nBlocksPerPeer = gridDim.x / nPeersInNode;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  const int remoteRankIdx = peerIdx < localRankId ? peerIdx : peerIdx + 1;
  mscclpp::MemoryChannelDeviceHandle memChan = memChans[peerIdx];
  mscclpp::PortChannelDeviceHandle portChan = portChans[localRankId];
  const int tid = threadIdx.x + localBlockIdx * blockDim.x;
  // double buffering
  size_t scratchBaseOffset = (flag & 1) ? 0 : nPkts * sizeof(mscclpp::LLPacket);
  size_t putBaseOffset = (flag & 1) ? 0 : nPktsPerLocalRank * sizeof(mscclpp::LLPacket);
  void* scratchBuff = (void*)((char*)scratch + scratchBaseOffset);
  size_t scratchOffset = scratchBaseOffset + localRankId * nPktsPerLocalRank * sizeof(mscclpp::LLPacket);
  size_t scratchResultOffset =
      (flag & 1) ? 2 * nPkts * sizeof(mscclpp::LLPacket) : 3 * nPkts * sizeof(mscclpp::LLPacket);
  size_t srcOffset = remoteRankIdx * nelemsPerLocalRank * sizeof(int);
  uint2* src = (uint2*)((char*)buff + localRankId * nelemsPerLocalRank * sizeof(int));
  uint2* dst = (uint2*)((char*)resultBuff + localRankId * nelemsPerLocalRank * sizeof(int));

  // step 1: write to scratch buffer
  if (nRanksPerNode > 1) {
    memChan.putPackets(
        scratchOffset, srcOffset, nelemsPerLocalRank * sizeof(int), tid, blockDim.x * nBlocksPerPeer, flag);
  }
  // step 2: get data from scratch buffer, do local reduce-scatter in each node.
  mscclpp::LLPacket* putPkt = (mscclpp::LLPacket*)((char*)putBuff + putBaseOffset);
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPktsPerLocalRank; idx += blockDim.x * gridDim.x) {
    uint2 data = make_uint2(0, 0);
    for (int index = 0; index < nPeersInNode; index++) {
      const int remoteRank = index < localRankId ? index : index + 1;
      mscclpp::LLPacket* dstPkt = (mscclpp::LLPacket*)scratchBuff + remoteRank * nPktsPerLocalRank;
      uint2 val = dstPkt[idx].read(flag);
      data = add_vectors<TYPE>(val, data);
    }
    data = add_vectors<TYPE>(data, src[idx]);
    putPkt[idx].write(data.x, data.y, flag);
    dst[idx] = data;
  }
  deviceSyncer.sync(gridDim.x);
  // step 3. send local reduced data to remote node.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    portChan.put(scratchOffset, putBaseOffset, nPktsPerLocalRank * sizeof(mscclpp::LLPacket));
    if ((flag & 63) == 0) {
      portChan.flush();
    }
  }
  // step 4. try to read the data from scratch buffer and write to local peers
  mscclpp::LLPacket* dstPkt = (mscclpp::LLPacket*)scratchBuff + localRankId * nPktsPerLocalRank;
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPktsPerLocalRank; idx += blockDim.x * gridDim.x) {
    uint2 res = dst[idx];
    uint2 val = dstPkt[idx].read(flag);
    res = add_vectors<TYPE>(res, val);

    mscclpp::LLPacket packet;
    packet.data1 = res.x;
    packet.flag1 = flag;
    packet.data2 = res.y;
    packet.flag2 = flag;
    size_t offset = scratchResultOffset / sizeof(mscclpp::LLPacket) + (idx + localRankId * nPktsPerLocalRank);
    for (int index = 0; index < nPeersInNode; index++) {
      memChans[index].write(offset, packet);
    }
    dst[idx] = res;
  }

  // step 5: get data result from scratch buffer
  dstPkt = (mscclpp::LLPacket*)((char*)scratch + scratchResultOffset);
  const int dstOffset = remoteRankIdx * nPktsPerLocalRank;
  uint2* result = (uint2*)((char*)resultBuff + remoteRankIdx * nelemsPerLocalRank * sizeof(int));
  if (nRanksPerNode > 1) {
    for (int idx = threadIdx.x + localBlockIdx * blockDim.x; idx < nPktsPerLocalRank;
         idx += blockDim.x * nBlocksPerPeer) {
      uint2 data = dstPkt[idx + dstOffset].read(flag);
      result[idx] = data;
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    globalFlag += 1;
  }
}

static const mscclpp::Transport IBs[] = {
    mscclpp::Transport::IB0,
    mscclpp::Transport::IB1,
    mscclpp::Transport::IB2,
    mscclpp::Transport::IB3,
    mscclpp::Transport::IB4,
    mscclpp::Transport::IB5,
    mscclpp::Transport::IB6,
    mscclpp::Transport::IB7};

class MscclCommGroup {
 public:
  std::shared_ptr<mscclpp::Communicator> comm_;
  const size_t rank_;
  const size_t world_size_;
  const std::vector<int64_t> rank_to_node_;
  const std::vector<int64_t> rank_to_ib_;
  MscclCommGroup(
      mscclpp::UniqueId unique_id,
      const size_t rank,
      const size_t world_size,
      const std::vector<int64_t>& rank_to_node,
      const std::vector<int64_t>& rank_to_ib)
      : rank_(rank), world_size_(world_size), rank_to_node_(rank_to_node), rank_to_ib_(rank_to_ib) {
    auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world_size);
    bootstrap->initialize(unique_id);
    comm_ = std::make_shared<mscclpp::Communicator>(bootstrap);
  }
  template <typename T>
  void allreduce(cudaStream_t stream, T* output, size_t input_numel, int threads = 512, int block_limit = 21) {
    throw std::runtime_error("you should not call allreduce of a base context");
  }
  bool is_same_node(int r1, int r2) {
    return rank_to_node_[r1] == rank_to_node_[r2];
  }

  void make_connection(
      std::unordered_map<int, std::shared_ptr<mscclpp::Connection>>& same_node_connections,
      std::unordered_map<int, std::shared_ptr<mscclpp::Connection>>& cross_node_connections) {
    same_node_connections.clear();
    cross_node_connections.clear();
    std::unordered_map<int, mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> conn_futures;
    for (int r = 0; r < world_size_; ++r) {
      if (r == rank_) continue;
      mscclpp::Transport transport = is_same_node(r, rank_) ? mscclpp::Transport::CudaIpc : IBs[rank_to_ib_[r]];
      conn_futures.emplace(r, comm_->connectOnSetup(r, 0, transport));
    }
    comm_->setup();
    for (int r = 0; r < world_size_; ++r) {
      if (r == rank_) continue;
      if (is_same_node(r, rank_)) {
        same_node_connections.emplace(r, conn_futures[r].get());
      } else {
        cross_node_connections.emplace(r, conn_futures[r].get());
      }
    }
  }

  void make_memory_channels_with_scratch(
      void* tensor_ptr,
      const size_t tensor_bytes,
      void* scratch_ptr,
      const size_t scratch_bytes,
      const std::unordered_map<int, std::shared_ptr<mscclpp::Connection>>& connections,
      std::unordered_map<int, std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>>& semaphores,
      std::unordered_map<int, mscclpp::RegisteredMemory>& registered_memories,
      std::unordered_map<int, mscclpp::MemoryChannel>& channels) {
    channels.clear();
    make_semaphores<mscclpp::MemoryDevice2DeviceSemaphore>(connections, semaphores);
    register_tensor_with_connections(scratch_ptr, scratch_bytes, connections, registered_memories);
    for (const auto& [peer, _] : connections) {
      channels.emplace(
          peer, mscclpp::MemoryChannel(semaphores[peer], registered_memories[peer], tensor_ptr, scratch_ptr));
    }
  }
  void make_port_channels_with_scratch(
      std::shared_ptr<mscclpp::ProxyService> proxyService,
      void* tensor_ptr,
      const size_t tensor_bytes,
      void* scratch_ptr,
      const size_t scratch_bytes,
      const std::unordered_map<int, std::shared_ptr<mscclpp::Connection>>& connections,
      std::unordered_map<int, std::shared_ptr<mscclpp::Host2DeviceSemaphore>>& semaphores,
      std::unordered_map<int, mscclpp::RegisteredMemory>& registered_memories,
      std::unordered_map<int, mscclpp::PortChannel>& channels) {
    channels.clear();
    make_semaphores<mscclpp::Host2DeviceSemaphore>(connections, semaphores);

    mscclpp::TransportFlags flags;
    for (const auto& [_, conn] : connections) {
      flags |= conn->transport();
    }
    auto local_reg_memory = comm_->registerMemory(tensor_ptr, tensor_bytes, flags);

    register_tensor_with_connections(scratch_ptr, scratch_bytes, connections, registered_memories);
    std::unordered_map<int, mscclpp::SemaphoreId> semaphore_ids;
    std::unordered_map<int, size_t> memory_ids;
    memory_ids[rank_] = proxyService->addMemory(local_reg_memory);
    for (const auto& [peer, memory] : registered_memories) {
      if (peer == rank_) continue;
      memory_ids[peer] = proxyService->addMemory(memory);
    }
    for (const auto& [peer, semaphore] : semaphores) {
      semaphore_ids[peer] = proxyService->addSemaphore(semaphore);
    }

    for (const auto& [peer, _] : connections) {
      channels.emplace(peer, proxyService->portChannel(semaphore_ids[peer], memory_ids[peer], memory_ids[rank_]));
    }
  }

  template <typename SemaphoreType>
  void make_semaphores(
      const std::unordered_map<int, std::shared_ptr<mscclpp::Connection>>& connections,
      std::unordered_map<int, std::shared_ptr<SemaphoreType>>& semaphores) {
    semaphores.clear();
    for (const auto& [peer, conn] : connections) {
      semaphores[peer] = std::make_shared<SemaphoreType>(*comm_, conn);
    }
    comm_->setup();
  }

  void register_tensor_with_connections(
      void* tensor_ptr,
      size_t tensor_bytes,
      const std::unordered_map<int, std::shared_ptr<mscclpp::Connection>>& connections,
      std::unordered_map<int, mscclpp::RegisteredMemory>& registered_memories) {
    registered_memories.clear();
    mscclpp::TransportFlags all_transports;
    for (const auto& [_, connection] : connections) {
      all_transports |= connection->transport();
    }
    mscclpp::RegisteredMemory buf_reg_mem = comm_->registerMemory(tensor_ptr, tensor_bytes, all_transports);
    registered_memories[rank_] = buf_reg_mem;

    std::unordered_map<int, mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remote_mem_futures;
    for (const auto& [r, connection] : connections) {
      comm_->sendMemoryOnSetup(buf_reg_mem, r, 0);
      auto remoteMemory = comm_->recvMemoryOnSetup(r, 0);
      remote_mem_futures.emplace(r, remoteMemory);
    }
    comm_->setup();
    for (auto& [r, mem_feature] : remote_mem_futures) {
      registered_memories.emplace(r, mem_feature.get());
    }
  }

  void make_device_memory_handle_base_on_new_ptr(
      const std::unordered_map<int, mscclpp::MemoryChannel>& old_memory_channels,
      std::unordered_map<int, mscclpp::RegisteredMemory>& registered_sm_memories,
      std::unordered_map<int, std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>>& memory_semaphores,
      std::unordered_map<int, mscclpp::MemoryChannel>& memory_channels,
      mscclpp::GpuBuffer<mscclpp::MemoryChannelDeviceHandle>& device_memory_handle,
      void* input,
      void* scratch,
      const cudaStream_t stream) {
    memory_channels.clear();
    for (const auto& [peer, channel] : old_memory_channels) {
      memory_channels.emplace(
          peer, mscclpp::MemoryChannel(memory_semaphores[peer], registered_sm_memories[peer], input, scratch));
    }
    std::vector<mscclpp::MemoryChannel> memory_channels_list;
    for (int r = 0; r < world_size_; r++) {
      if (r == rank_) continue;
      if (is_same_node(r, rank_)) {
        memory_channels_list.push_back(memory_channels[r]);
      }
    }
    std::vector<mscclpp::MemoryChannelDeviceHandle> memory_channel_handlers(memory_channels_list.size());
    std::transform(
        memory_channels_list.begin(),
        memory_channels_list.end(),
        memory_channel_handlers.begin(),
        [](const mscclpp::MemoryChannel& channel) { return channel.deviceHandle(); });
    mscclpp::gpuMemcpyAsync<mscclpp::MemoryChannelDeviceHandle>(
        device_memory_handle.data(),
        memory_channel_handlers.data(),
        memory_channel_handlers.size(),
        stream,
        cudaMemcpyHostToDevice);
  }
};

class Msccl1NodeLLcontext {
 private:
  std::shared_ptr<MscclCommGroup> comm_group_ = nullptr;
  void* scratch_;
  const size_t scratch_bytes_;
  std::unordered_map<int, std::shared_ptr<mscclpp::Connection>> same_node_connections_;
  std::unordered_map<int, std::shared_ptr<mscclpp::Connection>> cross_node_connections_;

  std::unordered_map<int, mscclpp::RegisteredMemory> registered_sm_memories_;
  std::unordered_map<int, std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> memory_semaphores_;
  std::unordered_map<int, mscclpp::MemoryChannel> memory_channels_;
  mscclpp::GpuBuffer<mscclpp::MemoryChannelDeviceHandle> d_memHandles_;
  std::unordered_map<void*, std::unordered_map<int, mscclpp::MemoryChannel>> input_ptr2memory_channels_;
  std::unordered_map<void*, mscclpp::GpuBuffer<mscclpp::MemoryChannelDeviceHandle>> input_ptr2d_memHandles_;
  cudaStream_t h2d_stream;
  const size_t nranks_per_node_;

 public:
  Msccl1NodeLLcontext(
      mscclpp::UniqueId unique_id,
      const size_t rank,
      const size_t world_size,
      void* scratch,
      const size_t scratch_bytes,
      const size_t nranks_per_node,
      const std::vector<int64_t>& rank_to_node,
      const std::vector<int64_t>& rank_to_ib)
      : scratch_(scratch),
        scratch_bytes_(scratch_bytes),
        nranks_per_node_(nranks_per_node),
        d_memHandles_(nranks_per_node - 1) {
    CHECK_CUDA_SUCCESS(cudaStreamCreateWithFlags(&h2d_stream, cudaStreamNonBlocking));
    comm_group_ = std::make_shared<MscclCommGroup>(unique_id, rank, world_size, rank_to_node, rank_to_ib);
    comm_group_->make_connection(same_node_connections_, cross_node_connections_);
    comm_group_->make_memory_channels_with_scratch(
        scratch_,
        scratch_bytes_,
        scratch_,
        scratch_bytes_,
        same_node_connections_,
        memory_semaphores_,
        registered_sm_memories_,
        memory_channels_);
    std::vector<mscclpp::MemoryChannel> memory_channels_list;
    for (int r = 0; r < comm_group_->world_size_; r++) {
      if (r == comm_group_->rank_) continue;
      memory_channels_list.push_back(memory_channels_[r]);
    }
    std::vector<mscclpp::MemoryChannelDeviceHandle> memory_channel_handlers(memory_channels_list.size());
    std::transform(
        memory_channels_list.begin(),
        memory_channels_list.end(),
        memory_channel_handlers.begin(),
        [](const mscclpp::MemoryChannel& channel) { return channel.deviceHandle(); });
    mscclpp::gpuMemcpy<mscclpp::MemoryChannelDeviceHandle>(
        d_memHandles_.data(), memory_channel_handlers.data(), memory_channel_handlers.size(), cudaMemcpyHostToDevice);
  }

  ~Msccl1NodeLLcontext() {
    CHECK_CUDA_SUCCESS(cudaStreamDestroy(h2d_stream));
  }

  template <typename T>
  void allreduce(cudaStream_t stream, T* input, T* output, size_t input_numel, int nthreads = 512, int nblocks = 21) {
    dim3 nthrs(nthreads);
    dim3 nblks(nblocks);
    cudaStreamCaptureStatus capturing_status;
    CHECK_CUDA_SUCCESS(cudaStreamIsCapturing(stream, &capturing_status));
    mscclpp::MemoryChannelDeviceHandle* memChans;
    if (capturing_status != cudaStreamCaptureStatusActive) {
      std::unordered_map<int, mscclpp::MemoryChannel> memory_channels;
      comm_group_->make_device_memory_handle_base_on_new_ptr(
          memory_channels_,
          registered_sm_memories_,
          memory_semaphores_,
          memory_channels,
          d_memHandles_,
          input,
          scratch_,
          h2d_stream);
      CHECK_CUDA_SUCCESS(cudaStreamSynchronize(h2d_stream));
      memChans = d_memHandles_.data();
    } else {
      void* input_void_ptr = reinterpret_cast<void*>(input);
      if (input_ptr2d_memHandles_.find(input_void_ptr) == input_ptr2d_memHandles_.end()) {
        std::unordered_map<int, mscclpp::MemoryChannel> memory_channels;
        mscclpp::GpuBuffer<mscclpp::MemoryChannelDeviceHandle> device_memory_handle(comm_group_->world_size_ - 1);
        comm_group_->make_device_memory_handle_base_on_new_ptr(
            memory_channels_,
            registered_sm_memories_,
            memory_semaphores_,
            memory_channels,
            device_memory_handle,
            input,
            scratch_,
            h2d_stream);
        input_ptr2memory_channels_.emplace(input_void_ptr, memory_channels);
        input_ptr2d_memHandles_.emplace(input_void_ptr, device_memory_handle);
      }
      auto it = input_ptr2d_memHandles_.find(input_void_ptr);
      memChans = it->second.data();
    }
    allreduce_LL_1node<T><<<nblks, nthrs, 0, stream>>>(
        memChans, (T*)input, (T*)scratch_, output, comm_group_->rank_, comm_group_->world_size_, input_numel);

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
      printf("rank: %lu failed to launch allreduce_LL_1node: %s\n", comm_group_->rank_, cudaGetErrorString(status));
    }
  }
};

class Msccl2NodeLLcontext {
 private:
  std::shared_ptr<MscclCommGroup> comm_group_ = nullptr;
  void* scratch_;
  const size_t scratch_bytes_;
  void* put_buffer_;
  const size_t put_buffer_bytes_;
  std::unordered_map<int, std::shared_ptr<mscclpp::Connection>> same_node_connections_;
  std::unordered_map<int, std::shared_ptr<mscclpp::Connection>> cross_node_connections_;

  std::unordered_map<int, mscclpp::RegisteredMemory> registered_sm_memories_;
  std::unordered_map<int, mscclpp::RegisteredMemory> registered_port_memories_;

  std::unordered_map<int, std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> memory_semaphores_;
  std::unordered_map<int, std::shared_ptr<mscclpp::Host2DeviceSemaphore>> port_semaphores_;

  std::unordered_map<int, mscclpp::MemoryChannel> memory_channels_;
  std::unordered_map<int, mscclpp::PortChannel> port_channels_;

  mscclpp::GpuBuffer<mscclpp::MemoryChannelDeviceHandle> d_memHandles_;
  mscclpp::GpuBuffer<mscclpp::PortChannelDeviceHandle> d_portHandles_;

  std::shared_ptr<mscclpp::ProxyService> proxyService;
  cudaStream_t h2d_stream;
  const size_t nranks_per_node_;

  std::unordered_map<void*, std::unordered_map<int, mscclpp::MemoryChannel>> input_ptr2memory_channels_;
  std::unordered_map<void*, mscclpp::GpuBuffer<mscclpp::MemoryChannelDeviceHandle>> input_ptr2d_memHandles_;

 public:
  Msccl2NodeLLcontext(
      mscclpp::UniqueId unique_id,
      const size_t rank,
      const size_t world_size,
      void* scratch,
      const size_t scratch_bytes,
      void* put_buffer,
      const size_t put_buffer_bytes,
      const size_t nranks_per_node,
      const std::vector<int64_t>& rank_to_node,
      const std::vector<int64_t>& rank_to_ib)
      : scratch_(scratch),
        scratch_bytes_(scratch_bytes),
        put_buffer_(put_buffer),
        put_buffer_bytes_(put_buffer_bytes),
        nranks_per_node_(nranks_per_node),
        d_memHandles_(nranks_per_node - 1),
        d_portHandles_(world_size - nranks_per_node) {
    CHECK_CUDA_SUCCESS(cudaStreamCreateWithFlags(&h2d_stream, cudaStreamNonBlocking));
    comm_group_ = std::make_shared<MscclCommGroup>(unique_id, rank, world_size, rank_to_node, rank_to_ib);
    proxyService = std::make_shared<mscclpp::ProxyService>();
    proxyService->startProxy();
    comm_group_->make_connection(same_node_connections_, cross_node_connections_);
    comm_group_->make_memory_channels_with_scratch(
        scratch_,
        scratch_bytes_,
        scratch_,
        scratch_bytes_,
        same_node_connections_,
        memory_semaphores_,
        registered_sm_memories_,
        memory_channels_);
    comm_group_->make_port_channels_with_scratch(
        proxyService,
        put_buffer_,
        put_buffer_bytes_,
        scratch_,
        scratch_bytes_,
        cross_node_connections_,
        port_semaphores_,
        registered_port_memories_,
        port_channels_);
    std::vector<mscclpp::MemoryChannel> memory_channels_list;
    std::vector<mscclpp::PortChannel> port_channels_list;
    for (int r = 0; r < comm_group_->world_size_; r++) {
      if (r == comm_group_->rank_) continue;
      if (comm_group_->is_same_node(r, comm_group_->rank_)) {
        memory_channels_list.push_back(memory_channels_[r]);
      } else {
        port_channels_list.push_back(port_channels_[r]);
      }
    }
    std::vector<mscclpp::MemoryChannelDeviceHandle> memory_channel_handlers(memory_channels_list.size());
    std::transform(
        memory_channels_list.begin(),
        memory_channels_list.end(),
        memory_channel_handlers.begin(),
        [](const mscclpp::MemoryChannel& channel) { return channel.deviceHandle(); });
    mscclpp::gpuMemcpy<mscclpp::MemoryChannelDeviceHandle>(
        d_memHandles_.data(), memory_channel_handlers.data(), memory_channel_handlers.size(), cudaMemcpyHostToDevice);

    std::vector<mscclpp::PortChannelDeviceHandle> port_channel_handlers(port_channels_list.size());
    std::transform(
        port_channels_list.begin(),
        port_channels_list.end(),
        port_channel_handlers.begin(),
        [](const mscclpp::PortChannel& channel) { return channel.deviceHandle(); });
    mscclpp::gpuMemcpy<mscclpp::PortChannelDeviceHandle>(
        d_portHandles_.data(), port_channel_handlers.data(), port_channel_handlers.size(), cudaMemcpyHostToDevice);
  }

  ~Msccl2NodeLLcontext() {
    CHECK_CUDA_SUCCESS(cudaStreamDestroy(h2d_stream));
    if (proxyService) {
      proxyService->stopProxy();
    }
  }

  template <typename T>
  void
  allreduce(cudaStream_t stream, T* input, T* output, const size_t input_numel, int nthreads = 512, int nblocks = 21) {
    dim3 nthrs(nthreads);
    dim3 nblks(nblocks);
    cudaStreamCaptureStatus capturing_status;
    CHECK_CUDA_SUCCESS(cudaStreamIsCapturing(stream, &capturing_status));
    mscclpp::MemoryChannelDeviceHandle* memChans;
    if (capturing_status != cudaStreamCaptureStatusActive) {
      std::unordered_map<int, mscclpp::MemoryChannel> memory_channels;
      comm_group_->make_device_memory_handle_base_on_new_ptr(
          memory_channels_,
          registered_sm_memories_,
          memory_semaphores_,
          memory_channels,
          d_memHandles_,
          input,
          scratch_,
          h2d_stream);
      CHECK_CUDA_SUCCESS(cudaStreamSynchronize(h2d_stream));
      memChans = d_memHandles_.data();
    } else {
      void* input_void_ptr = reinterpret_cast<void*>(input);
      if (input_ptr2d_memHandles_.find(input_void_ptr) == input_ptr2d_memHandles_.end()) {
        std::unordered_map<int, mscclpp::MemoryChannel> memory_channels;
        mscclpp::GpuBuffer<mscclpp::MemoryChannelDeviceHandle> device_memory_handle(7);
        comm_group_->make_device_memory_handle_base_on_new_ptr(
            memory_channels_,
            registered_sm_memories_,
            memory_semaphores_,
            memory_channels,
            device_memory_handle,
            input,
            scratch_,
            h2d_stream);
        input_ptr2memory_channels_.emplace(input_void_ptr, memory_channels);
        input_ptr2d_memHandles_.emplace(input_void_ptr, device_memory_handle);
      }
      auto it = input_ptr2d_memHandles_.find(input_void_ptr);
      memChans = it->second.data();
    }
    allreduce_LL_2node<T><<<nblks, nthrs, 0, stream>>>(
        memChans,
        d_portHandles_.data(),
        (T*)input,
        (T*)scratch_,
        (T*)put_buffer_,
        output,
        comm_group_->rank_,
        nranks_per_node_,
        comm_group_->world_size_,
        input_numel);

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
      printf("rank: %lu failed to launch allreduce_LL_2node: %s\n", comm_group_->rank_, cudaGetErrorString(status));
    }
  }
};

}  // namespace sglang
