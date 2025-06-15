#include <cuda.h>
#include <mutex>
#include <numeric>

#include "vtensor.h"

void init_shared_phy_blocks(int num_blocks, size_t block_size) {
  int device_id = -1;
  DRV_CALL(cuCtxGetDevice(&device_id));
  for (int i = 0; i < num_blocks; i++) {
    std::shared_ptr<PhyBlock> phy_block_pre =
        std::make_shared<PhyBlock>(device_id, block_size);
    if (phy_block_pre->status != CUDA_SUCCESS) {
      WARN(0, "init_shared_phy_blocks failed");
      return;
    }
    shared_phy_blocks_pre.emplace_back(phy_block_pre);
    std::shared_ptr<PhyBlock> phy_block_post =
        std::make_shared<PhyBlock>(device_id, block_size);
    if (phy_block_post->status != CUDA_SUCCESS) {
      WARN(0, "init_shared_phy_blocks failed");
      return;
    }
    shared_phy_blocks_post.emplace_back(phy_block_post);
  }
}

void init_unique_phy_blocks(int num_blocks, size_t block_size) {
  int device_id = -1;
  DRV_CALL(cuCtxGetDevice(&device_id));
  for (int i = 0; i < num_blocks; i++) {
    std::unique_ptr<PhyBlock> phy_block =
        std::make_unique<PhyBlock>(device_id, block_size);
    if (phy_block->status != CUDA_SUCCESS) {
      WARN(0, "init_unique_phy_blocks failed");
      return;
    }
    unique_phy_blocks.emplace_back(std::move(phy_block));
  }
}

void release_shared_phy_blocks() {
  int blocks_size = shared_phy_blocks_pre.size();
  for (int i = 0; i < blocks_size; i++) {
    auto tmp_pre = std::move(shared_phy_blocks_pre[blocks_size - i - 1]);
    shared_phy_blocks_pre.pop_back();
    auto tmp_post = std::move(shared_phy_blocks_post[blocks_size - i - 1]);
    shared_phy_blocks_post.pop_back();
  }
}

PhyBlock::PhyBlock(int device_id, size_t block_size) {
  this->device_id = device_id;
  this->block_size = block_size;
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;

  status = cuMemCreate(&alloc_handle, block_size, &prop, 0ULL);
}

PhyBlock::~PhyBlock() {
  if (status == CUDA_SUCCESS) {
    status = cuMemRelease(alloc_handle);
    DRV_CALL(status);
  }
}

VmmTensor::VmmTensor(std::vector<int64_t> shape, torch::Dtype dtype,
                     int offset_index, int world_size, int pre_flag)
    : device_id(-1), used_size(0), world_size(world_size) {
  if (device_id == -1) {
    DRV_CALL(cuCtxGetDevice(&device_id));
  }

  size_t dtype_size = torch::elementSize(dtype);
  actual_size = std::accumulate(shape.begin(), shape.end(), dtype_size,
                                std::multiplies<int64_t>());

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  size_t granularity;
  DRV_CALL(cuMemGetAllocationGranularity(&granularity, &prop,
                                         CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  padded_size = ROUND_UP(actual_size, granularity);
  DRV_CALL(cuMemAddressReserve(&v_ptr, padded_size, 0ULL, 0ULL, 0ULL));

  AllocMemory(offset_index, world_size, pre_flag);

  tensor = GetTensor(shape, dtype);
}

void VmmTensor::AllocMemory(int offset_index, int world_size, int pre_flag) {
  // Avoid concurrency issues caused by retries or others
  std::lock_guard<std::mutex> lock(mtx);

  size_t offset_size = padded_size / world_size;
  int shared_phy_index = 0;
  for (int i = 0; i < world_size; i++) {
    char *offset_addr = (char *)v_ptr + i * offset_size;
    if (i == offset_index) {
      assert(unique_phy_blocks.size() >= 0);
      this->u_p_block =
          std::move(unique_phy_blocks[unique_phy_blocks.size() - 1]);
      unique_phy_blocks.pop_back();
      DRV_CALL(cuMemMap(reinterpret_cast<CUdeviceptr>(offset_addr), offset_size,
                        0ULL, this->u_p_block->alloc_handle, 0ULL));
      CUmemAccessDesc accessDesc = {};
      accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      accessDesc.location.id = this->device_id;
      accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      DRV_CALL(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(offset_addr),
                              offset_size, &accessDesc, 1));
    } else {
      std::shared_ptr<PhyBlock> phy_block;
      if (pre_flag) {
        assert(shared_phy_index < shared_phy_blocks_pre.size());
        phy_block = shared_phy_blocks_pre[shared_phy_index];
      } else {
        assert(shared_phy_index < shared_phy_blocks_post.size());
        phy_block = shared_phy_blocks_post[shared_phy_index];
      }
      DRV_CALL(cuMemMap(reinterpret_cast<CUdeviceptr>(offset_addr), offset_size,
                        0ULL, phy_block->alloc_handle, 0ULL));
      CUmemAccessDesc accessDesc = {};
      accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      accessDesc.location.id = this->device_id;
      accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      DRV_CALL(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(offset_addr),
                              offset_size, &accessDesc, 1));
      shared_phy_index++;
    }
  }
  used_size = actual_size;
}

torch::Tensor VmmTensor::SplitTensor(std::vector<int64_t> shape,
                                     torch::Dtype dtype, int offset_index) {
  if (offset_v_ptr != 0) {
    throw std::runtime_error("SplitTensor already called");
  }
  offset_size = padded_size / world_size;
  DRV_CALL(cuMemAddressReserve(&offset_v_ptr, offset_size, 0ULL, 0ULL, 0ULL));
  DRV_CALL(cuMemMap(offset_v_ptr, offset_size, 0ULL,
                    this->u_p_block->alloc_handle, 0ULL));
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = this->device_id;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  DRV_CALL(cuMemSetAccess(offset_v_ptr, offset_size, &accessDesc, 1));
  std::vector<int64_t> stride(shape.size());
  stride[stride.size() - 1] = 1;
  for (int i = stride.size() - 2; i >= 0; i--) {
    stride[i] = shape[i + 1] * stride[i + 1];
  }
  torch::TensorOptions options =
      torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device_id);
  offset_tensor = torch::from_blob(
      reinterpret_cast<void *>(offset_v_ptr), shape, stride,
      [](void *offset_v_ptr) {}, options);
  return offset_tensor;
}

torch::Tensor VmmTensor::GetTensor() { return this->tensor; }

torch::Tensor VmmTensor::GetTensor(std::vector<int64_t> &shape,
                                   torch::Dtype dtype) {
  std::vector<int64_t> stride(shape.size());
  stride[stride.size() - 1] = 1;
  for (int i = stride.size() - 2; i >= 0; i--) {
    stride[i] = shape[i + 1] * stride[i + 1];
  }

  torch::TensorOptions options =
      torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device_id);
  torch::Tensor tensor = torch::from_blob(
      reinterpret_cast<void *>(v_ptr), shape, stride, [](void *v_ptr) {},
      options);

  return tensor;
}

VmmTensor::~VmmTensor() {
  if (v_ptr) {
    DRV_CALL(cuMemUnmap(v_ptr, padded_size));
    DRV_CALL(cuMemAddressFree(v_ptr, padded_size));
  }
  if (offset_v_ptr != 0) {
    DRV_CALL(cuMemUnmap(offset_v_ptr, offset_size));
    DRV_CALL(cuMemAddressFree(offset_v_ptr, offset_size));
  }
  auto tmp = std::move(u_p_block);
}
