#pragma once

#include <iostream>
#include <vector>
#include <cuda.h>
#include <torch/torch.h>
#include <torch/extension.h>

#define LOGE(format, ...) fprintf(stdout, "L%d:" format "\n", __LINE__, ##__VA_ARGS__); fflush(stdout);
#define ASSERT(cond, ...) { if(!(cond)) { LOGE(__VA_ARGS__); assert(0); } }
#define WARN(cond, ...) { if(!(cond)) { LOGE(__VA_ARGS__); } }

#define ROUND_UP(x, n) (((x) + ((n) - 1)) / (n) * (n))

#define DRV_CALL(call)                                                                                 \
    {                                                                                                  \
        CUresult result = (call);                                                                      \
        if (CUDA_SUCCESS != result)                                                    \
        {                                                                                              \
            const char *errMsg; cuGetErrorString(result, &errMsg);                                     \
            ASSERT(0, "Error when exec " #call " %s-%d code:%d err:%s", __FUNCTION__, __LINE__, result, errMsg); \
        }                                                                                              \
    }

class PhyBlock {
    public:
      PhyBlock(int device_id, size_t block_size);
      ~PhyBlock();

      size_t block_size;
      int device_id;
      CUmemGenericAllocationHandle alloc_handle;
      CUresult status;
};

class VmmTensor {
public:
    VmmTensor(std::vector<int64_t> shape, torch::Dtype dtype, int offset_index, int world_size, int pre_flag);
    ~VmmTensor();

    void AllocMemory(int offset_index, int world_size, int pre_flag);
    torch::Tensor GetTensor();
    torch::Tensor SplitTensor(std::vector<int64_t> shape, torch::Dtype dtype, int offset_idnex);
    torch::Tensor GetTensor(std::vector<int64_t>& shape, torch::Dtype dtype);

private:
    int device_id;
    size_t padded_size;
    size_t actual_size;
    size_t used_size;
    int world_size;

    std::mutex mtx;
    torch::Tensor tensor;
    torch::Tensor offset_tensor;

    std::unique_ptr<PhyBlock> u_p_block;
    CUdeviceptr v_ptr;
    CUdeviceptr offset_v_ptr = 0;
    size_t offset_size = 0;
};

std::vector<std::shared_ptr<PhyBlock>> shared_phy_blocks_pre;
std::vector<std::shared_ptr<PhyBlock>> shared_phy_blocks_post;
std::vector<std::unique_ptr<PhyBlock>> unique_phy_blocks;
void init_shared_phy_blocks(int num_blocks, size_t block_size);
void init_unique_phy_blocks(int num_blocks, size_t block_size);
void release_shared_phy_blocks();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m ){
    m.doc() = "vTensor";

    pybind11::class_<VmmTensor>(m, "tensor")
        .def(pybind11::init<std::vector<int64_t>, torch::Dtype, int, int, int>() )
        .def("realloc_memory", &VmmTensor::AllocMemory)
        .def("split_tensor", &VmmTensor::SplitTensor)
        .def("to_torch_tensor", py::overload_cast<>(&VmmTensor::GetTensor));

    m.def("init_shared_phy_blocks", &init_shared_phy_blocks, "init_shared_phy_blocks");
    m.def("init_unique_phy_blocks", &init_unique_phy_blocks, "init_unique_phy_blocks");
    m.def("release_shared_phy_blocks", &release_shared_phy_blocks, "release_shared_phy_blocks");
}
