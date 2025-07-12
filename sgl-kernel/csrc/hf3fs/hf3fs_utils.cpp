#include <torch/all.h>

#include <cstring>
#include <vector>

void read_shm(const torch::Tensor& shm, std::vector<torch::Tensor> dst, int64_t page_bytes) {
  char* src_ptr = static_cast<char*>(shm.data_ptr());
  for (size_t i = 0; i < dst.size(); ++i) {
    auto& t = dst[i];
    size_t t_bytes = t.numel() * t.element_size();
    char* dst_ptr = static_cast<char*>(t.data_ptr());
    std::memcpy(dst_ptr, src_ptr + i * page_bytes, t_bytes);
  }
}

void write_shm(const std::vector<torch::Tensor> src, torch::Tensor& shm, int64_t page_bytes) {
  char* dst_ptr = static_cast<char*>(shm.data_ptr());
  for (size_t i = 0; i < src.size(); ++i) {
    auto& t = src[i];
    size_t t_bytes = t.numel() * t.element_size();
    char* src_ptr = static_cast<char*>(t.data_ptr());
    std::memcpy(dst_ptr + i * page_bytes, src_ptr, t_bytes);
  }
}
