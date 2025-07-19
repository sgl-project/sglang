#include <torch/all.h>

#include <cstring>
#include <vector>

void read_shm(const torch::Tensor& shm, std::vector<torch::Tensor> dst) {
  char* src_ptr = static_cast<char*>(shm.data_ptr());
  size_t current = 0;
  for (size_t i = 0; i < dst.size(); ++i) {
    auto& t = dst[i];
    size_t t_bytes = t.numel() * t.element_size();
    char* dst_ptr = static_cast<char*>(t.data_ptr());
    std::memcpy(dst_ptr, src_ptr + current, t_bytes);
    current += t_bytes;
  }
}

void write_shm(const std::vector<torch::Tensor> src, torch::Tensor& shm) {
  char* dst_ptr = static_cast<char*>(shm.data_ptr());
  size_t current = 0;
  for (size_t i = 0; i < src.size(); ++i) {
    auto& t = src[i];
    size_t t_bytes = t.numel() * t.element_size();
    char* src_ptr = static_cast<char*>(t.data_ptr());
    std::memcpy(dst_ptr + current, src_ptr, t_bytes);
    current += t_bytes;
  }
}
