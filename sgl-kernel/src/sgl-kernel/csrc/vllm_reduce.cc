#include <torch/extension.h>

using fptr_t = int64_t;
fptr_t init_custom_ar(const std::vector<int64_t>& fake_ipc_ptrs,
                      torch::Tensor& rank_data, int64_t rank, bool full_nvlink);
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                fptr_t reg_buffer, int64_t reg_buffer_sz_bytes);
void dispose(fptr_t _fa);
int64_t meta_size();
void register_buffer(fptr_t _fa, const std::vector<int64_t>& fake_ipc_ptrs);
std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa);
void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_custom_ar", &init_custom_ar, "Init Custom AllReduce (CUDA)");
  m.def("all_reduce", &all_reduce, "Do AllReduce (CUDA)");
  m.def("dispose", &dispose, "Dispose Custom AllReduce Meta");
  m.def("meta_size", &meta_size, "Get Custom AllReduce Meta Size");
  m.def("register_buffer", &register_buffer, "Register Custom AllReduce Buffer");
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta, "Get Custom AllReduce Graph Meta Size");
  m.def("register_graph_buffers", &register_graph_buffers, "Register Custom AllReduce Graph Buffer");
}
