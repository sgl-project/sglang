#include <torch/extension.h>

using fptr_t = int64_t;
fptr_t init_custom_ar(int64_t rank_id, int64_t world_size, const std::vector<fptr_t>& buffers,
                      const std::vector<fptr_t>& barrier_in, const std::vector<fptr_t>& barrier_out);
void dispose(fptr_t _fa);
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_custom_ar", &init_custom_ar, "init custom allreduce meta (CUDA)");
  m.def("dispose", &dispose, "dispose custom allreduce meta");
  m.def("all_reduce", &all_reduce, "custom all reduce (CUDA)");
}
