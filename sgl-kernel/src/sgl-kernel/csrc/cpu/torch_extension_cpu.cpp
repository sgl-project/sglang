/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch/extension.h>

#include "shm.h"

// rmsnorm
void rmsnorm_cpu(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps);

// fused_add_rmsnorm
void fused_add_rmsnorm_cpu(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps);

// shared memory init
void initialize(int size, int rank);

// shared mmeory all_reduce
void shm_allreduce(at::Tensor& data, c10::intrusive_ptr<c10d::ProcessGroup> process_group, py::object op);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // norm
  m.def("rmsnorm_cpu", &rmsnorm_cpu, "Root mean square normalization for CPU");
  m.def("fused_add_rmsnorm_cpu", &fused_add_rmsnorm_cpu, "Fused add root mean square normalization for CPU");

  // all reduce
  m.def("initialize", &initialize, "shared memory initialization for CPU");
  m.def("shm_allreduce", &shm_allreduce, "low latency all_reduce implementation for CPU");
}
