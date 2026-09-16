#pragma once

#include <stdexcept>
#include <torch/version.h>

#if TORCH_VERSION_MAJOR > 2 ||                                                 \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 13)
// Keep tch 0.24's removed alignment wrappers as explicit runtime errors.
#define align_as(...)                                                          \
  alias();                                                                     \
  throw std::runtime_error("align_as is unavailable in PyTorch 2.13+")
#define align_tensors(...)                                                     \
  autograd::variable_list{};                                                   \
  throw std::runtime_error("align_tensors is unavailable in PyTorch 2.13+")
#endif
