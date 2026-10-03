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

// PT 2.14-specific shims live in their own file so they can be dropped when
// tch-rs ships a build against PyTorch 2.14; the include is a no-op on older
// torch versions.
#include "torch_2_14_compat.h"
