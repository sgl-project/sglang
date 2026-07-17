#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "metal_common.h"

namespace nb = nanobind;

NB_MODULE(_metal, m) {
  m.def("register_library", &sglang::metal_common::register_library, nb::arg("path"));
  register_rope_pool_fused(m);
  register_paged_attention(m);
}
