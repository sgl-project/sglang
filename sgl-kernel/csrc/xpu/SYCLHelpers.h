#pragma once

#include <sycl/sycl.hpp>

template <typename ker_t, int dim>
static inline void sycl_kernel_submit(::sycl::range<dim> range, ::sycl::queue q, ker_t ker) {
  q.parallel_for<ker_t>(range, ker);
}

// Additional convention of SYCL kernel configuration. Besides construct kernel
// functor, SYCL has some additional conventions required during setup of
// SYCL command group handler, e.g. declaring SYCL local accessor when the
// kernel requires shared local memory usage. Helpers below help simpilfiy
// submission of SYCL kernels requiring additional conventions.

// If a kernel functor defines additional SYCL requirements that must be added
// during the submission, the kernel functor must define a method
// 'sycl_ker_config_convention'  that takes a SYCL handler object.
// Note that virtual functions cannot be used to enforce the convention
// due to sycl::is_device_copyable restrictions
// (https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#device-copyable),
// so the code enforces this by requiring the struct to be a parent
// of any functor used for the SYCL kernel submissions.
struct __SYCL_KER_CONFIG_CONVENTION__ {};

template <typename ker_t, int dim>
static inline typename std::enable_if<std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>, void>::type
sycl_kernel_submit(::sycl::range<dim> global_range, ::sycl::range<dim> local_range, ::sycl::queue q, ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    ker.sycl_ker_config_convention(cgh);
    cgh.parallel_for<ker_t>(::sycl::nd_range<dim>(global_range, local_range), ker);
  };
  q.submit(cgf);
}

template <typename ker_t, int dim>
static inline typename std::enable_if<!std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>, void>::type
sycl_kernel_submit(::sycl::range<dim> global_range, ::sycl::range<dim> local_range, ::sycl::queue q, ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    cgh.parallel_for<ker_t>(::sycl::nd_range<dim>(global_range, local_range), ker);
  };
  q.submit(cgf);
}

template <typename ker_t>
static inline typename std::enable_if<std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>, void>::type
sycl_kernel_submit(int64_t global_range, int64_t local_range, ::sycl::queue q, ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    ker.sycl_ker_config_convention(cgh);
    cgh.parallel_for<ker_t>(::sycl::nd_range<1>(::sycl::range<1>(global_range), ::sycl::range<1>(local_range)), ker);
  };
  q.submit(cgf);
}

template <typename ker_t>
static inline typename std::enable_if<!std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>, void>::type
sycl_kernel_submit(int64_t global_range, int64_t local_range, ::sycl::queue q, ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    cgh.parallel_for<ker_t>(::sycl::nd_range<1>(::sycl::range<1>(global_range), ::sycl::range<1>(local_range)), ker);
  };
  q.submit(cgf);
}

#define SYCL_KERNEL_STRING(var, str) static const __attribute__((opencl_constant)) char var[] = str;
#define SYCL_KERNEL_PRINTF sycl::ext::oneapi::experimental::printf

#define SYCL_PRINT(fmt_str, ...)                \
  {                                             \
    SYCL_KERNEL_STRING(fmt_var, fmt_str);       \
    SYCL_KERNEL_PRINTF(fmt_var, ##__VA_ARGS__); \
  }
