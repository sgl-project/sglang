#include <sgl_kernel/logging.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/extra/stl.h>
#include <tvm/ffi/reflection/registry.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <functional>
#include <optional>
#include <vector>

namespace sglang {

/**
 * \brief Pinned host memory at VMM granularity (2MB), shareable by POSIX fd.
 *
 * One physical allocation on the device's host NUMA node, mapped with device
 * and host read-write access: the loader fills it with plain CPU writes and
 * kernels read the same VA. Peers import the exported fd (sent over a UDS,
 * never by number) and map the same physical pages.
 */
struct VMMHostMemory : public tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.VMMHostMemory", VMMHostMemory, tvm::ffi::Object);
  static constexpr bool _type_mutable = true;

  struct CleanUpStack {
   public:
    CleanUpStack() = default;
    CleanUpStack(const CleanUpStack&) = delete;
    CleanUpStack(CleanUpStack&&) = default;
    CleanUpStack& operator=(const CleanUpStack&) = delete;
    CleanUpStack& operator=(CleanUpStack&&) = default;

    void append(std::function<void()> func) {
      m_func.push_back(std::move(func));
    }
    void reserve(std::size_t size) {
      m_func.reserve(size);
    }
    void clear() {
      while (!m_func.empty()) {
        m_func.back()();
        m_func.pop_back();
      }
    }
    ~CleanUpStack() {
      this->clear();
    }

   private:
    std::vector<std::function<void()>> m_func;
  };

  VMMHostMemory(const VMMHostMemory&) = delete;
  VMMHostMemory& operator=(const VMMHostMemory&) = delete;
  // The cleanup lambdas capture `this`; keep the object unmovable.
  VMMHostMemory(VMMHostMemory&&) = delete;
  VMMHostMemory& operator=(VMMHostMemory&&) = delete;

  /// \brief Create a new allocation, or import the peer's exported `fd`
  /// (ownership of the fd stays with the caller).
  explicit VMMHostMemory(std::size_t size, std::optional<int> fd) {
    // use current numa
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    int host_numa = 0;
    CHECK_CUDA(cuDeviceGetAttribute(&host_numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device));
    CHECK_HOST(host_numa != -1) << "system does not support NUMA";

    const auto prop = [=] {
      CUmemAllocationProp p = {};
      p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      p.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
      p.location.id = host_numa;
      p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
      return p;
    }();

    std::size_t gmin = 0, grec = 0;
    CHECK_CUDA(cuMemGetAllocationGranularity(&gmin, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    CHECK_CUDA(cuMemGetAllocationGranularity(&grec, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    const auto granularity = std::max(gmin, grec);
    m_size = host::div_ceil(size, granularity) * granularity;
    CleanUpStack cleanup;
    cleanup.reserve(3);
    if (fd.has_value()) {
      // import from external fd: create from fd
      CHECK_HOST(m_size == size) << "External size must be aligned to granularity: "  //
                                 << size << ", " << m_size << ", " << granularity;
      const auto fd_ptr = std::bit_cast<void*>(static_cast<intptr_t>(fd.value()));
      CHECK_CUDA(cuMemImportFromShareableHandle(&m_handle, fd_ptr, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    } else {
      // normal VMM allocation: create
      CHECK_CUDA(cuMemCreate(&m_handle, m_size, &prop, 0));
    }
    cleanup.append([this] {
      if (const auto err = cuMemRelease(m_handle); host::runtime::is_cuda_error(err)) {
        SGL_LOG_WARNING << "Failed to release memory handle: " << host::runtime::get_cuda_error_string(err)
                        << ". Silently ignoring this error.";
      }
    });

    CHECK_CUDA(cuMemAddressReserve(&m_ptr, m_size, granularity, 0, 0));
    cleanup.append([this] {
      if (const auto err = cuMemAddressFree(m_ptr, m_size); host::runtime::is_cuda_error(err)) {
        SGL_LOG_WARNING << "Failed to free memory address: " << host::runtime::get_cuda_error_string(err)
                        << ". Silently ignoring this error.";
      }
    });

    CHECK_CUDA(cuMemMap(m_ptr, m_size, 0, m_handle, 0));
    cleanup.append([this] {
      if (const auto err = cuMemUnmap(m_ptr, m_size); host::runtime::is_cuda_error(err)) {
        SGL_LOG_WARNING << "Failed to unmap memory: " << host::runtime::get_cuda_error_string(err)
                        << ". Silently ignoring this error.";
      }
    });

    CUmemAccessDesc access[2] = {};
    access[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access[0].location.id = device;
    access[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    access[1].location.id = host_numa;
    access[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CUDA(cuMemSetAccess(m_ptr, m_size, access, 2));

    m_cleanup = std::move(cleanup);
  }

  /// \brief A fresh fd for a peer to import; the caller owns closing it.
  int export_fd() const {
    int fd = 0;
    CHECK_CUDA(cuMemExportToShareableHandle(&fd, m_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    return fd;
  }

  std::uintptr_t ptr() const {
    return static_cast<std::uintptr_t>(m_ptr);
  }

  /// \brief The granularity-rounded size; peers must pass exactly this.
  std::size_t size() const {
    return m_size;
  }

  /// \brief Unmap and release now; idempotent (the moved-from stack is empty).
  void close() {
    m_cleanup.clear();
    m_ptr = 0;
    m_size = 0;
  }

  CUmemGenericAllocationHandle m_handle;
  CUdeviceptr m_ptr;
  std::size_t m_size;
  CleanUpStack m_cleanup;
};

inline void register_vmm_host_memory() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<VMMHostMemory>()
      .def(refl::init<std::size_t, std::optional<int>>(), "__init__")
      .def("export_fd", &VMMHostMemory::export_fd)
      .def("ptr", &VMMHostMemory::ptr)
      .def("size", &VMMHostMemory::size)
      .def("close", &VMMHostMemory::close);
}

inline bool is_vmm_available() {
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  // Host-NUMA VMM locations exist since CUDA 12.2; older drivers reject cuMemCreate.
  int driver = 0;
  CHECK_CUDA(cuDriverGetVersion(&driver));
  if (driver < 12020) return false;
  int host_numa = -1, posix_fd = 0;
  CHECK_CUDA(cuDeviceGetAttribute(&host_numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device));
  CHECK_CUDA(cuDeviceGetAttribute(&posix_fd, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
  return host_numa != -1 && posix_fd != 0;
}

}  // namespace sglang
