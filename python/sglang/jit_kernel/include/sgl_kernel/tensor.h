#pragma once
#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#ifdef __CUDACC__
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

namespace host {

namespace details {

inline constexpr auto kAnyDeviceID = -1;
inline constexpr auto kAnySize = static_cast<int64_t>(-1);
inline constexpr auto kNullSize = static_cast<int64_t>(-1);
inline constexpr auto kNullDType = static_cast<DLDataTypeCode>(18u);
inline constexpr auto kNullDevice = static_cast<DLDeviceType>(-1);

struct SizeRef;
struct DTypeRef;
struct DeviceRef;

template <typename T>
struct dtype_trait {};

template <std::integral T>
struct dtype_trait<T> {
  inline static constexpr DLDataType value = {
      .code = std::is_signed_v<T> ? DLDataTypeCode::kDLInt : DLDataTypeCode::kDLUInt,
      .bits = static_cast<std::uint8_t>(sizeof(T) * 8),
      .lanes = 1};
};

template <std::floating_point T>
struct dtype_trait<T> {
  inline static constexpr DLDataType value = {
      .code = DLDataTypeCode::kDLFloat, .bits = static_cast<std::uint8_t>(sizeof(T) * 8), .lanes = 1};
};

#ifdef __CUDACC__
template <>
struct dtype_trait<__half> {
  inline static constexpr DLDataType value = {.code = DLDataTypeCode::kDLFloat, .bits = 16, .lanes = 1};
};
template <>
struct dtype_trait<__nv_bfloat16> {
  inline static constexpr DLDataType value = {.code = DLDataTypeCode::kDLBfloat, .bits = 16, .lanes = 1};
};
#endif

template <DLDeviceType Code>
struct device_trait {
  inline static constexpr DLDevice value = {.device_type = Code, .device_id = kAnyDeviceID};
};

template <typename... Ts>
inline constexpr auto kDTypeList = std::array<DLDataType, sizeof...(Ts)>{dtype_trait<Ts>::value...};

template <DLDeviceType... Codes>
inline constexpr auto kDeviceList = std::array<DLDevice, sizeof...(Codes)>{device_trait<Codes>::value...};

template <typename T>
struct PrintAbleSpan {
  explicit PrintAbleSpan(std::span<const T> data) : data(data) {}
  std::span<const T> data;
};

// define DLDataType comparison and printing in root namespace
inline constexpr auto kDeviceStringMap = [] {
  constexpr auto map = std::array<std::pair<DLDeviceType, const char*>, 16>{
      std::pair{DLDeviceType::kDLCPU, "cpu"},
      std::pair{DLDeviceType::kDLCUDA, "cuda"},
      std::pair{DLDeviceType::kDLCUDAHost, "cuda_host"},
      std::pair{DLDeviceType::kDLOpenCL, "opencl"},
      std::pair{DLDeviceType::kDLVulkan, "vulkan"},
      std::pair{DLDeviceType::kDLMetal, "metal"},
      std::pair{DLDeviceType::kDLVPI, "vpi"},
      std::pair{DLDeviceType::kDLROCM, "rocm"},
      std::pair{DLDeviceType::kDLROCMHost, "rocm_host"},
      std::pair{DLDeviceType::kDLExtDev, "ext_dev"},
      std::pair{DLDeviceType::kDLCUDAManaged, "cuda_managed"},
      std::pair{DLDeviceType::kDLOneAPI, "oneapi"},
      std::pair{DLDeviceType::kDLWebGPU, "webgpu"},
      std::pair{DLDeviceType::kDLHexagon, "hexagon"},
      std::pair{DLDeviceType::kDLMAIA, "maia"},
      std::pair{DLDeviceType::kDLTrn, "trn"},
  };
  constexpr auto max_type = stdr::max(map | stdv::keys);
  auto result = std::array<std::string_view, max_type + 1>{};
  for (const auto& [code, name] : map) {
    result[static_cast<std::size_t>(code)] = name;
  }
  return result;
}();

struct PrintableDevice {
  DLDevice device;
};

inline auto& operator<<(std::ostream& os, DLDevice device) {
  const auto& mapping = kDeviceStringMap;
  const auto entry = static_cast<std::size_t>(device.device_type);
  RuntimeCheck(entry < mapping.size());
  const auto name = mapping[entry];
  RuntimeCheck(!name.empty(), "Unknown device: ", int(device.device_type));
  os << name;
  if (device.device_id != kAnyDeviceID && device.device_type != DLDeviceType::kDLCPU) {
    os << ":" << device.device_id;
  }
  return os;
}

inline auto& operator<<(std::ostream& os, PrintableDevice pd) {
  return os << pd.device;
}

template <typename T>
inline auto& operator<<(std::ostream& os, PrintAbleSpan<T> span) {
  os << "[";
  for (const auto i : irange(span.data.size())) {
    if (i > 0) {
      os << ", ";
    }
    os << span.data[i];
  }
  os << "]";
  return os;
}

}  // namespace details

struct SymbolicSize {
 public:
  SymbolicSize(std::string_view annotation = {}) : m_value(details::kNullSize), m_annotation(annotation) {}
  SymbolicSize(const SymbolicSize&) = delete;
  SymbolicSize& operator=(const SymbolicSize&) = delete;

  auto get_name() const -> std::string_view {
    return m_annotation;
  }

  auto set_value(int64_t value) -> void {
    RuntimeCheck(!this->has_value(), "Size value already set");
    m_value = value;
  }

  auto has_value() const -> bool {
    return m_value != details::kNullSize;
  }

  auto get_value() const -> std::optional<int64_t> {
    return this->has_value() ? std::optional{m_value} : std::nullopt;
  }

  auto unwrap(DebugInfo info = {}) const -> int64_t {
    RuntimeCheck(info, this->has_value(), "Size value is not set");
    return m_value;
  }

  auto verify(int64_t value, const char* prefix, int64_t dim) -> void {
    if (this->has_value()) {
      if (m_value != value) {
        [[unlikely]];
        Panic("Size mismatch for ", m_name_str(prefix, dim), ": expected ", m_value, " but got ", value);
      }
    } else {
      this->set_value(value);
    }
  }

  auto value_or_name(const char* prefix, int64_t dim) const -> std::string {
    if (const auto value = this->get_value()) {
      return std::to_string(*value);
    } else {
      return m_name_str(prefix, dim);
    }
  }

 private:
  auto m_name_str(const char* prefix, int64_t dim) const -> std::string {
    std::ostringstream os;
    os << prefix << '#' << dim;
    if (!m_annotation.empty()) os << "('" << m_annotation << "')";
    return std::move(os).str();
  }

  std::int64_t m_value;
  std::string_view m_annotation;
};

inline auto operator==(DLDevice lhs, DLDevice rhs) -> bool {
  return lhs.device_type == rhs.device_type && lhs.device_id == rhs.device_id;
}

struct SymbolicDType {
 public:
  SymbolicDType() : m_value({details::kNullDType, 0, 0}) {}
  SymbolicDType(const SymbolicDType&) = delete;
  SymbolicDType& operator=(const SymbolicDType&) = delete;

  auto set_value(DLDataType value) -> void {
    RuntimeCheck(!this->has_value(), "Dtype value already set");
    RuntimeCheck(
        m_check(value), "Dtype value [", value, "] not in the allowed options: ", details::PrintAbleSpan{m_options});
    m_value = value;
  }

  auto has_value() const -> bool {
    return m_value.code != details::kNullDType;
  }

  auto get_value() const -> std::optional<DLDataType> {
    return this->has_value() ? std::optional{m_value} : std::nullopt;
  }

  auto unwrap(DebugInfo info = {}) const -> DLDataType {
    RuntimeCheck(info, this->has_value(), "Dtype value is not set");
    return m_value;
  }

  auto set_options(std::span<const DLDataType> options) -> void {
    m_options = options;
  }

  template <typename... Ts>
  auto set_options() -> void {
    m_options = details::kDTypeList<Ts...>;
  }

  auto verify(DLDataType dtype) -> void {
    if (this->has_value()) {
      RuntimeCheck(m_value == dtype, "DType mismatch: expected ", m_value, " but got ", dtype);
    } else {
      this->set_value(dtype);
    }
  }

 private:
  auto m_check(DLDataType value) const -> bool {
    return stdr::empty(m_options) || (stdr::find(m_options, value) != stdr::end(m_options));
  }

  std::span<const DLDataType> m_options;
  DLDataType m_value;
};

struct SymbolicDevice {
 public:
  SymbolicDevice() : m_value({details::kNullDevice, details::kAnyDeviceID}) {}
  SymbolicDevice(const SymbolicDevice&) = delete;
  SymbolicDevice& operator=(const SymbolicDevice&) = delete;

  auto set_value(DLDevice value) -> void {
    RuntimeCheck(!this->has_value(), "Device value already set");
    RuntimeCheck(
        m_check(value),
        "Device value [",
        details::PrintableDevice{value},
        "] not in the allowed options: ",
        details::PrintAbleSpan{m_options});
    m_value = value;
  }

  auto has_value() const -> bool {
    return m_value.device_type != details::kNullDevice;
  }

  auto get_value() const -> std::optional<DLDevice> {
    return this->has_value() ? std::optional{m_value} : std::nullopt;
  }

  auto unwrap(DebugInfo info = {}) const -> DLDevice {
    RuntimeCheck(info, this->has_value(), "Device value is not set");
    return m_value;
  }

  auto set_options(std::span<const DLDevice> options) -> void {
    m_options = options;
  }

  template <DLDeviceType... Codes>
  auto set_options() -> void {
    m_options = details::kDeviceList<Codes...>;
  }

  auto verify(DLDevice device) -> void {
    if (this->has_value()) {
      RuntimeCheck(
          m_value == device,
          "Device mismatch: expected ",
          details::PrintableDevice{m_value},
          " but got ",
          details::PrintableDevice{device});
    } else {
      this->set_value(device);
    }
  }

 private:
  auto m_check(DLDevice value) const -> bool {
    return stdr::empty(m_options) || (stdr::any_of(m_options, [value](const DLDevice& opt) {
             // device type must exactly match
             if (opt.device_type != value.device_type) return false;
             // device id can be wildcarded
             return opt.device_id == details::kAnyDeviceID || opt.device_id == value.device_id;
           }));
  }

  std::span<const DLDevice> m_options;
  DLDevice m_value;
};

namespace details {

template <typename T>
struct BaseRef {
 public:
  BaseRef(const BaseRef&) = delete;
  BaseRef& operator=(const BaseRef&) = delete;

  auto operator->() const -> T* {
    return m_ref;
  }
  auto operator*() const -> T& {
    return *m_ref;
  }
  auto rebind(T& other) -> void {
    m_ref = &other;
  }

  explicit BaseRef() : m_ref(&m_cache), m_cache() {}
  BaseRef(T& size) : m_ref(&size), m_cache() {}

 private:
  T* m_ref;
  T m_cache;
};

struct SizeRef : BaseRef<SymbolicSize> {
  using BaseRef::BaseRef;
  SizeRef(int64_t value) {
    if (value != kAnySize) {
      (**this).set_value(value);
    } else {
      // otherwise, we can match any size
    }
  }
};

struct DTypeRef : BaseRef<SymbolicDType> {
  using BaseRef::BaseRef;
  DTypeRef(DLDataType options) {
    (**this).set_value(options);
  }
  DTypeRef(std::initializer_list<DLDataType> options) {
    (**this).set_options(options);
  }
  DTypeRef(std::span<const DLDataType> options) {
    (**this).set_options(options);
  }
};

struct DeviceRef : BaseRef<SymbolicDevice> {
  using BaseRef::BaseRef;
  DeviceRef(DLDevice options) {
    (**this).set_value(options);
  }
  DeviceRef(std::initializer_list<DLDevice> options) {
    (**this).set_options(options);
  }
  DeviceRef(std::span<const DLDevice> options) {
    (**this).set_options(options);
  }
};

}  // namespace details

struct TensorMatcher {
 private:
  using SizeRef = details::SizeRef;
  using DTypeRef = details::DTypeRef;
  using DeviceRef = details::DeviceRef;

 public:
  TensorMatcher(const TensorMatcher&) = delete;
  TensorMatcher& operator=(const TensorMatcher&) = delete;

  explicit TensorMatcher(std::initializer_list<SizeRef> shape) : m_shape(shape), m_strides(), m_dtype() {}

  auto with_strides(std::initializer_list<SizeRef> strides) && -> TensorMatcher&& {
    // no partial update allowed
    RuntimeCheck(m_strides.size() == 0, "Strides already specified");
    RuntimeCheck(m_shape.size() == strides.size(), "Strides size must match shape size");
    m_strides = strides;
    return std::move(*this);
  }

  template <typename... Ts>
  auto with_dtype(DTypeRef&& dtype) && -> TensorMatcher&& {
    m_init_dtype();
    m_dtype.rebind(*dtype);
    m_dtype->set_options<Ts...>();
    return std::move(*this);
  }

  template <typename... Ts>
  auto with_dtype() && -> TensorMatcher&& {
    static_assert(sizeof...(Ts) > 0, "At least one dtype option must be specified");
    m_init_dtype();
    m_dtype->set_options<Ts...>();
    return std::move(*this);
  }

  template <DLDeviceType... Codes>
  auto with_device(DeviceRef&& device) && -> TensorMatcher&& {
    m_init_device();
    m_device.rebind(*device);
    m_device->set_options<Codes...>();
    return std::move(*this);
  }

  template <DLDeviceType... Codes>
  auto with_device() && -> TensorMatcher&& {
    static_assert(sizeof...(Codes) > 0, "At least one device option must be specified");
    m_init_device();
    m_device->set_options<Codes...>();
    return std::move(*this);
  }

  // once we start verification, we cannot modify anymore
  auto verify(tvm::ffi::TensorView view, DebugInfo info = {}) const&& -> const TensorMatcher&& {
    try {
      m_verify_impl(view);
    } catch (PanicError& e) {
      auto oss = std::ostringstream{};
      oss << "Tensor match failed for ";
      s_print_tensor(oss, view);
      oss << " at " << info.file_name() << ":" << info.line() << "\n- Root cause: " << e.root_cause();
      throw PanicError(std::move(oss).str());
    }
    return std::move(*this);
  }

 private:
  static auto s_print_tensor(std::ostringstream& oss, tvm::ffi::TensorView view) -> void {
    oss << "Tensor<";
    int64_t dim = 0;
    for (const auto& size : view.shape()) {
      if (dim++ > 0) oss << ", ";
      oss << size;
    }
    oss << ">[strides=<";
    dim = 0;
    for (const auto& stride : view.strides()) {
      if (dim++ > 0) {
        oss << ", ";
      }
      oss << stride;
    }
    oss << ">, dtype=" << view.dtype();
    oss << ", device=" << details::PrintableDevice{view.device()} << "]";
  }

  auto m_verify_impl(tvm::ffi::TensorView view) const -> void {
    const auto dim = static_cast<std::size_t>(view.dim());
    RuntimeCheck(dim == m_shape.size(), "Tensor dimension mismatch: expected ", m_shape.size(), " but got ", dim);
    for (const auto i : irange(dim)) {
      m_shape[i]->verify(view.size(i), "shape", i);
    }
    if (m_has_strides()) {
      for (const auto i : irange(dim)) {
        if (view.size(i) != 1 || !m_strides[i]->has_value()) {
          // skip stride check for size 1 dimension
          m_strides[i]->verify(view.stride(i), "stride", i);
        }
      }
    } else {
      RuntimeCheck(view.is_contiguous(), "Tensor is not contiguous as expected");
    }
    // since we may double verify, we will force to check
    m_dtype->verify(view.dtype());
    m_device->verify(view.device());
  }

  auto m_init_dtype() -> void {
    RuntimeCheck(!m_has_dtype, "DType already specified");
    m_has_dtype = true;
  }

  auto m_init_device() -> void {
    RuntimeCheck(!m_has_device, "Device already specified");
    m_has_device = true;
  }

  auto m_has_strides() const -> bool {
    return !m_strides.empty();
  }

  std::span<const SizeRef> m_shape;
  std::span<const SizeRef> m_strides;
  DTypeRef m_dtype;
  DeviceRef m_device;
  bool m_has_dtype = false;
  bool m_has_device = false;
};

}  // namespace host
