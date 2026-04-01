#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

inline uint64_t twos_to_uint(int64_t x, int bits) {
  const uint64_t mask = (bits == 64) ? ~uint64_t(0) : ((uint64_t(1) << bits) - 1);
  return static_cast<uint64_t>(x) & mask;
}

inline int32_t uint_to_twos(uint64_t u, int bits) {
  const uint64_t sign_bit = uint64_t(1) << (bits - 1);
  const uint64_t mask = (uint64_t(1) << bits) - 1;
  u &= mask;
  return (u & sign_bit) ? static_cast<int32_t>(static_cast<int64_t>(u) - (int64_t(1) << bits))
                        : static_cast<int32_t>(u);
}

torch::Tensor ensure_cpu_contiguous(torch::Tensor t, torch::ScalarType dtype) {
  if (t.scalar_type() != dtype) {
    t = t.to(dtype);
  }
  if (!t.device().is_cpu()) {
    t = t.cpu();
  }
  if (!t.is_contiguous()) {
    t = t.contiguous();
  }
  return t;
}

py::bytes pack_tensor_ints_twos_complement_cpp(torch::Tensor values, torch::Tensor bits) {
  values = ensure_cpu_contiguous(values.flatten(), torch::kInt32);
  bits = ensure_cpu_contiguous(bits.flatten(), torch::kInt16);
  if (values.numel() != bits.numel()) {
    throw std::runtime_error("values and bits must have same numel");
  }

  auto* v_ptr = values.data_ptr<int32_t>();
  auto* b_ptr = bits.data_ptr<int16_t>();
  const int64_t n = values.numel();

  int64_t total_bits = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (b_ptr[i] > 0) {
      total_bits += b_ptr[i];
    }
  }
  std::string out;
  out.resize((total_bits + 7) / 8, '\0');

  uint64_t bitbuf = 0;
  int bitcount = 0;
  size_t out_idx = 0;
  for (int64_t i = 0; i < n; ++i) {
    const int b = static_cast<int>(b_ptr[i]);
    if (b <= 0) {
      continue;
    }
    const uint64_t u = twos_to_uint(v_ptr[i], b);
    bitbuf |= (u << bitcount);
    bitcount += b;
    while (bitcount >= 8) {
      out[out_idx++] = static_cast<char>(bitbuf & 0xFFu);
      bitbuf >>= 8;
      bitcount -= 8;
    }
  }
  if (bitcount > 0 && out_idx < out.size()) {
    out[out_idx] = static_cast<char>(bitbuf & 0xFFu);
  }
  return py::bytes(out);
}

torch::Tensor unpack_tensor_ints_twos_complement_cpp(py::bytes blob, torch::Tensor bits) {
  bits = ensure_cpu_contiguous(bits.flatten(), torch::kInt16);
  auto* b_ptr = bits.data_ptr<int16_t>();
  const int64_t n = bits.numel();

  std::string in = blob;
  const auto* data = reinterpret_cast<const uint8_t*>(in.data());
  const size_t data_size = in.size();

  auto out = torch::empty({n}, torch::dtype(torch::kInt32).device(torch::kCPU));
  auto* out_ptr = out.data_ptr<int32_t>();

  uint64_t bitbuf = 0;
  int bitcount = 0;
  size_t idx = 0;
  for (int64_t i = 0; i < n; ++i) {
    const int b = static_cast<int>(b_ptr[i]);
    if (b <= 0) {
      out_ptr[i] = 0;
      continue;
    }
    while (bitcount < b && idx < data_size) {
      bitbuf |= (uint64_t(data[idx]) << bitcount);
      bitcount += 8;
      ++idx;
    }
    if (bitcount < b) {
      throw std::runtime_error("Not enough bits in blob for requested unpack");
    }
    const uint64_t mask = (uint64_t(1) << b) - 1;
    const uint64_t u = bitbuf & mask;
    bitbuf >>= b;
    bitcount -= b;
    out_ptr[i] = uint_to_twos(u, b);
  }
  return out;
}

py::bytes pack_bits_cpp(torch::Tensor mask) {
  mask = ensure_cpu_contiguous(mask.flatten(), torch::kBool);
  auto* ptr = mask.data_ptr<bool>();
  const int64_t n = mask.numel();
  std::string out;
  out.resize((n + 7) / 8, '\0');
  for (int64_t i = 0; i < n; ++i) {
    if (ptr[i]) {
      out[i / 8] = static_cast<char>(static_cast<uint8_t>(out[i / 8]) | (1u << (i % 8)));
    }
  }
  return py::bytes(out);
}

torch::Tensor unpack_bits_cpp(py::bytes blob, int64_t nbits) {
  std::string in = blob;
  const auto* data = reinterpret_cast<const uint8_t*>(in.data());
  const size_t data_size = in.size();
  auto out = torch::zeros({nbits}, torch::dtype(torch::kBool).device(torch::kCPU));
  auto* ptr = out.data_ptr<bool>();
  for (int64_t i = 0; i < nbits; ++i) {
    const size_t byte_idx = static_cast<size_t>(i / 8);
    if (byte_idx >= data_size) {
      break;
    }
    ptr[i] = ((data[byte_idx] >> (i % 8)) & 1u) != 0;
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(bitpack_cpp, m) {
  m.def("pack_tensor_ints_twos_complement_cpp", &pack_tensor_ints_twos_complement_cpp);
  m.def("unpack_tensor_ints_twos_complement_cpp", &unpack_tensor_ints_twos_complement_cpp);
  m.def("pack_bits_cpp", &pack_bits_cpp);
  m.def("unpack_bits_cpp", &unpack_bits_cpp);
}

