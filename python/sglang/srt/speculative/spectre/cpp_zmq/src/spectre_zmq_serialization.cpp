#include "spectre_zmq_serialization.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/stl.h>

namespace {

struct StringStream {
  explicit StringStream(std::string &s) : str(s) {}

  void write(const char *buf, size_t len) { str.append(buf, len); }

  std::string &str;
};

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
constexpr bool kNativeLittleEndian = false;
#else
constexpr bool kNativeLittleEndian = true;
#endif

constexpr uint32_t kSpectreRequestFieldCount = 15;

template <typename T>
inline void pack_optional_value(msgpack::packer<StringStream> &pk,
                                const std::optional<T> &value) {
  if (value) {
    pk.pack(*value);
  } else {
    pk.pack_nil();
  }
}

inline uint32_t spectre_load_le_u32(const char *src) {
  return static_cast<uint32_t>(static_cast<unsigned char>(src[0])) |
         (static_cast<uint32_t>(static_cast<unsigned char>(src[1])) << 8) |
         (static_cast<uint32_t>(static_cast<unsigned char>(src[2])) << 16) |
         (static_cast<uint32_t>(static_cast<unsigned char>(src[3])) << 24);
}

void pack_optional_int_vector_bin(
    msgpack::packer<StringStream> &pk,
    const std::optional<std::vector<int>> &values) {
  if (!values) {
    pk.pack_nil();
    return;
  }

  const auto &vec = *values;
  const uint32_t byte_size =
      static_cast<uint32_t>(vec.size() * sizeof(int32_t));
  pk.pack_bin(byte_size);

  if (vec.empty()) {
    pk.pack_bin_body("", 0);
    return;
  }

  if constexpr (sizeof(int) == sizeof(int32_t)) {
    if (kNativeLittleEndian) {
      pk.pack_bin_body(reinterpret_cast<const char *>(vec.data()), byte_size);
      return;
    }
  }

  thread_local std::string scratch;
  scratch.resize(byte_size);
  char *dst = scratch.data();
  for (size_t i = 0; i < vec.size(); ++i) {
    int32_t value = static_cast<int32_t>(vec[i]);
    size_t offset = i * sizeof(int32_t);
    dst[offset + 0] = static_cast<char>(value & 0xff);
    dst[offset + 1] = static_cast<char>((value >> 8) & 0xff);
    dst[offset + 2] = static_cast<char>((value >> 16) & 0xff);
    dst[offset + 3] = static_cast<char>((value >> 24) & 0xff);
  }
  pk.pack_bin_body(scratch.data(), byte_size);
}

void pack_optional_float_vector_bin(
    msgpack::packer<StringStream> &pk,
    const std::optional<std::vector<float>> &values) {
  if (!values) {
    pk.pack_nil();
    return;
  }

  const auto &vec = *values;
  const uint32_t byte_size = static_cast<uint32_t>(vec.size() * sizeof(float));
  pk.pack_bin(byte_size);

  if (vec.empty()) {
    pk.pack_bin_body("", 0);
    return;
  }

  if constexpr (sizeof(float) == 4) {
    if (kNativeLittleEndian) {
      pk.pack_bin_body(reinterpret_cast<const char *>(vec.data()), byte_size);
      return;
    }
  }

  thread_local std::string scratch;
  scratch.resize(byte_size);
  char *dst = scratch.data();
  for (size_t i = 0; i < vec.size(); ++i) {
    uint32_t bits = 0;
    std::memcpy(&bits, &vec[i], sizeof(bits));
    size_t offset = i * sizeof(float);
    dst[offset + 0] = static_cast<char>(bits & 0xff);
    dst[offset + 1] = static_cast<char>((bits >> 8) & 0xff);
    dst[offset + 2] = static_cast<char>((bits >> 16) & 0xff);
    dst[offset + 3] = static_cast<char>((bits >> 24) & 0xff);
  }
  pk.pack_bin_body(scratch.data(), byte_size);
}

void pack_spectre_request(msgpack::packer<StringStream> &pk,
                          const spectre::SpectreRequest &req) {
  pk.pack_array(kSpectreRequestFieldCount);
  pack_optional_value(pk, req.request_id);
  pack_optional_value(pk, req.spec_cnt);
  pk.pack(req.action);
  pk.pack(req.spec_type);
  pack_optional_int_vector_bin(pk, req.draft_token_ids);
  pack_optional_int_vector_bin(pk, req.input_ids);
  pack_optional_int_vector_bin(pk, req.output_ids);
  pack_optional_value(pk, req.num_draft_tokens);
  pack_optional_value(pk, req.sampling_params);
  pack_optional_value(pk, req.grammar);
  pk.pack(req.target_send_time);
  pk.pack(req.target_recv_time);
  pack_optional_float_vector_bin(pk, req.draft_logprobs);
  pk.pack(req.draft_recv_time);
  pk.pack(req.draft_send_time);
}

template <typename T>
inline void unpack_optional_value(const msgpack::object &obj,
                                  std::optional<T> &value) {
  if (obj.is_nil()) {
    value.reset();
    return;
  }
  value = obj.as<T>();
}

void unpack_optional_int_vector_bin(const msgpack::object &obj,
                                    std::optional<std::vector<int>> &values) {
  if (obj.is_nil()) {
    values.reset();
    return;
  }
  if (obj.type == msgpack::type::ARRAY) {
    values = obj.as<std::vector<int>>();
    return;
  }
  if (obj.type != msgpack::type::BIN) {
    throw std::runtime_error("invalid int vector payload type");
  }

  const auto &bin = obj.via.bin;
  if (bin.size % sizeof(int32_t) != 0) {
    throw std::runtime_error("invalid int vector payload size");
  }

  auto &vec = values.emplace();
  vec.resize(bin.size / sizeof(int32_t));
  if (vec.empty()) {
    return;
  }

  if constexpr (sizeof(int) == sizeof(int32_t)) {
    if (kNativeLittleEndian) {
      std::memcpy(vec.data(), bin.ptr, bin.size);
      return;
    }
  }

  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = static_cast<int>(static_cast<int32_t>(
        spectre_load_le_u32(bin.ptr + i * sizeof(int32_t))));
  }
}

void unpack_optional_float_vector_bin(
    const msgpack::object &obj, std::optional<std::vector<float>> &values) {
  if (obj.is_nil()) {
    values.reset();
    return;
  }
  if (obj.type == msgpack::type::ARRAY) {
    values = obj.as<std::vector<float>>();
    return;
  }
  if (obj.type != msgpack::type::BIN) {
    throw std::runtime_error("invalid float vector payload type");
  }

  const auto &bin = obj.via.bin;
  if (bin.size % sizeof(float) != 0) {
    throw std::runtime_error("invalid float vector payload size");
  }

  auto &vec = values.emplace();
  vec.resize(bin.size / sizeof(float));
  if (vec.empty()) {
    return;
  }

  if constexpr (sizeof(float) == 4) {
    if (kNativeLittleEndian) {
      std::memcpy(vec.data(), bin.ptr, bin.size);
      return;
    }
  }

  for (size_t i = 0; i < vec.size(); ++i) {
    uint32_t bits = spectre_load_le_u32(bin.ptr + i * sizeof(float));
    std::memcpy(&vec[i], &bits, sizeof(bits));
  }
}

void unpack_spectre_request(const msgpack::object &obj,
                            spectre::SpectreRequest &req) {
  if (obj.type != msgpack::type::ARRAY ||
      obj.via.array.size < kSpectreRequestFieldCount) {
    throw std::runtime_error("invalid Spectre request payload");
  }

  const msgpack::object *fields = obj.via.array.ptr;
  unpack_optional_value(fields[0], req.request_id);
  unpack_optional_value(fields[1], req.spec_cnt);
  req.action = fields[2].as<spectre::SpectreAction>();
  req.spec_type = fields[3].as<spectre::SpecType>();
  unpack_optional_int_vector_bin(fields[4], req.draft_token_ids);
  unpack_optional_int_vector_bin(fields[5], req.input_ids);
  unpack_optional_int_vector_bin(fields[6], req.output_ids);
  unpack_optional_value(fields[7], req.num_draft_tokens);
  unpack_optional_value(fields[8], req.sampling_params);
  unpack_optional_value(fields[9], req.grammar);
  req.target_send_time = fields[10].as<double>();
  req.target_recv_time = fields[11].as<double>();
  unpack_optional_float_vector_bin(fields[12], req.draft_logprobs);
  req.draft_recv_time = fields[13].as<double>();
  req.draft_send_time = fields[14].as<double>();
}

} // namespace

std::string
pack_spectre_batch_payload(const std::vector<spectre::SpectreRequest> &objs) {
  thread_local size_t cached_capacity = 4096;
  std::string raw_data;
  raw_data.reserve(cached_capacity);
  StringStream ss(raw_data);
  msgpack::packer<StringStream> pk(ss);
  pk.pack_array(objs.size());
  for (const auto &obj : objs) {
    pack_spectre_request(pk, obj);
  }
  if (raw_data.capacity() > cached_capacity) {
    cached_capacity = raw_data.capacity();
  }
  return raw_data;
}

bool unpack_spectre_batch_payload(msgpack::unpacker &unpacker, const void *data,
                                  size_t len,
                                  std::vector<spectre::SpectreRequest> &objs) {
  try {
    objs.clear();
    unpacker.reserve_buffer(len);
    std::memcpy(unpacker.buffer(), data, len);
    unpacker.buffer_consumed(len);

    msgpack::object_handle oh;
    if (!unpacker.next(oh)) {
      unpacker.remove_nonparsed_buffer();
      unpacker.reset();
      unpacker.reset_zone();
      return false;
    }

    const auto &root = oh.get();
    if (root.type != msgpack::type::ARRAY) {
      throw std::runtime_error("invalid Spectre batch payload");
    }

    objs.reserve(root.via.array.size);
    for (uint32_t i = 0; i < root.via.array.size; ++i) {
      objs.emplace_back();
      unpack_spectre_request(root.via.array.ptr[i], objs.back());
    }
    unpacker.remove_nonparsed_buffer();
    unpacker.reset_zone();
    return true;
  } catch (...) {
    unpacker.remove_nonparsed_buffer();
    unpacker.reset();
    unpacker.reset_zone();
    throw;
  }
}

std::vector<spectre::SpectreRequest> from_py_list(const pybind11::list &objs) {
  std::vector<spectre::SpectreRequest> out;
  out.reserve(static_cast<size_t>(pybind11::len(objs)));
  for (auto item : objs) {
    out.push_back(spectre::from_py_dict(item.cast<pybind11::dict>()));
  }
  return out;
}
