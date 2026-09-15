#include <Python.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <openssl/sha.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace py = pybind11;

namespace {

constexpr std::size_t kDigestLen = SHA256_DIGEST_LENGTH;
constexpr std::size_t kHexLen = SHA256_DIGEST_LENGTH * 2;

inline std::uint32_t checked_u32(std::uint64_t value) {
  if (value > UINT32_MAX) {
    throw std::out_of_range("token id does not fit in uint32");
  }
  return static_cast<std::uint32_t>(value);
}

inline void digest_to_hex_chars(const unsigned char *digest, char *out) {
  static constexpr char kHex[] = "0123456789abcdef";
  for (std::size_t i = 0; i < kDigestLen; ++i) {
    const unsigned char byte = digest[i];
    out[i * 2] = kHex[byte >> 4];
    out[i * 2 + 1] = kHex[byte & 0x0f];
  }
}

std::string digest_to_hex_string(const unsigned char *digest) {
  std::string out(kHexLen, '\0');
  digest_to_hex_chars(digest, out.data());
  return out;
}

std::array<unsigned char, kDigestLen>
parse_prior_digest(py::object prior_digest_obj, bool *has_prior_digest) {
  std::array<unsigned char, kDigestLen> prior_digest{};
  *has_prior_digest = false;
  if (!prior_digest_obj.is_none()) {
    std::string prior = prior_digest_obj.cast<std::string>();
    if (prior.size() != kDigestLen) {
      throw std::invalid_argument("prior_digest must be exactly 32 bytes");
    }
    std::copy(prior.begin(), prior.end(), prior_digest.begin());
    *has_prior_digest = true;
  }
  return prior_digest;
}

inline void hash_page(const unsigned char *data, std::size_t len,
                      bool &has_prior_digest,
                      std::array<unsigned char, kDigestLen> &prior_digest) {
  SHA256_CTX ctx;
  SHA256_Init(&ctx);
  if (has_prior_digest) {
    SHA256_Update(&ctx, prior_digest.data(), prior_digest.size());
  }
  if (len > 0) {
    SHA256_Update(&ctx, data, len);
  }
  SHA256_Final(prior_digest.data(), &ctx);
  has_prior_digest = true;
}

template <typename RawToken>
inline void fill_regular_page(const RawToken *raw, std::size_t start,
                              std::size_t count, std::uint32_t *out) {
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = checked_u32(raw[start + i]);
  }
}

template <>
inline void
fill_regular_page<std::uint32_t>(const std::uint32_t *raw, std::size_t start,
                                 std::size_t count, std::uint32_t *out) {
  std::copy(raw + start, raw + start + count, out);
}

template <>
inline void
fill_regular_page<std::uint64_t>(const std::uint64_t *raw, std::size_t start,
                                 std::size_t count, std::uint32_t *out) {
#if defined(__AVX2__)
  std::size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    const __m256i v =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(raw + start + i));
    const __m256i high = _mm256_srli_epi64(v, 32);
    if (!_mm256_testz_si256(high, high)) {
      throw std::out_of_range("token id does not fit in uint32");
    }
    const __m256i low_pairs = _mm256_shuffle_epi32(v, _MM_SHUFFLE(2, 0, 2, 0));
    const __m128i lane0 = _mm256_castsi256_si128(low_pairs);
    const __m128i lane1 = _mm256_extracti128_si256(low_pairs, 1);
    _mm_storel_epi64(reinterpret_cast<__m128i *>(out + i), lane0);
    _mm_storel_epi64(reinterpret_cast<__m128i *>(out + i + 2), lane1);
  }
  for (; i < count; ++i) {
    out[i] = checked_u32(raw[start + i]);
  }
#else
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = checked_u32(raw[start + i]);
  }
#endif
}

template <typename RawToken>
inline void fill_bigram_page(const RawToken *raw, std::size_t start,
                             std::size_t count, std::uint32_t *out) {
  std::uint32_t prev = checked_u32(raw[start]);
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint32_t next = checked_u32(raw[start + i + 1]);
    out[i * 2] = prev;
    out[i * 2 + 1] = next;
    prev = next;
  }
}

template <>
inline void
fill_bigram_page<std::uint32_t>(const std::uint32_t *raw, std::size_t start,
                                std::size_t count, std::uint32_t *out) {
  std::uint32_t prev = raw[start];
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint32_t next = raw[start + i + 1];
    out[i * 2] = prev;
    out[i * 2 + 1] = next;
    prev = next;
  }
}

template <>
inline void
fill_bigram_page<std::uint64_t>(const std::uint64_t *raw, std::size_t start,
                                std::size_t count, std::uint32_t *out) {
  std::uint32_t prev = checked_u32(raw[start]);
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint32_t next = checked_u32(raw[start + i + 1]);
    out[i * 2] = prev;
    out[i * 2 + 1] = next;
    prev = next;
  }
}

struct RawTokenBuffer {
  py::buffer_info info;
  std::size_t logical_len;
  bool is_bigram;
};

RawTokenBuffer get_raw_token_buffer(const py::buffer &raw_tokens,
                                    std::size_t logical_len,
                                    std::size_t unit_width, bool is_bigram) {
  py::buffer_info info = raw_tokens.request();
  if (info.ndim != 1) {
    throw std::invalid_argument("raw_tokens must be a one-dimensional buffer");
  }
  if (info.itemsize != 4 && info.itemsize != 8) {
    throw std::invalid_argument("raw_tokens itemsize must be 4 or 8 bytes");
  }
  const std::size_t need_raw_tokens =
      is_bigram && logical_len > 0 ? logical_len + 1 : logical_len * unit_width;
  if (static_cast<std::size_t>(info.size) < need_raw_tokens) {
    throw std::invalid_argument("raw_tokens is shorter than logical_len");
  }
  return RawTokenBuffer{std::move(info), logical_len, is_bigram};
}

template <typename RawToken>
void hash_pages_to_hex_blob(const RawToken *raw, std::size_t logical_len,
                            std::size_t page_size, std::size_t unit_width,
                            bool is_bigram, bool has_prior_digest,
                            std::array<unsigned char, kDigestLen> prior_digest,
                            std::string &hex_blob) {
  if (page_size == 0) {
    throw std::invalid_argument("page_size must be positive");
  }

  if (is_bigram) {
    unit_width = 2;
  }
  const bool can_hash_raw_bytes =
      std::is_same_v<RawToken, std::uint32_t> && !is_bigram;
  std::vector<std::uint32_t> page_words;
  if (!can_hash_raw_bytes) {
    page_words.resize(page_size * unit_width);
  }

  for (std::size_t start = 0, page_idx = 0; start < logical_len;
       start += page_size, ++page_idx) {
    const std::size_t page_units = std::min(page_size, logical_len - start);
    const std::size_t page_bytes =
        page_units * unit_width * sizeof(std::uint32_t);
    const unsigned char *bytes = nullptr;

    if (can_hash_raw_bytes) {
      bytes = reinterpret_cast<const unsigned char *>(raw + start * unit_width);
    } else {
      if (is_bigram) {
        fill_bigram_page(raw, start, page_units, page_words.data());
      } else {
        fill_regular_page(raw, start * unit_width, page_units * unit_width,
                          page_words.data());
      }
      bytes = reinterpret_cast<const unsigned char *>(page_words.data());
    }

    hash_page(bytes, page_bytes, has_prior_digest, prior_digest);
    digest_to_hex_chars(prior_digest.data(),
                        hex_blob.data() + page_idx * kHexLen);
  }
}

template <typename RawToken>
std::string hash_all(const RawToken *raw, std::size_t logical_len,
                     std::size_t unit_width, bool is_bigram,
                     bool has_prior_digest,
                     std::array<unsigned char, kDigestLen> prior_digest) {
  if (is_bigram) {
    unit_width = 2;
  }
  const bool can_hash_raw_bytes =
      std::is_same_v<RawToken, std::uint32_t> && !is_bigram;
  const unsigned char *bytes = nullptr;
  std::size_t num_bytes = logical_len * unit_width * sizeof(std::uint32_t);

  std::vector<std::uint32_t> words;
  if (can_hash_raw_bytes) {
    bytes = reinterpret_cast<const unsigned char *>(raw);
  } else {
    words.resize(logical_len * unit_width);
    if (logical_len > 0) {
      if (is_bigram) {
        fill_bigram_page(raw, 0, logical_len, words.data());
      } else {
        fill_regular_page(raw, 0, logical_len * unit_width, words.data());
      }
    }
    bytes = reinterpret_cast<const unsigned char *>(words.data());
  }

  hash_page(bytes, num_bytes, has_prior_digest, prior_digest);
  return digest_to_hex_string(prior_digest.data());
}

py::object hex_blob_to_pylist(const std::string &hex_blob) {
  const std::size_t num_pages = hex_blob.size() / kHexLen;
  if (num_pages >
      static_cast<std::size_t>(std::numeric_limits<Py_ssize_t>::max())) {
    throw std::overflow_error("too many hash pages");
  }

  PyObject *raw_list = PyList_New(static_cast<Py_ssize_t>(num_pages));
  if (raw_list == nullptr) {
    throw py::error_already_set();
  }
  py::object list = py::reinterpret_steal<py::object>(raw_list);

  const char *data = hex_blob.data();
  for (std::size_t i = 0; i < num_pages; ++i) {
    PyObject *item = PyUnicode_FromStringAndSize(
        data + i * kHexLen, static_cast<Py_ssize_t>(kHexLen));
    if (item == nullptr) {
      throw py::error_already_set();
    }
    PyList_SET_ITEM(raw_list, static_cast<Py_ssize_t>(i), item);
  }
  return list;
}

std::string hash_str(const py::buffer &raw_tokens, std::size_t logical_len,
                     std::size_t unit_width, bool is_bigram,
                     py::object prior_digest_obj) {
  RawTokenBuffer buffer =
      get_raw_token_buffer(raw_tokens, logical_len, unit_width, is_bigram);
  bool has_prior_digest = false;
  auto prior_digest = parse_prior_digest(prior_digest_obj, &has_prior_digest);

  py::gil_scoped_release release;
  if (buffer.info.itemsize == 4) {
    return hash_all(static_cast<const std::uint32_t *>(buffer.info.ptr),
                    logical_len, unit_width, is_bigram, has_prior_digest,
                    prior_digest);
  }
  return hash_all(static_cast<const std::uint64_t *>(buffer.info.ptr),
                  logical_len, unit_width, is_bigram, has_prior_digest,
                  prior_digest);
}

py::object pages_hashes(const py::buffer &raw_tokens, std::size_t logical_len,
                        std::size_t page_size, std::size_t unit_width,
                        bool is_bigram, py::object prior_digest_obj) {
  RawTokenBuffer buffer =
      get_raw_token_buffer(raw_tokens, logical_len, unit_width, is_bigram);
  const std::size_t num_pages =
      page_size == 0 ? 0 : (logical_len + page_size - 1) / page_size;
  bool has_prior_digest = false;
  auto prior_digest = parse_prior_digest(prior_digest_obj, &has_prior_digest);
  std::string hex_blob(num_pages * kHexLen, '\0');

  {
    py::gil_scoped_release release;
    if (buffer.info.itemsize == 4) {
      hash_pages_to_hex_blob(
          static_cast<const std::uint32_t *>(buffer.info.ptr), logical_len,
          page_size, unit_width, is_bigram, has_prior_digest, prior_digest,
          hex_blob);
    } else {
      hash_pages_to_hex_blob(
          static_cast<const std::uint64_t *>(buffer.info.ptr), logical_len,
          page_size, unit_width, is_bigram, has_prior_digest, prior_digest,
          hex_blob);
    }
  }

  return hex_blob_to_pylist(hex_blob);
}

py::object get_hash(const py::buffer &raw_tokens, std::size_t logical_len,
                    std::size_t unit_width, bool is_bigram,
                    py::object prior_digest_obj, py::object page_size_obj) {
  if (page_size_obj.is_none()) {
    return py::cast(hash_str(raw_tokens, logical_len, unit_width, is_bigram,
                             prior_digest_obj));
  }
  return pages_hashes(raw_tokens, logical_len,
                      page_size_obj.cast<std::size_t>(), unit_width, is_bigram,
                      prior_digest_obj);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_hash", &get_hash, py::arg("raw_tokens"), py::arg("logical_len"),
        py::arg("unit_width"), py::arg("is_bigram"), py::arg("prior_digest"),
        py::arg("page_size") = py::none());
}
