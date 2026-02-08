// Formatting library for C++ - scanning API proof of concept
//
// Copyright (c) 2019 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#include <array>
#include <cassert>
#include <climits>
#include <tuple>

#include "fmt/format-inl.h"

FMT_BEGIN_NAMESPACE
namespace detail {

inline auto is_whitespace(char c) -> bool { return c == ' ' || c == '\n'; }

// If c is a hex digit returns its numeric value, otherwise -1.
inline auto to_hex_digit(char c) -> int {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

struct maybe_contiguous_range {
  const char* begin;
  const char* end;

  explicit operator bool() const { return begin != nullptr; }
};

class scan_buffer {
 private:
  const char* ptr_;
  const char* end_;
  bool contiguous_;

 protected:
  scan_buffer(const char* ptr, const char* end, bool contiguous)
      : ptr_(ptr), end_(end), contiguous_(contiguous) {}
  ~scan_buffer() = default;

  void set(span<const char> buf) {
    ptr_ = buf.data;
    end_ = buf.data + buf.size;
  }

  auto ptr() const -> const char* { return ptr_; }

 public:
  scan_buffer(const scan_buffer&) = delete;
  void operator=(const scan_buffer&) = delete;

  // Fills the buffer with more input if available.
  virtual void consume() = 0;

  class sentinel {};

  class iterator {
   private:
    const char** ptr_;
    scan_buffer* buf_;  // This could be merged with ptr_.
    char value_;

    static auto get_sentinel() -> const char** {
      static const char* ptr = nullptr;
      return &ptr;
    }

    friend class scan_buffer;

    friend auto operator==(iterator lhs, sentinel) -> bool {
      return *lhs.ptr_ == nullptr;
    }
    friend auto operator!=(iterator lhs, sentinel) -> bool {
      return *lhs.ptr_ != nullptr;
    }

    iterator(scan_buffer* buf) : buf_(buf) {
      if (buf->ptr_ == buf->end_) {
        ptr_ = get_sentinel();
        return;
      }
      ptr_ = &buf->ptr_;
      value_ = *buf->ptr_;
    }

    friend scan_buffer& get_buffer(iterator it) { return *it.buf_; }

   public:
    iterator() : ptr_(get_sentinel()), buf_(nullptr) {}

    auto operator++() -> iterator& {
      if (!buf_->try_consume()) ptr_ = get_sentinel();
      value_ = *buf_->ptr_;
      return *this;
    }
    auto operator++(int) -> iterator {
      iterator copy = *this;
      ++*this;
      return copy;
    }
    auto operator*() const -> char { return value_; }

    auto base() const -> const char* { return buf_->ptr_; }

    friend auto to_contiguous(iterator it) -> maybe_contiguous_range;
    friend auto advance(iterator it, size_t n) -> iterator;
  };

  friend auto to_contiguous(iterator it) -> maybe_contiguous_range {
    if (it.buf_->is_contiguous()) return {it.buf_->ptr_, it.buf_->end_};
    return {nullptr, nullptr};
  }
  friend auto advance(iterator it, size_t n) -> iterator {
    FMT_ASSERT(it.buf_->is_contiguous(), "");
    const char*& ptr = it.buf_->ptr_;
    ptr += n;
    it.value_ = *ptr;
    if (ptr == it.buf_->end_) it.ptr_ = iterator::get_sentinel();
    return it;
  }

  auto begin() -> iterator { return this; }
  auto end() -> sentinel { return {}; }

  auto is_contiguous() const -> bool { return contiguous_; }

  // Tries consuming a single code unit. Returns true iff there is more input.
  auto try_consume() -> bool {
    FMT_ASSERT(ptr_ != end_, "");
    ++ptr_;
    if (ptr_ != end_) return true;
    consume();
    return ptr_ != end_;
  }
};

using scan_iterator = scan_buffer::iterator;
using scan_sentinel = scan_buffer::sentinel;

class string_scan_buffer final : public scan_buffer {
 private:
  void consume() override {}

 public:
  explicit string_scan_buffer(string_view s)
      : scan_buffer(s.begin(), s.end(), true) {}
};

class file_scan_buffer final : public scan_buffer {
 private:
  template <typename F, FMT_ENABLE_IF(sizeof(F::_IO_read_ptr) != 0 &&
                                      !FMT_USE_FALLBACK_FILE)>
  static auto get_file(F* f, int) -> glibc_file<F> {
    return f;
  }
  template <typename F,
            FMT_ENABLE_IF(sizeof(F::_p) != 0 && !FMT_USE_FALLBACK_FILE)>
  static auto get_file(F* f, int) -> apple_file<F> {
    return f;
  }
  static auto get_file(FILE* f, ...) -> fallback_file<FILE> { return f; }

  decltype(get_file(static_cast<FILE*>(nullptr), 0)) file_;

  // Fills the buffer if it is empty.
  void fill() {
    span<const char> buf = file_.get_read_buffer();
    if (buf.size == 0) {
      int c = file_.get();
      // Put the character back since we are only filling the buffer.
      if (c != EOF) file_.unget(static_cast<char>(c));
      buf = file_.get_read_buffer();
    }
    set(buf);
  }

  void consume() override {
    // Consume the current buffer content.
    size_t n = to_unsigned(ptr() - file_.get_read_buffer().data);
    for (size_t i = 0; i != n; ++i) file_.get();
    fill();
  }

 public:
  explicit file_scan_buffer(FILE* f)
      : scan_buffer(nullptr, nullptr, false), file_(f) {
    flockfile(f);
    fill();
  }
  ~file_scan_buffer() {
    FILE* f = file_;
    funlockfile(f);
  }
};
}  // namespace detail

template <typename T, typename Char = char> struct scanner {
  // A deleted default constructor indicates a disabled scanner.
  scanner() = delete;
};

class scan_parse_context {
 private:
  string_view format_;

 public:
  using iterator = string_view::iterator;

  FMT_CONSTEXPR explicit scan_parse_context(string_view format)
      : format_(format) {}

  FMT_CONSTEXPR auto begin() const -> iterator { return format_.begin(); }
  FMT_CONSTEXPR auto end() const -> iterator { return format_.end(); }

  void advance_to(iterator it) {
    format_.remove_prefix(detail::to_unsigned(it - begin()));
  }
};

namespace detail {
enum class scan_type {
  none_type,
  int_type,
  uint_type,
  long_long_type,
  ulong_long_type,
  double_type,
  float_type,
  string_type,
  string_view_type,
  custom_type
};

template <typename Context> struct custom_scan_arg {
  void* value;
  void (*scan)(void* arg, scan_parse_context& parse_ctx, Context& ctx);
};
}  // namespace detail

// A scan argument. Context is a template parameter for the compiled API where
// output can be unbuffered.
template <typename Context> class basic_scan_arg {
 private:
  using scan_type = detail::scan_type;
  scan_type type_;
  union {
    int* int_value_;
    unsigned* uint_value_;
    long long* long_long_value_;
    unsigned long long* ulong_long_value_;
    double* double_value_;
    float* float_value_;
    std::string* string_;
    string_view* string_view_;
    detail::custom_scan_arg<Context> custom_;
    // TODO: more types
  };

  template <typename T>
  static void scan_custom_arg(void* arg, scan_parse_context& parse_ctx,
                              Context& ctx) {
    auto s = scanner<T>();
    parse_ctx.advance_to(s.parse(parse_ctx));
    ctx.advance_to(s.scan(*static_cast<T*>(arg), ctx));
  }

 public:
  FMT_CONSTEXPR basic_scan_arg()
      : type_(scan_type::none_type), int_value_(nullptr) {}
  FMT_CONSTEXPR basic_scan_arg(int& value)
      : type_(scan_type::int_type), int_value_(&value) {}
  FMT_CONSTEXPR basic_scan_arg(unsigned& value)
      : type_(scan_type::uint_type), uint_value_(&value) {}
  FMT_CONSTEXPR basic_scan_arg(long long& value)
      : type_(scan_type::long_long_type), long_long_value_(&value) {}
  FMT_CONSTEXPR basic_scan_arg(unsigned long long& value)
      : type_(scan_type::ulong_long_type), ulong_long_value_(&value) {}
  FMT_CONSTEXPR basic_scan_arg(double& value)
      : type_(scan_type::double_type), double_value_(&value) {}
  FMT_CONSTEXPR basic_scan_arg(float& value)
      : type_(scan_type::float_type), float_value_(&value) {}
  FMT_CONSTEXPR basic_scan_arg(std::string& value)
      : type_(scan_type::string_type), string_(&value) {}
  FMT_CONSTEXPR basic_scan_arg(string_view& value)
      : type_(scan_type::string_view_type), string_view_(&value) {}
  template <typename T>
  FMT_CONSTEXPR basic_scan_arg(T& value) : type_(scan_type::custom_type) {
    custom_.value = &value;
    custom_.scan = scan_custom_arg<T>;
  }

  constexpr explicit operator bool() const noexcept {
    return type_ != scan_type::none_type;
  }

  auto type() const -> detail::scan_type { return type_; }

  template <typename Visitor>
  auto visit(Visitor&& vis) -> decltype(vis(monostate())) {
    switch (type_) {
    case scan_type::none_type:
      break;
    case scan_type::int_type:
      return vis(*int_value_);
    case scan_type::uint_type:
      return vis(*uint_value_);
    case scan_type::long_long_type:
      return vis(*long_long_value_);
    case scan_type::ulong_long_type:
      return vis(*ulong_long_value_);
    case scan_type::double_type:
      return vis(*double_value_);
    case scan_type::float_type:
      return vis(*float_value_);
    case scan_type::string_type:
      return vis(*string_);
    case scan_type::string_view_type:
      return vis(*string_view_);
    case scan_type::custom_type:
      break;
    }
    return vis(monostate());
  }

  auto scan_custom(const char* parse_begin, scan_parse_context& parse_ctx,
                   Context& ctx) const -> bool {
    if (type_ != scan_type::custom_type) return false;
    parse_ctx.advance_to(parse_begin);
    custom_.scan(custom_.value, parse_ctx, ctx);
    return true;
  }
};

class scan_context;
using scan_arg = basic_scan_arg<scan_context>;

struct scan_args {
  int size;
  const scan_arg* data;

  template <size_t N>
  FMT_CONSTEXPR scan_args(const std::array<scan_arg, N>& store)
      : size(N), data(store.data()) {
    static_assert(N < INT_MAX, "too many arguments");
  }
};

class scan_context {
 private:
  detail::scan_buffer& buf_;
  scan_args args_;

 public:
  using iterator = detail::scan_iterator;
  using sentinel = detail::scan_sentinel;

  FMT_CONSTEXPR explicit scan_context(detail::scan_buffer& buf, scan_args args)
      : buf_(buf), args_(args) {}

  FMT_CONSTEXPR auto arg(int id) const -> scan_arg {
    return id < args_.size ? args_.data[id] : scan_arg();
  }

  auto begin() const -> iterator { return buf_.begin(); }
  auto end() const -> sentinel { return {}; }

  void advance_to(iterator) { buf_.consume(); }
};

namespace detail {

const char* parse_scan_specs(const char* begin, const char* end,
                             format_specs& specs, scan_type) {
  while (begin != end) {
    switch (to_ascii(*begin)) {
    // TODO: parse more scan format specifiers
    case 'x':
      specs.set_type(presentation_type::hex);
      ++begin;
      break;
    case '}':
      return begin;
    }
  }
  return begin;
}

template <typename T, FMT_ENABLE_IF(std::is_unsigned<T>::value)>
auto read(scan_iterator it, T& value) -> scan_iterator {
  if (it == scan_sentinel()) return it;
  char c = *it;
  if (c < '0' || c > '9') report_error("invalid input");

  int num_digits = 0;
  T n = 0, prev = 0;
  char prev_digit = c;
  do {
    prev = n;
    n = n * 10 + static_cast<unsigned>(c - '0');
    prev_digit = c;
    c = *++it;
    ++num_digits;
    if (c < '0' || c > '9') break;
  } while (it != scan_sentinel());

  // Check overflow.
  if (num_digits <= std::numeric_limits<int>::digits10) {
    value = n;
    return it;
  }
  unsigned max = to_unsigned((std::numeric_limits<int>::max)());
  if (num_digits == std::numeric_limits<int>::digits10 + 1 &&
      prev * 10ull + unsigned(prev_digit - '0') <= max) {
    value = n;
  } else {
    report_error("number is too big");
  }
  return it;
}

template <typename T, FMT_ENABLE_IF(std::is_unsigned<T>::value)>
auto read_hex(scan_iterator it, T& value) -> scan_iterator {
  if (it == scan_sentinel()) return it;
  int digit = to_hex_digit(*it);
  if (digit < 0) report_error("invalid input");

  int num_digits = 0;
  T n = 0;
  do {
    n = (n << 4) + static_cast<unsigned>(digit);
    ++num_digits;
    digit = to_hex_digit(*++it);
    if (digit < 0) break;
  } while (it != scan_sentinel());

  // Check overflow.
  if (num_digits <= (std::numeric_limits<T>::digits >> 2))
    value = n;
  else
    report_error("number is too big");
  return it;
}

template <typename T, FMT_ENABLE_IF(std::is_unsigned<T>::value)>
auto read(scan_iterator it, T& value, const format_specs& specs)
    -> scan_iterator {
  if (specs.type() == presentation_type::hex) return read_hex(it, value);
  return read(it, value);
}

template <typename T, FMT_ENABLE_IF(std::is_signed<T>::value)>
auto read(scan_iterator it, T& value, const format_specs& specs = {})
    -> scan_iterator {
  bool negative = it != scan_sentinel() && *it == '-';
  if (negative) {
    ++it;
    if (it == scan_sentinel()) report_error("invalid input");
  }
  using unsigned_type = typename std::make_unsigned<T>::type;
  unsigned_type abs_value = 0;
  it = read(it, abs_value, specs);
  auto n = static_cast<T>(abs_value);
  value = negative ? -n : n;
  return it;
}

auto read(scan_iterator it, double& value, const format_specs& = {})
    -> scan_iterator {
  if (it == scan_sentinel()) return it;
  
  // Simple floating-point parsing
  bool negative = *it == '-';
  if (negative) {
    ++it;
    if (it == scan_sentinel()) report_error("invalid input");
  }
  
  double result = 0.0;
  // Parse integer part
  while (it != scan_sentinel() && *it >= '0' && *it <= '9') {
    result = result * 10.0 + (*it - '0');
    ++it;
  }
  
  // Parse decimal part if present
  if (it != scan_sentinel() && *it == '.') {
    ++it;
    double fraction = 0.1;
    while (it != scan_sentinel() && *it >= '0' && *it <= '9') {
      result += (*it - '0') * fraction;
      fraction *= 0.1;
      ++it;
    }
  }
  
  value = negative ? -result : result;
  return it;
}

auto read(scan_iterator it, float& value, const format_specs& specs = {})
    -> scan_iterator {
  double temp;
  it = read(it, temp, specs);
  value = static_cast<float>(temp);
  return it;
}

auto read(scan_iterator it, std::string& value, const format_specs& = {})
    -> scan_iterator {
  while (it != scan_sentinel() && *it != ' ') value.push_back(*it++);
  return it;
}

auto read(scan_iterator it, string_view& value, const format_specs& = {})
    -> scan_iterator {
  auto range = to_contiguous(it);
  // This could also be checked at compile time in scan.
  if (!range) report_error("string_view requires contiguous input");
  auto p = range.begin;
  while (p != range.end && *p != ' ') ++p;
  size_t size = to_unsigned(p - range.begin);
  value = {range.begin, size};
  return advance(it, size);
}

auto read(scan_iterator it, monostate, const format_specs& = {})
    -> scan_iterator {
  return it;
}

// An argument scanner that uses the default format, e.g. decimal for integers.
struct default_arg_scanner {
  scan_iterator it;

  template <typename T> FMT_INLINE auto operator()(T&& value) -> scan_iterator {
    return read(it, value);
  }
};

// An argument scanner with format specifiers.
struct arg_scanner {
  scan_iterator it;
  const format_specs& specs;

  template <typename T> auto operator()(T&& value) -> scan_iterator {
    return read(it, value, specs);
  }
};

struct scan_handler {
 private:
  scan_parse_context parse_ctx_;
  scan_context scan_ctx_;
  int next_arg_id_;

  using sentinel = scan_buffer::sentinel;

 public:
  FMT_CONSTEXPR scan_handler(string_view format, scan_buffer& buf,
                             scan_args args)
      : parse_ctx_(format), scan_ctx_(buf, args), next_arg_id_(0) {}

  auto pos() const -> scan_buffer::iterator { return scan_ctx_.begin(); }

  void on_text(const char* begin, const char* end) {
    if (begin == end) return;
    auto it = scan_ctx_.begin();
    for (; begin != end; ++begin, ++it) {
      if (it == sentinel() || *begin != *it) on_error("invalid input");
    }
    scan_ctx_.advance_to(it);
  }

  FMT_CONSTEXPR auto on_arg_id() -> int { return on_arg_id(next_arg_id_++); }
  FMT_CONSTEXPR auto on_arg_id(int id) -> int {
    if (!scan_ctx_.arg(id)) on_error("argument index out of range");
    return id;
  }
  FMT_CONSTEXPR auto on_arg_id(string_view id) -> int {
    if (id.data()) on_error("invalid format");
    return 0;
  }

  void on_replacement_field(int arg_id, const char* begin) {
    scan_arg arg = scan_ctx_.arg(arg_id);
    if (arg.scan_custom(begin, parse_ctx_, scan_ctx_)) return;
    auto it = scan_ctx_.begin();
    while (it != sentinel() && is_whitespace(*it)) ++it;
    scan_ctx_.advance_to(arg.visit(default_arg_scanner{it}));
  }

  auto on_format_specs(int arg_id, const char* begin, const char* end) -> const
      char* {
    scan_arg arg = scan_ctx_.arg(arg_id);
    if (arg.scan_custom(begin, parse_ctx_, scan_ctx_))
      return parse_ctx_.begin();
    auto specs = format_specs();
    begin = parse_scan_specs(begin, end, specs, arg.type());
    if (begin == end || *begin != '}') on_error("missing '}' in format string");
    scan_ctx_.advance_to(arg.visit(arg_scanner{scan_ctx_.begin(), specs}));
    return begin;
  }

  FMT_NORETURN void on_error(const char* message) { report_error(message); }
};

void vscan(detail::scan_buffer& buf, string_view fmt, scan_args args) {
  auto h = detail::scan_handler(fmt, buf, args);
  detail::parse_format_string(fmt, h);
}

template <size_t I, typename... T, FMT_ENABLE_IF(I == sizeof...(T))>
void make_args(std::array<scan_arg, sizeof...(T)>&, std::tuple<T...>&) {}

template <size_t I, typename... T, FMT_ENABLE_IF(I < sizeof...(T))>
void make_args(std::array<scan_arg, sizeof...(T)>& args,
               std::tuple<T...>& values) {
  using element_type = typename std::tuple_element<I, std::tuple<T...>>::type;
  static_assert(std::is_same<remove_cvref_t<element_type>, element_type>::value,
                "");
  args[I] = std::get<I>(values);
  make_args<I + 1>(args, values);
}
}  // namespace detail

template <typename Range, typename... T> class scan_data {
 private:
  std::tuple<T...> values_;
  Range range_;

 public:
  scan_data() = default;
  scan_data(T... values) : values_(std::move(values)...) {}

  auto value() const -> decltype(std::get<0>(values_)) {
    return std::get<0>(values_);
  }

  auto values() const -> const std::tuple<T...>& { return values_; }

  auto make_args() -> std::array<scan_arg, sizeof...(T)> {
    auto args = std::array<scan_arg, sizeof...(T)>();
    detail::make_args<0>(args, values_);
    return args;
  }

  auto range() const -> Range { return range_; }

  auto begin() const -> decltype(range_.begin()) { return range_.begin(); }
  auto end() const -> decltype(range_.end()) { return range_.end(); }
};

template <typename... T>
auto make_scan_args(T&... args) -> std::array<scan_arg, sizeof...(T)> {
  return {{args...}};
}

class scan_error {};

// A rudimentary version of std::expected for testing the API shape.
template <typename T, typename E> class expected {
 private:
  T value_;
  bool has_value_ = true;

 public:
  expected(T value) : value_(std::move(value)) {}

  explicit operator bool() const { return has_value_; }

  auto operator->() const -> const T* { return &value_; }

  auto error() -> E const { return E(); }
};

template <typename Range, typename... T>
using scan_result = expected<scan_data<Range, T...>, scan_error>;

auto vscan(string_view input, string_view fmt, scan_args args)
    -> string_view::iterator {
  auto&& buf = detail::string_scan_buffer(input);
  detail::vscan(buf, fmt, args);
  return input.begin() + (buf.begin().base() - input.data());
}

// Scans the input and stores the results (in)to args.
template <typename... T>
auto scan_to(string_view input, string_view fmt, T&... args)
    -> string_view::iterator {
  return vscan(input, fmt, make_scan_args(args...));
}

template <typename... T>
auto scan(string_view input, string_view fmt)
    -> scan_result<string_view, T...> {
  auto data = scan_data<string_view, T...>();
  vscan(input, fmt, data.make_args());
  return data;
}

template <typename Range, typename... T,
          FMT_ENABLE_IF(!std::is_convertible<Range, string_view>::value)>
auto scan_to(Range&& input, string_view fmt, T&... args)
    -> decltype(std::begin(input)) {
  auto it = std::begin(input);
  detail::vscan(get_buffer(it), fmt, make_scan_args(args...));
  return it;
}

template <typename... T>
auto scan_to(FILE* f, string_view fmt, T&... args) -> bool {
  auto&& buf = detail::file_scan_buffer(f);
  detail::vscan(buf, fmt, make_scan_args(args...));
  return buf.begin() != buf.end();
}

FMT_END_NAMESPACE
