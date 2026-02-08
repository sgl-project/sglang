# API Reference

The {fmt} library API consists of the following components:

- [`fmt/base.h`](#base-api): the base API providing main formatting functions
  for `char`/UTF-8 with C++20 compile-time checks and minimal dependencies
- [`fmt/format.h`](#format-api): `fmt::format` and other formatting functions
  as well as locale support
- [`fmt/ranges.h`](#ranges-api): formatting of ranges and tuples
- [`fmt/chrono.h`](#chrono-api): date and time formatting
- [`fmt/std.h`](#std-api): formatters for standard library types
- [`fmt/compile.h`](#compile-api): format string compilation
- [`fmt/color.h`](#color-api): terminal colors and text styles
- [`fmt/os.h`](#os-api): system APIs
- [`fmt/ostream.h`](#ostream-api): `std::ostream` support
- [`fmt/args.h`](#args-api): dynamic argument lists
- [`fmt/printf.h`](#printf-api): safe `printf`
- [`fmt/xchar.h`](#xchar-api): optional `wchar_t` support

All functions and types provided by the library reside in namespace `fmt`
and macros have prefix `FMT_`.

## Base API

`fmt/base.h` defines the base API which provides main formatting functions
for `char`/UTF-8 with C++20 compile-time checks. It has minimal include
dependencies for better compile times. This header is only beneficial when
using {fmt} as a library (the default) and not in the header-only mode.
It also provides `formatter` specializations for the following types:

- `int`, `long long`,
- `unsigned`, `unsigned long long`
- `float`, `double`, `long double`
- `bool`
- `char`
- `const char*`, [`fmt::string_view`](#basic_string_view)
- `const void*`

The following functions use [format string syntax](syntax.md) similar to that
of [str.format](https://docs.python.org/3/library/stdtypes.html#str.format)
in Python. They take *fmt* and *args* as arguments.

*fmt* is a format string that contains literal text and replacement fields
surrounded by braces `{}`. The fields are replaced with formatted arguments
in the resulting string. [`fmt::format_string`](#format_string) is a format
string which can be implicitly constructed from a string literal or a
`constexpr` string and is checked at compile time in C++20. To pass a runtime
format string wrap it in [`fmt::runtime`](#runtime).

*args* is an argument list representing objects to be formatted.

I/O errors are reported as [`std::system_error`](
https://en.cppreference.com/w/cpp/error/system_error) exceptions unless
specified otherwise.

::: print(format_string<T...>, T&&...)

::: print(FILE*, format_string<T...>, T&&...)

::: println(format_string<T...>, T&&...)

::: println(FILE*, format_string<T...>, T&&...)

::: format_to(OutputIt&&, format_string<T...>, T&&...)

::: format_to_n(OutputIt, size_t, format_string<T...>, T&&...)

::: format_to_n_result

::: formatted_size(format_string<T...>, T&&...)

<a id="udt"></a>
### Formatting User-Defined Types

The {fmt} library provides formatters for many standard C++ types.
See [`fmt/ranges.h`](#ranges-api) for ranges and tuples including standard
containers such as `std::vector`, [`fmt/chrono.h`](#chrono-api) for date and
time formatting and [`fmt/std.h`](#std-api) for other standard library types.

There are two ways to make a user-defined type formattable: providing a
`format_as` function or specializing the `formatter` struct template.
Formatting of non-void pointer types is intentionally disallowed and they
cannot be made formattable via either extension API.

Use `format_as` if you want to make your type formattable as some other
type with the same format specifiers. The `format_as` function should
take an object of your type and return an object of a formattable type.
It should be defined in the same namespace as your type.

Example ([run](https://godbolt.org/z/nvME4arz8)):

    #include <fmt/format.h>

    namespace kevin_namespacy {

    enum class film {
      house_of_cards, american_beauty, se7en = 7
    };

    auto format_as(film f) { return fmt::underlying(f); }

    }

    int main() {
      fmt::print("{}\n", kevin_namespacy::film::se7en); // Output: 7
    }

Using specialization is more complex but gives you full control over
parsing and formatting. To use this method specialize the `formatter`
struct template for your type and implement `parse` and `format`
methods.

The recommended way of defining a formatter is by reusing an existing
one via inheritance or composition. This way you can support standard
format specifiers without implementing them yourself. For example:

```c++
// color.h:
#include <fmt/base.h>

enum class color {red, green, blue};

template <> struct fmt::formatter<color>: formatter<string_view> {
  // parse is inherited from formatter<string_view>.

  auto format(color c, format_context& ctx) const
    -> format_context::iterator;
};
```

```c++
// color.cc:
#include "color.h"
#include <fmt/format.h>

auto fmt::formatter<color>::format(color c, format_context& ctx) const
    -> format_context::iterator {
  string_view name = "unknown";
  switch (c) {
  case color::red:   name = "red"; break;
  case color::green: name = "green"; break;
  case color::blue:  name = "blue"; break;
  }
  return formatter<string_view>::format(name, ctx);
}
```

Note that `formatter<string_view>::format` is defined in `fmt/format.h`
so it has to be included in the source file. Since `parse` is inherited
from `formatter<string_view>` it will recognize all string format
specifications, for example

```c++
fmt::format("{:>10}", color::blue)
```

will return `"      blue"`.

<!-- The experimental `nested_formatter` provides an easy way of applying a
formatter to one or more subobjects.

For example:

    #include <fmt/format.h>

    struct point {
      double x, y;
    };

    template <>
    struct fmt::formatter<point> : nested_formatter<double> {
      auto format(point p, format_context& ctx) const {
        return write_padded(ctx, [=](auto out) {
          return format_to(out, "({}, {})", this->nested(p.x),
                           this->nested(p.y));
        });
      }
    };

    int main() {
      fmt::print("[{:>20.2f}]", point{1, 2});
    }

prints:

    [          (1.00, 2.00)]

Notice that fill, align and width are applied to the whole object which
is the recommended behavior while the remaining specifiers apply to
elements. -->

In general the formatter has the following form:

    template <> struct fmt::formatter<T> {
      // Parses format specifiers and stores them in the formatter.
      //
      // [ctx.begin(), ctx.end()) is a, possibly empty, character range that
      // contains a part of the format string starting from the format
      // specifications to be parsed, e.g. in
      //
      //   fmt::format("{:f} continued", ...);
      //
      // the range will contain "f} continued". The formatter should parse
      // specifiers until '}' or the end of the range. In this example the
      // formatter should parse the 'f' specifier and return an iterator
      // pointing to '}'.
      constexpr auto parse(format_parse_context& ctx)
        -> format_parse_context::iterator;

      // Formats value using the parsed format specification stored in this
      // formatter and writes the output to ctx.out().
      auto format(const T& value, format_context& ctx) const
        -> format_context::iterator;
    };

It is recommended to at least support fill, align and width that apply
to the whole object and have the same semantics as in standard
formatters.

You can also write a formatter for a hierarchy of classes:

```c++
// demo.h:
#include <type_traits>
#include <fmt/format.h>

struct A {
  virtual ~A() {}
  virtual std::string name() const { return "A"; }
};

struct B : A {
  virtual std::string name() const { return "B"; }
};

template <typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<A, T>, char>> :
    fmt::formatter<std::string> {
  auto format(const A& a, format_context& ctx) const {
    return formatter<std::string>::format(a.name(), ctx);
  }
};
```

```c++
// demo.cc:
#include "demo.h"
#include <fmt/format.h>

int main() {
  B b;
  A& a = b;
  fmt::print("{}", a); // Output: B
}
```

Providing both a `formatter` specialization and a `format_as` overload is
disallowed.

::: basic_format_parse_context

::: context

::: format_context

### Compile-Time Checks

Compile-time format string checks are enabled by default on compilers
that support C++20 `consteval`. On older compilers you can use the
[FMT_STRING](#legacy-checks) macro defined in `fmt/format.h` instead.

Unused arguments are allowed as in Python's `str.format` and ordinary functions.

See [Type Erasure](#type-erasure) for an example of how to enable compile-time
checks in your own functions with `fmt::format_string` while avoiding template
bloat.

::: fstring

::: format_string

::: runtime(string_view)

### Type Erasure

You can create your own formatting function with compile-time checks and
small binary footprint, for example ([run](https://godbolt.org/z/b9Pbasvzc)):

```c++
#include <fmt/format.h>

void vlog(const char* file, int line,
          fmt::string_view fmt, fmt::format_args args) {
  fmt::print("{}: {}: {}", file, line, fmt::vformat(fmt, args));
}

template <typename... T>
void log(const char* file, int line,
         fmt::format_string<T...> fmt, T&&... args) {
  vlog(file, line, fmt, fmt::make_format_args(args...));
}

#define MY_LOG(fmt, ...) log(__FILE__, __LINE__, fmt, __VA_ARGS__)

MY_LOG("invalid squishiness: {}", 42);
```

Note that `vlog` is not parameterized on argument types which improves
compile times and reduces binary code size compared to a fully
parameterized version.

::: make_format_args(T&...)

::: basic_format_args

::: format_args

::: basic_format_arg

### Named Arguments

::: arg(const Char*, const T&)

Named arguments are not supported in compile-time checks at the moment.

### Compatibility

::: basic_string_view

::: string_view

## Format API

`fmt/format.h` defines the full format API providing additional
formatting functions and locale support.

<a id="format"></a>
::: format(format_string<T...>, T&&...)

::: vformat(string_view, format_args)

::: operator""_a()

### Utilities

::: ptr(T)

::: underlying(Enum)

::: to_string(const T&)

::: group_digits(T)

::: detail::buffer

::: basic_memory_buffer

### System Errors

{fmt} does not use `errno` to communicate errors to the user, but it may
call system functions which set `errno`. Users should not make any
assumptions about the value of `errno` being preserved by library
functions.

::: system_error

::: format_system_error

### Custom Allocators

The {fmt} library supports custom dynamic memory allocators. A custom
allocator class can be specified as a template argument to
[`fmt::basic_memory_buffer`](#basic_memory_buffer):

    using custom_memory_buffer = 
      fmt::basic_memory_buffer<char, fmt::inline_buffer_size, custom_allocator>;

It is also possible to write a formatting function that uses a custom
allocator:

    using custom_string =
      std::basic_string<char, std::char_traits<char>, custom_allocator>;

    auto vformat(custom_allocator alloc, fmt::string_view fmt,
                 fmt::format_args args) -> custom_string {
      auto buf = custom_memory_buffer(alloc);
      fmt::vformat_to(std::back_inserter(buf), fmt, args);
      return custom_string(buf.data(), buf.size(), alloc);
    }

    template <typename ...Args>
    auto format(custom_allocator alloc, fmt::string_view fmt,
                const Args& ... args) -> custom_string {
      return vformat(alloc, fmt, fmt::make_format_args(args...));
    }

The allocator will be used for the output container only. Formatting
functions normally don't do any allocations for built-in and string
types except for non-default floating-point formatting that occasionally
falls back on `sprintf`.

### Locale

All formatting is locale-independent by default. Use the `'L'` format
specifier to insert the appropriate number separator characters from the
locale:

    #include <fmt/format.h>
    #include <locale>

    std::locale::global(std::locale("en_US.UTF-8"));
    auto s = fmt::format("{:L}", 1000000);  // s == "1,000,000"

`fmt/format.h` provides the following overloads of formatting functions
that take `std::locale` as a parameter. The locale type is a template
parameter to avoid the expensive `<locale>` include.

::: format(const Locale&, format_string<T...>, T&&...)

::: format_to(OutputIt, const Locale&, format_string<T...>, T&&...)

::: formatted_size(const Locale&, format_string<T...>, T&&...)

<a id="legacy-checks"></a>
### Legacy Compile-Time Checks

`FMT_STRING` enables compile-time checks on older compilers. It requires
C++14 or later and is a no-op in C++11.

::: FMT_STRING

To force the use of legacy compile-time checks, define the preprocessor
variable `FMT_ENFORCE_COMPILE_STRING`. When set, functions accepting
`FMT_STRING` will fail to compile with regular strings.

<a id="ranges-api"></a>
## Range and Tuple Formatting

`fmt/ranges.h` provides formatting support for ranges and tuples:

    #include <fmt/ranges.h>

    fmt::print("{}", std::tuple<char, int>{'a', 42});
    // Output: ('a', 42)

Using `fmt::join`, you can separate tuple elements with a custom separator:

    #include <fmt/ranges.h>

    auto t = std::tuple<int, char>{1, 'a'};
    fmt::print("{}", fmt::join(t, ", "));
    // Output: 1, a

::: join(Range&&, string_view)

::: join(It, Sentinel, string_view)

::: join(std::initializer_list<T>, string_view)

<a id="chrono-api"></a>
## Date and Time Formatting

`fmt/chrono.h` provides formatters for

- [`std::chrono::duration`](https://en.cppreference.com/w/cpp/chrono/duration)
- [`std::chrono::time_point`](
  https://en.cppreference.com/w/cpp/chrono/time_point)
- [`std::tm`](https://en.cppreference.com/w/cpp/chrono/c/tm)

The format syntax is described in [Chrono Format Specifications](syntax.md#
chrono-format-specifications).

**Example**:

    #include <fmt/chrono.h>

    int main() {
      auto now = std::chrono::system_clock::now();

      fmt::print("The date is {:%Y-%m-%d}.\n", now);
      // Output: The date is 2020-11-07.
      // (with 2020-11-07 replaced by the current date)

      using namespace std::literals::chrono_literals;

      fmt::print("Default format: {} {}\n", 42s, 100ms);
      // Output: Default format: 42s 100ms

      fmt::print("strftime-like format: {:%H:%M:%S}\n", 3h + 15min + 30s);
      // Output: strftime-like format: 03:15:30
    }

::: gmtime(std::time_t)

<a id="std-api"></a>
## Standard Library Types Formatting

`fmt/std.h` provides formatters for:

- [`std::atomic`](https://en.cppreference.com/w/cpp/atomic/atomic)
- [`std::atomic_flag`](https://en.cppreference.com/w/cpp/atomic/atomic_flag)
- [`std::bitset`](https://en.cppreference.com/w/cpp/utility/bitset)
- [`std::error_code`](https://en.cppreference.com/w/cpp/error/error_code)
- [`std::exception`](https://en.cppreference.com/w/cpp/error/exception)
- [`std::filesystem::path`](https://en.cppreference.com/w/cpp/filesystem/path)
- [`std::monostate`](
  https://en.cppreference.com/w/cpp/utility/variant/monostate)
- [`std::optional`](https://en.cppreference.com/w/cpp/utility/optional)
- [`std::source_location`](
  https://en.cppreference.com/w/cpp/utility/source_location)
- [`std::thread::id`](https://en.cppreference.com/w/cpp/thread/thread/id)
- [`std::variant`](https://en.cppreference.com/w/cpp/utility/variant/variant)

::: ptr(const std::unique_ptr<T, Deleter>&)

::: ptr(const std::shared_ptr<T>&)

### Variants

A `std::variant` is only formattable if every variant alternative is
formattable, and requires the `__cpp_lib_variant` [library
feature](https://en.cppreference.com/w/cpp/feature_test).

**Example**:

    #include <fmt/std.h>

    fmt::print("{}", std::variant<char, float>('x'));
    // Output: variant('x')

    fmt::print("{}", std::variant<std::monostate, char>());
    // Output: variant(monostate)

## Bit-Fields and Packed Structs

To format a bit-field or a field of a struct with `__attribute__((packed))`
applied to it, you need to convert it to the underlying or compatible type via
a cast or a unary `+` ([godbolt](https://www.godbolt.org/z/3qKKs6T5Y)):

```c++
struct smol {
  int bit : 1;
};

auto s = smol();
fmt::print("{}", +s.bit);
```

This is a known limitation of "perfect" forwarding in C++.

<a id="compile-api"></a>
## Format String Compilation

`fmt/compile.h` provides format string compilation and compile-time
(`constexpr`) formatting enabled via the `FMT_COMPILE` macro or the `_cf`
user-defined literal defined in namespace `fmt::literals`. Format strings
marked with `FMT_COMPILE` or `_cf` are parsed, checked and converted into
efficient formatting code at compile-time. This supports arguments of built-in
and string types as well as user-defined types with `format` functions taking
the format context type as a template parameter in their `formatter`
specializations. For example:

    template <> struct fmt::formatter<point> {
      constexpr auto parse(format_parse_context& ctx);

      template <typename FormatContext>
      auto format(const point& p, FormatContext& ctx) const;
    };

Format string compilation can generate more binary code compared to the
default API and is only recommended in places where formatting is a
performance bottleneck.

::: FMT_COMPILE

::: operator""_cf

<a id="color-api"></a>
## Terminal Colors and Text Styles

`fmt/color.h` provides support for terminal color and text style output.

::: print(text_style, format_string<T...>, T&&...)

::: fg(detail::color_type)

::: bg(detail::color_type)

::: styled(const T&, text_style)

<a id="os-api"></a>
## System APIs

::: ostream

::: windows_error

<a id="ostream-api"></a>
## `std::ostream` Support

`fmt/ostream.h` provides `std::ostream` support including formatting of
user-defined types that have an overloaded insertion operator
(`operator<<`). In order to make a type formattable via `std::ostream`
you should provide a `formatter` specialization inherited from
`ostream_formatter`:

    #include <fmt/ostream.h>

    struct date {
      int year, month, day;

      friend std::ostream& operator<<(std::ostream& os, const date& d) {
        return os << d.year << '-' << d.month << '-' << d.day;
      }
    };

    template <> struct fmt::formatter<date> : ostream_formatter {};

    std::string s = fmt::format("The date is {}", date{2012, 12, 9});
    // s == "The date is 2012-12-9"

::: streamed(const T&)

::: print(std::ostream&, format_string<T...>, T&&...)

<a id="args-api"></a>
## Dynamic Argument Lists

The header `fmt/args.h` provides `dynamic_format_arg_store`, a builder-like API
that can be used to construct format argument lists dynamically.

::: dynamic_format_arg_store

<a id="printf-api"></a>
## Safe `printf`

The header `fmt/printf.h` provides `printf`-like formatting
functionality. The following functions use [printf format string
syntax](https://pubs.opengroup.org/onlinepubs/009695399/functions/fprintf.html)
with the POSIX extension for positional arguments. Unlike their standard
counterparts, the `fmt` functions are type-safe and throw an exception
if an argument type doesn't match its format specification.

::: printf(string_view, const T&...)

::: fprintf(std::FILE*, const S&, const T&...)

::: sprintf(const S&, const T&...)

<a id="xchar-api"></a>
## Wide Strings

The optional header `fmt/xchar.h` provides support for `wchar_t` and
exotic character types.

::: is_char

::: wstring_view

::: wformat_context

::: to_wstring(const T&)

## Compatibility with C++20 `std::format`

{fmt} implements nearly all of the [C++20 formatting
library](https://en.cppreference.com/w/cpp/utility/format) with the
following differences:

- Names are defined in the `fmt` namespace instead of `std` to avoid
  collisions with standard library implementations.

- Width calculation doesn't use grapheme clusterization. The latter has
  been implemented in a separate branch but hasn't been integrated yet.

- The default floating-point representation in {fmt} uses the smallest
  precision that provides round-trip guarantees similarly to other languages
  like Java and Python. `std::format` is currently specified in terms of
  `std::to_chars` which tries to generate the smallest number of characters
  (ignoring redundant digits and sign in exponent) and may procude more
  decimal digits than necessary.
