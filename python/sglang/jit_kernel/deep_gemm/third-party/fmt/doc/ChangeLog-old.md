# 7.1.3 - 2020-11-24

- Fixed handling of buffer boundaries in `format_to_n`
  (https://github.com/fmtlib/fmt/issues/1996,
  https://github.com/fmtlib/fmt/issues/2029).
- Fixed linkage errors when linking with a shared library
  (https://github.com/fmtlib/fmt/issues/2011).
- Reintroduced ostream support to range formatters
  (https://github.com/fmtlib/fmt/issues/2014).
- Worked around an issue with mixing std versions in gcc
  (https://github.com/fmtlib/fmt/issues/2017).

# 7.1.2 - 2020-11-04

- Fixed floating point formatting with large precision
  (https://github.com/fmtlib/fmt/issues/1976).

# 7.1.1 - 2020-11-01

- Fixed ABI compatibility with 7.0.x
  (https://github.com/fmtlib/fmt/issues/1961).
- Added the `FMT_ARM_ABI_COMPATIBILITY` macro to work around ABI
  incompatibility between GCC and Clang on ARM
  (https://github.com/fmtlib/fmt/issues/1919).
- Worked around a SFINAE bug in GCC 8
  (https://github.com/fmtlib/fmt/issues/1957).
- Fixed linkage errors when building with GCC\'s LTO
  (https://github.com/fmtlib/fmt/issues/1955).
- Fixed a compilation error when building without `__builtin_clz` or
  equivalent (https://github.com/fmtlib/fmt/pull/1968).
  Thanks @tohammer.
- Fixed a sign conversion warning
  (https://github.com/fmtlib/fmt/pull/1964). Thanks @OptoCloud.

# 7.1.0 - 2020-10-25

- Switched from
  [Grisu3](https://www.cs.tufts.edu/~nr/cs257/archive/florian-loitsch/printf.pdf)
  to [Dragonbox](https://github.com/jk-jeon/dragonbox) for the default
  floating-point formatting which gives the shortest decimal
  representation with round-trip guarantee and correct rounding
  (https://github.com/fmtlib/fmt/pull/1882,
  https://github.com/fmtlib/fmt/pull/1887,
  https://github.com/fmtlib/fmt/pull/1894). This makes {fmt}
  up to 20-30x faster than common implementations of
  `std::ostringstream` and `sprintf` on
  [dtoa-benchmark](https://github.com/fmtlib/dtoa-benchmark) and
  faster than double-conversion and Ryū:

  ![](https://user-images.githubusercontent.com/576385/95684665-11719600-0ba8-11eb-8e5b-972ff4e49428.png)

  It is possible to get even better performance at the cost of larger
  binary size by compiling with the `FMT_USE_FULL_CACHE_DRAGONBOX`
  macro set to 1.

  Thanks @jk-jeon.

- Added an experimental unsynchronized file output API which, together
  with [format string
  compilation](https://fmt.dev/latest/api.html#compile-api), can give
  [5-9 times speed up compared to
  fprintf](https://www.zverovich.net/2020/08/04/optimal-file-buffer-size.html)
  on common platforms ([godbolt](https://godbolt.org/z/nsTcG8)):

  ```c++
  #include <fmt/os.h>

  int main() {
    auto f = fmt::output_file("guide");
    f.print("The answer is {}.", 42);
  }
  ```

- Added a formatter for `std::chrono::time_point<system_clock>`
  (https://github.com/fmtlib/fmt/issues/1819,
  https://github.com/fmtlib/fmt/pull/1837). For example
  ([godbolt](https://godbolt.org/z/c4M6fh)):

  ```c++
  #include <fmt/chrono.h>

  int main() {
    auto now = std::chrono::system_clock::now();
    fmt::print("The time is {:%H:%M:%S}.\n", now);
  }
  ```

  Thanks @adamburgess.

- Added support for ranges with non-const `begin`/`end` to `fmt::join`
  (https://github.com/fmtlib/fmt/issues/1784,
  https://github.com/fmtlib/fmt/pull/1786). For example
  ([godbolt](https://godbolt.org/z/jP63Tv)):

  ```c++
  #include <fmt/ranges.h>
  #include <range/v3/view/filter.hpp>

  int main() {
    using std::literals::string_literals::operator""s;
    auto strs = std::array{"a"s, "bb"s, "ccc"s};
    auto range = strs | ranges::views::filter(
      [] (const std::string &x) { return x.size() != 2; }
    );
    fmt::print("{}\n", fmt::join(range, ""));
  }
  ```

  prints \"accc\".

  Thanks @tonyelewis.

- Added a `memory_buffer::append` overload that takes a range
  (https://github.com/fmtlib/fmt/pull/1806). Thanks @BRevzin.

- Improved handling of single code units in `FMT_COMPILE`. For
  example:

  ```c++
  #include <fmt/compile.h>

  char* f(char* buf) {
    return fmt::format_to(buf, FMT_COMPILE("x{}"), 42);
  }
  ```

  compiles to just ([godbolt](https://godbolt.org/z/5vncz3)):

  ```asm
  _Z1fPc:
    movb $120, (%rdi)
    xorl %edx, %edx
    cmpl $42, _ZN3fmt2v76detail10basic_dataIvE23zero_or_powers_of_10_32E+8(%rip)
    movl $3, %eax
    seta %dl
    subl %edx, %eax
    movzwl _ZN3fmt2v76detail10basic_dataIvE6digitsE+84(%rip), %edx
    cltq
    addq %rdi, %rax
    movw %dx, -2(%rax)
    ret
  ```

  Here a single `mov` instruction writes `'x'` (`$120`) to the output
  buffer.

- Added dynamic width support to format string compilation
  (https://github.com/fmtlib/fmt/issues/1809).

- Improved error reporting for unformattable types: now you\'ll get
  the type name directly in the error message instead of the note:

  ```c++
  #include <fmt/core.h>

  struct how_about_no {};

  int main() {
    fmt::print("{}", how_about_no());
  }
  ```

  Error ([godbolt](https://godbolt.org/z/GoxM4e)):

  `fmt/core.h:1438:3: error: static_assert failed due to requirement 'fmt::v7::formattable<how_about_no>()' "Cannot format an argument. To make type T formattable provide a formatter<T> specialization: https://fmt.dev/latest/api.html#udt" ...`

- Added the
  [make_args_checked](https://fmt.dev/7.1.0/api.html#argument-lists)
  function template that allows you to write formatting functions with
  compile-time format string checks and avoid binary code bloat
  ([godbolt](https://godbolt.org/z/PEf9qr)):

  ```c++
  void vlog(const char* file, int line, fmt::string_view format,
            fmt::format_args args) {
    fmt::print("{}: {}: ", file, line);
    fmt::vprint(format, args);
  }

  template <typename S, typename... Args>
  void log(const char* file, int line, const S& format, Args&&... args) {
    vlog(file, line, format,
        fmt::make_args_checked<Args...>(format, args...));
  }

  #define MY_LOG(format, ...) \
    log(__FILE__, __LINE__, FMT_STRING(format), __VA_ARGS__)

  MY_LOG("invalid squishiness: {}", 42);
  ```

- Replaced `snprintf` fallback with a faster internal IEEE 754 `float`
  and `double` formatter for arbitrary precision. For example
  ([godbolt](https://godbolt.org/z/dPhWvj)):

  ```c++
  #include <fmt/core.h>

  int main() {
    fmt::print("{:.500}\n", 4.9406564584124654E-324);
  }
  ```

  prints

  `4.9406564584124654417656879286822137236505980261432476442558568250067550727020875186529983636163599237979656469544571773092665671035593979639877479601078187812630071319031140452784581716784898210368871863605699873072305000638740915356498438731247339727316961514003171538539807412623856559117102665855668676818703956031062493194527159149245532930545654440112748012970999954193198940908041656332452475714786901472678015935523861155013480352649347201937902681071074917033322268447533357208324319360923829e-324`.

- Made `format_to_n` and `formatted_size` part of the [core
  API](https://fmt.dev/latest/api.html#core-api)
  ([godbolt](https://godbolt.org/z/sPjY1K)):

  ```c++
  #include <fmt/core.h>

  int main() {
    char buffer[10];
    auto result = fmt::format_to_n(buffer, sizeof(buffer), "{}", 42);
  }
  ```

- Added `fmt::format_to_n` overload with format string compilation
  (https://github.com/fmtlib/fmt/issues/1764,
  https://github.com/fmtlib/fmt/pull/1767,
  https://github.com/fmtlib/fmt/pull/1869). For example
  ([godbolt](https://godbolt.org/z/93h86q)):

  ```c++
  #include <fmt/compile.h>

  int main() {
    char buffer[8];
    fmt::format_to_n(buffer, sizeof(buffer), FMT_COMPILE("{}"), 42);
  }
  ```

  Thanks @Kurkin and @alexezeder.

- Added `fmt::format_to` overload that take `text_style`
  (https://github.com/fmtlib/fmt/issues/1593,
  https://github.com/fmtlib/fmt/issues/1842,
  https://github.com/fmtlib/fmt/pull/1843). For example
  ([godbolt](https://godbolt.org/z/91153r)):

  ```c++
  #include <fmt/color.h>

  int main() {
    std::string out;
    fmt::format_to(std::back_inserter(out),
                   fmt::emphasis::bold | fg(fmt::color::red),
                   "The answer is {}.", 42);
  }
  ```

  Thanks @Naios.

- Made the `'#'` specifier emit trailing zeros in addition to the
  decimal point (https://github.com/fmtlib/fmt/issues/1797).
  For example ([godbolt](https://godbolt.org/z/bhdcW9)):

  ```c++
  #include <fmt/core.h>

  int main() {
    fmt::print("{:#.2g}", 0.5);
  }
  ```

  prints `0.50`.

- Changed the default floating point format to not include `.0` for
  consistency with `std::format` and `std::to_chars`
  (https://github.com/fmtlib/fmt/issues/1893,
  https://github.com/fmtlib/fmt/issues/1943). It is possible
  to get the decimal point and trailing zero with the `#` specifier.

- Fixed an issue with floating-point formatting that could result in
  addition of a non-significant trailing zero in rare cases e.g.
  `1.00e-34` instead of `1.0e-34`
  (https://github.com/fmtlib/fmt/issues/1873,
  https://github.com/fmtlib/fmt/issues/1917).

- Made `fmt::to_string` fallback on `ostream` insertion operator if
  the `formatter` specialization is not provided
  (https://github.com/fmtlib/fmt/issues/1815,
  https://github.com/fmtlib/fmt/pull/1829). Thanks @alexezeder.

- Added support for the append mode to the experimental file API and
  improved `fcntl.h` detection.
  (https://github.com/fmtlib/fmt/pull/1847,
  https://github.com/fmtlib/fmt/pull/1848). Thanks @t-wiser.

- Fixed handling of types that have both an implicit conversion
  operator and an overloaded `ostream` insertion operator
  (https://github.com/fmtlib/fmt/issues/1766).

- Fixed a slicing issue in an internal iterator type
  (https://github.com/fmtlib/fmt/pull/1822). Thanks @BRevzin.

- Fixed an issue in locale-specific integer formatting
  (https://github.com/fmtlib/fmt/issues/1927).

- Fixed handling of exotic code unit types
  (https://github.com/fmtlib/fmt/issues/1870,
  https://github.com/fmtlib/fmt/issues/1932).

- Improved `FMT_ALWAYS_INLINE`
  (https://github.com/fmtlib/fmt/pull/1878). Thanks @jk-jeon.

- Removed dependency on `windows.h`
  (https://github.com/fmtlib/fmt/pull/1900). Thanks @bernd5.

- Optimized counting of decimal digits on MSVC
  (https://github.com/fmtlib/fmt/pull/1890). Thanks @mwinterb.

- Improved documentation
  (https://github.com/fmtlib/fmt/issues/1772,
  https://github.com/fmtlib/fmt/pull/1775,
  https://github.com/fmtlib/fmt/pull/1792,
  https://github.com/fmtlib/fmt/pull/1838,
  https://github.com/fmtlib/fmt/pull/1888,
  https://github.com/fmtlib/fmt/pull/1918,
  https://github.com/fmtlib/fmt/pull/1939).
  Thanks @leolchat, @pepsiman, @Klaim, @ravijanjam, @francesco-st and @udnaan.

- Added the `FMT_REDUCE_INT_INSTANTIATIONS` CMake option that reduces
  the binary code size at the cost of some integer formatting
  performance. This can be useful for extremely memory-constrained
  embedded systems
  (https://github.com/fmtlib/fmt/issues/1778,
  https://github.com/fmtlib/fmt/pull/1781). Thanks @kammce.

- Added the `FMT_USE_INLINE_NAMESPACES` macro to control usage of
  inline namespaces
  (https://github.com/fmtlib/fmt/pull/1945). Thanks @darklukee.

- Improved build configuration
  (https://github.com/fmtlib/fmt/pull/1760,
  https://github.com/fmtlib/fmt/pull/1770,
  https://github.com/fmtlib/fmt/issues/1779,
  https://github.com/fmtlib/fmt/pull/1783,
  https://github.com/fmtlib/fmt/pull/1823).
  Thanks @dvetutnev, @xvitaly, @tambry, @medithe and @martinwuehrer.

- Fixed various warnings and compilation issues
  (https://github.com/fmtlib/fmt/pull/1790,
  https://github.com/fmtlib/fmt/pull/1802,
  https://github.com/fmtlib/fmt/pull/1808,
  https://github.com/fmtlib/fmt/issues/1810,
  https://github.com/fmtlib/fmt/issues/1811,
  https://github.com/fmtlib/fmt/pull/1812,
  https://github.com/fmtlib/fmt/pull/1814,
  https://github.com/fmtlib/fmt/pull/1816,
  https://github.com/fmtlib/fmt/pull/1817,
  https://github.com/fmtlib/fmt/pull/1818,
  https://github.com/fmtlib/fmt/issues/1825,
  https://github.com/fmtlib/fmt/pull/1836,
  https://github.com/fmtlib/fmt/pull/1855,
  https://github.com/fmtlib/fmt/pull/1856,
  https://github.com/fmtlib/fmt/pull/1860,
  https://github.com/fmtlib/fmt/pull/1877,
  https://github.com/fmtlib/fmt/pull/1879,
  https://github.com/fmtlib/fmt/pull/1880,
  https://github.com/fmtlib/fmt/issues/1896,
  https://github.com/fmtlib/fmt/pull/1897,
  https://github.com/fmtlib/fmt/pull/1898,
  https://github.com/fmtlib/fmt/issues/1904,
  https://github.com/fmtlib/fmt/pull/1908,
  https://github.com/fmtlib/fmt/issues/1911,
  https://github.com/fmtlib/fmt/issues/1912,
  https://github.com/fmtlib/fmt/issues/1928,
  https://github.com/fmtlib/fmt/pull/1929,
  https://github.com/fmtlib/fmt/issues/1935,
  https://github.com/fmtlib/fmt/pull/1937,
  https://github.com/fmtlib/fmt/pull/1942,
  https://github.com/fmtlib/fmt/issues/1949).
  Thanks @TheQwertiest, @medithe, @martinwuehrer, @n16h7hunt3r, @Othereum,
  @gsjaardema, @AlexanderLanin, @gcerretani, @chronoxor, @noizefloor,
  @akohlmey, @jk-jeon, @rimathia, @rglarix, @moiwi, @heckad, @MarcDirven.
  @BartSiwek and @darklukee.

# 7.0.3 - 2020-08-06

- Worked around broken `numeric_limits` for 128-bit integers
  (https://github.com/fmtlib/fmt/issues/1787).
- Added error reporting on missing named arguments
  (https://github.com/fmtlib/fmt/issues/1796).
- Stopped using 128-bit integers with clang-cl
  (https://github.com/fmtlib/fmt/pull/1800). Thanks @Kingcom.
- Fixed issues in locale-specific integer formatting
  (https://github.com/fmtlib/fmt/issues/1782,
  https://github.com/fmtlib/fmt/issues/1801).

# 7.0.2 - 2020-07-29

- Worked around broken `numeric_limits` for 128-bit integers
  (https://github.com/fmtlib/fmt/issues/1725).
- Fixed compatibility with CMake 3.4
  (https://github.com/fmtlib/fmt/issues/1779).
- Fixed handling of digit separators in locale-specific formatting
  (https://github.com/fmtlib/fmt/issues/1782).

# 7.0.1 - 2020-07-07

- Updated the inline version namespace name.
- Worked around a gcc bug in mangling of alias templates
  (https://github.com/fmtlib/fmt/issues/1753).
- Fixed a linkage error on Windows
  (https://github.com/fmtlib/fmt/issues/1757). Thanks @Kurkin.
- Fixed minor issues with the documentation.

# 7.0.0 - 2020-07-05

- Reduced the library size. For example, on macOS a stripped test
  binary statically linked with {fmt} [shrank from \~368k to less than
  100k](http://www.zverovich.net/2020/05/21/reducing-library-size.html).

- Added a simpler and more efficient [format string compilation
  API](https://fmt.dev/7.0.0/api.html#compile-api):

  ```c++
  #include <fmt/compile.h>

  // Converts 42 into std::string using the most efficient method and no
  // runtime format string processing.
  std::string s = fmt::format(FMT_COMPILE("{}"), 42);
  ```

  The old `fmt::compile` API is now deprecated.

- Optimized integer formatting: `format_to` with format string
  compilation and a stack-allocated buffer is now [faster than
  to_chars on both libc++ and
  libstdc++](http://www.zverovich.net/2020/06/13/fast-int-to-string-revisited.html).

- Optimized handling of small format strings. For example,

  ```c++
  fmt::format("Result: {}: ({},{},{},{})", str1, str2, str3, str4, str5)
  ```

  is now \~40% faster
  (https://github.com/fmtlib/fmt/issues/1685).

- Applied extern templates to improve compile times when using the
  core API and `fmt/format.h`
  (https://github.com/fmtlib/fmt/issues/1452). For example,
  on macOS with clang the compile time of a test translation unit
  dropped from 2.3s to 0.3s with `-O2` and from 0.6s to 0.3s with the
  default settings (`-O0`).

  Before (`-O2`):

      % time c++ -c test.cc -I include -std=c++17 -O2
      c++ -c test.cc -I include -std=c++17 -O2  2.22s user 0.08s system 99% cpu 2.311 total

  After (`-O2`):

      % time c++ -c test.cc -I include -std=c++17 -O2
      c++ -c test.cc -I include -std=c++17 -O2  0.26s user 0.04s system 98% cpu 0.303 total

  Before (default):

      % time c++ -c test.cc -I include -std=c++17
      c++ -c test.cc -I include -std=c++17  0.53s user 0.06s system 98% cpu 0.601 total

  After (default):

      % time c++ -c test.cc -I include -std=c++17
      c++ -c test.cc -I include -std=c++17  0.24s user 0.06s system 98% cpu 0.301 total

  It is still recommended to use `fmt/core.h` instead of
  `fmt/format.h` but the compile time difference is now smaller.
  Thanks @alex3d for the suggestion.

- Named arguments are now stored on stack (no dynamic memory
  allocations) and the compiled code is more compact and efficient.
  For example

  ```c++
  #include <fmt/core.h>

  int main() {
    fmt::print("The answer is {answer}\n", fmt::arg("answer", 42));
  }
  ```

  compiles to just ([godbolt](https://godbolt.org/z/NcfEp_))

  ```asm
  .LC0:
          .string "answer"
  .LC1:
          .string "The answer is {answer}\n"
  main:
          sub     rsp, 56
          mov     edi, OFFSET FLAT:.LC1
          mov     esi, 23
          movabs  rdx, 4611686018427387905
          lea     rax, [rsp+32]
          lea     rcx, [rsp+16]
          mov     QWORD PTR [rsp+8], 1
          mov     QWORD PTR [rsp], rax
          mov     DWORD PTR [rsp+16], 42
          mov     QWORD PTR [rsp+32], OFFSET FLAT:.LC0
          mov     DWORD PTR [rsp+40], 0
          call    fmt::v6::vprint(fmt::v6::basic_string_view<char>,
                                  fmt::v6::format_args)
          xor     eax, eax
          add     rsp, 56
          ret

      .L.str.1:
              .asciz  "answer"
  ```

- Implemented compile-time checks for dynamic width and precision
  (https://github.com/fmtlib/fmt/issues/1614):

  ```c++
  #include <fmt/format.h>

  int main() {
    fmt::print(FMT_STRING("{0:{1}}"), 42);
  }
  ```

  now gives a compilation error because argument 1 doesn\'t exist:

      In file included from test.cc:1:
      include/fmt/format.h:2726:27: error: constexpr variable 'invalid_format' must be
      initialized by a constant expression
        FMT_CONSTEXPR_DECL bool invalid_format =
                                ^
      ...
      include/fmt/core.h:569:26: note: in call to
      '&checker(s, {}).context_->on_error(&"argument not found"[0])'
          if (id >= num_args_) on_error("argument not found");
                              ^

- Added sentinel support to `fmt::join`
  (https://github.com/fmtlib/fmt/pull/1689)

  ```c++
  struct zstring_sentinel {};
  bool operator==(const char* p, zstring_sentinel) { return *p == '\0'; }
  bool operator!=(const char* p, zstring_sentinel) { return *p != '\0'; }

  struct zstring {
    const char* p;
    const char* begin() const { return p; }
    zstring_sentinel end() const { return {}; }
  };

  auto s = fmt::format("{}", fmt::join(zstring{"hello"}, "_"));
  // s == "h_e_l_l_o"
  ```

  Thanks @BRevzin.

- Added support for named arguments, `clear` and `reserve` to
  `dynamic_format_arg_store`
  (https://github.com/fmtlib/fmt/issues/1655,
  https://github.com/fmtlib/fmt/pull/1663,
  https://github.com/fmtlib/fmt/pull/1674,
  https://github.com/fmtlib/fmt/pull/1677). Thanks @vsolontsov-ll.

- Added support for the `'c'` format specifier to integral types for
  compatibility with `std::format`
  (https://github.com/fmtlib/fmt/issues/1652).

- Replaced the `'n'` format specifier with `'L'` for compatibility
  with `std::format`
  (https://github.com/fmtlib/fmt/issues/1624). The `'n'`
  specifier can be enabled via the `FMT_DEPRECATED_N_SPECIFIER` macro.

- The `'='` format specifier is now disabled by default for
  compatibility with `std::format`. It can be enabled via the
  `FMT_DEPRECATED_NUMERIC_ALIGN` macro.

- Removed the following deprecated APIs:

  -   `FMT_STRING_ALIAS` and `fmt` macros - replaced by `FMT_STRING`
  -   `fmt::basic_string_view::char_type` - replaced by
      `fmt::basic_string_view::value_type`
  -   `convert_to_int`
  -   `format_arg_store::types`
  -   `*parse_context` - replaced by `*format_parse_context`
  -   `FMT_DEPRECATED_INCLUDE_OS`
  -   `FMT_DEPRECATED_PERCENT` - incompatible with `std::format`
  -   `*writer` - replaced by compiled format API

- Renamed the `internal` namespace to `detail`
  (https://github.com/fmtlib/fmt/issues/1538). The former is
  still provided as an alias if the `FMT_USE_INTERNAL` macro is
  defined.

- Improved compatibility between `fmt::printf` with the standard specs
  (https://github.com/fmtlib/fmt/issues/1595,
  https://github.com/fmtlib/fmt/pull/1682,
  https://github.com/fmtlib/fmt/pull/1683,
  https://github.com/fmtlib/fmt/pull/1687,
  https://github.com/fmtlib/fmt/pull/1699). Thanks @rimathia.

- Fixed handling of `operator<<` overloads that use `copyfmt`
  (https://github.com/fmtlib/fmt/issues/1666).

- Added the `FMT_OS` CMake option to control inclusion of OS-specific
  APIs in the fmt target. This can be useful for embedded platforms
  (https://github.com/fmtlib/fmt/issues/1654,
  https://github.com/fmtlib/fmt/pull/1656). Thanks @kwesolowski.

- Replaced `FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION` with the
  `FMT_FUZZ` macro to prevent interfering with fuzzing of projects
  using {fmt} (https://github.com/fmtlib/fmt/pull/1650).
  Thanks @asraa.

- Fixed compatibility with emscripten
  (https://github.com/fmtlib/fmt/issues/1636,
  https://github.com/fmtlib/fmt/pull/1637). Thanks @ArthurSonzogni.

- Improved documentation
  (https://github.com/fmtlib/fmt/issues/704,
  https://github.com/fmtlib/fmt/pull/1643,
  https://github.com/fmtlib/fmt/pull/1660,
  https://github.com/fmtlib/fmt/pull/1681,
  https://github.com/fmtlib/fmt/pull/1691,
  https://github.com/fmtlib/fmt/pull/1706,
  https://github.com/fmtlib/fmt/pull/1714,
  https://github.com/fmtlib/fmt/pull/1721,
  https://github.com/fmtlib/fmt/pull/1739,
  https://github.com/fmtlib/fmt/pull/1740,
  https://github.com/fmtlib/fmt/pull/1741,
  https://github.com/fmtlib/fmt/pull/1751).
  Thanks @senior7515, @lsr0, @puetzk, @fpelliccioni, Alexey Kuzmenko, @jelly,
  @claremacrae, @jiapengwen, @gsjaardema and @alexey-milovidov.

- Implemented various build configuration fixes and improvements
  (https://github.com/fmtlib/fmt/pull/1603,
  https://github.com/fmtlib/fmt/pull/1657,
  https://github.com/fmtlib/fmt/pull/1702,
  https://github.com/fmtlib/fmt/pull/1728).
  Thanks @scramsby, @jtojnar, @orivej and @flagarde.

- Fixed various warnings and compilation issues
  (https://github.com/fmtlib/fmt/pull/1616,
  https://github.com/fmtlib/fmt/issues/1620,
  https://github.com/fmtlib/fmt/issues/1622,
  https://github.com/fmtlib/fmt/issues/1625,
  https://github.com/fmtlib/fmt/pull/1627,
  https://github.com/fmtlib/fmt/issues/1628,
  https://github.com/fmtlib/fmt/pull/1629,
  https://github.com/fmtlib/fmt/issues/1631,
  https://github.com/fmtlib/fmt/pull/1633,
  https://github.com/fmtlib/fmt/pull/1649,
  https://github.com/fmtlib/fmt/issues/1658,
  https://github.com/fmtlib/fmt/pull/1661,
  https://github.com/fmtlib/fmt/pull/1667,
  https://github.com/fmtlib/fmt/issues/1668,
  https://github.com/fmtlib/fmt/pull/1669,
  https://github.com/fmtlib/fmt/issues/1692,
  https://github.com/fmtlib/fmt/pull/1696,
  https://github.com/fmtlib/fmt/pull/1697,
  https://github.com/fmtlib/fmt/issues/1707,
  https://github.com/fmtlib/fmt/pull/1712,
  https://github.com/fmtlib/fmt/pull/1716,
  https://github.com/fmtlib/fmt/pull/1722,
  https://github.com/fmtlib/fmt/issues/1724,
  https://github.com/fmtlib/fmt/pull/1729,
  https://github.com/fmtlib/fmt/pull/1738,
  https://github.com/fmtlib/fmt/issues/1742,
  https://github.com/fmtlib/fmt/issues/1743,
  https://github.com/fmtlib/fmt/pull/1744,
  https://github.com/fmtlib/fmt/issues/1747,
  https://github.com/fmtlib/fmt/pull/1750).
  Thanks @gsjaardema, @gabime, @johnor, @Kurkin, @invexed, @peterbell10,
  @daixtrose, @petrutlucian94, @Neargye, @ambitslix, @gabime,
  @tohammer and @0x8000-0000.

# 6.2.1 - 2020-05-09

- Fixed ostream support in `sprintf`
  (https://github.com/fmtlib/fmt/issues/1631).
- Fixed type detection when using implicit conversion to `string_view`
  and ostream `operator<<` inconsistently
  (https://github.com/fmtlib/fmt/issues/1662).

# 6.2.0 - 2020-04-05

- Improved error reporting when trying to format an object of a
  non-formattable type:

  ```c++
  fmt::format("{}", S());
  ```

  now gives:

      include/fmt/core.h:1015:5: error: static_assert failed due to requirement
      'formattable' "Cannot format argument. To make type T formattable provide a
      formatter<T> specialization:
      https://fmt.dev/latest/api.html#formatting-user-defined-types"
          static_assert(
          ^
      ...
      note: in instantiation of function template specialization
      'fmt::v6::format<char [3], S, char>' requested here
        fmt::format("{}", S());
             ^

  if `S` is not formattable.

- Reduced the library size by \~10%.

- Always print decimal point if `#` is specified
  (https://github.com/fmtlib/fmt/issues/1476,
  https://github.com/fmtlib/fmt/issues/1498):

  ```c++
  fmt::print("{:#.0f}", 42.0);
  ```

  now prints `42.`

- Implemented the `'L'` specifier for locale-specific numeric
  formatting to improve compatibility with `std::format`. The `'n'`
  specifier is now deprecated and will be removed in the next major
  release.

- Moved OS-specific APIs such as `windows_error` from `fmt/format.h`
  to `fmt/os.h`. You can define `FMT_DEPRECATED_INCLUDE_OS` to
  automatically include `fmt/os.h` from `fmt/format.h` for
  compatibility but this will be disabled in the next major release.

- Added precision overflow detection in floating-point formatting.

- Implemented detection of invalid use of `fmt::arg`.

- Used `type_identity` to block unnecessary template argument
  deduction. Thanks Tim Song.

- Improved UTF-8 handling
  (https://github.com/fmtlib/fmt/issues/1109):

  ```c++
  fmt::print("┌{0:─^{2}}┐\n"
             "│{1: ^{2}}│\n"
             "└{0:─^{2}}┘\n", "", "Прывітанне, свет!", 21);
  ```

  now prints:

      ┌─────────────────────┐
      │  Прывітанне, свет!  │
      └─────────────────────┘

  on systems that support Unicode.

- Added experimental dynamic argument storage
  (https://github.com/fmtlib/fmt/issues/1170,
  https://github.com/fmtlib/fmt/pull/1584):

  ```c++
  fmt::dynamic_format_arg_store<fmt::format_context> store;
  store.push_back("answer");
  store.push_back(42);
  fmt::vprint("The {} is {}.\n", store);
  ```

  prints:

      The answer is 42.

  Thanks @vsolontsov-ll.

- Made `fmt::join` accept `initializer_list`
  (https://github.com/fmtlib/fmt/pull/1591). Thanks @Rapotkinnik.

- Fixed handling of empty tuples
  (https://github.com/fmtlib/fmt/issues/1588).

- Fixed handling of output iterators in `format_to_n`
  (https://github.com/fmtlib/fmt/issues/1506).

- Fixed formatting of `std::chrono::duration` types to wide output
  (https://github.com/fmtlib/fmt/pull/1533). Thanks @zeffy.

- Added const `begin` and `end` overload to buffers
  (https://github.com/fmtlib/fmt/pull/1553). Thanks @dominicpoeschko.

- Added the ability to disable floating-point formatting via
  `FMT_USE_FLOAT`, `FMT_USE_DOUBLE` and `FMT_USE_LONG_DOUBLE` macros
  for extremely memory-constrained embedded system
  (https://github.com/fmtlib/fmt/pull/1590). Thanks @albaguirre.

- Made `FMT_STRING` work with `constexpr` `string_view`
  (https://github.com/fmtlib/fmt/pull/1589). Thanks @scramsby.

- Implemented a minor optimization in the format string parser
  (https://github.com/fmtlib/fmt/pull/1560). Thanks @IkarusDeveloper.

- Improved attribute detection
  (https://github.com/fmtlib/fmt/pull/1469,
  https://github.com/fmtlib/fmt/pull/1475,
  https://github.com/fmtlib/fmt/pull/1576).
  Thanks @federico-busato, @chronoxor and @refnum.

- Improved documentation
  (https://github.com/fmtlib/fmt/pull/1481,
  https://github.com/fmtlib/fmt/pull/1523).
  Thanks @JackBoosY and @imba-tjd.

- Fixed symbol visibility on Linux when compiling with
  `-fvisibility=hidden`
  (https://github.com/fmtlib/fmt/pull/1535). Thanks @milianw.

- Implemented various build configuration fixes and improvements
  (https://github.com/fmtlib/fmt/issues/1264,
  https://github.com/fmtlib/fmt/issues/1460,
  https://github.com/fmtlib/fmt/pull/1534,
  https://github.com/fmtlib/fmt/issues/1536,
  https://github.com/fmtlib/fmt/issues/1545,
  https://github.com/fmtlib/fmt/pull/1546,
  https://github.com/fmtlib/fmt/issues/1566,
  https://github.com/fmtlib/fmt/pull/1582,
  https://github.com/fmtlib/fmt/issues/1597,
  https://github.com/fmtlib/fmt/pull/1598).
  Thanks @ambitslix, @jwillikers and @stac47.

- Fixed various warnings and compilation issues
  (https://github.com/fmtlib/fmt/pull/1433,
  https://github.com/fmtlib/fmt/issues/1461,
  https://github.com/fmtlib/fmt/pull/1470,
  https://github.com/fmtlib/fmt/pull/1480,
  https://github.com/fmtlib/fmt/pull/1485,
  https://github.com/fmtlib/fmt/pull/1492,
  https://github.com/fmtlib/fmt/issues/1493,
  https://github.com/fmtlib/fmt/issues/1504,
  https://github.com/fmtlib/fmt/pull/1505,
  https://github.com/fmtlib/fmt/pull/1512,
  https://github.com/fmtlib/fmt/issues/1515,
  https://github.com/fmtlib/fmt/pull/1516,
  https://github.com/fmtlib/fmt/pull/1518,
  https://github.com/fmtlib/fmt/pull/1519,
  https://github.com/fmtlib/fmt/pull/1520,
  https://github.com/fmtlib/fmt/pull/1521,
  https://github.com/fmtlib/fmt/pull/1522,
  https://github.com/fmtlib/fmt/issues/1524,
  https://github.com/fmtlib/fmt/pull/1530,
  https://github.com/fmtlib/fmt/issues/1531,
  https://github.com/fmtlib/fmt/pull/1532,
  https://github.com/fmtlib/fmt/issues/1539,
  https://github.com/fmtlib/fmt/issues/1547,
  https://github.com/fmtlib/fmt/issues/1548,
  https://github.com/fmtlib/fmt/pull/1554,
  https://github.com/fmtlib/fmt/issues/1567,
  https://github.com/fmtlib/fmt/pull/1568,
  https://github.com/fmtlib/fmt/pull/1569,
  https://github.com/fmtlib/fmt/pull/1571,
  https://github.com/fmtlib/fmt/pull/1573,
  https://github.com/fmtlib/fmt/pull/1575,
  https://github.com/fmtlib/fmt/pull/1581,
  https://github.com/fmtlib/fmt/issues/1583,
  https://github.com/fmtlib/fmt/issues/1586,
  https://github.com/fmtlib/fmt/issues/1587,
  https://github.com/fmtlib/fmt/issues/1594,
  https://github.com/fmtlib/fmt/pull/1596,
  https://github.com/fmtlib/fmt/issues/1604,
  https://github.com/fmtlib/fmt/pull/1606,
  https://github.com/fmtlib/fmt/issues/1607,
  https://github.com/fmtlib/fmt/issues/1609).
  Thanks @marti4d, @iPherian, @parkertomatoes, @gsjaardema, @chronoxor,
  @DanielaE, @torsten48, @tohammer, @lefticus, @ryusakki, @adnsv, @fghzxm,
  @refnum, @pramodk, @Spirrwell and @scramsby.

# 6.1.2 - 2019-12-11

- Fixed ABI compatibility with `libfmt.so.6.0.0`
  (https://github.com/fmtlib/fmt/issues/1471).
- Fixed handling types convertible to `std::string_view`
  (https://github.com/fmtlib/fmt/pull/1451). Thanks @denizevrenci.
- Made CUDA test an opt-in enabled via the `FMT_CUDA_TEST` CMake
  option.
- Fixed sign conversion warnings
  (https://github.com/fmtlib/fmt/pull/1440). Thanks @0x8000-0000.

# 6.1.1 - 2019-12-04

- Fixed shared library build on Windows
  (https://github.com/fmtlib/fmt/pull/1443,
  https://github.com/fmtlib/fmt/issues/1445,
  https://github.com/fmtlib/fmt/pull/1446,
  https://github.com/fmtlib/fmt/issues/1450).
  Thanks @egorpugin and @bbolli.
- Added a missing decimal point in exponent notation with trailing
  zeros.
- Removed deprecated `format_arg_store::TYPES`.

# 6.1.0 - 2019-12-01

- {fmt} now formats IEEE 754 `float` and `double` using the shortest
  decimal representation with correct rounding by default:

  ```c++
  #include <cmath>
  #include <fmt/core.h>

  int main() {
    fmt::print("{}", M_PI);
  }
  ```

  prints `3.141592653589793`.

- Made the fast binary to decimal floating-point formatter the
  default, simplified it and improved performance. {fmt} is now 15
  times faster than libc++\'s `std::ostringstream`, 11 times faster
  than `printf` and 10% faster than double-conversion on
  [dtoa-benchmark](https://github.com/fmtlib/dtoa-benchmark):

  | Function      | Time (ns) | Speedup |
  | ------------- | --------: | ------: |
  | ostringstream | 1,346.30  | 1.00x   |
  | ostrstream    | 1,195.74  | 1.13x   |
  | sprintf       | 995.08    | 1.35x   |
  | doubleconv    | 99.10     | 13.59x  |
  | fmt           | 88.34     | 15.24x  |

  ![](https://user-images.githubusercontent.com/576385/69767160-cdaca400-112f-11ea-9fc5-347c9f83caad.png)

- {fmt} no longer converts `float` arguments to `double`. In
  particular this improves the default (shortest) representation of
  floats and makes `fmt::format` consistent with `std::format` specs
  (https://github.com/fmtlib/fmt/issues/1336,
  https://github.com/fmtlib/fmt/issues/1353,
  https://github.com/fmtlib/fmt/pull/1360,
  https://github.com/fmtlib/fmt/pull/1361):

  ```c++
  fmt::print("{}", 0.1f);
  ```

  prints `0.1` instead of `0.10000000149011612`.

  Thanks @orivej.

- Made floating-point formatting output consistent with
  `printf`/iostreams
  (https://github.com/fmtlib/fmt/issues/1376,
  https://github.com/fmtlib/fmt/issues/1417).

- Added support for 128-bit integers
  (https://github.com/fmtlib/fmt/pull/1287):

  ```c++
  fmt::print("{}", std::numeric_limits<__int128_t>::max());
  ```

  prints `170141183460469231731687303715884105727`.

  Thanks @denizevrenci.

- The overload of `print` that takes `text_style` is now atomic, i.e.
  the output from different threads doesn\'t interleave
  (https://github.com/fmtlib/fmt/pull/1351). Thanks @tankiJong.

- Made compile time in the header-only mode \~20% faster by reducing
  the number of template instantiations. `wchar_t` overload of
  `vprint` was moved from `fmt/core.h` to `fmt/format.h`.

- Added an overload of `fmt::join` that works with tuples
  (https://github.com/fmtlib/fmt/issues/1322,
  https://github.com/fmtlib/fmt/pull/1330):

  ```c++
  #include <tuple>
  #include <fmt/ranges.h>

  int main() {
    std::tuple<char, int, float> t{'a', 1, 2.0f};
    fmt::print("{}", t);
  }
  ```

  prints `('a', 1, 2.0)`.

  Thanks @jeremyong.

- Changed formatting of octal zero with prefix from \"00\" to \"0\":

  ```c++
  fmt::print("{:#o}", 0);
  ```

  prints `0`.

- The locale is now passed to ostream insertion (`<<`) operators
  (https://github.com/fmtlib/fmt/pull/1406):

  ```c++
  #include <fmt/locale.h>
  #include <fmt/ostream.h>

  struct S {
    double value;
  };

  std::ostream& operator<<(std::ostream& os, S s) {
    return os << s.value;
  }

  int main() {
    auto s = fmt::format(std::locale("fr_FR.UTF-8"), "{}", S{0.42});
    // s == "0,42"
  }
  ```

  Thanks @dlaugt.

- Locale-specific number formatting now uses grouping
  (https://github.com/fmtlib/fmt/issues/1393,
  https://github.com/fmtlib/fmt/pull/1394). Thanks @skrdaniel.

- Fixed handling of types with deleted implicit rvalue conversion to
  `const char**` (https://github.com/fmtlib/fmt/issues/1421):

  ```c++
  struct mystring {
    operator const char*() const&;
    operator const char*() &;
    operator const char*() const&& = delete;
    operator const char*() && = delete;
  };
  mystring str;
  fmt::print("{}", str); // now compiles
  ```

- Enums are now mapped to correct underlying types instead of `int`
  (https://github.com/fmtlib/fmt/pull/1286). Thanks @agmt.

- Enum classes are no longer implicitly converted to `int`
  (https://github.com/fmtlib/fmt/issues/1424).

- Added `basic_format_parse_context` for consistency with C++20
  `std::format` and deprecated `basic_parse_context`.

- Fixed handling of UTF-8 in precision
  (https://github.com/fmtlib/fmt/issues/1389,
  https://github.com/fmtlib/fmt/pull/1390). Thanks @tajtiattila.

- {fmt} can now be installed on Linux, macOS and Windows with
  [Conda](https://docs.conda.io/en/latest/) using its
  [conda-forge](https://conda-forge.org)
  [package](https://github.com/conda-forge/fmt-feedstock)
  (https://github.com/fmtlib/fmt/pull/1410):

      conda install -c conda-forge fmt

  Thanks @tdegeus.

- Added a CUDA test (https://github.com/fmtlib/fmt/pull/1285,
  https://github.com/fmtlib/fmt/pull/1317).
  Thanks @luncliff and @risa2000.

- Improved documentation
  (https://github.com/fmtlib/fmt/pull/1276,
  https://github.com/fmtlib/fmt/issues/1291,
  https://github.com/fmtlib/fmt/issues/1296,
  https://github.com/fmtlib/fmt/pull/1315,
  https://github.com/fmtlib/fmt/pull/1332,
  https://github.com/fmtlib/fmt/pull/1337,
  https://github.com/fmtlib/fmt/issues/1395
  https://github.com/fmtlib/fmt/pull/1418).
  Thanks @waywardmonkeys, @pauldreik and @jackoalan.

- Various code improvements
  (https://github.com/fmtlib/fmt/pull/1358,
  https://github.com/fmtlib/fmt/pull/1407).
  Thanks @orivej and @dpacbach.

- Fixed compile-time format string checks for user-defined types
  (https://github.com/fmtlib/fmt/issues/1292).

- Worked around a false positive in `unsigned-integer-overflow` sanitizer
  (https://github.com/fmtlib/fmt/issues/1377).

- Fixed various warnings and compilation issues
  (https://github.com/fmtlib/fmt/issues/1273,
  https://github.com/fmtlib/fmt/pull/1278,
  https://github.com/fmtlib/fmt/pull/1280,
  https://github.com/fmtlib/fmt/issues/1281,
  https://github.com/fmtlib/fmt/issues/1288,
  https://github.com/fmtlib/fmt/pull/1290,
  https://github.com/fmtlib/fmt/pull/1301,
  https://github.com/fmtlib/fmt/issues/1305,
  https://github.com/fmtlib/fmt/issues/1306,
  https://github.com/fmtlib/fmt/issues/1309,
  https://github.com/fmtlib/fmt/pull/1312,
  https://github.com/fmtlib/fmt/issues/1313,
  https://github.com/fmtlib/fmt/issues/1316,
  https://github.com/fmtlib/fmt/issues/1319,
  https://github.com/fmtlib/fmt/pull/1320,
  https://github.com/fmtlib/fmt/pull/1326,
  https://github.com/fmtlib/fmt/pull/1328,
  https://github.com/fmtlib/fmt/issues/1344,
  https://github.com/fmtlib/fmt/pull/1345,
  https://github.com/fmtlib/fmt/pull/1347,
  https://github.com/fmtlib/fmt/pull/1349,
  https://github.com/fmtlib/fmt/issues/1354,
  https://github.com/fmtlib/fmt/issues/1362,
  https://github.com/fmtlib/fmt/issues/1366,
  https://github.com/fmtlib/fmt/pull/1364,
  https://github.com/fmtlib/fmt/pull/1370,
  https://github.com/fmtlib/fmt/pull/1371,
  https://github.com/fmtlib/fmt/issues/1385,
  https://github.com/fmtlib/fmt/issues/1388,
  https://github.com/fmtlib/fmt/pull/1397,
  https://github.com/fmtlib/fmt/pull/1414,
  https://github.com/fmtlib/fmt/pull/1416,
  https://github.com/fmtlib/fmt/issues/1422
  https://github.com/fmtlib/fmt/pull/1427,
  https://github.com/fmtlib/fmt/issues/1431,
  https://github.com/fmtlib/fmt/pull/1433).
  Thanks @hhb, @gsjaardema, @gabime, @neheb, @vedranmiletic, @dkavolis,
  @mwinterb, @orivej, @denizevrenci, @leonklingele, @chronoxor, @kent-tri,
  @0x8000-0000 and @marti4d.

# 6.0.0 - 2019-08-26

- Switched to the [MIT license](
  https://github.com/fmtlib/fmt/blob/5a4b24613ba16cc689977c3b5bd8274a3ba1dd1f/LICENSE.rst)
  with an optional exception that allows distributing binary code
  without attribution.

- Floating-point formatting is now locale-independent by default:

  ```c++
  #include <locale>
  #include <fmt/core.h>

  int main() {
    std::locale::global(std::locale("ru_RU.UTF-8"));
    fmt::print("value = {}", 4.2);
  }
  ```

  prints \"value = 4.2\" regardless of the locale.

  For locale-specific formatting use the `n` specifier:

  ```c++
  std::locale::global(std::locale("ru_RU.UTF-8"));
  fmt::print("value = {:n}", 4.2);
  ```

  prints \"value = 4,2\".

- Added an experimental Grisu floating-point formatting algorithm
  implementation (disabled by default). To enable it compile with the
  `FMT_USE_GRISU` macro defined to 1:

  ```c++
  #define FMT_USE_GRISU 1
  #include <fmt/format.h>

  auto s = fmt::format("{}", 4.2); // formats 4.2 using Grisu
  ```

  With Grisu enabled, {fmt} is 13x faster than `std::ostringstream`
  (libc++) and 10x faster than `sprintf` on
  [dtoa-benchmark](https://github.com/fmtlib/dtoa-benchmark) ([full
  results](https://fmt.dev/unknown_mac64_clang10.0.html)):

  ![](https://user-images.githubusercontent.com/576385/54883977-9fe8c000-4e28-11e9-8bde-272d122e7c52.jpg)

- Separated formatting and parsing contexts for consistency with
  [C++20 std::format](http://eel.is/c++draft/format), removing the
  undocumented `basic_format_context::parse_context()` function.

- Added [oss-fuzz](https://github.com/google/oss-fuzz) support
  (https://github.com/fmtlib/fmt/pull/1199). Thanks @pauldreik.

- `formatter` specializations now always take precedence over
  `operator<<` (https://github.com/fmtlib/fmt/issues/952):

  ```c++
  #include <iostream>
  #include <fmt/ostream.h>

  struct S {};

  std::ostream& operator<<(std::ostream& os, S) {
    return os << 1;
  }

  template <>
  struct fmt::formatter<S> : fmt::formatter<int> {
    auto format(S, format_context& ctx) {
      return formatter<int>::format(2, ctx);
    }
  };

  int main() {
    std::cout << S() << "\n"; // prints 1 using operator<<
    fmt::print("{}\n", S());  // prints 2 using formatter
  }
  ```

- Introduced the experimental `fmt::compile` function that does format
  string compilation
  (https://github.com/fmtlib/fmt/issues/618,
  https://github.com/fmtlib/fmt/issues/1169,
  https://github.com/fmtlib/fmt/pull/1171):

  ```c++
  #include <fmt/compile.h>

  auto f = fmt::compile<int>("{}");
  std::string s = fmt::format(f, 42); // can be called multiple times to
                                      // format different values
  // s == "42"
  ```

  It moves the cost of parsing a format string outside of the format
  function which can be beneficial when identically formatting many
  objects of the same types. Thanks @stryku.

- Added experimental `%` format specifier that formats floating-point
  values as percentages
  (https://github.com/fmtlib/fmt/pull/1060,
  https://github.com/fmtlib/fmt/pull/1069,
  https://github.com/fmtlib/fmt/pull/1071):

  ```c++
  auto s = fmt::format("{:.1%}", 0.42); // s == "42.0%"
  ```

  Thanks @gawain-bolton.

- Implemented precision for floating-point durations
  (https://github.com/fmtlib/fmt/issues/1004,
  https://github.com/fmtlib/fmt/pull/1012):

  ```c++
  auto s = fmt::format("{:.1}", std::chrono::duration<double>(1.234));
  // s == 1.2s
  ```

  Thanks @DanielaE.

- Implemented `chrono` format specifiers `%Q` and `%q` that give the
  value and the unit respectively
  (https://github.com/fmtlib/fmt/pull/1019):

  ```c++
  auto value = fmt::format("{:%Q}", 42s); // value == "42"
  auto unit  = fmt::format("{:%q}", 42s); // unit == "s"
  ```

  Thanks @DanielaE.

- Fixed handling of dynamic width in chrono formatter:

  ```c++
  auto s = fmt::format("{0:{1}%H:%M:%S}", std::chrono::seconds(12345), 12);
  //                        ^ width argument index                     ^ width
  // s == "03:25:45    "
  ```

  Thanks Howard Hinnant.

- Removed deprecated `fmt/time.h`. Use `fmt/chrono.h` instead.

- Added `fmt::format` and `fmt::vformat` overloads that take
  `text_style` (https://github.com/fmtlib/fmt/issues/993,
  https://github.com/fmtlib/fmt/pull/994):

  ```c++
  #include <fmt/color.h>

  std::string message = fmt::format(fmt::emphasis::bold | fg(fmt::color::red),
                                    "The answer is {}.", 42);
  ```

  Thanks @Naios.

- Removed the deprecated color API (`print_colored`). Use the new API,
  namely `print` overloads that take `text_style` instead.

- Made `std::unique_ptr` and `std::shared_ptr` formattable as pointers
  via `fmt::ptr` (https://github.com/fmtlib/fmt/pull/1121):

  ```c++
  std::unique_ptr<int> p = ...;
  fmt::print("{}", fmt::ptr(p)); // prints p as a pointer
  ```

  Thanks @sighingnow.

- Made `print` and `vprint` report I/O errors
  (https://github.com/fmtlib/fmt/issues/1098,
  https://github.com/fmtlib/fmt/pull/1099). Thanks @BillyDonahue.

- Marked deprecated APIs with the `[[deprecated]]` attribute and
  removed internal uses of deprecated APIs
  (https://github.com/fmtlib/fmt/pull/1022). Thanks @eliaskosunen.

- Modernized the codebase using more C++11 features and removing
  workarounds. Most importantly, `buffer_context` is now an alias
  template, so use `buffer_context<T>` instead of
  `buffer_context<T>::type`. These features require GCC 4.8 or later.

- `formatter` specializations now always take precedence over implicit
  conversions to `int` and the undocumented `convert_to_int` trait is
  now deprecated.

- Moved the undocumented `basic_writer`, `writer`, and `wwriter` types
  to the `internal` namespace.

- Removed deprecated `basic_format_context::begin()`. Use `out()`
  instead.

- Disallowed passing the result of `join` as an lvalue to prevent
  misuse.

- Refactored the undocumented structs that represent parsed format
  specifiers to simplify the API and allow multibyte fill.

- Moved SFINAE to template parameters to reduce symbol sizes.

- Switched to `fputws` for writing wide strings so that it\'s no
  longer required to call `_setmode` on Windows
  (https://github.com/fmtlib/fmt/issues/1229,
  https://github.com/fmtlib/fmt/pull/1243). Thanks @jackoalan.

- Improved literal-based API
  (https://github.com/fmtlib/fmt/pull/1254). Thanks @sylveon.

- Added support for exotic platforms without `uintptr_t` such as IBM i
  (AS/400) which has 128-bit pointers and only 64-bit integers
  (https://github.com/fmtlib/fmt/issues/1059).

- Added [Sublime Text syntax highlighting config](
  https://github.com/fmtlib/fmt/blob/master/support/C%2B%2B.sublime-syntax)
  (https://github.com/fmtlib/fmt/issues/1037). Thanks @Kronuz.

- Added the `FMT_ENFORCE_COMPILE_STRING` macro to enforce the use of
  compile-time format strings
  (https://github.com/fmtlib/fmt/pull/1231). Thanks @jackoalan.

- Stopped setting `CMAKE_BUILD_TYPE` if {fmt} is a subproject
  (https://github.com/fmtlib/fmt/issues/1081).

- Various build improvements
  (https://github.com/fmtlib/fmt/pull/1039,
  https://github.com/fmtlib/fmt/pull/1078,
  https://github.com/fmtlib/fmt/pull/1091,
  https://github.com/fmtlib/fmt/pull/1103,
  https://github.com/fmtlib/fmt/pull/1177).
  Thanks @luncliff, @jasonszang, @olafhering, @Lecetem and @pauldreik.

- Improved documentation
  (https://github.com/fmtlib/fmt/issues/1049,
  https://github.com/fmtlib/fmt/pull/1051,
  https://github.com/fmtlib/fmt/pull/1083,
  https://github.com/fmtlib/fmt/pull/1113,
  https://github.com/fmtlib/fmt/pull/1114,
  https://github.com/fmtlib/fmt/issues/1146,
  https://github.com/fmtlib/fmt/issues/1180,
  https://github.com/fmtlib/fmt/pull/1250,
  https://github.com/fmtlib/fmt/pull/1252,
  https://github.com/fmtlib/fmt/pull/1265).
  Thanks @mikelui, @foonathan, @BillyDonahue, @jwakely, @kaisbe and
  @sdebionne.

- Fixed ambiguous formatter specialization in `fmt/ranges.h`
  (https://github.com/fmtlib/fmt/issues/1123).

- Fixed formatting of a non-empty `std::filesystem::path` which is an
  infinitely deep range of its components
  (https://github.com/fmtlib/fmt/issues/1268).

- Fixed handling of general output iterators when formatting
  characters (https://github.com/fmtlib/fmt/issues/1056,
  https://github.com/fmtlib/fmt/pull/1058). Thanks @abolz.

- Fixed handling of output iterators in `formatter` specialization for
  ranges (https://github.com/fmtlib/fmt/issues/1064).

- Fixed handling of exotic character types
  (https://github.com/fmtlib/fmt/issues/1188).

- Made chrono formatting work with exceptions disabled
  (https://github.com/fmtlib/fmt/issues/1062).

- Fixed DLL visibility issues
  (https://github.com/fmtlib/fmt/pull/1134,
  https://github.com/fmtlib/fmt/pull/1147). Thanks @denchat.

- Disabled the use of UDL template extension on GCC 9
  (https://github.com/fmtlib/fmt/issues/1148).

- Removed misplaced `format` compile-time checks from `printf`
  (https://github.com/fmtlib/fmt/issues/1173).

- Fixed issues in the experimental floating-point formatter
  (https://github.com/fmtlib/fmt/issues/1072,
  https://github.com/fmtlib/fmt/issues/1129,
  https://github.com/fmtlib/fmt/issues/1153,
  https://github.com/fmtlib/fmt/pull/1155,
  https://github.com/fmtlib/fmt/issues/1210,
  https://github.com/fmtlib/fmt/issues/1222). Thanks @alabuzhev.

- Fixed bugs discovered by fuzzing or during fuzzing integration
  (https://github.com/fmtlib/fmt/issues/1124,
  https://github.com/fmtlib/fmt/issues/1127,
  https://github.com/fmtlib/fmt/issues/1132,
  https://github.com/fmtlib/fmt/pull/1135,
  https://github.com/fmtlib/fmt/issues/1136,
  https://github.com/fmtlib/fmt/issues/1141,
  https://github.com/fmtlib/fmt/issues/1142,
  https://github.com/fmtlib/fmt/issues/1178,
  https://github.com/fmtlib/fmt/issues/1179,
  https://github.com/fmtlib/fmt/issues/1194). Thanks @pauldreik.

- Fixed building tests on FreeBSD and Hurd
  (https://github.com/fmtlib/fmt/issues/1043). Thanks @jackyf.

- Fixed various warnings and compilation issues
  (https://github.com/fmtlib/fmt/pull/998,
  https://github.com/fmtlib/fmt/pull/1006,
  https://github.com/fmtlib/fmt/issues/1008,
  https://github.com/fmtlib/fmt/issues/1011,
  https://github.com/fmtlib/fmt/issues/1025,
  https://github.com/fmtlib/fmt/pull/1027,
  https://github.com/fmtlib/fmt/pull/1028,
  https://github.com/fmtlib/fmt/pull/1029,
  https://github.com/fmtlib/fmt/pull/1030,
  https://github.com/fmtlib/fmt/pull/1031,
  https://github.com/fmtlib/fmt/pull/1054,
  https://github.com/fmtlib/fmt/issues/1063,
  https://github.com/fmtlib/fmt/pull/1068,
  https://github.com/fmtlib/fmt/pull/1074,
  https://github.com/fmtlib/fmt/pull/1075,
  https://github.com/fmtlib/fmt/pull/1079,
  https://github.com/fmtlib/fmt/pull/1086,
  https://github.com/fmtlib/fmt/issues/1088,
  https://github.com/fmtlib/fmt/pull/1089,
  https://github.com/fmtlib/fmt/pull/1094,
  https://github.com/fmtlib/fmt/issues/1101,
  https://github.com/fmtlib/fmt/pull/1102,
  https://github.com/fmtlib/fmt/issues/1105,
  https://github.com/fmtlib/fmt/pull/1107,
  https://github.com/fmtlib/fmt/issues/1115,
  https://github.com/fmtlib/fmt/issues/1117,
  https://github.com/fmtlib/fmt/issues/1118,
  https://github.com/fmtlib/fmt/issues/1120,
  https://github.com/fmtlib/fmt/issues/1123,
  https://github.com/fmtlib/fmt/pull/1139,
  https://github.com/fmtlib/fmt/issues/1140,
  https://github.com/fmtlib/fmt/issues/1143,
  https://github.com/fmtlib/fmt/pull/1144,
  https://github.com/fmtlib/fmt/pull/1150,
  https://github.com/fmtlib/fmt/pull/1151,
  https://github.com/fmtlib/fmt/issues/1152,
  https://github.com/fmtlib/fmt/issues/1154,
  https://github.com/fmtlib/fmt/issues/1156,
  https://github.com/fmtlib/fmt/pull/1159,
  https://github.com/fmtlib/fmt/issues/1175,
  https://github.com/fmtlib/fmt/issues/1181,
  https://github.com/fmtlib/fmt/issues/1186,
  https://github.com/fmtlib/fmt/pull/1187,
  https://github.com/fmtlib/fmt/pull/1191,
  https://github.com/fmtlib/fmt/issues/1197,
  https://github.com/fmtlib/fmt/issues/1200,
  https://github.com/fmtlib/fmt/issues/1203,
  https://github.com/fmtlib/fmt/issues/1205,
  https://github.com/fmtlib/fmt/pull/1206,
  https://github.com/fmtlib/fmt/issues/1213,
  https://github.com/fmtlib/fmt/issues/1214,
  https://github.com/fmtlib/fmt/pull/1217,
  https://github.com/fmtlib/fmt/issues/1228,
  https://github.com/fmtlib/fmt/pull/1230,
  https://github.com/fmtlib/fmt/issues/1232,
  https://github.com/fmtlib/fmt/pull/1235,
  https://github.com/fmtlib/fmt/pull/1236,
  https://github.com/fmtlib/fmt/issues/1240).
  Thanks @DanielaE, @mwinterb, @eliaskosunen, @morinmorin, @ricco19,
  @waywardmonkeys, @chronoxor, @remyabel, @pauldreik, @gsjaardema, @rcane,
  @mocabe, @denchat, @cjdb, @HazardyKnusperkeks, @vedranmiletic, @jackoalan,
  @DaanDeMeyer and @starkmapper.

# 5.3.0 - 2018-12-28

- Introduced experimental chrono formatting support:

  ```c++
  #include <fmt/chrono.h>

  int main() {
    using namespace std::literals::chrono_literals;
    fmt::print("Default format: {} {}\n", 42s, 100ms);
    fmt::print("strftime-like format: {:%H:%M:%S}\n", 3h + 15min + 30s);
  }
  ```

  prints:

      Default format: 42s 100ms
      strftime-like format: 03:15:30

- Added experimental support for emphasis (bold, italic, underline,
  strikethrough), colored output to a file stream, and improved
  colored formatting API
  (https://github.com/fmtlib/fmt/pull/961,
  https://github.com/fmtlib/fmt/pull/967,
  https://github.com/fmtlib/fmt/pull/973):

  ```c++
  #include <fmt/color.h>

  int main() {
    fmt::print(fg(fmt::color::crimson) | fmt::emphasis::bold,
               "Hello, {}!\n", "world");
    fmt::print(fg(fmt::color::floral_white) | bg(fmt::color::slate_gray) |
               fmt::emphasis::underline, "Olá, {}!\n", "Mundo");
    fmt::print(fg(fmt::color::steel_blue) | fmt::emphasis::italic,
               "你好{}！\n", "世界");
  }
  ```

  prints the following on modern terminals with RGB color support:

  ![](https://github.com/fmtlib/fmt/assets/%0A576385/2a93c904-d6fa-4aa6-b453-2618e1c327d7)

  Thanks @Rakete1111.

- Added support for 4-bit terminal colors
  (https://github.com/fmtlib/fmt/issues/968,
  https://github.com/fmtlib/fmt/pull/974)

  ```c++
  #include <fmt/color.h>

  int main() {
    print(fg(fmt::terminal_color::red), "stop\n");
  }
  ```

  Note that these colors vary by terminal:

  ![](https://user-images.githubusercontent.com/576385/50405925-dbfc7e00-0770-11e9-9b85-333fab0af9ac.png)

  Thanks @Rakete1111.

- Parameterized formatting functions on the type of the format string
  (https://github.com/fmtlib/fmt/issues/880,
  https://github.com/fmtlib/fmt/pull/881,
  https://github.com/fmtlib/fmt/pull/883,
  https://github.com/fmtlib/fmt/pull/885,
  https://github.com/fmtlib/fmt/pull/897,
  https://github.com/fmtlib/fmt/issues/920). Any object of
  type `S` that has an overloaded `to_string_view(const S&)` returning
  `fmt::string_view` can be used as a format string:

  ```c++
  namespace my_ns {
  inline string_view to_string_view(const my_string& s) {
    return {s.data(), s.length()};
  }
  }

  std::string message = fmt::format(my_string("The answer is {}."), 42);
  ```

  Thanks @DanielaE.

- Made `std::string_view` work as a format string
  (https://github.com/fmtlib/fmt/pull/898):

  ```c++
  auto message = fmt::format(std::string_view("The answer is {}."), 42);
  ```

  Thanks @DanielaE.

- Added wide string support to compile-time format string checks
  (https://github.com/fmtlib/fmt/pull/924):

  ```c++
  print(fmt(L"{:f}"), 42); // compile-time error: invalid type specifier
  ```

  Thanks @XZiar.

- Made colored print functions work with wide strings
  (https://github.com/fmtlib/fmt/pull/867):

  ```c++
  #include <fmt/color.h>

  int main() {
    print(fg(fmt::color::red), L"{}\n", 42);
  }
  ```

  Thanks @DanielaE.

- Introduced experimental Unicode support
  (https://github.com/fmtlib/fmt/issues/628,
  https://github.com/fmtlib/fmt/pull/891):

  ```c++
  using namespace fmt::literals;
  auto s = fmt::format("{:*^5}"_u, "🤡"_u); // s == "**🤡**"_u
  ```

- Improved locale support:

  ```c++
  #include <fmt/locale.h>

  struct numpunct : std::numpunct<char> {
   protected:
    char do_thousands_sep() const override { return '~'; }
  };

  std::locale loc;
  auto s = fmt::format(std::locale(loc, new numpunct()), "{:n}", 1234567);
  // s == "1~234~567"
  ```

- Constrained formatting functions on proper iterator types
  (https://github.com/fmtlib/fmt/pull/921). Thanks @DanielaE.

- Added `make_printf_args` and `make_wprintf_args` functions
  (https://github.com/fmtlib/fmt/pull/934). Thanks @tnovotny.

- Deprecated `fmt::visit`, `parse_context`, and `wparse_context`. Use
  `fmt::visit_format_arg`, `format_parse_context`, and
  `wformat_parse_context` instead.

- Removed undocumented `basic_fixed_buffer` which has been superseded
  by the iterator-based API
  (https://github.com/fmtlib/fmt/issues/873,
  https://github.com/fmtlib/fmt/pull/902). Thanks @superfunc.

- Disallowed repeated leading zeros in an argument ID:

  ```c++
  fmt::print("{000}", 42); // error
  ```

- Reintroduced support for gcc 4.4.

- Fixed compilation on platforms with exotic `double`
  (https://github.com/fmtlib/fmt/issues/878).

- Improved documentation
  (https://github.com/fmtlib/fmt/issues/164,
  https://github.com/fmtlib/fmt/issues/877,
  https://github.com/fmtlib/fmt/pull/901,
  https://github.com/fmtlib/fmt/pull/906,
  https://github.com/fmtlib/fmt/pull/979).
  Thanks @kookjr, @DarkDimius and @HecticSerenity.

- Added pkgconfig support which makes it easier to consume the library
  from meson and other build systems
  (https://github.com/fmtlib/fmt/pull/916). Thanks @colemickens.

- Various build improvements
  (https://github.com/fmtlib/fmt/pull/909,
  https://github.com/fmtlib/fmt/pull/926,
  https://github.com/fmtlib/fmt/pull/937,
  https://github.com/fmtlib/fmt/pull/953,
  https://github.com/fmtlib/fmt/pull/959).
  Thanks @tchaikov, @luncliff, @AndreasSchoenle, @hotwatermorning and @Zefz.

- Improved `string_view` construction performance
  (https://github.com/fmtlib/fmt/pull/914). Thanks @gabime.

- Fixed non-matching char types
  (https://github.com/fmtlib/fmt/pull/895). Thanks @DanielaE.

- Fixed `format_to_n` with `std::back_insert_iterator`
  (https://github.com/fmtlib/fmt/pull/913). Thanks @DanielaE.

- Fixed locale-dependent formatting
  (https://github.com/fmtlib/fmt/issues/905).

- Fixed various compiler warnings and errors
  (https://github.com/fmtlib/fmt/pull/882,
  https://github.com/fmtlib/fmt/pull/886,
  https://github.com/fmtlib/fmt/pull/933,
  https://github.com/fmtlib/fmt/pull/941,
  https://github.com/fmtlib/fmt/issues/931,
  https://github.com/fmtlib/fmt/pull/943,
  https://github.com/fmtlib/fmt/pull/954,
  https://github.com/fmtlib/fmt/pull/956,
  https://github.com/fmtlib/fmt/pull/962,
  https://github.com/fmtlib/fmt/issues/965,
  https://github.com/fmtlib/fmt/issues/977,
  https://github.com/fmtlib/fmt/pull/983,
  https://github.com/fmtlib/fmt/pull/989).
  Thanks @Luthaf, @stevenhoving, @christinaa, @lgritz, @DanielaE,
  @0x8000-0000 and @liuping1997.

# 5.2.1 - 2018-09-21

- Fixed `visit` lookup issues on gcc 7 & 8
  (https://github.com/fmtlib/fmt/pull/870). Thanks @medithe.
- Fixed linkage errors on older gcc.
- Prevented `fmt/range.h` from specializing `fmt::basic_string_view`
  (https://github.com/fmtlib/fmt/issues/865,
  https://github.com/fmtlib/fmt/pull/868). Thanks @hhggit.
- Improved error message when formatting unknown types
  (https://github.com/fmtlib/fmt/pull/872). Thanks @foonathan.
- Disabled templated user-defined literals when compiled under nvcc
  (https://github.com/fmtlib/fmt/pull/875). Thanks @CandyGumdrop.
- Fixed `format_to` formatting to `wmemory_buffer`
  (https://github.com/fmtlib/fmt/issues/874).

# 5.2.0 - 2018-09-13

- Optimized format string parsing and argument processing which
  resulted in up to 5x speed up on long format strings and significant
  performance boost on various benchmarks. For example, version 5.2 is
  2.22x faster than 5.1 on decimal integer formatting with `format_to`
  (macOS, clang-902.0.39.2):

  | Method                     | Time, s         | Speedup |
  | -------------------------- | --------------: | ------: |
  | fmt::format 5.1            | 0.58            |         |
  | fmt::format 5.2            | 0.35            |   1.66x |
  | fmt::format_to 5.1         | 0.51            |         |
  | fmt::format_to 5.2         | 0.23            |   2.22x |
  | sprintf                    | 0.71            |         |
  | std::to_string             | 1.01            |         |
  | std::stringstream          | 1.73            |         |

- Changed the `fmt` macro from opt-out to opt-in to prevent name
  collisions. To enable it define the `FMT_STRING_ALIAS` macro to 1
  before including `fmt/format.h`:

  ```c++
  #define FMT_STRING_ALIAS 1
  #include <fmt/format.h>
  std::string answer = format(fmt("{}"), 42);
  ```

- Added compile-time format string checks to `format_to` overload that
  takes `fmt::memory_buffer`
  (https://github.com/fmtlib/fmt/issues/783):

  ```c++
  fmt::memory_buffer buf;
  // Compile-time error: invalid type specifier.
  fmt::format_to(buf, fmt("{:d}"), "foo");
  ```

- Moved experimental color support to `fmt/color.h` and enabled the
  new API by default. The old API can be enabled by defining the
  `FMT_DEPRECATED_COLORS` macro.

- Added formatting support for types explicitly convertible to
  `fmt::string_view`:

  ```c++
  struct foo {
    explicit operator fmt::string_view() const { return "foo"; }
  };
  auto s = format("{}", foo());
  ```

  In particular, this makes formatting function work with
  `folly::StringPiece`.

- Implemented preliminary support for `char*_t` by replacing the
  `format` function overloads with a single function template
  parameterized on the string type.

- Added support for dynamic argument lists
  (https://github.com/fmtlib/fmt/issues/814,
  https://github.com/fmtlib/fmt/pull/819). Thanks @MikePopoloski.

- Reduced executable size overhead for embedded targets using newlib
  nano by making locale dependency optional
  (https://github.com/fmtlib/fmt/pull/839). Thanks @teajay-fr.

- Keep `noexcept` specifier when exceptions are disabled
  (https://github.com/fmtlib/fmt/issues/801,
  https://github.com/fmtlib/fmt/pull/810). Thanks @qis.

- Fixed formatting of user-defined types providing `operator<<` with
  `format_to_n` (https://github.com/fmtlib/fmt/pull/806).
  Thanks @mkurdej.

- Fixed dynamic linkage of new symbols
  (https://github.com/fmtlib/fmt/issues/808).

- Fixed global initialization issue
  (https://github.com/fmtlib/fmt/issues/807):

  ```c++
  // This works on compilers with constexpr support.
  static const std::string answer = fmt::format("{}", 42);
  ```

- Fixed various compiler warnings and errors
  (https://github.com/fmtlib/fmt/pull/804,
  https://github.com/fmtlib/fmt/issues/809,
  https://github.com/fmtlib/fmt/pull/811,
  https://github.com/fmtlib/fmt/issues/822,
  https://github.com/fmtlib/fmt/pull/827,
  https://github.com/fmtlib/fmt/issues/830,
  https://github.com/fmtlib/fmt/pull/838,
  https://github.com/fmtlib/fmt/issues/843,
  https://github.com/fmtlib/fmt/pull/844,
  https://github.com/fmtlib/fmt/issues/851,
  https://github.com/fmtlib/fmt/pull/852,
  https://github.com/fmtlib/fmt/pull/854).
  Thanks @henryiii, @medithe, and @eliasdaler.

# 5.1.0 - 2018-07-05

- Added experimental support for RGB color output enabled with the
  `FMT_EXTENDED_COLORS` macro:

  ```c++
  #define FMT_EXTENDED_COLORS
  #define FMT_HEADER_ONLY // or compile fmt with FMT_EXTENDED_COLORS defined
  #include <fmt/format.h>

  fmt::print(fmt::color::steel_blue, "Some beautiful text");
  ```

  The old API (the `print_colored` and `vprint_colored` functions and
  the `color` enum) is now deprecated.
  (https://github.com/fmtlib/fmt/issues/762
  https://github.com/fmtlib/fmt/pull/767). thanks @Remotion.

- Added quotes to strings in ranges and tuples
  (https://github.com/fmtlib/fmt/pull/766). Thanks @Remotion.

- Made `format_to` work with `basic_memory_buffer`
  (https://github.com/fmtlib/fmt/issues/776).

- Added `vformat_to_n` and `wchar_t` overload of `format_to_n`
  (https://github.com/fmtlib/fmt/issues/764,
  https://github.com/fmtlib/fmt/issues/769).

- Made `is_range` and `is_tuple_like` part of public (experimental)
  API to allow specialization for user-defined types
  (https://github.com/fmtlib/fmt/issues/751,
  https://github.com/fmtlib/fmt/pull/759). Thanks @drrlvn.

- Added more compilers to continuous integration and increased
  `FMT_PEDANTIC` warning levels
  (https://github.com/fmtlib/fmt/pull/736). Thanks @eliaskosunen.

- Fixed compilation with MSVC 2013.

- Fixed handling of user-defined types in `format_to`
  (https://github.com/fmtlib/fmt/issues/793).

- Forced linking of inline `vformat` functions into the library
  (https://github.com/fmtlib/fmt/issues/795).

- Fixed incorrect call to on_align in `'{:}='`
  (https://github.com/fmtlib/fmt/issues/750).

- Fixed floating-point formatting to a non-back_insert_iterator with
  sign & numeric alignment specified
  (https://github.com/fmtlib/fmt/issues/756).

- Fixed formatting to an array with `format_to_n`
  (https://github.com/fmtlib/fmt/issues/778).

- Fixed formatting of more than 15 named arguments
  (https://github.com/fmtlib/fmt/issues/754).

- Fixed handling of compile-time strings when including
  `fmt/ostream.h`. (https://github.com/fmtlib/fmt/issues/768).

- Fixed various compiler warnings and errors
  (https://github.com/fmtlib/fmt/issues/742,
  https://github.com/fmtlib/fmt/issues/748,
  https://github.com/fmtlib/fmt/issues/752,
  https://github.com/fmtlib/fmt/issues/770,
  https://github.com/fmtlib/fmt/pull/775,
  https://github.com/fmtlib/fmt/issues/779,
  https://github.com/fmtlib/fmt/pull/780,
  https://github.com/fmtlib/fmt/pull/790,
  https://github.com/fmtlib/fmt/pull/792,
  https://github.com/fmtlib/fmt/pull/800).
  Thanks @Remotion, @gabime, @foonathan, @Dark-Passenger and @0x8000-0000.

# 5.0.0 - 2018-05-21

- Added a requirement for partial C++11 support, most importantly
  variadic templates and type traits, and dropped `FMT_VARIADIC_*`
  emulation macros. Variadic templates are available since GCC 4.4,
  Clang 2.9 and MSVC 18.0 (2013). For older compilers use {fmt}
  [version 4.x](https://github.com/fmtlib/fmt/releases/tag/4.1.0)
  which continues to be maintained and works with C++98 compilers.

- Renamed symbols to follow standard C++ naming conventions and
  proposed a subset of the library for standardization in [P0645R2
  Text Formatting](https://wg21.link/P0645).

- Implemented `constexpr` parsing of format strings and [compile-time
  format string
  checks](https://fmt.dev/latest/api.html#compile-time-format-string-checks).
  For example

  ```c++
  #include <fmt/format.h>

  std::string s = format(fmt("{:d}"), "foo");
  ```

  gives a compile-time error because `d` is an invalid specifier for
  strings ([godbolt](https://godbolt.org/g/rnCy9Q)):

      ...
      <source>:4:19: note: in instantiation of function template specialization 'fmt::v5::format<S, char [4]>' requested here
        std::string s = format(fmt("{:d}"), "foo");
                        ^
      format.h:1337:13: note: non-constexpr function 'on_error' cannot be used in a constant expression
          handler.on_error("invalid type specifier");

  Compile-time checks require relaxed `constexpr` (C++14 feature)
  support. If the latter is not available, checks will be performed at
  runtime.

- Separated format string parsing and formatting in the extension API
  to enable compile-time format string processing. For example

  ```c++
  struct Answer {};

  namespace fmt {
  template <>
  struct formatter<Answer> {
    constexpr auto parse(parse_context& ctx) {
      auto it = ctx.begin();
      spec = *it;
      if (spec != 'd' && spec != 's')
        throw format_error("invalid specifier");
      return ++it;
    }

    template <typename FormatContext>
    auto format(Answer, FormatContext& ctx) {
      return spec == 's' ?
        format_to(ctx.begin(), "{}", "fourty-two") :
        format_to(ctx.begin(), "{}", 42);
    }

    char spec = 0;
  };
  }

  std::string s = format(fmt("{:x}"), Answer());
  ```

  gives a compile-time error due to invalid format specifier
  ([godbolt](https://godbolt.org/g/2jQ1Dv)):

      ...
      <source>:12:45: error: expression '<throw-expression>' is not a constant expression
             throw format_error("invalid specifier");

- Added [iterator
  support](https://fmt.dev/latest/api.html#output-iterator-support):

  ```c++
  #include <vector>
  #include <fmt/format.h>

  std::vector<char> out;
  fmt::format_to(std::back_inserter(out), "{}", 42);
  ```

- Added the
  [format_to_n](https://fmt.dev/latest/api.html#_CPPv2N3fmt11format_to_nE8OutputItNSt6size_tE11string_viewDpRK4Args)
  function that restricts the output to the specified number of
  characters (https://github.com/fmtlib/fmt/issues/298):

  ```c++
  char out[4];
  fmt::format_to_n(out, sizeof(out), "{}", 12345);
  // out == "1234" (without terminating '\0')
  ```

- Added the [formatted_size](
  https://fmt.dev/latest/api.html#_CPPv2N3fmt14formatted_sizeE11string_viewDpRK4Args)
  function for computing the output size:

  ```c++
  #include <fmt/format.h>

  auto size = fmt::formatted_size("{}", 12345); // size == 5
  ```

- Improved compile times by reducing dependencies on standard headers
  and providing a lightweight [core
  API](https://fmt.dev/latest/api.html#core-api):

  ```c++
  #include <fmt/core.h>

  fmt::print("The answer is {}.", 42);
  ```

  See [Compile time and code
  bloat](https://github.com/fmtlib/fmt#compile-time-and-code-bloat).

- Added the [make_format_args](
  https://fmt.dev/latest/api.html#_CPPv2N3fmt16make_format_argsEDpRK4Args)
  function for capturing formatting arguments:

  ```c++
  // Prints formatted error message.
  void vreport_error(const char *format, fmt::format_args args) {
    fmt::print("Error: ");
    fmt::vprint(format, args);
  }
  template <typename... Args>
  void report_error(const char *format, const Args & ... args) {
    vreport_error(format, fmt::make_format_args(args...));
  }
  ```

- Added the `make_printf_args` function for capturing `printf`
  arguments (https://github.com/fmtlib/fmt/issues/687,
  https://github.com/fmtlib/fmt/pull/694). Thanks @Kronuz.

- Added prefix `v` to non-variadic functions taking `format_args` to
  distinguish them from variadic ones:

  ```c++
  std::string vformat(string_view format_str, format_args args);

  template <typename... Args>
  std::string format(string_view format_str, const Args & ... args);
  ```

- Added experimental support for formatting ranges, containers and
  tuple-like types in `fmt/ranges.h`
  (https://github.com/fmtlib/fmt/pull/735):

  ```c++
  #include <fmt/ranges.h>

  std::vector<int> v = {1, 2, 3};
  fmt::print("{}", v); // prints {1, 2, 3}
  ```

  Thanks @Remotion.

- Implemented `wchar_t` date and time formatting
  (https://github.com/fmtlib/fmt/pull/712):

  ```c++
  #include <fmt/time.h>

  std::time_t t = std::time(nullptr);
  auto s = fmt::format(L"The date is {:%Y-%m-%d}.", *std::localtime(&t));
  ```

  Thanks @DanielaE.

- Provided more wide string overloads
  (https://github.com/fmtlib/fmt/pull/724). Thanks @DanielaE.

- Switched from a custom null-terminated string view class to
  `string_view` in the format API and provided `fmt::string_view`
  which implements a subset of `std::string_view` API for pre-C++17
  systems.

- Added support for `std::experimental::string_view`
  (https://github.com/fmtlib/fmt/pull/607):

  ```c++
  #include <fmt/core.h>
  #include <experimental/string_view>

  fmt::print("{}", std::experimental::string_view("foo"));
  ```

  Thanks @virgiliofornazin.

- Allowed mixing named and automatic arguments:

  ```c++
  fmt::format("{} {two}", 1, fmt::arg("two", 2));
  ```

- Removed the write API in favor of the [format
  API](https://fmt.dev/latest/api.html#format-api) with compile-time
  handling of format strings.

- Disallowed formatting of multibyte strings into a wide character
  target (https://github.com/fmtlib/fmt/pull/606).

- Improved documentation
  (https://github.com/fmtlib/fmt/pull/515,
  https://github.com/fmtlib/fmt/issues/614,
  https://github.com/fmtlib/fmt/pull/617,
  https://github.com/fmtlib/fmt/pull/661,
  https://github.com/fmtlib/fmt/pull/680).
  Thanks @ibell, @mihaitodor and @johnthagen.

- Implemented more efficient handling of large number of format
  arguments.

- Introduced an inline namespace for symbol versioning.

- Added debug postfix `d` to the `fmt` library name
  (https://github.com/fmtlib/fmt/issues/636).

- Removed unnecessary `fmt/` prefix in includes
  (https://github.com/fmtlib/fmt/pull/397). Thanks @chronoxor.

- Moved `fmt/*.h` to `include/fmt/*.h` to prevent irrelevant files and
  directories appearing on the include search paths when fmt is used
  as a subproject and moved source files to the `src` directory.

- Added qmake project file `support/fmt.pro`
  (https://github.com/fmtlib/fmt/pull/641). Thanks @cowo78.

- Added Gradle build file `support/build.gradle`
  (https://github.com/fmtlib/fmt/pull/649). Thanks @luncliff.

- Removed `FMT_CPPFORMAT` CMake option.

- Fixed a name conflict with the macro `CHAR_WIDTH` in glibc
  (https://github.com/fmtlib/fmt/pull/616). Thanks @aroig.

- Fixed handling of nested braces in `fmt::join`
  (https://github.com/fmtlib/fmt/issues/638).

- Added `SOURCELINK_SUFFIX` for compatibility with Sphinx 1.5
  (https://github.com/fmtlib/fmt/pull/497). Thanks @ginggs.

- Added a missing `inline` in the header-only mode
  (https://github.com/fmtlib/fmt/pull/626). Thanks @aroig.

- Fixed various compiler warnings
  (https://github.com/fmtlib/fmt/pull/640,
  https://github.com/fmtlib/fmt/pull/656,
  https://github.com/fmtlib/fmt/pull/679,
  https://github.com/fmtlib/fmt/pull/681,
  https://github.com/fmtlib/fmt/pull/705,
  https://github.com/fmtlib/fmt/issues/715,
  https://github.com/fmtlib/fmt/pull/717,
  https://github.com/fmtlib/fmt/pull/720,
  https://github.com/fmtlib/fmt/pull/723,
  https://github.com/fmtlib/fmt/pull/726,
  https://github.com/fmtlib/fmt/pull/730,
  https://github.com/fmtlib/fmt/pull/739).
  Thanks @peterbell10, @LarsGullik, @foonathan, @eliaskosunen,
  @christianparpart, @DanielaE and @mwinterb.

- Worked around an MSVC bug and fixed several warnings
  (https://github.com/fmtlib/fmt/pull/653). Thanks @alabuzhev.

- Worked around GCC bug 67371
  (https://github.com/fmtlib/fmt/issues/682).

- Fixed compilation with `-fno-exceptions`
  (https://github.com/fmtlib/fmt/pull/655). Thanks @chenxiaolong.

- Made `constexpr remove_prefix` gcc version check tighter
  (https://github.com/fmtlib/fmt/issues/648).

- Renamed internal type enum constants to prevent collision with
  poorly written C libraries
  (https://github.com/fmtlib/fmt/issues/644).

- Added detection of `wostream operator<<`
  (https://github.com/fmtlib/fmt/issues/650).

- Fixed compilation on OpenBSD
  (https://github.com/fmtlib/fmt/pull/660). Thanks @hubslave.

- Fixed compilation on FreeBSD 12
  (https://github.com/fmtlib/fmt/pull/732). Thanks @dankm.

- Fixed compilation when there is a mismatch between `-std` options
  between the library and user code
  (https://github.com/fmtlib/fmt/issues/664).

- Fixed compilation with GCC 7 and `-std=c++11`
  (https://github.com/fmtlib/fmt/issues/734).

- Improved generated binary code on GCC 7 and older
  (https://github.com/fmtlib/fmt/issues/668).

- Fixed handling of numeric alignment with no width
  (https://github.com/fmtlib/fmt/issues/675).

- Fixed handling of empty strings in UTF8/16 converters
  (https://github.com/fmtlib/fmt/pull/676). Thanks @vgalka-sl.

- Fixed formatting of an empty `string_view`
  (https://github.com/fmtlib/fmt/issues/689).

- Fixed detection of `string_view` on libc++
  (https://github.com/fmtlib/fmt/issues/686).

- Fixed DLL issues (https://github.com/fmtlib/fmt/pull/696).
  Thanks @sebkoenig.

- Fixed compile checks for mixing narrow and wide strings
  (https://github.com/fmtlib/fmt/issues/690).

- Disabled unsafe implicit conversion to `std::string`
  (https://github.com/fmtlib/fmt/issues/729).

- Fixed handling of reused format specs (as in `fmt::join`) for
  pointers (https://github.com/fmtlib/fmt/pull/725). Thanks @mwinterb.

- Fixed installation of `fmt/ranges.h`
  (https://github.com/fmtlib/fmt/pull/738). Thanks @sv1990.

# 4.1.0 - 2017-12-20

- Added `fmt::to_wstring()` in addition to `fmt::to_string()`
  (https://github.com/fmtlib/fmt/pull/559). Thanks @alabuzhev.
- Added support for C++17 `std::string_view`
  (https://github.com/fmtlib/fmt/pull/571 and
  https://github.com/fmtlib/fmt/pull/578).
  Thanks @thelostt and @mwinterb.
- Enabled stream exceptions to catch errors
  (https://github.com/fmtlib/fmt/issues/581). Thanks @crusader-mike.
- Allowed formatting of class hierarchies with `fmt::format_arg()`
  (https://github.com/fmtlib/fmt/pull/547). Thanks @rollbear.
- Removed limitations on character types
  (https://github.com/fmtlib/fmt/pull/563). Thanks @Yelnats321.
- Conditionally enabled use of `std::allocator_traits`
  (https://github.com/fmtlib/fmt/pull/583). Thanks @mwinterb.
- Added support for `const` variadic member function emulation with
  `FMT_VARIADIC_CONST`
  (https://github.com/fmtlib/fmt/pull/591). Thanks @ludekvodicka.
- Various bugfixes: bad overflow check, unsupported implicit type
  conversion when determining formatting function, test segfaults
  (https://github.com/fmtlib/fmt/issues/551), ill-formed
  macros (https://github.com/fmtlib/fmt/pull/542) and
  ambiguous overloads
  (https://github.com/fmtlib/fmt/issues/580). Thanks @xylosper.
- Prevented warnings on MSVC
  (https://github.com/fmtlib/fmt/pull/605,
  https://github.com/fmtlib/fmt/pull/602, and
  https://github.com/fmtlib/fmt/pull/545), clang
  (https://github.com/fmtlib/fmt/pull/582), GCC
  (https://github.com/fmtlib/fmt/issues/573), various
  conversion warnings (https://github.com/fmtlib/fmt/pull/609,
  https://github.com/fmtlib/fmt/pull/567,
  https://github.com/fmtlib/fmt/pull/553 and
  https://github.com/fmtlib/fmt/pull/553), and added
  `override` and `[[noreturn]]`
  (https://github.com/fmtlib/fmt/pull/549 and
  https://github.com/fmtlib/fmt/issues/555).
  Thanks @alabuzhev, @virgiliofornazin, @alexanderbock, @yumetodo, @VaderY,
  @jpcima, @thelostt and @Manu343726.
- Improved CMake: Used `GNUInstallDirs` to set installation location
  (https://github.com/fmtlib/fmt/pull/610) and fixed warnings
  (https://github.com/fmtlib/fmt/pull/536 and
  https://github.com/fmtlib/fmt/pull/556).
  Thanks @mikecrowe, @evgen231 and @henryiii.

# 4.0.0 - 2017-06-27

- Removed old compatibility headers `cppformat/*.h` and CMake options
  (https://github.com/fmtlib/fmt/pull/527). Thanks @maddinat0r.

- Added `string.h` containing `fmt::to_string()` as alternative to
  `std::to_string()` as well as other string writer functionality
  (https://github.com/fmtlib/fmt/issues/326 and
  https://github.com/fmtlib/fmt/pull/441):

  ```c++
  #include "fmt/string.h"

  std::string answer = fmt::to_string(42);
  ```

  Thanks @glebov-andrey.

- Moved `fmt::printf()` to new `printf.h` header and allowed `%s` as
  generic specifier (https://github.com/fmtlib/fmt/pull/453),
  made `%.f` more conformant to regular `printf()`
  (https://github.com/fmtlib/fmt/pull/490), added custom
  writer support (https://github.com/fmtlib/fmt/issues/476)
  and implemented missing custom argument formatting
  (https://github.com/fmtlib/fmt/pull/339 and
  https://github.com/fmtlib/fmt/pull/340):

  ```c++
  #include "fmt/printf.h"

  // %s format specifier can be used with any argument type.
  fmt::printf("%s", 42);
  ```

  Thanks @mojoBrendan, @manylegged and @spacemoose.
  See also https://github.com/fmtlib/fmt/issues/360,
  https://github.com/fmtlib/fmt/issues/335 and
  https://github.com/fmtlib/fmt/issues/331.

- Added `container.h` containing a `BasicContainerWriter` to write to
  containers like `std::vector`
  (https://github.com/fmtlib/fmt/pull/450). Thanks @polyvertex.

- Added `fmt::join()` function that takes a range and formats its
  elements separated by a given string
  (https://github.com/fmtlib/fmt/pull/466):

  ```c++
  #include "fmt/format.h"

  std::vector<double> v = {1.2, 3.4, 5.6};
  // Prints "(+01.20, +03.40, +05.60)".
  fmt::print("({:+06.2f})", fmt::join(v.begin(), v.end(), ", "));
  ```

  Thanks @olivier80.

- Added support for custom formatting specifications to simplify
  customization of built-in formatting
  (https://github.com/fmtlib/fmt/pull/444). Thanks @polyvertex.
  See also https://github.com/fmtlib/fmt/issues/439.

- Added `fmt::format_system_error()` for error code formatting
  (https://github.com/fmtlib/fmt/issues/323 and
  https://github.com/fmtlib/fmt/pull/526). Thanks @maddinat0r.

- Added thread-safe `fmt::localtime()` and `fmt::gmtime()` as
  replacement for the standard version to `time.h`
  (https://github.com/fmtlib/fmt/pull/396). Thanks @codicodi.

- Internal improvements to `NamedArg` and `ArgLists`
  (https://github.com/fmtlib/fmt/pull/389 and
  https://github.com/fmtlib/fmt/pull/390). Thanks @chronoxor.

- Fixed crash due to bug in `FormatBuf`
  (https://github.com/fmtlib/fmt/pull/493). Thanks @effzeh. See also
  https://github.com/fmtlib/fmt/issues/480 and
  https://github.com/fmtlib/fmt/issues/491.

- Fixed handling of wide strings in `fmt::StringWriter`.

- Improved compiler error messages
  (https://github.com/fmtlib/fmt/issues/357).

- Fixed various warnings and issues with various compilers
  (https://github.com/fmtlib/fmt/pull/494,
  https://github.com/fmtlib/fmt/pull/499,
  https://github.com/fmtlib/fmt/pull/483,
  https://github.com/fmtlib/fmt/pull/485,
  https://github.com/fmtlib/fmt/pull/482,
  https://github.com/fmtlib/fmt/pull/475,
  https://github.com/fmtlib/fmt/pull/473 and
  https://github.com/fmtlib/fmt/pull/414).
  Thanks @chronoxor, @zhaohuaxishi, @pkestene, @dschmidt and @0x414c.

- Improved CMake: targets are now namespaced
  (https://github.com/fmtlib/fmt/pull/511 and
  https://github.com/fmtlib/fmt/pull/513), supported
  header-only `printf.h`
  (https://github.com/fmtlib/fmt/pull/354), fixed issue with
  minimal supported library subset
  (https://github.com/fmtlib/fmt/issues/418,
  https://github.com/fmtlib/fmt/pull/419 and
  https://github.com/fmtlib/fmt/pull/420).
  Thanks @bjoernthiel, @niosHD, @LogicalKnight and @alabuzhev.

- Improved documentation (https://github.com/fmtlib/fmt/pull/393).
  Thanks @pwm1234.

# 3.0.2 - 2017-06-14

- Added `FMT_VERSION` macro
  (https://github.com/fmtlib/fmt/issues/411).
- Used `FMT_NULL` instead of literal `0`
  (https://github.com/fmtlib/fmt/pull/409). Thanks @alabuzhev.
- Added extern templates for `format_float`
  (https://github.com/fmtlib/fmt/issues/413).
- Fixed implicit conversion issue
  (https://github.com/fmtlib/fmt/issues/507).
- Fixed signbit detection
  (https://github.com/fmtlib/fmt/issues/423).
- Fixed naming collision
  (https://github.com/fmtlib/fmt/issues/425).
- Fixed missing intrinsic for C++/CLI
  (https://github.com/fmtlib/fmt/pull/457). Thanks @calumr.
- Fixed Android detection
  (https://github.com/fmtlib/fmt/pull/458). Thanks @Gachapen.
- Use lean `windows.h` if not in header-only mode
  (https://github.com/fmtlib/fmt/pull/503). Thanks @Quentin01.
- Fixed issue with CMake exporting C++11 flag
  (https://github.com/fmtlib/fmt/pull/455). Thanks @EricWF.
- Fixed issue with nvcc and MSVC compiler bug and MinGW
  (https://github.com/fmtlib/fmt/issues/505).
- Fixed DLL issues (https://github.com/fmtlib/fmt/pull/469 and
  https://github.com/fmtlib/fmt/pull/502).
  Thanks @richardeakin and @AndreasSchoenle.
- Fixed test compilation under FreeBSD
  (https://github.com/fmtlib/fmt/issues/433).
- Fixed various warnings
  (https://github.com/fmtlib/fmt/pull/403,
  https://github.com/fmtlib/fmt/pull/410 and
  https://github.com/fmtlib/fmt/pull/510).
  Thanks @Lecetem, @chenhayat and @trozen.
- Worked around a broken `__builtin_clz` in clang with MS codegen
  (https://github.com/fmtlib/fmt/issues/519).
- Removed redundant include
  (https://github.com/fmtlib/fmt/issues/479).
- Fixed documentation issues.

# 3.0.1 - 2016-11-01

- Fixed handling of thousands separator
  (https://github.com/fmtlib/fmt/issues/353).
- Fixed handling of `unsigned char` strings
  (https://github.com/fmtlib/fmt/issues/373).
- Corrected buffer growth when formatting time
  (https://github.com/fmtlib/fmt/issues/367).
- Removed warnings under MSVC and clang
  (https://github.com/fmtlib/fmt/issues/318,
  https://github.com/fmtlib/fmt/issues/250, also merged
  https://github.com/fmtlib/fmt/pull/385 and
  https://github.com/fmtlib/fmt/pull/361).
  Thanks @jcelerier and @nmoehrle.
- Fixed compilation issues under Android
  (https://github.com/fmtlib/fmt/pull/327,
  https://github.com/fmtlib/fmt/issues/345 and
  https://github.com/fmtlib/fmt/pull/381), FreeBSD
  (https://github.com/fmtlib/fmt/pull/358), Cygwin
  (https://github.com/fmtlib/fmt/issues/388), MinGW
  (https://github.com/fmtlib/fmt/issues/355) as well as other
  issues (https://github.com/fmtlib/fmt/issues/350,
  https://github.com/fmtlib/fmt/issues/355,
  https://github.com/fmtlib/fmt/pull/348,
  https://github.com/fmtlib/fmt/pull/402,
  https://github.com/fmtlib/fmt/pull/405).
  Thanks @dpantele, @hghwng, @arvedarved, @LogicalKnight and @JanHellwig.
- Fixed some documentation issues and extended specification
  (https://github.com/fmtlib/fmt/issues/320,
  https://github.com/fmtlib/fmt/pull/333,
  https://github.com/fmtlib/fmt/issues/347,
  https://github.com/fmtlib/fmt/pull/362). Thanks @smellman.

# 3.0.0 - 2016-05-07

- The project has been renamed from C++ Format (cppformat) to fmt for
  consistency with the used namespace and macro prefix
  (https://github.com/fmtlib/fmt/issues/307). Library headers
  are now located in the `fmt` directory:

  ```c++
  #include "fmt/format.h"
  ```

  Including `format.h` from the `cppformat` directory is deprecated
  but works via a proxy header which will be removed in the next major
  version.

  The documentation is now available at <https://fmt.dev>.

- Added support for
  [strftime](http://en.cppreference.com/w/cpp/chrono/c/strftime)-like
  [date and time
  formatting](https://fmt.dev/3.0.0/api.html#date-and-time-formatting)
  (https://github.com/fmtlib/fmt/issues/283):

  ```c++
  #include "fmt/time.h"

  std::time_t t = std::time(nullptr);
  // Prints "The date is 2016-04-29." (with the current date)
  fmt::print("The date is {:%Y-%m-%d}.", *std::localtime(&t));
  ```

- `std::ostream` support including formatting of user-defined types
  that provide overloaded `operator<<` has been moved to
  `fmt/ostream.h`:

  ```c++
  #include "fmt/ostream.h"

  class Date {
    int year_, month_, day_;
  public:
    Date(int year, int month, int day) : year_(year), month_(month), day_(day) {}

    friend std::ostream &operator<<(std::ostream &os, const Date &d) {
      return os << d.year_ << '-' << d.month_ << '-' << d.day_;
    }
  };

  std::string s = fmt::format("The date is {}", Date(2012, 12, 9));
  // s == "The date is 2012-12-9"
  ```

- Added support for [custom argument
  formatters](https://fmt.dev/3.0.0/api.html#argument-formatters)
  (https://github.com/fmtlib/fmt/issues/235).

- Added support for locale-specific integer formatting with the `n`
  specifier (https://github.com/fmtlib/fmt/issues/305):

  ```c++
  std::setlocale(LC_ALL, "en_US.utf8");
  fmt::print("cppformat: {:n}\n", 1234567); // prints 1,234,567
  ```

- Sign is now preserved when formatting an integer with an incorrect
  `printf` format specifier
  (https://github.com/fmtlib/fmt/issues/265):

  ```c++
  fmt::printf("%lld", -42); // prints -42
  ```

  Note that it would be an undefined behavior in `std::printf`.

- Length modifiers such as `ll` are now optional in printf formatting
  functions and the correct type is determined automatically
  (https://github.com/fmtlib/fmt/issues/255):

  ```c++
  fmt::printf("%d", std::numeric_limits<long long>::max());
  ```

  Note that it would be an undefined behavior in `std::printf`.

- Added initial support for custom formatters
  (https://github.com/fmtlib/fmt/issues/231).

- Fixed detection of user-defined literal support on Intel C++
  compiler (https://github.com/fmtlib/fmt/issues/311,
  https://github.com/fmtlib/fmt/pull/312).
  Thanks @dean0x7d and @speth.

- Reduced compile time
  (https://github.com/fmtlib/fmt/pull/243,
  https://github.com/fmtlib/fmt/pull/249,
  https://github.com/fmtlib/fmt/issues/317):

  ![](https://cloud.githubusercontent.com/assets/4831417/11614060/b9e826d2-9c36-11e5-8666-d4131bf503ef.png)

  ![](https://cloud.githubusercontent.com/assets/4831417/11614080/6ac903cc-9c37-11e5-8165-26df6efae364.png)

  Thanks @dean0x7d.

- Compile test fixes (https://github.com/fmtlib/fmt/pull/313).
  Thanks @dean0x7d.

- Documentation fixes (https://github.com/fmtlib/fmt/pull/239,
  https://github.com/fmtlib/fmt/issues/248,
  https://github.com/fmtlib/fmt/issues/252,
  https://github.com/fmtlib/fmt/pull/258,
  https://github.com/fmtlib/fmt/issues/260,
  https://github.com/fmtlib/fmt/issues/301,
  https://github.com/fmtlib/fmt/pull/309).
  Thanks @ReadmeCritic @Gachapen and @jwilk.

- Fixed compiler and sanitizer warnings
  (https://github.com/fmtlib/fmt/issues/244,
  https://github.com/fmtlib/fmt/pull/256,
  https://github.com/fmtlib/fmt/pull/259,
  https://github.com/fmtlib/fmt/issues/263,
  https://github.com/fmtlib/fmt/issues/274,
  https://github.com/fmtlib/fmt/pull/277,
  https://github.com/fmtlib/fmt/pull/286,
  https://github.com/fmtlib/fmt/issues/291,
  https://github.com/fmtlib/fmt/issues/296,
  https://github.com/fmtlib/fmt/issues/308).
  Thanks @mwinterb, @pweiskircher and @Naios.

- Improved compatibility with Windows Store apps
  (https://github.com/fmtlib/fmt/issues/280,
  https://github.com/fmtlib/fmt/pull/285) Thanks @mwinterb.

- Added tests of compatibility with older C++ standards
  (https://github.com/fmtlib/fmt/pull/273). Thanks @niosHD.

- Fixed Android build
  (https://github.com/fmtlib/fmt/pull/271). Thanks @newnon.

- Changed `ArgMap` to be backed by a vector instead of a map.
  (https://github.com/fmtlib/fmt/issues/261,
  https://github.com/fmtlib/fmt/pull/262). Thanks @mwinterb.

- Added `fprintf` overload that writes to a `std::ostream`
  (https://github.com/fmtlib/fmt/pull/251).
  Thanks @nickhutchinson.

- Export symbols when building a Windows DLL
  (https://github.com/fmtlib/fmt/pull/245).
  Thanks @macdems.

- Fixed compilation on Cygwin
  (https://github.com/fmtlib/fmt/issues/304).

- Implemented a workaround for a bug in Apple LLVM version 4.2 of
  clang (https://github.com/fmtlib/fmt/issues/276).

- Implemented a workaround for Google Test bug
  https://github.com/google/googletest/issues/705 on gcc 6
  (https://github.com/fmtlib/fmt/issues/268). Thanks @octoploid.

- Removed Biicode support because the latter has been discontinued.

# 2.1.1 - 2016-04-11

- The install location for generated CMake files is now configurable
  via the `FMT_CMAKE_DIR` CMake variable
  (https://github.com/fmtlib/fmt/pull/299). Thanks @niosHD.
- Documentation fixes
  (https://github.com/fmtlib/fmt/issues/252).

# 2.1.0 - 2016-03-21

- Project layout and build system improvements
  (https://github.com/fmtlib/fmt/pull/267):

  -   The code have been moved to the `cppformat` directory. Including
      `format.h` from the top-level directory is deprecated but works
      via a proxy header which will be removed in the next major
      version.
  -   C++ Format CMake targets now have proper interface definitions.
  -   Installed version of the library now supports the header-only
      configuration.
  -   Targets `doc`, `install`, and `test` are now disabled if C++
      Format is included as a CMake subproject. They can be enabled by
      setting `FMT_DOC`, `FMT_INSTALL`, and `FMT_TEST` in the parent
      project.

  Thanks @niosHD.

# 2.0.1 - 2016-03-13

- Improved CMake find and package support
  (https://github.com/fmtlib/fmt/issues/264). Thanks @niosHD.
- Fix compile error with Android NDK and mingw32
  (https://github.com/fmtlib/fmt/issues/241). Thanks @Gachapen.
- Documentation fixes
  (https://github.com/fmtlib/fmt/issues/248,
  https://github.com/fmtlib/fmt/issues/260).

# 2.0.0 - 2015-12-01

## General

- \[Breaking\] Named arguments
  (https://github.com/fmtlib/fmt/pull/169,
  https://github.com/fmtlib/fmt/pull/173,
  https://github.com/fmtlib/fmt/pull/174):

  ```c++
  fmt::print("The answer is {answer}.", fmt::arg("answer", 42));
  ```

  Thanks @jamboree.

- \[Experimental\] User-defined literals for format and named
  arguments (https://github.com/fmtlib/fmt/pull/204,
  https://github.com/fmtlib/fmt/pull/206,
  https://github.com/fmtlib/fmt/pull/207):

  ```c++
  using namespace fmt::literals;
  fmt::print("The answer is {answer}.", "answer"_a=42);
  ```

  Thanks @dean0x7d.

- \[Breaking\] Formatting of more than 16 arguments is now supported
  when using variadic templates
  (https://github.com/fmtlib/fmt/issues/141). Thanks @Shauren.

- Runtime width specification
  (https://github.com/fmtlib/fmt/pull/168):

  ```c++
  fmt::format("{0:{1}}", 42, 5); // gives "   42"
  ```

  Thanks @jamboree.

- \[Breaking\] Enums are now formatted with an overloaded
  `std::ostream` insertion operator (`operator<<`) if available
  (https://github.com/fmtlib/fmt/issues/232).

- \[Breaking\] Changed default `bool` format to textual, \"true\" or
  \"false\" (https://github.com/fmtlib/fmt/issues/170):

  ```c++
  fmt::print("{}", true); // prints "true"
  ```

  To print `bool` as a number use numeric format specifier such as
  `d`:

  ```c++
  fmt::print("{:d}", true); // prints "1"
  ```

- `fmt::printf` and `fmt::sprintf` now support formatting of `bool`
  with the `%s` specifier giving textual output, \"true\" or \"false\"
  (https://github.com/fmtlib/fmt/pull/223):

  ```c++
  fmt::printf("%s", true); // prints "true"
  ```

  Thanks @LarsGullik.

- \[Breaking\] `signed char` and `unsigned char` are now formatted as
  integers by default
  (https://github.com/fmtlib/fmt/pull/217).

- \[Breaking\] Pointers to C strings can now be formatted with the `p`
  specifier (https://github.com/fmtlib/fmt/pull/223):

  ```c++
  fmt::print("{:p}", "test"); // prints pointer value
  ```

  Thanks @LarsGullik.

- \[Breaking\] `fmt::printf` and `fmt::sprintf` now print null
  pointers as `(nil)` and null strings as `(null)` for consistency
  with glibc (https://github.com/fmtlib/fmt/pull/226).
  Thanks @LarsGullik.

- \[Breaking\] `fmt::(s)printf` now supports formatting of objects of
  user-defined types that provide an overloaded `std::ostream`
  insertion operator (`operator<<`)
  (https://github.com/fmtlib/fmt/issues/201):

  ```c++
  fmt::printf("The date is %s", Date(2012, 12, 9));
  ```

- \[Breaking\] The `Buffer` template is now part of the public API and
  can be used to implement custom memory buffers
  (https://github.com/fmtlib/fmt/issues/140). Thanks @polyvertex.

- \[Breaking\] Improved compatibility between `BasicStringRef` and
  [std::experimental::basic_string_view](
  http://en.cppreference.com/w/cpp/experimental/basic_string_view)
  (https://github.com/fmtlib/fmt/issues/100,
  https://github.com/fmtlib/fmt/issues/159,
  https://github.com/fmtlib/fmt/issues/183):

  -   Comparison operators now compare string content, not pointers
  -   `BasicStringRef::c_str` replaced by `BasicStringRef::data`
  -   `BasicStringRef` is no longer assumed to be null-terminated

  References to null-terminated strings are now represented by a new
  class, `BasicCStringRef`.

- Dependency on pthreads introduced by Google Test is now optional
  (https://github.com/fmtlib/fmt/issues/185).

- New CMake options `FMT_DOC`, `FMT_INSTALL` and `FMT_TEST` to control
  generation of `doc`, `install` and `test` targets respectively, on
  by default (https://github.com/fmtlib/fmt/issues/197,
  https://github.com/fmtlib/fmt/issues/198,
  https://github.com/fmtlib/fmt/issues/200). Thanks @maddinat0r.

- `noexcept` is now used when compiling with MSVC2015
  (https://github.com/fmtlib/fmt/pull/215). Thanks @dmkrepo.

- Added an option to disable use of `windows.h` when
  `FMT_USE_WINDOWS_H` is defined as 0 before including `format.h`
  (https://github.com/fmtlib/fmt/issues/171). Thanks @alfps.

- \[Breaking\] `windows.h` is now included with `NOMINMAX` unless
  `FMT_WIN_MINMAX` is defined. This is done to prevent breaking code
  using `std::min` and `std::max` and only affects the header-only
  configuration (https://github.com/fmtlib/fmt/issues/152,
  https://github.com/fmtlib/fmt/pull/153,
  https://github.com/fmtlib/fmt/pull/154). Thanks @DevO2012.

- Improved support for custom character types
  (https://github.com/fmtlib/fmt/issues/171). Thanks @alfps.

- Added an option to disable use of IOStreams when `FMT_USE_IOSTREAMS`
  is defined as 0 before including `format.h`
  (https://github.com/fmtlib/fmt/issues/205,
  https://github.com/fmtlib/fmt/pull/208). Thanks @JodiTheTigger.

- Improved detection of `isnan`, `isinf` and `signbit`.

## Optimization

- Made formatting of user-defined types more efficient with a custom
  stream buffer (https://github.com/fmtlib/fmt/issues/92,
  https://github.com/fmtlib/fmt/pull/230). Thanks @NotImplemented.
- Further improved performance of `fmt::Writer` on integer formatting
  and fixed a minor regression. Now it is \~7% faster than
  `karma::generate` on Karma\'s benchmark
  (https://github.com/fmtlib/fmt/issues/186).
- \[Breaking\] Reduced [compiled code
  size](https://github.com/fmtlib/fmt#compile-time-and-code-bloat)
  (https://github.com/fmtlib/fmt/issues/143,
  https://github.com/fmtlib/fmt/pull/149).

## Distribution

- \[Breaking\] Headers are now installed in
  `${CMAKE_INSTALL_PREFIX}/include/cppformat`
  (https://github.com/fmtlib/fmt/issues/178). Thanks @jackyf.

- \[Breaking\] Changed the library name from `format` to `cppformat`
  for consistency with the project name and to avoid potential
  conflicts (https://github.com/fmtlib/fmt/issues/178).
  Thanks @jackyf.

- C++ Format is now available in [Debian](https://www.debian.org/)
  GNU/Linux
  ([stretch](https://packages.debian.org/source/stretch/cppformat),
  [sid](https://packages.debian.org/source/sid/cppformat)) and derived
  distributions such as
  [Ubuntu](https://launchpad.net/ubuntu/+source/cppformat) 15.10 and
  later (https://github.com/fmtlib/fmt/issues/155):

      $ sudo apt-get install libcppformat1-dev

  Thanks @jackyf.

- [Packages for Fedora and
  RHEL](https://admin.fedoraproject.org/pkgdb/package/cppformat/) are
  now available. Thanks Dave Johansen.

- C++ Format can now be installed via [Homebrew](http://brew.sh/) on
  OS X (https://github.com/fmtlib/fmt/issues/157):

      $ brew install cppformat

  Thanks @ortho and Anatoliy Bulukin.

## Documentation

- Migrated from ReadTheDocs to GitHub Pages for better responsiveness
  and reliability (https://github.com/fmtlib/fmt/issues/128).
  New documentation address is <http://cppformat.github.io/>.
- Added [Building thedocumentation](
  https://fmt.dev/2.0.0/usage.html#building-the-documentation)
  section to the documentation.
- Documentation build script is now compatible with Python 3 and newer
  pip versions. (https://github.com/fmtlib/fmt/pull/189,
  https://github.com/fmtlib/fmt/issues/209).
  Thanks @JodiTheTigger and @xentec.
- Documentation fixes and improvements
  (https://github.com/fmtlib/fmt/issues/36,
  https://github.com/fmtlib/fmt/issues/75,
  https://github.com/fmtlib/fmt/issues/125,
  https://github.com/fmtlib/fmt/pull/160,
  https://github.com/fmtlib/fmt/pull/161,
  https://github.com/fmtlib/fmt/issues/162,
  https://github.com/fmtlib/fmt/issues/165,
  https://github.com/fmtlib/fmt/issues/210). 
  Thanks @syohex.
- Fixed out-of-tree documentation build
  (https://github.com/fmtlib/fmt/issues/177). Thanks @jackyf.

## Fixes

- Fixed `initializer_list` detection
  (https://github.com/fmtlib/fmt/issues/136). Thanks @Gachapen.

- \[Breaking\] Fixed formatting of enums with numeric format
  specifiers in `fmt::(s)printf`
  (https://github.com/fmtlib/fmt/issues/131,
  https://github.com/fmtlib/fmt/issues/139):

  ```c++
  enum { ANSWER = 42 };
  fmt::printf("%d", ANSWER);
  ```

  Thanks @Naios.

- Improved compatibility with old versions of MinGW
  (https://github.com/fmtlib/fmt/issues/129,
  https://github.com/fmtlib/fmt/pull/130,
  https://github.com/fmtlib/fmt/issues/132). Thanks @cstamford.

- Fixed a compile error on MSVC with disabled exceptions
  (https://github.com/fmtlib/fmt/issues/144).

- Added a workaround for broken implementation of variadic templates
  in MSVC2012 (https://github.com/fmtlib/fmt/issues/148).

- Placed the anonymous namespace within `fmt` namespace for the
  header-only configuration (https://github.com/fmtlib/fmt/issues/171).
  Thanks @alfps.

- Fixed issues reported by Coverity Scan
  (https://github.com/fmtlib/fmt/issues/187,
  https://github.com/fmtlib/fmt/issues/192).

- Implemented a workaround for a name lookup bug in MSVC2010
  (https://github.com/fmtlib/fmt/issues/188).

- Fixed compiler warnings
  (https://github.com/fmtlib/fmt/issues/95,
  https://github.com/fmtlib/fmt/issues/96,
  https://github.com/fmtlib/fmt/pull/114,
  https://github.com/fmtlib/fmt/issues/135,
  https://github.com/fmtlib/fmt/issues/142,
  https://github.com/fmtlib/fmt/issues/145,
  https://github.com/fmtlib/fmt/issues/146,
  https://github.com/fmtlib/fmt/issues/158,
  https://github.com/fmtlib/fmt/issues/163,
  https://github.com/fmtlib/fmt/issues/175,
  https://github.com/fmtlib/fmt/issues/190,
  https://github.com/fmtlib/fmt/pull/191,
  https://github.com/fmtlib/fmt/issues/194,
  https://github.com/fmtlib/fmt/pull/196,
  https://github.com/fmtlib/fmt/issues/216,
  https://github.com/fmtlib/fmt/pull/218,
  https://github.com/fmtlib/fmt/pull/220,
  https://github.com/fmtlib/fmt/pull/229,
  https://github.com/fmtlib/fmt/issues/233,
  https://github.com/fmtlib/fmt/issues/234,
  https://github.com/fmtlib/fmt/pull/236,
  https://github.com/fmtlib/fmt/issues/281,
  https://github.com/fmtlib/fmt/issues/289).
  Thanks @seanmiddleditch, @dixlorenz, @CarterLi, @Naios, @fmatthew5876,
  @LevskiWeng, @rpopescu, @gabime, @cubicool, @jkflying, @LogicalKnight,
  @inguin and @Jopie64.

- Fixed portability issues (mostly causing test failures) on ARM,
  ppc64, ppc64le, s390x and SunOS 5.11 i386
  (https://github.com/fmtlib/fmt/issues/138,
  https://github.com/fmtlib/fmt/issues/179,
  https://github.com/fmtlib/fmt/issues/180,
  https://github.com/fmtlib/fmt/issues/202,
  https://github.com/fmtlib/fmt/issues/225, [Red Hat Bugzilla
  Bug 1260297](https://bugzilla.redhat.com/show_bug.cgi?id=1260297)).
  Thanks @Naios, @jackyf and Dave Johansen.

- Fixed a name conflict with macro `free` defined in `crtdbg.h` when
  `_CRTDBG_MAP_ALLOC` is set (https://github.com/fmtlib/fmt/issues/211).

- Fixed shared library build on OS X
  (https://github.com/fmtlib/fmt/pull/212). Thanks @dean0x7d.

- Fixed an overload conflict on MSVC when `/Zc:wchar_t-` option is
  specified (https://github.com/fmtlib/fmt/pull/214).
  Thanks @slavanap.

- Improved compatibility with MSVC 2008
  (https://github.com/fmtlib/fmt/pull/236). Thanks @Jopie64.

- Improved compatibility with bcc32
  (https://github.com/fmtlib/fmt/issues/227).

- Fixed `static_assert` detection on Clang
  (https://github.com/fmtlib/fmt/pull/228). Thanks @dean0x7d.

# 1.1.0 - 2015-03-06

- Added `BasicArrayWriter`, a class template that provides operations
  for formatting and writing data into a fixed-size array
  (https://github.com/fmtlib/fmt/issues/105 and
  https://github.com/fmtlib/fmt/issues/122):

  ```c++
  char buffer[100];
  fmt::ArrayWriter w(buffer);
  w.write("The answer is {}", 42);
  ```

- Added [0 A.D.](http://play0ad.com/) and [PenUltima Online
  (POL)](http://www.polserver.com/) to the list of notable projects
  using C++ Format.

- C++ Format now uses MSVC intrinsics for better formatting performance
  (https://github.com/fmtlib/fmt/pull/115,
  https://github.com/fmtlib/fmt/pull/116,
  https://github.com/fmtlib/fmt/pull/118 and
  https://github.com/fmtlib/fmt/pull/121). Previously these
  optimizations where only used on GCC and Clang.
  Thanks @CarterLi and @objectx.

- CMake install target
  (https://github.com/fmtlib/fmt/pull/119). Thanks @TrentHouliston.

  You can now install C++ Format with `make install` command.

- Improved [Biicode](http://www.biicode.com/) support
  (https://github.com/fmtlib/fmt/pull/98 and
  https://github.com/fmtlib/fmt/pull/104).
  Thanks @MariadeAnton and @franramirez688.

- Improved support for building with [Android NDK](
  https://developer.android.com/tools/sdk/ndk/index.html)
  (https://github.com/fmtlib/fmt/pull/107). Thanks @newnon.

  The [android-ndk-example](https://github.com/fmtlib/android-ndk-example)
  repository provides and example of using C++ Format with Android NDK:

  ![](https://raw.githubusercontent.com/fmtlib/android-ndk-example/master/screenshot.png)

- Improved documentation of `SystemError` and `WindowsError`
  (https://github.com/fmtlib/fmt/issues/54).

- Various code improvements
  (https://github.com/fmtlib/fmt/pull/110,
  https://github.com/fmtlib/fmt/pull/111
  https://github.com/fmtlib/fmt/pull/112). Thanks @CarterLi.

- Improved compile-time errors when formatting wide into narrow
  strings (https://github.com/fmtlib/fmt/issues/117).

- Fixed `BasicWriter::write` without formatting arguments when C++11
  support is disabled
  (https://github.com/fmtlib/fmt/issues/109).

- Fixed header-only build on OS X with GCC 4.9
  (https://github.com/fmtlib/fmt/issues/124).

- Fixed packaging issues (https://github.com/fmtlib/fmt/issues/94).

- Added [changelog](https://github.com/fmtlib/fmt/blob/master/ChangeLog.md)
  (https://github.com/fmtlib/fmt/issues/103).

# 1.0.0 - 2015-02-05

- Add support for a header-only configuration when `FMT_HEADER_ONLY`
  is defined before including `format.h`:

  ```c++
  #define FMT_HEADER_ONLY
  #include "format.h"
  ```

- Compute string length in the constructor of `BasicStringRef` instead
  of the `size` method
  (https://github.com/fmtlib/fmt/issues/79). This eliminates
  size computation for string literals on reasonable optimizing
  compilers.

- Fix formatting of types with overloaded `operator <<` for
  `std::wostream` (https://github.com/fmtlib/fmt/issues/86):

  ```c++
  fmt::format(L"The date is {0}", Date(2012, 12, 9));
  ```

- Fix linkage of tests on Arch Linux
  (https://github.com/fmtlib/fmt/issues/89).

- Allow precision specifier for non-float arguments
  (https://github.com/fmtlib/fmt/issues/90):

  ```c++
  fmt::print("{:.3}\n", "Carpet"); // prints "Car"
  ```

- Fix build on Android NDK (https://github.com/fmtlib/fmt/issues/93).

- Improvements to documentation build procedure.

- Remove `FMT_SHARED` CMake variable in favor of standard [BUILD_SHARED_LIBS](
  http://www.cmake.org/cmake/help/v3.0/variable/BUILD_SHARED_LIBS.html).

- Fix error handling in `fmt::fprintf`.

- Fix a number of warnings.

# 0.12.0 - 2014-10-25

- \[Breaking\] Improved separation between formatting and buffer
  management. `Writer` is now a base class that cannot be instantiated
  directly. The new `MemoryWriter` class implements the default buffer
  management with small allocations done on stack. So `fmt::Writer`
  should be replaced with `fmt::MemoryWriter` in variable
  declarations.

  Old code:

  ```c++
  fmt::Writer w;
  ```

  New code:

  ```c++
  fmt::MemoryWriter w;
  ```

  If you pass `fmt::Writer` by reference, you can continue to do so:

  ```c++
  void f(fmt::Writer &w);
  ```

  This doesn\'t affect the formatting API.

- Support for custom memory allocators
  (https://github.com/fmtlib/fmt/issues/69)

- Formatting functions now accept [signed char]{.title-ref} and
  [unsigned char]{.title-ref} strings as arguments
  (https://github.com/fmtlib/fmt/issues/73):

  ```c++
  auto s = format("GLSL version: {}", glGetString(GL_VERSION));
  ```

- Reduced code bloat. According to the new [benchmark
  results](https://github.com/fmtlib/fmt#compile-time-and-code-bloat),
  cppformat is close to `printf` and by the order of magnitude better
  than Boost Format in terms of compiled code size.

- Improved appearance of the documentation on mobile by using the
  [Sphinx Bootstrap
  theme](http://ryan-roemer.github.io/sphinx-bootstrap-theme/):

  | Old | New |
  | --- | --- |
  | ![](https://cloud.githubusercontent.com/assets/576385/4792130/cd256436-5de3-11e4-9a62-c077d0c2b003.png) | ![](https://cloud.githubusercontent.com/assets/576385/4792131/cd29896c-5de3-11e4-8f59-cac952942bf0.png) |

# 0.11.0 - 2014-08-21

- Safe printf implementation with a POSIX extension for positional
  arguments:

  ```c++
  fmt::printf("Elapsed time: %.2f seconds", 1.23);
  fmt::printf("%1$s, %3$d %2$s", weekday, month, day);
  ```

- Arguments of `char` type can now be formatted as integers (Issue
  https://github.com/fmtlib/fmt/issues/55):

  ```c++
  fmt::format("0x{0:02X}", 'a');
  ```

- Deprecated parts of the API removed.

- The library is now built and tested on MinGW with Appveyor in
  addition to existing test platforms Linux/GCC, OS X/Clang,
  Windows/MSVC.

# 0.10.0 - 2014-07-01

**Improved API**

- All formatting methods are now implemented as variadic functions
  instead of using `operator<<` for feeding arbitrary arguments into a
  temporary formatter object. This works both with C++11 where
  variadic templates are used and with older standards where variadic
  functions are emulated by providing lightweight wrapper functions
  defined with the `FMT_VARIADIC` macro. You can use this macro for
  defining your own portable variadic functions:

  ```c++
  void report_error(const char *format, const fmt::ArgList &args) {
    fmt::print("Error: {}");
    fmt::print(format, args);
  }
  FMT_VARIADIC(void, report_error, const char *)

  report_error("file not found: {}", path);
  ```

  Apart from a more natural syntax, this also improves performance as
  there is no need to construct temporary formatter objects and
  control arguments\' lifetimes. Because the wrapper functions are
  very lightweight, this doesn\'t cause code bloat even in pre-C++11
  mode.

- Simplified common case of formatting an `std::string`. Now it
  requires a single function call:

  ```c++
  std::string s = format("The answer is {}.", 42);
  ```

  Previously it required 2 function calls:

  ```c++
  std::string s = str(Format("The answer is {}.") << 42);
  ```

  Instead of unsafe `c_str` function, `fmt::Writer` should be used
  directly to bypass creation of `std::string`:

  ```c++
  fmt::Writer w;
  w.write("The answer is {}.", 42);
  w.c_str();  // returns a C string
  ```

  This doesn\'t do dynamic memory allocation for small strings and is
  less error prone as the lifetime of the string is the same as for
  `std::string::c_str` which is well understood (hopefully).

- Improved consistency in naming functions that are a part of the
  public API. Now all public functions are lowercase following the
  standard library conventions. Previously it was a combination of
  lowercase and CapitalizedWords. Issue
  https://github.com/fmtlib/fmt/issues/50.

- Old functions are marked as deprecated and will be removed in the
  next release.

**Other Changes**

- Experimental support for printf format specifications (work in
  progress):

  ```c++
  fmt::printf("The answer is %d.", 42);
  std::string s = fmt::sprintf("Look, a %s!", "string");
  ```

- Support for hexadecimal floating point format specifiers `a` and
  `A`:

  ```c++
  print("{:a}", -42.0); // Prints -0x1.5p+5
  print("{:A}", -42.0); // Prints -0X1.5P+5
  ```

- CMake option `FMT_SHARED` that specifies whether to build format as
  a shared library (off by default).

# 0.9.0 - 2014-05-13

- More efficient implementation of variadic formatting functions.

- `Writer::Format` now has a variadic overload:

  ```c++
  Writer out;
  out.Format("Look, I'm {}!", "variadic");
  ```

- For efficiency and consistency with other overloads, variadic
  overload of the `Format` function now returns `Writer` instead of
  `std::string`. Use the `str` function to convert it to
  `std::string`:

  ```c++
  std::string s = str(Format("Look, I'm {}!", "variadic"));
  ```

- Replaced formatter actions with output sinks: `NoAction` -\>
  `NullSink`, `Write` -\> `FileSink`, `ColorWriter` -\>
  `ANSITerminalSink`. This improves naming consistency and shouldn\'t
  affect client code unless these classes are used directly which
  should be rarely needed.

- Added `ThrowSystemError` function that formats a message and throws
  `SystemError` containing the formatted message and system-specific
  error description. For example, the following code

  ```c++
  FILE *f = fopen(filename, "r");
  if (!f)
    ThrowSystemError(errno, "Failed to open file '{}'") << filename;
  ```

  will throw `SystemError` exception with description \"Failed to open
  file \'\<filename\>\': No such file or directory\" if file doesn\'t
  exist.

- Support for AppVeyor continuous integration platform.

- `Format` now throws `SystemError` in case of I/O errors.

- Improve test infrastructure. Print functions are now tested by
  redirecting the output to a pipe.

# 0.8.0 - 2014-04-14

- Initial release
