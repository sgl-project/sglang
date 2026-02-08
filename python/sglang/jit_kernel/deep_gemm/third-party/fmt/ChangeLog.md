# 11.2.0 - 2025-05-03

- Added the `s` specifier for `std::error_code`. It allows formatting an error
  message as a string. For example:

  ```c++
  #include <fmt/std.h>

  int main() {
    auto ec = std::make_error_code(std::errc::no_such_file_or_directory);
    fmt::print("{:s}\n", ec);
  }
  ```

  prints

  ```
  No such file or directory
  ```
  (The actual message is platform-specific.)

- Fixed formatting of `std::chrono::local_time` and `tm`
  (https://github.com/fmtlib/fmt/issues/3815,
  https://github.com/fmtlib/fmt/issues/4350).
  For example ([godbolt](https://www.godbolt.org/z/8o4b1PPn5)):

  ```c++
  #include <fmt/chrono.h>

  int main() {
    std::chrono::zoned_time zt(
      std::chrono::current_zone(),
      std::chrono::system_clock::now());
    fmt::print("{}", zt.get_local_time());
  }
  ```

  is now formatted consistenly across platforms.

- Added diagnostics for cases when timezone information is not available.
  For example:

  ```c++
  fmt::print("{:Z}", std::chrono::local_seconds());
  ```

  now gives a compile-time error.

- Deprecated `fmt::localtime` in favor of `std::localtime`.

- Fixed compilation with GCC 15 and C++20 modules enabled
  (https://github.com/fmtlib/fmt/pull/4347). Thanks @tkhyn.

- Fixed handling of named arguments in format specs
  (https://github.com/fmtlib/fmt/issues/4360,
  https://github.com/fmtlib/fmt/pull/4361). Thanks @dinomight.

- Added error reporting for duplicate named arguments
  (https://github.com/fmtlib/fmt/pull/4367). Thanks @dinomight.

- Fixed formatting of `long` with `FMT_BUILTIN_TYPES=0`
  (https://github.com/fmtlib/fmt/issues/4375,
  https://github.com/fmtlib/fmt/issues/4394).

- Optimized `text_style` using bit packing
  (https://github.com/fmtlib/fmt/pull/4363). Thanks @LocalSpook.

- Added support for incomplete types (https://github.com/fmtlib/fmt/issues/3180,
  https://github.com/fmtlib/fmt/pull/4383). Thanks @LocalSpook.

- Fixed a flush issue in `fmt::print` when using libstdc++
  (https://github.com/fmtlib/fmt/issues/4398).

- Fixed `fmt::println` usage with `FMT_ENFORCE_COMPILE_STRING` and legacy
  compile-time checks (https://github.com/fmtlib/fmt/pull/4407).
  Thanks @madmaxoft.

- Removed legacy header `fmt/core.h` from docs
  (https://github.com/fmtlib/fmt/pull/4421,
  https://github.com/fmtlib/fmt/pull/4422). Thanks @krzysztofkortas.

- Worked around limitations of `__builtin_strlen` during constant evaluation
  (https://github.com/fmtlib/fmt/issues/4423,
  https://github.com/fmtlib/fmt/pull/4429). Thanks @BRevzin.

- Worked around a bug in MSVC v141 (https://github.com/fmtlib/fmt/issues/4412,
  https://github.com/fmtlib/fmt/pull/4413). Thanks @hirohira9119.

- Removed the `fmt_detail` namespace
  (https://github.com/fmtlib/fmt/issues/4324).

- Removed specializations of `std::is_floating_point` in tests
  (https://github.com/fmtlib/fmt/issues/4417).

- Fixed a CMake error when setting `CMAKE_MODULE_PATH` in the pedantic mode
  (https://github.com/fmtlib/fmt/pull/4426). Thanks @rlalik.

- Updated the Bazel config (https://github.com/fmtlib/fmt/pull/4400).
  Thanks @Vertexwahn.

# 11.1.4 - 2025-02-26

- Fixed ABI compatibility with earlier 11.x versions on Windows
  (https://github.com/fmtlib/fmt/issues/4359).

- Improved the logic of switching between fixed and exponential format for
  `float` (https://github.com/fmtlib/fmt/issues/3649).

- Moved `is_compiled_string` to the public API
  (https://github.com/fmtlib/fmt/issues/4342). Thanks @SwooshyCueb.

- Simplified implementation of `operator""_cf`
  (https://github.com/fmtlib/fmt/pull/4349). Thanks @LocalSpook.

- Fixed `__builtin_strlen` detection (https://github.com/fmtlib/fmt/pull/4329).
  Thanks @LocalSpook.

- Fixed handling of BMI paths with the Ninja generator
  (https://github.com/fmtlib/fmt/pull/4344). Thanks @tkhyn.

- Fixed gcc 8.3 compile errors (https://github.com/fmtlib/fmt/issues/4331,
  https://github.com/fmtlib/fmt/pull/4336). Thanks @sergiud.

- Fixed a bogus MSVC warning (https://github.com/fmtlib/fmt/pull/4356).
  Thanks @dinomight.

# 11.1.3 - 2025-01-25

- Fixed compilation on GCC 9.4 (https://github.com/fmtlib/fmt/issues/4313).

- Worked around an internal compiler error when using C++20 modules with GCC
  14.2 and earlier (https://github.com/fmtlib/fmt/issues/4295).

- Worked around a bug in GCC 6 (https://github.com/fmtlib/fmt/issues/4318).

- Fixed an issue caused by instantiating `formatter<const T>`
  (https://github.com/fmtlib/fmt/issues/4303,
  https://github.com/fmtlib/fmt/pull/4325). Thanks @timsong-cpp.

- Fixed formatting into `std::ostreambuf_iterator` when using format string
  compilation (https://github.com/fmtlib/fmt/issues/4309,
  https://github.com/fmtlib/fmt/pull/4312). Thanks @phprus.

- Restored a constraint on the map formatter so that it correctly reports as
  unformattable when the element is (https://github.com/fmtlib/fmt/pull/4326).
  Thanks @timsong-cpp.

- Reduced the size of format specs (https://github.com/fmtlib/fmt/issues/4298).

- Readded `args()` to `fmt::format_context`
  (https://github.com/fmtlib/fmt/issues/4307,
  https://github.com/fmtlib/fmt/pull/4310). Thanks @Erroneous1.

- Fixed a bogus MSVC warning (https://github.com/fmtlib/fmt/issues/4314,
  https://github.com/fmtlib/fmt/pull/4322). Thanks @ZehMatt.

- Fixed a pedantic mode error in the CMake config
  (https://github.com/fmtlib/fmt/pull/4327). Thanks @rlalik.

# 11.1.2 - 2025-01-12

- Fixed ABI compatibility with earlier 11.x versions
  (https://github.com/fmtlib/fmt/issues/4292).

- Added `wchar_t` support to the `std::bitset` formatter
  (https://github.com/fmtlib/fmt/issues/4285,
  https://github.com/fmtlib/fmt/pull/4286,
  https://github.com/fmtlib/fmt/issues/4289,
  https://github.com/fmtlib/fmt/pull/4290). Thanks @phprus.

- Prefixed CMake components with `fmt-` to simplify usage of {fmt} via
  `add_subdirectory` (https://github.com/fmtlib/fmt/issues/4283).

- Updated docs for meson (https://github.com/fmtlib/fmt/pull/4291).
  Thanks @trim21.

- Fixed a compilation error in chrono on nvcc
  (https://github.com/fmtlib/fmt/issues/4297,
  https://github.com/fmtlib/fmt/pull/4301). Thanks @breyerml.

- Fixed various warnings
  (https://github.com/fmtlib/fmt/pull/4288,
  https://github.com/fmtlib/fmt/pull/4299). Thanks @GamesTrap and @edo9300.

# 11.1.1 - 2024-12-27

- Fixed ABI compatibility with earlier 11.x versions
  (https://github.com/fmtlib/fmt/issues/4278).

- Defined CMake components (`core` and `doc`) to allow docs to be installed
  separately (https://github.com/fmtlib/fmt/pull/4276).
  Thanks @carlsmedstad.

# 11.1.0 - 2024-12-25

- Improved C++20 module support
  (https://github.com/fmtlib/fmt/issues/4081,
  https://github.com/fmtlib/fmt/pull/4083,
  https://github.com/fmtlib/fmt/pull/4084,
  https://github.com/fmtlib/fmt/pull/4152,
  https://github.com/fmtlib/fmt/issues/4153,
  https://github.com/fmtlib/fmt/pull/4169,
  https://github.com/fmtlib/fmt/issues/4190,
  https://github.com/fmtlib/fmt/issues/4234,
  https://github.com/fmtlib/fmt/pull/4239).
  Thanks @kamrann and @Arghnews.

- Reduced debug (unoptimized) binary code size and the number of template
  instantiations when passing formatting arguments. For example, unoptimized
  binary code size for `fmt::print("{}", 42)` was reduced by ~40% on GCC and
  ~60% on clang (x86-64).

  GCC:
  - Before: 161 instructions of which 105 are in reusable functions
    ([godbolt](https://www.godbolt.org/z/s9bGoo4ze)).
  - After: 116 instructions of which 60 are in reusable functions
    ([godbolt](https://www.godbolt.org/z/r7GGGxMs6)).

  Clang:
  - Before: 310 instructions of which 251 are in reusable functions
    ([godbolt](https://www.godbolt.org/z/Ts88b7M9o)).
  - After: 194 instructions of which 135 are in reusable functions
    ([godbolt](https://www.godbolt.org/z/vcrjP8ceW)).

- Added an experimental `fmt::writer` API that can be used for writing to
  different destinations such as files or strings
  (https://github.com/fmtlib/fmt/issues/2354).
  For example ([godbolt](https://www.godbolt.org/z/rWoKfbP7e)):

  ```c++
  #include <fmt/os.h>

  void write_text(fmt::writer w) {
    w.print("The answer is {}.", 42);
  }

  int main() {
    // Write to FILE.
    write_text(stdout);

    // Write to fmt::ostream.
    auto f = fmt::output_file("myfile");
    write_text(f);

    // Write to std::string.
    auto sb = fmt::string_buffer();
    write_text(sb);
    std::string s = sb.str();
  }
  ```

- Added width and alignment support to the formatter of `std::error_code`.

- Made `std::expected<void, E>` formattable
  (https://github.com/fmtlib/fmt/issues/4145,
  https://github.com/fmtlib/fmt/pull/4148).
  For example ([godbolt](https://www.godbolt.org/z/hrj5c6G86)):

  ```c++
  fmt::print("{}", std::expected<void, int>());
  ```

  prints

  ```
  expected()
  ```

  Thanks @phprus.

- Made `fmt::is_formattable<void>` SFINAE-friendly
  (https://github.com/fmtlib/fmt/issues/4147).

- Added support for `_BitInt` formatting when using clang
  (https://github.com/fmtlib/fmt/issues/4007,
  https://github.com/fmtlib/fmt/pull/4072,
  https://github.com/fmtlib/fmt/issues/4140,
  https://github.com/fmtlib/fmt/issues/4173,
  https://github.com/fmtlib/fmt/pull/4176).
  For example ([godbolt](https://www.godbolt.org/z/KWjbWec5z)):

  ```c++
  using int42 = _BitInt(42);
  fmt::print("{}", int42(100));
  ```

  Thanks @Arghnews.

- Added the `n` specifier for tuples and pairs
  (https://github.com/fmtlib/fmt/pull/4107). Thanks @someonewithpc.

- Added support for tuple-like types to `fmt::join`
  (https://github.com/fmtlib/fmt/issues/4226,
  https://github.com/fmtlib/fmt/pull/4230). Thanks @phprus.

- Made more types formattable at compile time
  (https://github.com/fmtlib/fmt/pull/4127). Thanks @AnthonyVH.

- Implemented a more efficient compile-time `fmt::formatted_size`
  (https://github.com/fmtlib/fmt/issues/4102,
  https://github.com/fmtlib/fmt/pull/4103). Thanks @phprus.

- Fixed compile-time formatting of some string types
  (https://github.com/fmtlib/fmt/pull/4065). Thanks @torshepherd.

- Made compiled version of `fmt::format_to` work with
  `std::back_insert_iterator<std::vector<char>>`
  (https://github.com/fmtlib/fmt/issues/4206,
  https://github.com/fmtlib/fmt/pull/4211). Thanks @phprus.

- Added a formatter for `std::reference_wrapper`
  (https://github.com/fmtlib/fmt/pull/4163,
  https://github.com/fmtlib/fmt/pull/4164). Thanks @yfeldblum and @phprus.

- Added experimental padding support (glibc `strftime` extension) to `%m`, `%j`
  and `%Y` (https://github.com/fmtlib/fmt/pull/4161). Thanks @KKhanhH.

- Made microseconds formatted as `us` instead of `µs` if the Unicode support is
  disabled (https://github.com/fmtlib/fmt/issues/4088).

- Fixed an unreleased regression in transcoding of surrogate pairs
  (https://github.com/fmtlib/fmt/issues/4094,
  https://github.com/fmtlib/fmt/pull/4095). Thanks @phprus.

- Made `fmt::appender` satisfy `std::output_iterator` concept
  (https://github.com/fmtlib/fmt/issues/4092,
  https://github.com/fmtlib/fmt/pull/4093). Thanks @phprus.

- Made `std::iterator_traits<fmt::appender>` standard-conforming
  (https://github.com/fmtlib/fmt/pull/4185). Thanks @CaseyCarter.

- Made it easier to reuse `fmt::formatter<std::string_view>` for types with
  an implicit conversion to `std::string_view`
  (https://github.com/fmtlib/fmt/issues/4036,
  https://github.com/fmtlib/fmt/pull/4055). Thanks @Arghnews.

- Made it possible to disable `<filesystem>` use via `FMT_CPP_LIB_FILESYSTEM`
  for compatibility with some video game console SDKs, e.g. Nintendo Switch SDK
  (https://github.com/fmtlib/fmt/issues/4257,
  https://github.com/fmtlib/fmt/pull/4258,
  https://github.com/fmtlib/fmt/pull/4259). Thanks @W4RH4WK and @phprus.

- Fixed compatibility with platforms that use 80-bit `long double`
  (https://github.com/fmtlib/fmt/issues/4245,
  https://github.com/fmtlib/fmt/pull/4246). Thanks @jsirpoma.

- Added support for UTF-32 code units greater than `0xFFFF` in fill
  (https://github.com/fmtlib/fmt/issues/4201).

- Fixed handling of legacy encodings on Windows with GCC
  (https://github.com/fmtlib/fmt/issues/4162).

- Made `fmt::to_string` take `fmt::basic_memory_buffer` by const reference
  (https://github.com/fmtlib/fmt/issues/4261,
  https://github.com/fmtlib/fmt/pull/4262). Thanks @sascha-devel.

- Added `fmt::dynamic_format_arg_store::size`
  (https://github.com/fmtlib/fmt/pull/4270). Thanks @hannes-harnisch.

- Removed the ability to control locale usage via an undocumented
  `FMT_STATIC_THOUSANDS_SEPARATOR` in favor of `FMT_USE_LOCALE`.

- Renamed `FMT_EXCEPTIONS` to `FMT_USE_EXCEPTIONS` for consistency with other
  similar macros.

- Improved include directory ordering to reduce the chance of including
  incorrect headers when using multiple versions of {fmt}
  (https://github.com/fmtlib/fmt/pull/4116). Thanks @cdzhan.

- Made it possible to compile a subset of {fmt} without the C++ runtime.

- Improved documentation and README
  (https://github.com/fmtlib/fmt/pull/4066,
  https://github.com/fmtlib/fmt/issues/4117,
  https://github.com/fmtlib/fmt/issues/4203,
  https://github.com/fmtlib/fmt/pull/4235). Thanks @zyctree and @nikola-sh.

- Improved the documentation generator (https://github.com/fmtlib/fmt/pull/4110,
  https://github.com/fmtlib/fmt/pull/4115). Thanks @rturrado.

- Improved CI (https://github.com/fmtlib/fmt/pull/4155,
  https://github.com/fmtlib/fmt/pull/4151). Thanks @phprus.

- Fixed various warnings and compilation issues
  (https://github.com/fmtlib/fmt/issues/2708,
  https://github.com/fmtlib/fmt/issues/4091,
  https://github.com/fmtlib/fmt/issues/4109,
  https://github.com/fmtlib/fmt/issues/4113,
  https://github.com/fmtlib/fmt/issues/4125,
  https://github.com/fmtlib/fmt/issues/4129,
  https://github.com/fmtlib/fmt/pull/4130,
  https://github.com/fmtlib/fmt/pull/4131,
  https://github.com/fmtlib/fmt/pull/4132,
  https://github.com/fmtlib/fmt/issues/4133,
  https://github.com/fmtlib/fmt/issues/4144,
  https://github.com/fmtlib/fmt/issues/4150,
  https://github.com/fmtlib/fmt/issues/4158,
  https://github.com/fmtlib/fmt/pull/4159,
  https://github.com/fmtlib/fmt/issues/4160,
  https://github.com/fmtlib/fmt/pull/4170,
  https://github.com/fmtlib/fmt/issues/4177,
  https://github.com/fmtlib/fmt/pull/4187,
  https://github.com/fmtlib/fmt/pull/4188,
  https://github.com/fmtlib/fmt/pull/4194,
  https://github.com/fmtlib/fmt/pull/4200,
  https://github.com/fmtlib/fmt/issues/4205,
  https://github.com/fmtlib/fmt/issues/4207,
  https://github.com/fmtlib/fmt/pull/4208,
  https://github.com/fmtlib/fmt/pull/4210,
  https://github.com/fmtlib/fmt/issues/4220,
  https://github.com/fmtlib/fmt/issues/4231,
  https://github.com/fmtlib/fmt/issues/4232,
  https://github.com/fmtlib/fmt/pull/4233,
  https://github.com/fmtlib/fmt/pull/4236,
  https://github.com/fmtlib/fmt/pull/4267,
  https://github.com/fmtlib/fmt/pull/4271).
  Thanks @torsten48, @Arghnews, @tinfoilboy, @aminya, @Ottani, @zeroomega,
  @c4v4, @kongy, @vinayyadav3016, @sergio-nsk, @phprus and @YexuanXiao.

# 11.0.2 - 2024-07-20

- Fixed compatibility with non-POSIX systems
  (https://github.com/fmtlib/fmt/issues/4054,
  https://github.com/fmtlib/fmt/issues/4060).

- Fixed performance regressions when using `std::back_insert_iterator` with
  `fmt::format_to` (https://github.com/fmtlib/fmt/issues/4070).

- Fixed handling of `std::generator` and move-only iterators
  (https://github.com/fmtlib/fmt/issues/4053,
  https://github.com/fmtlib/fmt/pull/4057). Thanks @Arghnews.

- Made `formatter<std::string_view>::parse` work with types convertible to
  `std::string_view` (https://github.com/fmtlib/fmt/issues/4036,
  https://github.com/fmtlib/fmt/pull/4055). Thanks @Arghnews.

- Made `volatile void*` formattable
  (https://github.com/fmtlib/fmt/issues/4049,
  https://github.com/fmtlib/fmt/pull/4056). Thanks @Arghnews.

- Made `Glib::ustring` not be confused with `std::string`
  (https://github.com/fmtlib/fmt/issues/4052).

- Made `fmt::context` iterator compatible with STL algorithms that rely on
  iterator category (https://github.com/fmtlib/fmt/issues/4079).

# 11.0.1 - 2024-07-05

- Fixed version number in the inline namespace
  (https://github.com/fmtlib/fmt/issues/4047).

- Fixed disabling Unicode support via CMake
  (https://github.com/fmtlib/fmt/issues/4051).

- Fixed deprecated `visit_format_arg` (https://github.com/fmtlib/fmt/pull/4043).
  Thanks @nebkat.

- Fixed handling of a sign and improved the `std::complex` formater
  (https://github.com/fmtlib/fmt/pull/4034,
  https://github.com/fmtlib/fmt/pull/4050). Thanks @tesch1 and @phprus.

- Fixed ADL issues in `fmt::printf` when using C++20
  (https://github.com/fmtlib/fmt/pull/4042). Thanks @toge.

- Removed a redundant check in the formatter for `std::expected`
  (https://github.com/fmtlib/fmt/pull/4040). Thanks @phprus.

# 11.0.0 - 2024-07-01

- Added `fmt/base.h` which provides a subset of the API with minimal include
  dependencies and enough functionality to replace all uses of the `printf`
  family of functions. This brings the compile time of code using {fmt} much
  closer to the equivalent `printf` code as shown on the following benchmark
  that compiles 100 source files:

  | Method       | Compile Time (s) |
  |--------------|------------------|
  | printf       | 1.6              |
  | IOStreams    | 25.9             |
  | fmt 10.x     | 19.0             |
  | fmt 11.0     | 4.8              |
  | tinyformat   | 29.1             |
  | Boost Format | 55.0             |

  This gives almost 4x improvement in build speed compared to version 10.
  Note that the benchmark is purely formatting code and includes. In real
  projects the difference from `printf` will be smaller partly because common
  standard headers will be included in almost any translation unit (TU) anyway.
  In particular, in every case except `printf` above ~1s is spent in total on
  including `<type_traits>` in all TUs.

- Optimized includes in other headers such as `fmt/format.h` which is now
  roughly equivalent to the old `fmt/core.h` in terms of build speed.

- Migrated the documentation at https://fmt.dev/ from Sphinx to MkDocs.

- Improved C++20 module support
  (https://github.com/fmtlib/fmt/issues/3990,
  https://github.com/fmtlib/fmt/pull/3991,
  https://github.com/fmtlib/fmt/issues/3993,
  https://github.com/fmtlib/fmt/pull/3994,
  https://github.com/fmtlib/fmt/pull/3997,
  https://github.com/fmtlib/fmt/pull/3998,
  https://github.com/fmtlib/fmt/pull/4004,
  https://github.com/fmtlib/fmt/pull/4005,
  https://github.com/fmtlib/fmt/pull/4006,
  https://github.com/fmtlib/fmt/pull/4013,
  https://github.com/fmtlib/fmt/pull/4027,
  https://github.com/fmtlib/fmt/pull/4029). In particular, native CMake support
  for modules is now used if available. Thanks @yujincheng08 and @matt77hias.

- Added an option to replace standard includes with `import std` enabled via
  the `FMT_IMPORT_STD` macro (https://github.com/fmtlib/fmt/issues/3921,
  https://github.com/fmtlib/fmt/pull/3928). Thanks @matt77hias.

- Exported `fmt::range_format`, `fmt::range_format_kind` and
  `fmt::compiled_string` from the `fmt` module
  (https://github.com/fmtlib/fmt/pull/3970,
  https://github.com/fmtlib/fmt/pull/3999).
  Thanks @matt77hias and @yujincheng08.

- Improved integration with stdio in `fmt::print`, enabling direct writes
  into a C stream buffer in common cases. This may give significant
  performance improvements ranging from tens of percent to [2x](
  https://stackoverflow.com/a/78457454/471164) and eliminates dynamic memory
  allocations on the buffer level. It is currently enabled for built-in and
  string types with wider availability coming up in future releases.

  For example, it gives ~24% improvement on a [simple benchmark](
  https://isocpp.org/files/papers/P3107R5.html#perf) compiled with Apple clang
  version 15.0.0 (clang-1500.1.0.2.5) and run on macOS 14.2.1:

  ```
  -------------------------------------------------------
  Benchmark             Time             CPU   Iterations
  -------------------------------------------------------
  printf             81.8 ns         81.5 ns      8496899
  fmt::print (10.x)  63.8 ns         61.9 ns     11524151
  fmt::print (11.0)  51.3 ns         51.0 ns     13846580
  ```

- Improved safety of `fmt::format_to` when writing to an array
  (https://github.com/fmtlib/fmt/pull/3805).
  For example ([godbolt](https://www.godbolt.org/z/cYrn8dWY8)):

  ```c++
  auto volkswagen = char[4];
  auto result = fmt::format_to(volkswagen, "elephant");
  ```

  no longer results in a buffer overflow. Instead the output will be truncated
  and you can get the end iterator and whether truncation occurred from the
  `result` object. Thanks @ThePhD.

- Enabled Unicode support by default in MSVC, bringing it on par with other
  compilers and making it unnecessary for users to enable it explicitly.
  Most of {fmt} is encoding-agnostic but this prevents mojibake in places
  where encoding matters such as path formatting and terminal output.
  You can control the Unicode support via the CMake `FMT_UNICODE` option.
  Note that some {fmt} packages such as the one in vcpkg have already been
  compiled with Unicode enabled.

- Added a formatter for `std::expected`
  (https://github.com/fmtlib/fmt/pull/3834). Thanks @dominicpoeschko.

- Added a formatter for `std::complex`
  (https://github.com/fmtlib/fmt/issues/1467,
  https://github.com/fmtlib/fmt/issues/3886,
  https://github.com/fmtlib/fmt/pull/3892,
  https://github.com/fmtlib/fmt/pull/3900). Thanks @phprus.

- Added a formatter for `std::type_info`
  (https://github.com/fmtlib/fmt/pull/3978). Thanks @matt77hias.

- Specialized `formatter` for `std::basic_string` types with custom traits
  and allocators (https://github.com/fmtlib/fmt/issues/3938,
  https://github.com/fmtlib/fmt/pull/3943). Thanks @dieram3.

- Added formatters for `std::chrono::day`, `std::chrono::month`,
  `std::chrono::year` and `std::chrono::year_month_day`
  (https://github.com/fmtlib/fmt/issues/3758,
  https://github.com/fmtlib/fmt/issues/3772,
  https://github.com/fmtlib/fmt/pull/3906,
  https://github.com/fmtlib/fmt/pull/3913). For example:

  ```c++
  #include <fmt/chrono.h>
  #include <fmt/color.h>

  int main() {
    fmt::print(fg(fmt::color::green), "{}\n", std::chrono::day(7));
  }
  ```

  prints a green day:

  <img width="306" alt="image" src="https://github.com/fmtlib/fmt/assets/576385/6e395f8b-451a-4cf7-bccc-ee92ca0dec65">

  Thanks @zivshek.

- Fixed handling of precision in `%S` (https://github.com/fmtlib/fmt/issues/3794,
  https://github.com/fmtlib/fmt/pull/3814). Thanks @js324.

- Added support for the `-` specifier (glibc `strftime` extension) to day of
  the month (`%d`) and week of the year (`%W`, `%U`, `%V`) specifiers
  (https://github.com/fmtlib/fmt/pull/3976). Thanks @ZaheenJ.

- Fixed the scope of the `-` extension in chrono formatting so that it doesn't
  apply to subsequent specifiers (https://github.com/fmtlib/fmt/issues/3811,
  https://github.com/fmtlib/fmt/pull/3812). Thanks @phprus.

- Improved handling of `time_point::min()`
  (https://github.com/fmtlib/fmt/issues/3282).

- Added support for character range formatting
  (https://github.com/fmtlib/fmt/issues/3857,
  https://github.com/fmtlib/fmt/pull/3863). Thanks @js324.

- Added `string` and `debug_string` range formatters
  (https://github.com/fmtlib/fmt/pull/3973,
  https://github.com/fmtlib/fmt/pull/4024). Thanks @matt77hias.

- Enabled ADL for `begin` and `end` in `fmt::join`
  (https://github.com/fmtlib/fmt/issues/3813,
  https://github.com/fmtlib/fmt/pull/3824). Thanks @bbolli.

- Made contiguous iterator optimizations apply to `std::basic_string` iterators
  (https://github.com/fmtlib/fmt/pull/3798). Thanks @phprus.

- Added support for ranges with mutable `begin` and `end`
  (https://github.com/fmtlib/fmt/issues/3752,
  https://github.com/fmtlib/fmt/pull/3800,
  https://github.com/fmtlib/fmt/pull/3955). Thanks @tcbrindle and @Arghnews.

- Added support for move-only iterators to `fmt::join`
  (https://github.com/fmtlib/fmt/issues/3802,
  https://github.com/fmtlib/fmt/pull/3946). Thanks @Arghnews.

- Moved range and iterator overloads of `fmt::join` to `fmt/ranges.h`, next
  to other overloads.

- Fixed handling of types with `begin` returning `void` such as Eigen matrices
  (https://github.com/fmtlib/fmt/issues/3839,
  https://github.com/fmtlib/fmt/pull/3964). Thanks @Arghnews.

- Added an `fmt::formattable` concept (https://github.com/fmtlib/fmt/pull/3974).
  Thanks @matt77hias.

- Added support for `__float128` (https://github.com/fmtlib/fmt/issues/3494).

- Fixed rounding issues when formatting `long double` with fixed precision
  (https://github.com/fmtlib/fmt/issues/3539).

- Made `fmt::isnan` not trigger floating-point exception for NaN values
  (https://github.com/fmtlib/fmt/issues/3948,
  https://github.com/fmtlib/fmt/pull/3951). Thanks @alexdewar.

- Removed dependency on `<memory>` for `std::allocator_traits` when possible
  (https://github.com/fmtlib/fmt/pull/3804). Thanks @phprus.

- Enabled compile-time checks in formatting functions that take text colors and
  styles.

- Deprecated wide stream overloads of `fmt::print` that take text styles.

- Made format string compilation work with clang 12 and later despite
  only partial non-type template parameter support
  (https://github.com/fmtlib/fmt/issues/4000,
  https://github.com/fmtlib/fmt/pull/4001). Thanks @yujincheng08.

- Made `fmt::iterator_buffer`'s move constructor `noexcept`
  (https://github.com/fmtlib/fmt/pull/3808). Thanks @waywardmonkeys.

- Started enforcing that `formatter::format` is const for compatibility
  with `std::format` (https://github.com/fmtlib/fmt/issues/3447).

- Added `fmt::basic_format_arg::visit` and deprecated `fmt::visit_format_arg`.

- Made `fmt::basic_string_view` not constructible from `nullptr` for
  consistency with `std::string_view` in C++23
  (https://github.com/fmtlib/fmt/pull/3846). Thanks @dalle.

- Fixed `fmt::group_digits` for negative integers
  (https://github.com/fmtlib/fmt/issues/3891,
  https://github.com/fmtlib/fmt/pull/3901). Thanks @phprus.

- Fixed handling of negative ids in `fmt::basic_format_args::get`
  (https://github.com/fmtlib/fmt/pull/3945). Thanks @marlenecota.

- Fixed handling of a buffer boundary on flush
  (https://github.com/fmtlib/fmt/issues/4229).

- Improved named argument validation
  (https://github.com/fmtlib/fmt/issues/3817).

- Disabled copy construction/assignment for `fmt::format_arg_store` and
  fixed moved construction (https://github.com/fmtlib/fmt/pull/3833).
  Thanks @ivafanas.

- Worked around a locale issue in RHEL/devtoolset
  (https://github.com/fmtlib/fmt/issues/3858,
  https://github.com/fmtlib/fmt/pull/3859). Thanks @g199209.

- Added RTTI detection for MSVC (https://github.com/fmtlib/fmt/pull/3821,
  https://github.com/fmtlib/fmt/pull/3963). Thanks @edo9300.

- Migrated the documentation from Sphinx to MkDocs.

- Improved documentation and README
  (https://github.com/fmtlib/fmt/issues/3775,
  https://github.com/fmtlib/fmt/pull/3784,
  https://github.com/fmtlib/fmt/issues/3788,
  https://github.com/fmtlib/fmt/pull/3789,
  https://github.com/fmtlib/fmt/pull/3793,
  https://github.com/fmtlib/fmt/issues/3818,
  https://github.com/fmtlib/fmt/pull/3820,
  https://github.com/fmtlib/fmt/pull/3822,
  https://github.com/fmtlib/fmt/pull/3843,
  https://github.com/fmtlib/fmt/pull/3890,
  https://github.com/fmtlib/fmt/issues/3894,
  https://github.com/fmtlib/fmt/pull/3895,
  https://github.com/fmtlib/fmt/pull/3905,
  https://github.com/fmtlib/fmt/issues/3942,
  https://github.com/fmtlib/fmt/pull/4008).
  Thanks @zencatalyst, WolleTD, @tupaschoal, @Dobiasd, @frank-weinberg, @bbolli,
  @phprus, @waywardmonkeys, @js324 and @tchaikov.

- Improved CI and tests
  (https://github.com/fmtlib/fmt/issues/3878,
  https://github.com/fmtlib/fmt/pull/3883,
  https://github.com/fmtlib/fmt/issues/3897,
  https://github.com/fmtlib/fmt/pull/3979,
  https://github.com/fmtlib/fmt/pull/3980,
  https://github.com/fmtlib/fmt/pull/3988,
  https://github.com/fmtlib/fmt/pull/4010,
  https://github.com/fmtlib/fmt/pull/4012,
  https://github.com/fmtlib/fmt/pull/4038).
  Thanks @vgorrX, @waywardmonkeys, @tchaikov and @phprus.

- Fixed buffer overflow when using format string compilation with debug format
  and `std::back_insert_iterator` (https://github.com/fmtlib/fmt/issues/3795,
  https://github.com/fmtlib/fmt/pull/3797). Thanks @phprus.

- Improved Bazel support
  (https://github.com/fmtlib/fmt/pull/3792,
  https://github.com/fmtlib/fmt/pull/3801,
  https://github.com/fmtlib/fmt/pull/3962,
  https://github.com/fmtlib/fmt/pull/3965). Thanks @Vertexwahn.

- Improved/fixed the CMake config
  (https://github.com/fmtlib/fmt/issues/3777,
  https://github.com/fmtlib/fmt/pull/3783,
  https://github.com/fmtlib/fmt/issues/3847,
  https://github.com/fmtlib/fmt/pull/3907). Thanks @phprus and @xTachyon.

- Fixed various warnings and compilation issues
  (https://github.com/fmtlib/fmt/issues/3685,
  https://github.com/fmtlib/fmt/issues/3769,
  https://github.com/fmtlib/fmt/issues/3796,
  https://github.com/fmtlib/fmt/issues/3803,
  https://github.com/fmtlib/fmt/pull/3806,
  https://github.com/fmtlib/fmt/pull/3807,
  https://github.com/fmtlib/fmt/issues/3809,
  https://github.com/fmtlib/fmt/pull/3810,
  https://github.com/fmtlib/fmt/issues/3830,
  https://github.com/fmtlib/fmt/pull/3832,
  https://github.com/fmtlib/fmt/issues/3835,
  https://github.com/fmtlib/fmt/pull/3844,
  https://github.com/fmtlib/fmt/issues/3854,
  https://github.com/fmtlib/fmt/pull/3856,
  https://github.com/fmtlib/fmt/pull/3865,
  https://github.com/fmtlib/fmt/pull/3866,
  https://github.com/fmtlib/fmt/pull/3880,
  https://github.com/fmtlib/fmt/issues/3881,
  https://github.com/fmtlib/fmt/issues/3884,
  https://github.com/fmtlib/fmt/issues/3898,
  https://github.com/fmtlib/fmt/pull/3899,
  https://github.com/fmtlib/fmt/pull/3909,
  https://github.com/fmtlib/fmt/pull/3917,
  https://github.com/fmtlib/fmt/pull/3923,
  https://github.com/fmtlib/fmt/pull/3924,
  https://github.com/fmtlib/fmt/issues/3925,
  https://github.com/fmtlib/fmt/pull/3930,
  https://github.com/fmtlib/fmt/pull/3931,
  https://github.com/fmtlib/fmt/pull/3933,
  https://github.com/fmtlib/fmt/issues/3935,
  https://github.com/fmtlib/fmt/pull/3937,
  https://github.com/fmtlib/fmt/pull/3967,
  https://github.com/fmtlib/fmt/pull/3968,
  https://github.com/fmtlib/fmt/pull/3972,
  https://github.com/fmtlib/fmt/pull/3983,
  https://github.com/fmtlib/fmt/issues/3992,
  https://github.com/fmtlib/fmt/pull/3995,
  https://github.com/fmtlib/fmt/pull/4009,
  https://github.com/fmtlib/fmt/pull/4023).
  Thanks @hmbj, @phprus, @res2k, @Baardi, @matt77hias, @waywardmonkeys, @hmbj,
  @yakra, @prlw1, @Arghnews, @mtillmann0, @ShifftC, @eepp, @jimmy-park and
  @ChristianGebhardt.

# 10.2.1 - 2024-01-04

- Fixed ABI compatibility with earlier 10.x versions
  (https://github.com/fmtlib/fmt/issues/3785,
  https://github.com/fmtlib/fmt/pull/3786). Thanks @saraedum.

# 10.2.0 - 2024-01-01

- Added support for the `%j` specifier (the number of days) for
  `std::chrono::duration` (https://github.com/fmtlib/fmt/issues/3643,
  https://github.com/fmtlib/fmt/pull/3732). Thanks @intelfx.

- Added support for the chrono suffix for days and changed
  the suffix for minutes from "m" to the correct "min"
  (https://github.com/fmtlib/fmt/issues/3662,
  https://github.com/fmtlib/fmt/pull/3664).
  For example ([godbolt](https://godbolt.org/z/9KhMnq9ba)):

  ```c++
  #include <fmt/chrono.h>

  int main() {
    fmt::print("{}\n", std::chrono::days(42)); // prints "42d"
  }
  ```

  Thanks @Richardk2n.

- Fixed an overflow in `std::chrono::time_point` formatting with large dates
  (https://github.com/fmtlib/fmt/issues/3725,
  https://github.com/fmtlib/fmt/pull/3727). Thanks @cschreib.

- Added a formatter for `std::source_location`
  (https://github.com/fmtlib/fmt/pull/3730).
  For example ([godbolt](https://godbolt.org/z/YajfKjhhr)):

  ```c++
  #include <source_location>
  #include <fmt/std.h>

  int main() {
    fmt::print("{}\n", std::source_location::current());
  }
  ```

  prints

  ```
  /app/example.cpp:5:51: int main()
  ```

  Thanks @felix642.

- Added a formatter for `std::bitset`
  (https://github.com/fmtlib/fmt/pull/3660).
  For example ([godbolt](https://godbolt.org/z/bdEaGeYxe)):

  ```c++
  #include <bitset>
  #include <fmt/std.h>

  int main() {
    fmt::print("{}\n", std::bitset<6>(42)); // prints "101010"
  }
  ```

  Thanks @muggenhor.

- Added an experimental `nested_formatter` that provides an easy way of
  applying a formatter to one or more subobjects while automatically handling
  width, fill and alignment. For example:

  ```c++
  #include <fmt/format.h>

  struct point {
    double x, y;
  };

  template <>
  struct fmt::formatter<point> : nested_formatter<double> {
    auto format(point p, format_context& ctx) const {
      return write_padded(ctx, [=](auto out) {
        return format_to(out, "({}, {})", nested(p.x), nested(p.y));
      });
    }
  };

  int main() {
    fmt::print("[{:>20.2f}]", point{1, 2});
  }
  ```

  prints

  ```
  [          (1.00, 2.00)]
  ```

- Added the generic representation (`g`) to `std::filesystem::path`
  (https://github.com/fmtlib/fmt/issues/3715,
  https://github.com/fmtlib/fmt/pull/3729). For example:

  ```c++
  #include <filesystem>
  #include <fmt/std.h>

  int main() {
    fmt::print("{:g}\n", std::filesystem::path("C:\\foo"));
  }
  ```

  prints `"C:/foo"` on Windows.

  Thanks @js324.

- Made `format_as` work with references
  (https://github.com/fmtlib/fmt/pull/3739). Thanks @tchaikov.

- Fixed formatting of invalid UTF-8 with precision
  (https://github.com/fmtlib/fmt/issues/3284).

- Fixed an inconsistency between `fmt::to_string` and `fmt::format`
  (https://github.com/fmtlib/fmt/issues/3684).

- Disallowed unsafe uses of `fmt::styled`
  (https://github.com/fmtlib/fmt/issues/3625):

  ```c++
  auto s = fmt::styled(std::string("dangle"), fmt::emphasis::bold);
  fmt::print("{}\n", s); // compile error
  ```

  Pass `fmt::styled(...)` as a parameter instead.

- Added a null check when formatting a C string with the `s` specifier
  (https://github.com/fmtlib/fmt/issues/3706).

- Disallowed the `c` specifier for `bool`
  (https://github.com/fmtlib/fmt/issues/3726,
  https://github.com/fmtlib/fmt/pull/3734). Thanks @js324.

- Made the default formatting unlocalized in `fmt::ostream_formatter` for
  consistency with the rest of the library
  (https://github.com/fmtlib/fmt/issues/3460).

- Fixed localized formatting in bases other than decimal
  (https://github.com/fmtlib/fmt/issues/3693,
  https://github.com/fmtlib/fmt/pull/3750). Thanks @js324.

- Fixed a performance regression in experimental `fmt::ostream::print`
  (https://github.com/fmtlib/fmt/issues/3674).

- Added synchronization with the underlying output stream when writing to
  the Windows console
  (https://github.com/fmtlib/fmt/pull/3668,
  https://github.com/fmtlib/fmt/issues/3688,
  https://github.com/fmtlib/fmt/pull/3689).
  Thanks @Roman-Koshelev and @dimztimz.

- Changed to only export `format_error` when {fmt} is built as a shared
  library (https://github.com/fmtlib/fmt/issues/3626,
  https://github.com/fmtlib/fmt/pull/3627). Thanks @phprus.

- Made `fmt::streamed` `constexpr`.
  (https://github.com/fmtlib/fmt/pull/3650). Thanks @muggenhor.

- Made `fmt::format_int` `constexpr`
  (https://github.com/fmtlib/fmt/issues/4031,
  https://github.com/fmtlib/fmt/pull/4032). Thanks @dixlorenz.

- Enabled `consteval` on older versions of MSVC
  (https://github.com/fmtlib/fmt/pull/3757). Thanks @phprus.

- Added an option to build without `wchar_t` support on Windows
  (https://github.com/fmtlib/fmt/issues/3631,
  https://github.com/fmtlib/fmt/pull/3636). Thanks @glebm.

- Improved build and CI configuration
  (https://github.com/fmtlib/fmt/pull/3679,
  https://github.com/fmtlib/fmt/issues/3701,
  https://github.com/fmtlib/fmt/pull/3702,
  https://github.com/fmtlib/fmt/pull/3749).
  Thanks @jcar87, @pklima and @tchaikov.

- Fixed various warnings, compilation and test issues
  (https://github.com/fmtlib/fmt/issues/3607,
  https://github.com/fmtlib/fmt/pull/3610,
  https://github.com/fmtlib/fmt/pull/3624,
  https://github.com/fmtlib/fmt/pull/3630,
  https://github.com/fmtlib/fmt/pull/3634,
  https://github.com/fmtlib/fmt/pull/3638,
  https://github.com/fmtlib/fmt/issues/3645,
  https://github.com/fmtlib/fmt/issues/3646,
  https://github.com/fmtlib/fmt/pull/3647,
  https://github.com/fmtlib/fmt/pull/3652,
  https://github.com/fmtlib/fmt/issues/3654,
  https://github.com/fmtlib/fmt/pull/3663,
  https://github.com/fmtlib/fmt/issues/3670,
  https://github.com/fmtlib/fmt/pull/3680,
  https://github.com/fmtlib/fmt/issues/3694,
  https://github.com/fmtlib/fmt/pull/3695,
  https://github.com/fmtlib/fmt/pull/3699,
  https://github.com/fmtlib/fmt/issues/3705,
  https://github.com/fmtlib/fmt/issues/3710,
  https://github.com/fmtlib/fmt/issues/3712,
  https://github.com/fmtlib/fmt/pull/3713,
  https://github.com/fmtlib/fmt/issues/3714,
  https://github.com/fmtlib/fmt/pull/3716,
  https://github.com/fmtlib/fmt/pull/3723,
  https://github.com/fmtlib/fmt/issues/3738,
  https://github.com/fmtlib/fmt/issues/3740,
  https://github.com/fmtlib/fmt/pull/3741,
  https://github.com/fmtlib/fmt/pull/3743,
  https://github.com/fmtlib/fmt/issues/3745,
  https://github.com/fmtlib/fmt/pull/3747,
  https://github.com/fmtlib/fmt/pull/3748,
  https://github.com/fmtlib/fmt/pull/3751,
  https://github.com/fmtlib/fmt/pull/3754,
  https://github.com/fmtlib/fmt/pull/3755,
  https://github.com/fmtlib/fmt/issues/3760,
  https://github.com/fmtlib/fmt/pull/3762,
  https://github.com/fmtlib/fmt/issues/3763,
  https://github.com/fmtlib/fmt/pull/3764,
  https://github.com/fmtlib/fmt/issues/3774,
  https://github.com/fmtlib/fmt/pull/3779).
  Thanks @danakj, @vinayyadav3016, @cyyever, @phprus, @qimiko, @saschasc,
  @gsjaardema, @lazka, @Zhaojun-Liu, @carlsmedstad, @hotwatermorning,
  @cptFracassa, @kuguma, @PeterJohnson, @H1X4Dev, @asantoni, @eltociear,
  @msimberg, @tchaikov, @waywardmonkeys.

- Improved documentation and README
  (https://github.com/fmtlib/fmt/issues/2086,
  https://github.com/fmtlib/fmt/issues/3637,
  https://github.com/fmtlib/fmt/pull/3642,
  https://github.com/fmtlib/fmt/pull/3653,
  https://github.com/fmtlib/fmt/pull/3655,
  https://github.com/fmtlib/fmt/pull/3661,
  https://github.com/fmtlib/fmt/issues/3673,
  https://github.com/fmtlib/fmt/pull/3677,
  https://github.com/fmtlib/fmt/pull/3737,
  https://github.com/fmtlib/fmt/issues/3742,
  https://github.com/fmtlib/fmt/pull/3744).
  Thanks @idzm, @perlun, @joycebrum, @fennewald, @reinhardt1053, @GeorgeLS.

- Updated CI dependencies
  (https://github.com/fmtlib/fmt/pull/3615,
  https://github.com/fmtlib/fmt/pull/3622,
  https://github.com/fmtlib/fmt/pull/3623,
  https://github.com/fmtlib/fmt/pull/3666,
  https://github.com/fmtlib/fmt/pull/3696,
  https://github.com/fmtlib/fmt/pull/3697,
  https://github.com/fmtlib/fmt/pull/3759,
  https://github.com/fmtlib/fmt/pull/3782).

# 10.1.1 - 2023-08-28

- Added formatters for `std::atomic` and `atomic_flag`
  (https://github.com/fmtlib/fmt/pull/3574,
  https://github.com/fmtlib/fmt/pull/3594).
  Thanks @wangzw and @AlexGuteniev.
- Fixed an error about partial specialization of `formatter<string>`
  after instantiation when compiled with gcc and C++20
  (https://github.com/fmtlib/fmt/issues/3584).
- Fixed compilation as a C++20 module with gcc and clang
  (https://github.com/fmtlib/fmt/issues/3587,
  https://github.com/fmtlib/fmt/pull/3597,
  https://github.com/fmtlib/fmt/pull/3605).
  Thanks @MathewBensonCode.
- Made `fmt::to_string` work with types that have `format_as`
  overloads (https://github.com/fmtlib/fmt/pull/3575). Thanks @phprus.
- Made `formatted_size` work with integral format specifiers at
  compile time (https://github.com/fmtlib/fmt/pull/3591).
  Thanks @elbeno.
- Fixed a warning about the `no_unique_address` attribute on clang-cl
  (https://github.com/fmtlib/fmt/pull/3599). Thanks @lukester1975.
- Improved compatibility with the legacy GBK encoding
  (https://github.com/fmtlib/fmt/issues/3598,
  https://github.com/fmtlib/fmt/pull/3599). Thanks @YuHuanTin.
- Added OpenSSF Scorecard analysis
  (https://github.com/fmtlib/fmt/issues/3530,
  https://github.com/fmtlib/fmt/pull/3571). Thanks @joycebrum.
- Updated CI dependencies
  (https://github.com/fmtlib/fmt/pull/3591,
  https://github.com/fmtlib/fmt/pull/3592,
  https://github.com/fmtlib/fmt/pull/3593,
  https://github.com/fmtlib/fmt/pull/3602).

# 10.1.0 - 2023-08-12

- Optimized format string compilation resulting in up to 40% speed up
  in compiled `format_to` and \~4x speed up in compiled `format_to_n`
  on a concatenation benchmark
  (https://github.com/fmtlib/fmt/issues/3133,
  https://github.com/fmtlib/fmt/issues/3484).

  {fmt} 10.0:

      ---------------------------------------------------------
      Benchmark               Time             CPU   Iterations
      ---------------------------------------------------------
      BM_format_to         78.9 ns         78.9 ns      8881746
      BM_format_to_n        568 ns          568 ns      1232089

  {fmt} 10.1:

      ---------------------------------------------------------
      Benchmark               Time             CPU   Iterations
      ---------------------------------------------------------
      BM_format_to         54.9 ns         54.9 ns     12727944
      BM_format_to_n        133 ns          133 ns      5257795

- Optimized storage of an empty allocator in `basic_memory_buffer`
  (https://github.com/fmtlib/fmt/pull/3485). Thanks @Minty-Meeo.

- Added formatters for proxy references to elements of
  `std::vector<bool>` and `std::bitset<N>`
  (https://github.com/fmtlib/fmt/issues/3567,
  https://github.com/fmtlib/fmt/pull/3570). For example
  ([godbolt](https://godbolt.org/z/zYb79Pvn8)):

  ```c++
  #include <vector>
  #include <fmt/std.h>

  int main() {
    auto v = std::vector<bool>{true};
    fmt::print("{}", v[0]);
  }
  ```

  Thanks @phprus and @felix642.

- Fixed an ambiguous formatter specialization for containers that look
  like container adaptors such as `boost::flat_set`
  (https://github.com/fmtlib/fmt/issues/3556,
  https://github.com/fmtlib/fmt/pull/3561). Thanks @5chmidti.

- Fixed compilation when formatting durations not convertible from
  `std::chrono::seconds`
  (https://github.com/fmtlib/fmt/pull/3430). Thanks @patlkli.

- Made the `formatter` specialization for `char*` const-correct
  (https://github.com/fmtlib/fmt/pull/3432). Thanks @timsong-cpp.

- Made `{}` and `{:}` handled consistently during compile-time checks
  (https://github.com/fmtlib/fmt/issues/3526).

- Disallowed passing temporaries to `make_format_args` to improve API
  safety by preventing dangling references.

- Improved the compile-time error for unformattable types
  (https://github.com/fmtlib/fmt/pull/3478). Thanks @BRevzin.

- Improved the floating-point formatter
  (https://github.com/fmtlib/fmt/pull/3448,
  https://github.com/fmtlib/fmt/pull/3450).
  Thanks @florimond-collette.

- Fixed handling of precision for `long double` larger than 64 bits.
  (https://github.com/fmtlib/fmt/issues/3539,
  https://github.com/fmtlib/fmt/issues/3564).

- Made floating-point and chrono tests less platform-dependent
  (https://github.com/fmtlib/fmt/issues/3337,
  https://github.com/fmtlib/fmt/issues/3433,
  https://github.com/fmtlib/fmt/pull/3434). Thanks @phprus.

- Removed the remnants of the Grisu floating-point formatter that has
  been replaced by Dragonbox in earlier versions.

- Added `throw_format_error` to the public API
  (https://github.com/fmtlib/fmt/pull/3551). Thanks @mjerabek.

- Made `FMT_THROW` assert even if assertions are disabled when
  compiling with exceptions disabled
  (https://github.com/fmtlib/fmt/issues/3418,
  https://github.com/fmtlib/fmt/pull/3439). Thanks @BRevzin.

- Made `format_as` and `std::filesystem::path` formatter work with
  exotic code unit types.
  (https://github.com/fmtlib/fmt/pull/3457,
  https://github.com/fmtlib/fmt/pull/3476). Thanks @gix and @hmbj.

- Added support for the `?` format specifier to
  `std::filesystem::path` and made the default unescaped for
  consistency with strings.

- Deprecated the wide stream overload of `printf`.

- Removed unused `basic_printf_parse_context`.

- Improved RTTI detection used when formatting exceptions
  (https://github.com/fmtlib/fmt/pull/3468). Thanks @danakj.

- Improved compatibility with VxWorks7
  (https://github.com/fmtlib/fmt/pull/3467). Thanks @wenshan1.

- Improved documentation
  (https://github.com/fmtlib/fmt/issues/3174,
  https://github.com/fmtlib/fmt/issues/3423,
  https://github.com/fmtlib/fmt/pull/3454,
  https://github.com/fmtlib/fmt/issues/3458,
  https://github.com/fmtlib/fmt/pull/3461,
  https://github.com/fmtlib/fmt/issues/3487,
  https://github.com/fmtlib/fmt/pull/3515).
  Thanks @zencatalyst, @rlalik and @mikecrowe.

- Improved build and CI configurations
  (https://github.com/fmtlib/fmt/issues/3449,
  https://github.com/fmtlib/fmt/pull/3451,
  https://github.com/fmtlib/fmt/pull/3452,
  https://github.com/fmtlib/fmt/pull/3453,
  https://github.com/fmtlib/fmt/pull/3459,
  https://github.com/fmtlib/fmt/issues/3481,
  https://github.com/fmtlib/fmt/pull/3486,
  https://github.com/fmtlib/fmt/issues/3489,
  https://github.com/fmtlib/fmt/pull/3496,
  https://github.com/fmtlib/fmt/issues/3517,
  https://github.com/fmtlib/fmt/pull/3523,
  https://github.com/fmtlib/fmt/pull/3563).
  Thanks @joycebrum, @glebm, @phprus, @petrmanek, @setoye and @abouvier.

- Fixed various warnings and compilation issues
  (https://github.com/fmtlib/fmt/issues/3408,
  https://github.com/fmtlib/fmt/issues/3424,
  https://github.com/fmtlib/fmt/issues/3444,
  https://github.com/fmtlib/fmt/pull/3446,
  https://github.com/fmtlib/fmt/pull/3475,
  https://github.com/fmtlib/fmt/pull/3482,
  https://github.com/fmtlib/fmt/issues/3492,
  https://github.com/fmtlib/fmt/pull/3493,
  https://github.com/fmtlib/fmt/pull/3508,
  https://github.com/fmtlib/fmt/issues/3509,
  https://github.com/fmtlib/fmt/issues/3533,
  https://github.com/fmtlib/fmt/pull/3542,
  https://github.com/fmtlib/fmt/issues/3543,
  https://github.com/fmtlib/fmt/issues/3540,
  https://github.com/fmtlib/fmt/pull/3544,
  https://github.com/fmtlib/fmt/issues/3548,
  https://github.com/fmtlib/fmt/pull/3549,
  https://github.com/fmtlib/fmt/pull/3550,
  https://github.com/fmtlib/fmt/pull/3552).
  Thanks @adesitter, @hmbj, @Minty-Meeo, @phprus, @TobiSchluter,
  @kieranclancy, @alexeedm, @jurihock, @Ozomahtli and @razaqq.

# 10.0.0 - 2023-05-09

- Replaced Grisu with a new floating-point formatting algorithm for
  given precision (https://github.com/fmtlib/fmt/issues/3262,
  https://github.com/fmtlib/fmt/issues/2750,
  https://github.com/fmtlib/fmt/pull/3269,
  https://github.com/fmtlib/fmt/pull/3276). The new algorithm
  is based on Dragonbox already used for the shortest representation
  and gives substantial performance improvement:

  ![](https://user-images.githubusercontent.com/33922675/211956670-84891a09-6867-47d9-82fc-3230da7abe0f.png)

  -   Red: new algorithm
  -   Green: new algorithm with `FMT_USE_FULL_CACHE_DRAGONBOX` defined
      to 1
  -   Blue: old algorithm

  Thanks @jk-jeon.

- Replaced `snprintf`-based hex float formatter with an internal
  implementation (https://github.com/fmtlib/fmt/pull/3179,
  https://github.com/fmtlib/fmt/pull/3203). This removes the
  last usage of `s(n)printf` in {fmt}. Thanks @phprus.

- Fixed alignment of floating-point numbers with localization
  (https://github.com/fmtlib/fmt/issues/3263,
  https://github.com/fmtlib/fmt/pull/3272). Thanks @ShawnZhong.

- Made handling of `#` consistent with `std::format`.

- Improved C++20 module support
  (https://github.com/fmtlib/fmt/pull/3134,
  https://github.com/fmtlib/fmt/pull/3254,
  https://github.com/fmtlib/fmt/pull/3386,
  https://github.com/fmtlib/fmt/pull/3387,
  https://github.com/fmtlib/fmt/pull/3388,
  https://github.com/fmtlib/fmt/pull/3392,
  https://github.com/fmtlib/fmt/pull/3397,
  https://github.com/fmtlib/fmt/pull/3399,
  https://github.com/fmtlib/fmt/pull/3400).
  Thanks @laitingsheng, @Orvid and @DanielaE.
  
- Switched to the [modules CMake library](https://github.com/vitaut/modules)
  which allows building {fmt} as a C++20 module with clang:

      CXX=clang++ cmake -DFMT_MODULE=ON .
      make

- Made `format_as` work with any user-defined type and not just enums.
  For example ([godbolt](https://godbolt.org/z/b7rqhq5Kh)):

  ```c++
  #include <fmt/format.h>

  struct floaty_mc_floatface {
    double value;
  };

  auto format_as(floaty_mc_floatface f) { return f.value; }

  int main() {
    fmt::print("{:8}\n", floaty_mc_floatface{0.42}); // prints "    0.42"
  }
  ```

- Removed deprecated implicit conversions for enums and conversions to
  primitive types for compatibility with `std::format` and to prevent
  potential ODR violations. Use `format_as` instead.

- Added support for fill, align and width to the time point formatter
  (https://github.com/fmtlib/fmt/issues/3237,
  https://github.com/fmtlib/fmt/pull/3260,
  https://github.com/fmtlib/fmt/pull/3275). For example
  ([godbolt](https://godbolt.org/z/rKP6MGz6c)):

  ```c++
  #include <fmt/chrono.h>

  int main() {
    // prints "    2023"
    fmt::print("{:>8%Y}\n", std::chrono::system_clock::now());
  }
  ```

  Thanks @ShawnZhong.

- Implemented formatting of subseconds
  (https://github.com/fmtlib/fmt/issues/2207,
  https://github.com/fmtlib/fmt/issues/3117,
  https://github.com/fmtlib/fmt/pull/3115,
  https://github.com/fmtlib/fmt/pull/3143,
  https://github.com/fmtlib/fmt/pull/3144,
  https://github.com/fmtlib/fmt/pull/3349). For example
  ([godbolt](https://godbolt.org/z/45738oGEo)):

  ```c++
  #include <fmt/chrono.h>

  int main() {
    // prints 01.234567
    fmt::print("{:%S}\n", std::chrono::microseconds(1234567));
  }
  ```

  Thanks @patrickroocks @phprus and @BRevzin.

- Added precision support to `%S`
  (https://github.com/fmtlib/fmt/pull/3148). Thanks @SappyJoy

- Added support for `std::utc_time`
  (https://github.com/fmtlib/fmt/issues/3098,
  https://github.com/fmtlib/fmt/pull/3110). Thanks @patrickroocks.

- Switched formatting of `std::chrono::system_clock` from local time
  to UTC for compatibility with the standard
  (https://github.com/fmtlib/fmt/issues/3199,
  https://github.com/fmtlib/fmt/pull/3230). Thanks @ned14.

- Added support for `%Ez` and `%Oz` to chrono formatters.
  (https://github.com/fmtlib/fmt/issues/3220,
  https://github.com/fmtlib/fmt/pull/3222). Thanks @phprus.

- Improved validation of format specifiers for `std::chrono::duration`
  (https://github.com/fmtlib/fmt/issues/3219,
  https://github.com/fmtlib/fmt/pull/3232). Thanks @ShawnZhong.

- Fixed formatting of time points before the epoch
  (https://github.com/fmtlib/fmt/issues/3117,
  https://github.com/fmtlib/fmt/pull/3261). For example
  ([godbolt](https://godbolt.org/z/f7bcznb3W)):

  ```c++
  #include <fmt/chrono.h>

  int main() {
    auto t = std::chrono::system_clock::from_time_t(0) -
             std::chrono::milliseconds(250);
    fmt::print("{:%S}\n", t); // prints 59.750000000
  }
  ```

  Thanks @ShawnZhong.

- Experimental: implemented glibc extension for padding seconds,
  minutes and hours
  (https://github.com/fmtlib/fmt/issues/2959,
  https://github.com/fmtlib/fmt/pull/3271). Thanks @ShawnZhong.

- Added a formatter for `std::exception`
  (https://github.com/fmtlib/fmt/issues/2977,
  https://github.com/fmtlib/fmt/issues/3012,
  https://github.com/fmtlib/fmt/pull/3062,
  https://github.com/fmtlib/fmt/pull/3076,
  https://github.com/fmtlib/fmt/pull/3119). For example
  ([godbolt](https://godbolt.org/z/8xoWGs9e4)):

  ```c++
  #include <fmt/std.h>
  #include <vector>

  int main() {
    try {
      std::vector<bool>().at(0);
    } catch(const std::exception& e) {
      fmt::print("{}", e);
    }
  }
  ```

  prints:

      vector<bool>::_M_range_check: __n (which is 0) >= this->size() (which is 0)

  on libstdc++. Thanks @zach2good and @phprus.

- Moved `std::error_code` formatter from `fmt/os.h` to `fmt/std.h`.
  (https://github.com/fmtlib/fmt/pull/3125). Thanks @phprus.

- Added formatters for standard container adapters:
  `std::priority_queue`, `std::queue` and `std::stack`
  (https://github.com/fmtlib/fmt/issues/3215,
  https://github.com/fmtlib/fmt/pull/3279). For example
  ([godbolt](https://godbolt.org/z/74h1xY9qK)):

  ```c++
  #include <fmt/ranges.h>
  #include <stack>
  #include <vector>

  int main() {
    auto s = std::stack<bool, std::vector<bool>>();
    for (auto b: {true, false, true}) s.push(b);
    fmt::print("{}\n", s); // prints [true, false, true]
  }
  ```

  Thanks @ShawnZhong.

- Added a formatter for `std::optional` to `fmt/std.h`
  (https://github.com/fmtlib/fmt/issues/1367,
  https://github.com/fmtlib/fmt/pull/3303).
  Thanks @tom-huntington.

- Fixed formatting of valueless by exception variants
  (https://github.com/fmtlib/fmt/pull/3347). Thanks @TheOmegaCarrot.

- Made `fmt::ptr` accept `unique_ptr` with a custom deleter
  (https://github.com/fmtlib/fmt/pull/3177). Thanks @hmbj.

- Fixed formatting of noncopyable ranges and nested ranges of chars
  (https://github.com/fmtlib/fmt/pull/3158
  https://github.com/fmtlib/fmt/issues/3286,
  https://github.com/fmtlib/fmt/pull/3290). Thanks @BRevzin.

- Fixed issues with formatting of paths and ranges of paths
  (https://github.com/fmtlib/fmt/issues/3319,
  https://github.com/fmtlib/fmt/pull/3321
  https://github.com/fmtlib/fmt/issues/3322). Thanks @phprus.

- Improved handling of invalid Unicode in paths.

- Enabled compile-time checks on Apple clang 14 and later
  (https://github.com/fmtlib/fmt/pull/3331). Thanks @cloyce.

- Improved compile-time checks of named arguments
  (https://github.com/fmtlib/fmt/issues/3105,
  https://github.com/fmtlib/fmt/pull/3214). Thanks @rbrich.

- Fixed formatting when both alignment and `0` are given
  (https://github.com/fmtlib/fmt/issues/3236,
  https://github.com/fmtlib/fmt/pull/3248). Thanks @ShawnZhong.

- Improved Unicode support in the experimental file API on Windows
  (https://github.com/fmtlib/fmt/issues/3234,
  https://github.com/fmtlib/fmt/pull/3293). Thanks @Fros1er.

- Unified UTF transcoding
  (https://github.com/fmtlib/fmt/pull/3416). Thanks @phprus.

- Added support for UTF-8 digit separators via an experimental locale
  facet (https://github.com/fmtlib/fmt/issues/1861). For
  example ([godbolt](https://godbolt.org/z/f7bcznb3W)):

  ```c++
  auto loc = std::locale(
    std::locale(), new fmt::format_facet<std::locale>("’"));
  auto s = fmt::format(loc, "{:L}", 1000);
  ```

  where `’` is U+2019 used as a digit separator in the de_CH locale.

- Added an overload of `formatted_size` that takes a locale
  (https://github.com/fmtlib/fmt/issues/3084,
  https://github.com/fmtlib/fmt/pull/3087). Thanks @gerboengels.

- Removed the deprecated `FMT_DEPRECATED_OSTREAM`.

- Fixed a UB when using a null `std::string_view` with
  `fmt::to_string` or format string compilation
  (https://github.com/fmtlib/fmt/issues/3241,
  https://github.com/fmtlib/fmt/pull/3244). Thanks @phprus.

- Added `starts_with` to the fallback `string_view` implementation
  (https://github.com/fmtlib/fmt/pull/3080). Thanks @phprus.

- Added `fmt::basic_format_string::get()` for compatibility with
  `basic_format_string`
  (https://github.com/fmtlib/fmt/pull/3111). Thanks @huangqinjin.

- Added `println` for compatibility with C++23
  (https://github.com/fmtlib/fmt/pull/3267). Thanks @ShawnZhong.

- Renamed the `FMT_EXPORT` macro for shared library usage to
  `FMT_LIB_EXPORT`.

- Improved documentation
  (https://github.com/fmtlib/fmt/issues/3108,
  https://github.com/fmtlib/fmt/issues/3169,
  https://github.com/fmtlib/fmt/pull/3243).
  https://github.com/fmtlib/fmt/pull/3404,
  https://github.com/fmtlib/fmt/pull/4002).
  Thanks @Cleroth, @Vertexwahn and @yujincheng08.

- Improved build configuration and tests
  (https://github.com/fmtlib/fmt/pull/3118,
  https://github.com/fmtlib/fmt/pull/3120,
  https://github.com/fmtlib/fmt/pull/3188,
  https://github.com/fmtlib/fmt/issues/3189,
  https://github.com/fmtlib/fmt/pull/3198,
  https://github.com/fmtlib/fmt/pull/3205,
  https://github.com/fmtlib/fmt/pull/3207,
  https://github.com/fmtlib/fmt/pull/3210,
  https://github.com/fmtlib/fmt/pull/3240,
  https://github.com/fmtlib/fmt/pull/3256,
  https://github.com/fmtlib/fmt/pull/3264,
  https://github.com/fmtlib/fmt/issues/3299,
  https://github.com/fmtlib/fmt/pull/3302,
  https://github.com/fmtlib/fmt/pull/3312,
  https://github.com/fmtlib/fmt/issues/3317,
  https://github.com/fmtlib/fmt/pull/3328,
  https://github.com/fmtlib/fmt/pull/3333,
  https://github.com/fmtlib/fmt/pull/3369,
  https://github.com/fmtlib/fmt/issues/3373,
  https://github.com/fmtlib/fmt/pull/3395,
  https://github.com/fmtlib/fmt/pull/3406,
  https://github.com/fmtlib/fmt/pull/3411).
  Thanks @dimztimz, @phprus, @DavidKorczynski, @ChrisThrasher,
  @FrancoisCarouge, @kennyweiss, @luzpaz, @codeinred, @Mixaill, @joycebrum,
  @kevinhwang and @Vertexwahn.

- Fixed a regression in handling empty format specifiers after a colon
  (`{:}`) (https://github.com/fmtlib/fmt/pull/3086). Thanks @oxidase.

- Worked around a broken implementation of
  `std::is_constant_evaluated` in some versions of libstdc++ on clang
  (https://github.com/fmtlib/fmt/issues/3247,
  https://github.com/fmtlib/fmt/pull/3281). Thanks @phprus.

- Fixed formatting of volatile variables
  (https://github.com/fmtlib/fmt/pull/3068).

- Fixed various warnings and compilation issues
  (https://github.com/fmtlib/fmt/pull/3057,
  https://github.com/fmtlib/fmt/pull/3066,
  https://github.com/fmtlib/fmt/pull/3072,
  https://github.com/fmtlib/fmt/pull/3082,
  https://github.com/fmtlib/fmt/pull/3091,
  https://github.com/fmtlib/fmt/issues/3092,
  https://github.com/fmtlib/fmt/pull/3093,
  https://github.com/fmtlib/fmt/pull/3095,
  https://github.com/fmtlib/fmt/issues/3096,
  https://github.com/fmtlib/fmt/pull/3097,
  https://github.com/fmtlib/fmt/issues/3128,
  https://github.com/fmtlib/fmt/pull/3129,
  https://github.com/fmtlib/fmt/pull/3137,
  https://github.com/fmtlib/fmt/pull/3139,
  https://github.com/fmtlib/fmt/issues/3140,
  https://github.com/fmtlib/fmt/pull/3142,
  https://github.com/fmtlib/fmt/issues/3149,
  https://github.com/fmtlib/fmt/pull/3150,
  https://github.com/fmtlib/fmt/issues/3154,
  https://github.com/fmtlib/fmt/issues/3163,
  https://github.com/fmtlib/fmt/issues/3178,
  https://github.com/fmtlib/fmt/pull/3184,
  https://github.com/fmtlib/fmt/pull/3196,
  https://github.com/fmtlib/fmt/issues/3204,
  https://github.com/fmtlib/fmt/pull/3206,
  https://github.com/fmtlib/fmt/pull/3208,
  https://github.com/fmtlib/fmt/issues/3213,
  https://github.com/fmtlib/fmt/pull/3216,
  https://github.com/fmtlib/fmt/issues/3224,
  https://github.com/fmtlib/fmt/issues/3226,
  https://github.com/fmtlib/fmt/issues/3228,
  https://github.com/fmtlib/fmt/pull/3229,
  https://github.com/fmtlib/fmt/pull/3259,
  https://github.com/fmtlib/fmt/issues/3274,
  https://github.com/fmtlib/fmt/issues/3287,
  https://github.com/fmtlib/fmt/pull/3288,
  https://github.com/fmtlib/fmt/issues/3292,
  https://github.com/fmtlib/fmt/pull/3295,
  https://github.com/fmtlib/fmt/pull/3296,
  https://github.com/fmtlib/fmt/issues/3298,
  https://github.com/fmtlib/fmt/issues/3325,
  https://github.com/fmtlib/fmt/pull/3326,
  https://github.com/fmtlib/fmt/issues/3334,
  https://github.com/fmtlib/fmt/issues/3342,
  https://github.com/fmtlib/fmt/pull/3343,
  https://github.com/fmtlib/fmt/issues/3351,
  https://github.com/fmtlib/fmt/pull/3352,
  https://github.com/fmtlib/fmt/pull/3362,
  https://github.com/fmtlib/fmt/issues/3365,
  https://github.com/fmtlib/fmt/pull/3366,
  https://github.com/fmtlib/fmt/pull/3374,
  https://github.com/fmtlib/fmt/issues/3377,
  https://github.com/fmtlib/fmt/pull/3378,
  https://github.com/fmtlib/fmt/issues/3381,
  https://github.com/fmtlib/fmt/pull/3398,
  https://github.com/fmtlib/fmt/pull/3413,
  https://github.com/fmtlib/fmt/issues/3415).
  Thanks @phprus, @gsjaardema, @NewbieOrange, @EngineLessCC, @asmaloney,
  @HazardyKnusperkeks, @sergiud, @Youw, @thesmurph, @czudziakm,
  @Roman-Koshelev, @chronoxor, @ShawnZhong, @russelltg, @glebm, @tmartin-gh,
  @Zhaojun-Liu, @louiswins and @mogemimi.

# 9.1.0 - 2022-08-27

- `fmt::formatted_size` now works at compile time
  (https://github.com/fmtlib/fmt/pull/3026). For example
  ([godbolt](https://godbolt.org/z/1MW5rMdf8)):

  ```c++
  #include <fmt/compile.h>

  int main() {
    using namespace fmt::literals;
    constexpr size_t n = fmt::formatted_size("{}"_cf, 42);
    fmt::print("{}\n", n); // prints 2
  }
  ```

  Thanks @marksantaniello.

- Fixed handling of invalid UTF-8
  (https://github.com/fmtlib/fmt/pull/3038,
  https://github.com/fmtlib/fmt/pull/3044,
  https://github.com/fmtlib/fmt/pull/3056).
  Thanks @phprus and @skeeto.

- Improved Unicode support in `ostream` overloads of `print`
  (https://github.com/fmtlib/fmt/pull/2994,
  https://github.com/fmtlib/fmt/pull/3001,
  https://github.com/fmtlib/fmt/pull/3025). Thanks @dimztimz.

- Fixed handling of the sign specifier in localized formatting on
  systems with 32-bit `wchar_t`
  (https://github.com/fmtlib/fmt/issues/3041).

- Added support for wide streams to `fmt::streamed`
  (https://github.com/fmtlib/fmt/pull/2994). Thanks @phprus.

- Added the `n` specifier that disables the output of delimiters when
  formatting ranges (https://github.com/fmtlib/fmt/pull/2981,
  https://github.com/fmtlib/fmt/pull/2983). For example
  ([godbolt](https://godbolt.org/z/roKqGdj8c)):

  ```c++
  #include <fmt/ranges.h>
  #include <vector>

  int main() {
    auto v = std::vector{1, 2, 3};
    fmt::print("{:n}\n", v); // prints 1, 2, 3
  }
  ```

  Thanks @BRevzin.

- Worked around problematic `std::string_view` constructors introduced
  in C++23 (https://github.com/fmtlib/fmt/issues/3030,
  https://github.com/fmtlib/fmt/issues/3050). Thanks @strega-nil-ms.

- Improve handling (exclusion) of recursive ranges
  (https://github.com/fmtlib/fmt/issues/2968,
  https://github.com/fmtlib/fmt/pull/2974). Thanks @Dani-Hub.

- Improved error reporting in format string compilation
  (https://github.com/fmtlib/fmt/issues/3055).

- Improved the implementation of
  [Dragonbox](https://github.com/jk-jeon/dragonbox), the algorithm
  used for the default floating-point formatting
  (https://github.com/fmtlib/fmt/pull/2984). Thanks @jk-jeon.

- Fixed issues with floating-point formatting on exotic platforms.

- Improved the implementation of chrono formatting
  (https://github.com/fmtlib/fmt/pull/3010). Thanks @phprus.

- Improved documentation
  (https://github.com/fmtlib/fmt/pull/2966,
  https://github.com/fmtlib/fmt/pull/3009,
  https://github.com/fmtlib/fmt/issues/3020,
  https://github.com/fmtlib/fmt/pull/3037).
  Thanks @mwinterb, @jcelerier and @remiburtin.

- Improved build configuration
  (https://github.com/fmtlib/fmt/pull/2991,
  https://github.com/fmtlib/fmt/pull/2995,
  https://github.com/fmtlib/fmt/issues/3004,
  https://github.com/fmtlib/fmt/pull/3007,
  https://github.com/fmtlib/fmt/pull/3040).
  Thanks @dimztimz and @hwhsu1231.

- Fixed various warnings and compilation issues
  (https://github.com/fmtlib/fmt/issues/2969,
  https://github.com/fmtlib/fmt/pull/2971,
  https://github.com/fmtlib/fmt/issues/2975,
  https://github.com/fmtlib/fmt/pull/2982,
  https://github.com/fmtlib/fmt/pull/2985,
  https://github.com/fmtlib/fmt/issues/2988,
  https://github.com/fmtlib/fmt/issues/2989,
  https://github.com/fmtlib/fmt/issues/3000,
  https://github.com/fmtlib/fmt/issues/3006,
  https://github.com/fmtlib/fmt/issues/3014,
  https://github.com/fmtlib/fmt/issues/3015,
  https://github.com/fmtlib/fmt/pull/3021,
  https://github.com/fmtlib/fmt/issues/3023,
  https://github.com/fmtlib/fmt/pull/3024,
  https://github.com/fmtlib/fmt/pull/3029,
  https://github.com/fmtlib/fmt/pull/3043,
  https://github.com/fmtlib/fmt/issues/3052,
  https://github.com/fmtlib/fmt/pull/3053,
  https://github.com/fmtlib/fmt/pull/3054).
  Thanks @h-friederich, @dimztimz, @olupton, @bernhardmgruber and @phprus.

# 9.0.0 - 2022-07-04

- Switched to the internal floating point formatter for all decimal
  presentation formats. In particular this results in consistent
  rounding on all platforms and removing the `s[n]printf` fallback for
  decimal FP formatting.

- Compile-time floating point formatting no longer requires the
  header-only mode. For example
  ([godbolt](https://godbolt.org/z/G37PTeG3b)):

  ```c++
  #include <array>
  #include <fmt/compile.h>

  consteval auto compile_time_dtoa(double value) -> std::array<char, 10> {
    auto result = std::array<char, 10>();
    fmt::format_to(result.data(), FMT_COMPILE("{}"), value);
    return result;
  }

  constexpr auto answer = compile_time_dtoa(0.42);
  ```

  works with the default settings.

- Improved the implementation of
  [Dragonbox](https://github.com/jk-jeon/dragonbox), the algorithm
  used for the default floating-point formatting
  (https://github.com/fmtlib/fmt/pull/2713,
  https://github.com/fmtlib/fmt/pull/2750). Thanks @jk-jeon.

- Made `fmt::to_string` work with `__float128`. This uses the internal
  FP formatter and works even on system without `__float128` support
  in `[s]printf`.

- Disabled automatic `std::ostream` insertion operator (`operator<<`)
  discovery when `fmt/ostream.h` is included to prevent ODR
  violations. You can get the old behavior by defining
  `FMT_DEPRECATED_OSTREAM` but this will be removed in the next major
  release. Use `fmt::streamed` or `fmt::ostream_formatter` to enable
  formatting via `std::ostream` instead.

- Added `fmt::ostream_formatter` that can be used to write `formatter`
  specializations that perform formatting via `std::ostream`. For
  example ([godbolt](https://godbolt.org/z/5sEc5qMsf)):

  ```c++
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
  ```

- Added the `fmt::streamed` function that takes an object and formats
  it via `std::ostream`. For example
  ([godbolt](https://godbolt.org/z/5G3346G1f)):

  ```c++
  #include <thread>
  #include <fmt/ostream.h>

  int main() {
    fmt::print("Current thread id: {}\n",
               fmt::streamed(std::this_thread::get_id()));
  }
  ```

  Note that `fmt/std.h` provides a `formatter` specialization for
  `std::thread::id` so you don\'t need to format it via
  `std::ostream`.

- Deprecated implicit conversions of unscoped enums to integers for
  consistency with scoped enums.

- Added an argument-dependent lookup based `format_as` extension API
  to simplify formatting of enums.

- Added experimental `std::variant` formatting support
  (https://github.com/fmtlib/fmt/pull/2941). For example
  ([godbolt](https://godbolt.org/z/KG9z6cq68)):

  ```c++
  #include <variant>
  #include <fmt/std.h>

  int main() {
    auto v = std::variant<int, std::string>(42);
    fmt::print("{}\n", v);
  }
  ```

  prints:

      variant(42)

  Thanks @jehelset.

- Added experimental `std::filesystem::path` formatting support
  (https://github.com/fmtlib/fmt/issues/2865,
  https://github.com/fmtlib/fmt/pull/2902,
  https://github.com/fmtlib/fmt/issues/2917,
  https://github.com/fmtlib/fmt/pull/2918). For example
  ([godbolt](https://godbolt.org/z/o44dMexEb)):

  ```c++
  #include <filesystem>
  #include <fmt/std.h>

  int main() {
    fmt::print("There is no place like {}.", std::filesystem::path("/home"));
  }
  ```

  prints:

      There is no place like "/home".

  Thanks @phprus.

- Added a `std::thread::id` formatter to `fmt/std.h`. For example
  ([godbolt](https://godbolt.org/z/j1azbYf3E)):

  ```c++
  #include <thread>
  #include <fmt/std.h>

  int main() {
    fmt::print("Current thread id: {}\n", std::this_thread::get_id());
  }
  ```

- Added `fmt::styled` that applies a text style to an individual
  argument (https://github.com/fmtlib/fmt/pull/2793). For
  example ([godbolt](https://godbolt.org/z/vWGW7v5M6)):

  ```c++
  #include <fmt/chrono.h>
  #include <fmt/color.h>

  int main() {
    auto now = std::chrono::system_clock::now();
    fmt::print(
      "[{}] {}: {}\n",
      fmt::styled(now, fmt::emphasis::bold),
      fmt::styled("error", fg(fmt::color::red)),
      "something went wrong");
  }
  ```

  prints

  ![](https://user-images.githubusercontent.com/576385/175071215-12809244-dab0-4005-96d8-7cd911c964d5.png)

  Thanks @rbrugo.

- Made `fmt::print` overload for text styles correctly handle UTF-8
  (https://github.com/fmtlib/fmt/issues/2681,
  https://github.com/fmtlib/fmt/pull/2701). Thanks @AlexGuteniev.

- Fixed Unicode handling when writing to an ostream.

- Added support for nested specifiers to range formatting
  (https://github.com/fmtlib/fmt/pull/2673). For example
  ([godbolt](https://godbolt.org/z/xd3Gj38cf)):

  ```c++
  #include <vector>
  #include <fmt/ranges.h>

  int main() {
    fmt::print("{::#x}\n", std::vector{10, 20, 30});
  }
  ```

  prints `[0xa, 0x14, 0x1e]`.

  Thanks @BRevzin.

- Implemented escaping of wide strings in ranges
  (https://github.com/fmtlib/fmt/pull/2904). Thanks @phprus.

- Added support for ranges with `begin` / `end` found via the
  argument-dependent lookup
  (https://github.com/fmtlib/fmt/pull/2807). Thanks @rbrugo.

- Fixed formatting of certain kinds of ranges of ranges
  (https://github.com/fmtlib/fmt/pull/2787). Thanks @BRevzin.

- Fixed handling of maps with element types other than `std::pair`
  (https://github.com/fmtlib/fmt/pull/2944). Thanks @BrukerJWD.

- Made tuple formatter enabled only if elements are formattable
  (https://github.com/fmtlib/fmt/issues/2939,
  https://github.com/fmtlib/fmt/pull/2940). Thanks @jehelset.

- Made `fmt::join` compatible with format string compilation
  (https://github.com/fmtlib/fmt/issues/2719,
  https://github.com/fmtlib/fmt/pull/2720). Thanks @phprus.

- Made compile-time checks work with named arguments of custom types
  and `std::ostream` `print` overloads
  (https://github.com/fmtlib/fmt/issues/2816,
  https://github.com/fmtlib/fmt/issues/2817,
  https://github.com/fmtlib/fmt/pull/2819). Thanks @timsong-cpp.

- Removed `make_args_checked` because it is no longer needed for
  compile-time checks
  (https://github.com/fmtlib/fmt/pull/2760). Thanks @phprus.

- Removed the following deprecated APIs: `_format`, `arg_join`, the
  `format_to` overload that takes a memory buffer, `[v]fprintf` that
  takes an `ostream`.

- Removed the deprecated implicit conversion of `[const] signed char*`
  and `[const] unsigned char*` to C strings.

- Removed the deprecated `fmt/locale.h`.

- Replaced the deprecated `fileno()` with `descriptor()` in
  `buffered_file`.

- Moved `to_string_view` to the `detail` namespace since it\'s an
  implementation detail.

- Made access mode of a created file consistent with `fopen` by
  setting `S_IWGRP` and `S_IWOTH`
  (https://github.com/fmtlib/fmt/pull/2733). Thanks @arogge.

- Removed a redundant buffer resize when formatting to `std::ostream`
  (https://github.com/fmtlib/fmt/issues/2842,
  https://github.com/fmtlib/fmt/pull/2843). Thanks @jcelerier.

- Made precision computation for strings consistent with width
  (https://github.com/fmtlib/fmt/issues/2888).

- Fixed handling of locale separators in floating point formatting
  (https://github.com/fmtlib/fmt/issues/2830).

- Made sign specifiers work with `__int128_t`
  (https://github.com/fmtlib/fmt/issues/2773).

- Improved support for systems such as CHERI with extra data stored in
  pointers (https://github.com/fmtlib/fmt/pull/2932).
  Thanks @davidchisnall.

- Improved documentation
  (https://github.com/fmtlib/fmt/pull/2706,
  https://github.com/fmtlib/fmt/pull/2712,
  https://github.com/fmtlib/fmt/pull/2789,
  https://github.com/fmtlib/fmt/pull/2803,
  https://github.com/fmtlib/fmt/pull/2805,
  https://github.com/fmtlib/fmt/pull/2815,
  https://github.com/fmtlib/fmt/pull/2924).
  Thanks @BRevzin, @Pokechu22, @setoye, @rtobar, @rbrugo, @anoonD and
  @leha-bot.

- Improved build configuration
  (https://github.com/fmtlib/fmt/pull/2766,
  https://github.com/fmtlib/fmt/pull/2772,
  https://github.com/fmtlib/fmt/pull/2836,
  https://github.com/fmtlib/fmt/pull/2852,
  https://github.com/fmtlib/fmt/pull/2907,
  https://github.com/fmtlib/fmt/pull/2913,
  https://github.com/fmtlib/fmt/pull/2914).
  Thanks @kambala-decapitator, @mattiasljungstrom, @kieselnb, @nathannaveen
  and @Vertexwahn.

- Fixed various warnings and compilation issues
  (https://github.com/fmtlib/fmt/issues/2408,
  https://github.com/fmtlib/fmt/issues/2507,
  https://github.com/fmtlib/fmt/issues/2697,
  https://github.com/fmtlib/fmt/issues/2715,
  https://github.com/fmtlib/fmt/issues/2717,
  https://github.com/fmtlib/fmt/pull/2722,
  https://github.com/fmtlib/fmt/pull/2724,
  https://github.com/fmtlib/fmt/pull/2725,
  https://github.com/fmtlib/fmt/issues/2726,
  https://github.com/fmtlib/fmt/pull/2728,
  https://github.com/fmtlib/fmt/pull/2732,
  https://github.com/fmtlib/fmt/issues/2738,
  https://github.com/fmtlib/fmt/pull/2742,
  https://github.com/fmtlib/fmt/issues/2744,
  https://github.com/fmtlib/fmt/issues/2745,
  https://github.com/fmtlib/fmt/issues/2746,
  https://github.com/fmtlib/fmt/issues/2754,
  https://github.com/fmtlib/fmt/pull/2755,
  https://github.com/fmtlib/fmt/issues/2757,
  https://github.com/fmtlib/fmt/pull/2758,
  https://github.com/fmtlib/fmt/issues/2761,
  https://github.com/fmtlib/fmt/pull/2762,
  https://github.com/fmtlib/fmt/issues/2763,
  https://github.com/fmtlib/fmt/pull/2765,
  https://github.com/fmtlib/fmt/issues/2769,
  https://github.com/fmtlib/fmt/pull/2770,
  https://github.com/fmtlib/fmt/issues/2771,
  https://github.com/fmtlib/fmt/issues/2777,
  https://github.com/fmtlib/fmt/pull/2779,
  https://github.com/fmtlib/fmt/pull/2782,
  https://github.com/fmtlib/fmt/pull/2783,
  https://github.com/fmtlib/fmt/issues/2794,
  https://github.com/fmtlib/fmt/issues/2796,
  https://github.com/fmtlib/fmt/pull/2797,
  https://github.com/fmtlib/fmt/pull/2801,
  https://github.com/fmtlib/fmt/pull/2802,
  https://github.com/fmtlib/fmt/issues/2808,
  https://github.com/fmtlib/fmt/issues/2818,
  https://github.com/fmtlib/fmt/pull/2819,
  https://github.com/fmtlib/fmt/issues/2829,
  https://github.com/fmtlib/fmt/issues/2835,
  https://github.com/fmtlib/fmt/issues/2848,
  https://github.com/fmtlib/fmt/issues/2860,
  https://github.com/fmtlib/fmt/pull/2861,
  https://github.com/fmtlib/fmt/pull/2882,
  https://github.com/fmtlib/fmt/issues/2886,
  https://github.com/fmtlib/fmt/issues/2891,
  https://github.com/fmtlib/fmt/pull/2892,
  https://github.com/fmtlib/fmt/issues/2895,
  https://github.com/fmtlib/fmt/issues/2896,
  https://github.com/fmtlib/fmt/pull/2903,
  https://github.com/fmtlib/fmt/issues/2906,
  https://github.com/fmtlib/fmt/issues/2908,
  https://github.com/fmtlib/fmt/pull/2909,
  https://github.com/fmtlib/fmt/issues/2920,
  https://github.com/fmtlib/fmt/pull/2922,
  https://github.com/fmtlib/fmt/pull/2927,
  https://github.com/fmtlib/fmt/pull/2929,
  https://github.com/fmtlib/fmt/issues/2936,
  https://github.com/fmtlib/fmt/pull/2937,
  https://github.com/fmtlib/fmt/pull/2938,
  https://github.com/fmtlib/fmt/pull/2951,
  https://github.com/fmtlib/fmt/issues/2954,
  https://github.com/fmtlib/fmt/pull/2957,
  https://github.com/fmtlib/fmt/issues/2958,
  https://github.com/fmtlib/fmt/pull/2960).
  Thanks @matrackif @Tobi823, @ivan-volnov, @VasiliPupkin256,
  @federico-busato, @barcharcraz, @jk-jeon, @HazardyKnusperkeks, @dalboris,
  @seanm, @gsjaardema, @timsong-cpp, @seanm, @frithrah, @chronoxor, @Agga,
  @madmaxoft, @JurajX, @phprus and @Dani-Hub.

# 8.1.1 - 2022-01-06

- Restored ABI compatibility with version 8.0.x
  (https://github.com/fmtlib/fmt/issues/2695,
  https://github.com/fmtlib/fmt/pull/2696). Thanks @saraedum.
- Fixed chrono formatting on big endian systems
  (https://github.com/fmtlib/fmt/issues/2698,
  https://github.com/fmtlib/fmt/pull/2699).
  Thanks @phprus and @xvitaly.
- Fixed a linkage error with mingw
  (https://github.com/fmtlib/fmt/issues/2691,
  https://github.com/fmtlib/fmt/pull/2692). Thanks @rbberger.

# 8.1.0 - 2022-01-02

- Optimized chrono formatting
  (https://github.com/fmtlib/fmt/pull/2500,
  https://github.com/fmtlib/fmt/pull/2537,
  https://github.com/fmtlib/fmt/issues/2541,
  https://github.com/fmtlib/fmt/pull/2544,
  https://github.com/fmtlib/fmt/pull/2550,
  https://github.com/fmtlib/fmt/pull/2551,
  https://github.com/fmtlib/fmt/pull/2576,
  https://github.com/fmtlib/fmt/issues/2577,
  https://github.com/fmtlib/fmt/pull/2586,
  https://github.com/fmtlib/fmt/pull/2591,
  https://github.com/fmtlib/fmt/pull/2594,
  https://github.com/fmtlib/fmt/pull/2602,
  https://github.com/fmtlib/fmt/pull/2617,
  https://github.com/fmtlib/fmt/issues/2628,
  https://github.com/fmtlib/fmt/pull/2633,
  https://github.com/fmtlib/fmt/issues/2670,
  https://github.com/fmtlib/fmt/pull/2671).

  Processing of some specifiers such as `%z` and `%Y` is now up to
  10-20 times faster, for example on GCC 11 with libstdc++:

      ----------------------------------------------------------------------------
      Benchmark                                  Before             After
      ----------------------------------------------------------------------------
      FMTFormatter_z                             261 ns             26.3 ns
      FMTFormatterCompile_z                      246 ns             11.6 ns
      FMTFormatter_Y                             263 ns             26.1 ns
      FMTFormatterCompile_Y                      244 ns             10.5 ns
      ----------------------------------------------------------------------------

  Thanks @phprus and @toughengineer.

- Implemented subsecond formatting for chrono durations
  (https://github.com/fmtlib/fmt/pull/2623). For example
  ([godbolt](https://godbolt.org/z/es7vWTETe)):

  ```c++
  #include <fmt/chrono.h>

  int main() {
    fmt::print("{:%S}", std::chrono::milliseconds(1234));
  }
  ```

  prints \"01.234\".

  Thanks @matrackif.

- Fixed handling of precision 0 when formatting chrono durations
  (https://github.com/fmtlib/fmt/issues/2587,
  https://github.com/fmtlib/fmt/pull/2588). Thanks @lukester1975.

- Fixed an overflow on invalid inputs in the `tm` formatter
  (https://github.com/fmtlib/fmt/pull/2564). Thanks @phprus.

- Added `fmt::group_digits` that formats integers with a non-localized
  digit separator (comma) for groups of three digits. For example
  ([godbolt](https://godbolt.org/z/TxGxG9Poq)):

  ```c++
  #include <fmt/format.h>

  int main() {
    fmt::print("{} dollars", fmt::group_digits(1000000));
  }
  ```

  prints \"1,000,000 dollars\".

- Added support for faint, conceal, reverse and blink text styles
  (https://github.com/fmtlib/fmt/pull/2394):

  <https://user-images.githubusercontent.com/576385/147710227-c68f5317-f8fa-42c3-9123-7c4ba3c398cb.mp4>

  Thanks @benit8 and @data-man.

- Added experimental support for compile-time floating point
  formatting (https://github.com/fmtlib/fmt/pull/2426,
  https://github.com/fmtlib/fmt/pull/2470). It is currently
  limited to the header-only mode. Thanks @alexezeder.

- Added UDL-based named argument support to compile-time format string
  checks (https://github.com/fmtlib/fmt/issues/2640,
  https://github.com/fmtlib/fmt/pull/2649). For example
  ([godbolt](https://godbolt.org/z/ohGbbvonv)):

  ```c++
  #include <fmt/format.h>

  int main() {
    using namespace fmt::literals;
    fmt::print("{answer:s}", "answer"_a=42);
  }
  ```

  gives a compile-time error on compilers with C++20 `consteval` and
  non-type template parameter support (gcc 10+) because `s` is not a
  valid format specifier for an integer.

  Thanks @alexezeder.

- Implemented escaping of string range elements. For example
  ([godbolt](https://godbolt.org/z/rKvM1vKf3)):

  ```c++
  #include <fmt/ranges.h>
  #include <vector>

  int main() {
    fmt::print("{}", std::vector<std::string>{"\naan"});
  }
  ```

  is now printed as:

      ["\naan"]

  instead of:

      ["
      aan"]

- Added an experimental `?` specifier for escaping strings.
  (https://github.com/fmtlib/fmt/pull/2674). Thanks @BRevzin.

- Switched to JSON-like representation of maps and sets for
  consistency with Python\'s `str.format`. For example
  ([godbolt](https://godbolt.org/z/seKjoY9W5)):

  ```c++
  #include <fmt/ranges.h>
  #include <map>

  int main() {
    fmt::print("{}", std::map<std::string, int>{{"answer", 42}});
  }
  ```

  is now printed as:

      {"answer": 42}

- Extended `fmt::join` to support C++20-only ranges
  (https://github.com/fmtlib/fmt/pull/2549). Thanks @BRevzin.

- Optimized handling of non-const-iterable ranges and implemented
  initial support for non-const-formattable types.

- Disabled implicit conversions of scoped enums to integers that was
  accidentally introduced in earlier versions
  (https://github.com/fmtlib/fmt/pull/1841).

- Deprecated implicit conversion of `[const] signed char*` and
  `[const] unsigned char*` to C strings.

- Deprecated `_format`, a legacy UDL-based format API
  (https://github.com/fmtlib/fmt/pull/2646). Thanks @alexezeder.

- Marked `format`, `formatted_size` and `to_string` as `[[nodiscard]]`
  (https://github.com/fmtlib/fmt/pull/2612). @0x8000-0000.

- Added missing diagnostic when trying to format function and member
  pointers as well as objects convertible to pointers which is
  explicitly disallowed
  (https://github.com/fmtlib/fmt/issues/2598,
  https://github.com/fmtlib/fmt/pull/2609,
  https://github.com/fmtlib/fmt/pull/2610). Thanks @AlexGuteniev.

- Optimized writing to a contiguous buffer with `format_to_n`
  (https://github.com/fmtlib/fmt/pull/2489). Thanks @Roman-Koshelev.

- Optimized writing to non-`char` buffers
  (https://github.com/fmtlib/fmt/pull/2477). Thanks @Roman-Koshelev.

- Decimal point is now localized when using the `L` specifier.

- Improved floating point formatter implementation
  (https://github.com/fmtlib/fmt/pull/2498,
  https://github.com/fmtlib/fmt/pull/2499). Thanks @Roman-Koshelev.

- Fixed handling of very large precision in fixed format
  (https://github.com/fmtlib/fmt/pull/2616).

- Made a table of cached powers used in FP formatting static
  (https://github.com/fmtlib/fmt/pull/2509). Thanks @jk-jeon.

- Resolved a lookup ambiguity with C++20 format-related functions due
  to ADL (https://github.com/fmtlib/fmt/issues/2639,
  https://github.com/fmtlib/fmt/pull/2641). Thanks @mkurdej.

- Removed unnecessary inline namespace qualification
  (https://github.com/fmtlib/fmt/issues/2642,
  https://github.com/fmtlib/fmt/pull/2643). Thanks @mkurdej.

- Implemented argument forwarding in `format_to_n`
  (https://github.com/fmtlib/fmt/issues/2462,
  https://github.com/fmtlib/fmt/pull/2463). Thanks @owent.

- Fixed handling of implicit conversions in `fmt::to_string` and
  format string compilation
  (https://github.com/fmtlib/fmt/issues/2565).

- Changed the default access mode of files created by
  `fmt::output_file` to `-rw-r--r--` for consistency with `fopen`
  (https://github.com/fmtlib/fmt/issues/2530).

- Make `fmt::ostream::flush` public
  (https://github.com/fmtlib/fmt/issues/2435).

- Improved C++14/17 attribute detection
  (https://github.com/fmtlib/fmt/pull/2615). Thanks @AlexGuteniev.

- Improved `consteval` detection for MSVC
  (https://github.com/fmtlib/fmt/pull/2559). Thanks @DanielaE.

- Improved documentation
  (https://github.com/fmtlib/fmt/issues/2406,
  https://github.com/fmtlib/fmt/pull/2446,
  https://github.com/fmtlib/fmt/issues/2493,
  https://github.com/fmtlib/fmt/issues/2513,
  https://github.com/fmtlib/fmt/pull/2515,
  https://github.com/fmtlib/fmt/issues/2522,
  https://github.com/fmtlib/fmt/pull/2562,
  https://github.com/fmtlib/fmt/pull/2575,
  https://github.com/fmtlib/fmt/pull/2606,
  https://github.com/fmtlib/fmt/pull/2620,
  https://github.com/fmtlib/fmt/issues/2676).
  Thanks @sobolevn, @UnePierre, @zhsj, @phprus, @ericcurtin and @Lounarok.

- Improved fuzzers and added a fuzzer for chrono timepoint formatting
  (https://github.com/fmtlib/fmt/pull/2461,
  https://github.com/fmtlib/fmt/pull/2469). @pauldreik,

- Added the `FMT_SYSTEM_HEADERS` CMake option setting which marks
  {fmt}\'s headers as system. It can be used to suppress warnings
  (https://github.com/fmtlib/fmt/issues/2644,
  https://github.com/fmtlib/fmt/pull/2651). Thanks @alexezeder.

- Added the Bazel build system support
  (https://github.com/fmtlib/fmt/pull/2505,
  https://github.com/fmtlib/fmt/pull/2516). Thanks @Vertexwahn.

- Improved build configuration and tests
  (https://github.com/fmtlib/fmt/issues/2437,
  https://github.com/fmtlib/fmt/pull/2558,
  https://github.com/fmtlib/fmt/pull/2648,
  https://github.com/fmtlib/fmt/pull/2650,
  https://github.com/fmtlib/fmt/pull/2663,
  https://github.com/fmtlib/fmt/pull/2677).
  Thanks @DanielaE, @alexezeder and @phprus.

- Fixed various warnings and compilation issues
  (https://github.com/fmtlib/fmt/pull/2353,
  https://github.com/fmtlib/fmt/pull/2356,
  https://github.com/fmtlib/fmt/pull/2399,
  https://github.com/fmtlib/fmt/issues/2408,
  https://github.com/fmtlib/fmt/pull/2414,
  https://github.com/fmtlib/fmt/pull/2427,
  https://github.com/fmtlib/fmt/pull/2432,
  https://github.com/fmtlib/fmt/pull/2442,
  https://github.com/fmtlib/fmt/pull/2434,
  https://github.com/fmtlib/fmt/issues/2439,
  https://github.com/fmtlib/fmt/pull/2447,
  https://github.com/fmtlib/fmt/pull/2450,
  https://github.com/fmtlib/fmt/issues/2455,
  https://github.com/fmtlib/fmt/issues/2465,
  https://github.com/fmtlib/fmt/issues/2472,
  https://github.com/fmtlib/fmt/issues/2474,
  https://github.com/fmtlib/fmt/pull/2476,
  https://github.com/fmtlib/fmt/issues/2478,
  https://github.com/fmtlib/fmt/issues/2479,
  https://github.com/fmtlib/fmt/issues/2481,
  https://github.com/fmtlib/fmt/pull/2482,
  https://github.com/fmtlib/fmt/pull/2483,
  https://github.com/fmtlib/fmt/issues/2490,
  https://github.com/fmtlib/fmt/pull/2491,
  https://github.com/fmtlib/fmt/pull/2510,
  https://github.com/fmtlib/fmt/pull/2518,
  https://github.com/fmtlib/fmt/issues/2528,
  https://github.com/fmtlib/fmt/pull/2529,
  https://github.com/fmtlib/fmt/pull/2539,
  https://github.com/fmtlib/fmt/issues/2540,
  https://github.com/fmtlib/fmt/pull/2545,
  https://github.com/fmtlib/fmt/pull/2555,
  https://github.com/fmtlib/fmt/issues/2557,
  https://github.com/fmtlib/fmt/issues/2570,
  https://github.com/fmtlib/fmt/pull/2573,
  https://github.com/fmtlib/fmt/pull/2582,
  https://github.com/fmtlib/fmt/issues/2605,
  https://github.com/fmtlib/fmt/pull/2611,
  https://github.com/fmtlib/fmt/pull/2647,
  https://github.com/fmtlib/fmt/issues/2627,
  https://github.com/fmtlib/fmt/pull/2630,
  https://github.com/fmtlib/fmt/issues/2635,
  https://github.com/fmtlib/fmt/issues/2638,
  https://github.com/fmtlib/fmt/issues/2653,
  https://github.com/fmtlib/fmt/issues/2654,
  https://github.com/fmtlib/fmt/issues/2661,
  https://github.com/fmtlib/fmt/pull/2664,
  https://github.com/fmtlib/fmt/pull/2684).
  Thanks @DanielaE, @mwinterb, @cdacamar, @TrebledJ, @bodomartin, @cquammen,
  @white238, @mmarkeloff, @palacaze, @jcelerier, @mborn-adi, @BrukerJWD,
  @spyridon97, @phprus, @oliverlee, @joshessman-llnl, @akohlmey, @timkalu,
  @olupton, @Acretock, @alexezeder, @andrewcorrigan, @lucpelletier and
  @HazardyKnusperkeks.

# 8.0.1 - 2021-07-02

- Fixed the version number in the inline namespace
  (https://github.com/fmtlib/fmt/issues/2374).
- Added a missing presentation type check for `std::string`
  (https://github.com/fmtlib/fmt/issues/2402).
- Fixed a linkage error when mixing code built with clang and gcc
  (https://github.com/fmtlib/fmt/issues/2377).
- Fixed documentation issues
  (https://github.com/fmtlib/fmt/pull/2396,
  https://github.com/fmtlib/fmt/issues/2403,
  https://github.com/fmtlib/fmt/issues/2406). Thanks @mkurdej.
- Removed dead code in FP formatter (
  https://github.com/fmtlib/fmt/pull/2398). Thanks @javierhonduco.
- Fixed various warnings and compilation issues
  (https://github.com/fmtlib/fmt/issues/2351,
  https://github.com/fmtlib/fmt/issues/2359,
  https://github.com/fmtlib/fmt/pull/2365,
  https://github.com/fmtlib/fmt/issues/2368,
  https://github.com/fmtlib/fmt/pull/2370,
  https://github.com/fmtlib/fmt/pull/2376,
  https://github.com/fmtlib/fmt/pull/2381,
  https://github.com/fmtlib/fmt/pull/2382,
  https://github.com/fmtlib/fmt/issues/2386,
  https://github.com/fmtlib/fmt/pull/2389,
  https://github.com/fmtlib/fmt/pull/2395,
  https://github.com/fmtlib/fmt/pull/2397,
  https://github.com/fmtlib/fmt/issues/2400,
  https://github.com/fmtlib/fmt/issues/2401,
  https://github.com/fmtlib/fmt/pull/2407).
  Thanks @zx2c4, @AidanSun05, @mattiasljungstrom, @joemmett, @erengy,
  @patlkli, @gsjaardema and @phprus.

# 8.0.0 - 2021-06-21

- Enabled compile-time format string checks by default. For example
  ([godbolt](https://godbolt.org/z/sMxcohGjz)):

  ```c++
  #include <fmt/core.h>

  int main() {
    fmt::print("{:d}", "I am not a number");
  }
  ```

  gives a compile-time error on compilers with C++20 `consteval`
  support (gcc 10+, clang 11+) because `d` is not a valid format
  specifier for a string.

  To pass a runtime string wrap it in `fmt::runtime`:

  ```c++
  fmt::print(fmt::runtime("{:d}"), "I am not a number");
  ```

- Added compile-time formatting
  (https://github.com/fmtlib/fmt/pull/2019,
  https://github.com/fmtlib/fmt/pull/2044,
  https://github.com/fmtlib/fmt/pull/2056,
  https://github.com/fmtlib/fmt/pull/2072,
  https://github.com/fmtlib/fmt/pull/2075,
  https://github.com/fmtlib/fmt/issues/2078,
  https://github.com/fmtlib/fmt/pull/2129,
  https://github.com/fmtlib/fmt/pull/2326). For example
  ([godbolt](https://godbolt.org/z/Mxx9d89jM)):

  ```c++
  #include <fmt/compile.h>

  consteval auto compile_time_itoa(int value) -> std::array<char, 10> {
    auto result = std::array<char, 10>();
    fmt::format_to(result.data(), FMT_COMPILE("{}"), value);
    return result;
  }

  constexpr auto answer = compile_time_itoa(42);
  ```

  Most of the formatting functionality is available at compile time
  with a notable exception of floating-point numbers and pointers.
  Thanks @alexezeder.

- Optimized handling of format specifiers during format string
  compilation. For example, hexadecimal formatting (`"{:x}"`) is now
  3-7x faster than before when using `format_to` with format string
  compilation and a stack-allocated buffer
  (https://github.com/fmtlib/fmt/issues/1944).

  Before (7.1.3):

      ----------------------------------------------------------------------------
      Benchmark                                  Time             CPU   Iterations
      ----------------------------------------------------------------------------
      FMTCompileOld/0                         15.5 ns         15.5 ns     43302898
      FMTCompileOld/42                        16.6 ns         16.6 ns     43278267
      FMTCompileOld/273123                    18.7 ns         18.6 ns     37035861
      FMTCompileOld/9223372036854775807       19.4 ns         19.4 ns     35243000
      ----------------------------------------------------------------------------

  After (8.x):

      ----------------------------------------------------------------------------
      Benchmark                                  Time             CPU   Iterations
      ----------------------------------------------------------------------------
      FMTCompileNew/0                         1.99 ns         1.99 ns    360523686
      FMTCompileNew/42                        2.33 ns         2.33 ns    279865664
      FMTCompileNew/273123                    3.72 ns         3.71 ns    190230315
      FMTCompileNew/9223372036854775807       5.28 ns         5.26 ns    130711631
      ----------------------------------------------------------------------------

  It is even faster than `std::to_chars` from libc++ compiled with
  clang on macOS:

      ----------------------------------------------------------------------------
      Benchmark                                  Time             CPU   Iterations
      ----------------------------------------------------------------------------
      ToChars/0                               4.42 ns         4.41 ns    160196630
      ToChars/42                              5.00 ns         4.98 ns    140735201
      ToChars/273123                          7.26 ns         7.24 ns     95784130
      ToChars/9223372036854775807             8.77 ns         8.75 ns     75872534
      ----------------------------------------------------------------------------

  In other cases, especially involving `std::string` construction, the
  speed up is usually lower because handling format specifiers takes a
  smaller fraction of the total time.

- Added the `_cf` user-defined literal to represent a compiled format
  string. It can be used instead of the `FMT_COMPILE` macro
  (https://github.com/fmtlib/fmt/pull/2043,
  https://github.com/fmtlib/fmt/pull/2242):

  ```c++
  #include <fmt/compile.h>

  using namespace fmt::literals;
  auto s = fmt::format(FMT_COMPILE("{}"), 42); // 🙁 not modern
  auto s = fmt::format("{}"_cf, 42);           // 🙂 modern as hell
  ```

  It requires compiler support for class types in non-type template
  parameters (a C++20 feature) which is available in GCC 9.3+.
  Thanks @alexezeder.

- Format string compilation now requires `format` functions of
  `formatter` specializations for user-defined types to be `const`:

  ```c++
  template <> struct fmt::formatter<my_type>: formatter<string_view> {
    template <typename FormatContext>
    auto format(my_type obj, FormatContext& ctx) const {  // Note const here.
      // ...
    }
  };
  ```

- Added UDL-based named argument support to format string compilation
  (https://github.com/fmtlib/fmt/pull/2243,
  https://github.com/fmtlib/fmt/pull/2281). For example:

  ```c++
  #include <fmt/compile.h>

  using namespace fmt::literals;
  auto s = fmt::format(FMT_COMPILE("{answer}"), "answer"_a = 42);
  ```

  Here the argument named \"answer\" is resolved at compile time with
  no runtime overhead. Thanks @alexezeder.

- Added format string compilation support to `fmt::print`
  (https://github.com/fmtlib/fmt/issues/2280,
  https://github.com/fmtlib/fmt/pull/2304). Thanks @alexezeder.

- Added initial support for compiling {fmt} as a C++20 module
  (https://github.com/fmtlib/fmt/pull/2235,
  https://github.com/fmtlib/fmt/pull/2240,
  https://github.com/fmtlib/fmt/pull/2260,
  https://github.com/fmtlib/fmt/pull/2282,
  https://github.com/fmtlib/fmt/pull/2283,
  https://github.com/fmtlib/fmt/pull/2288,
  https://github.com/fmtlib/fmt/pull/2298,
  https://github.com/fmtlib/fmt/pull/2306,
  https://github.com/fmtlib/fmt/pull/2307,
  https://github.com/fmtlib/fmt/pull/2309,
  https://github.com/fmtlib/fmt/pull/2318,
  https://github.com/fmtlib/fmt/pull/2324,
  https://github.com/fmtlib/fmt/pull/2332,
  https://github.com/fmtlib/fmt/pull/2340). Thanks @DanielaE.

- Made symbols private by default reducing shared library size
  (https://github.com/fmtlib/fmt/pull/2301). For example
  there was a \~15% reported reduction on one platform. Thanks @sergiud.

- Optimized includes making the result of preprocessing `fmt/format.h`
  \~20% smaller with libstdc++/C++20 and slightly improving build
  times (https://github.com/fmtlib/fmt/issues/1998).

- Added support of ranges with non-const `begin` / `end`
  (https://github.com/fmtlib/fmt/pull/1953). Thanks @kitegi.

- Added support of `std::byte` and other formattable types to
  `fmt::join` (https://github.com/fmtlib/fmt/issues/1981,
  https://github.com/fmtlib/fmt/issues/2040,
  https://github.com/fmtlib/fmt/pull/2050,
  https://github.com/fmtlib/fmt/issues/2262). For example:

  ```c++
  #include <fmt/format.h>
  #include <cstddef>
  #include <vector>

  int main() {
    auto bytes = std::vector{std::byte(4), std::byte(2)};
    fmt::print("{}", fmt::join(bytes, ""));
  }
  ```

  prints \"42\".

  Thanks @kamibo.

- Implemented the default format for `std::chrono::system_clock`
  (https://github.com/fmtlib/fmt/issues/2319,
  https://github.com/fmtlib/fmt/pull/2345). For example:

  ```c++
  #include <fmt/chrono.h>

  int main() {
    fmt::print("{}", std::chrono::system_clock::now());
  }
  ```

  prints \"2021-06-18 15:22:00\" (the output depends on the current
  date and time). Thanks @sunmy2019.

- Made more chrono specifiers locale independent by default. Use the
  `'L'` specifier to get localized formatting. For example:

  ```c++
  #include <fmt/chrono.h>

  int main() {
    std::locale::global(std::locale("ru_RU.UTF-8"));
    auto monday = std::chrono::weekday(1);
    fmt::print("{}\n", monday);   // prints "Mon"
    fmt::print("{:L}\n", monday); // prints "пн"
  }
  ```

- Improved locale handling in chrono formatting
  (https://github.com/fmtlib/fmt/issues/2337,
  https://github.com/fmtlib/fmt/pull/2349,
  https://github.com/fmtlib/fmt/pull/2350). Thanks @phprus.

- Deprecated `fmt/locale.h` moving the formatting functions that take
  a locale to `fmt/format.h` (`char`) and `fmt/xchar` (other
  overloads). This doesn\'t introduce a dependency on `<locale>` so
  there is virtually no compile time effect.

- Deprecated an undocumented `format_to` overload that takes
  `basic_memory_buffer`.

- Made parameter order in `vformat_to` consistent with `format_to`
  (https://github.com/fmtlib/fmt/issues/2327).

- Added support for time points with arbitrary durations
  (https://github.com/fmtlib/fmt/issues/2208). For example:

  ```c++
  #include <fmt/chrono.h>

  int main() {
    using tp = std::chrono::time_point<
      std::chrono::system_clock, std::chrono::seconds>;
    fmt::print("{:%S}", tp(std::chrono::seconds(42)));
  }
  ```

  prints \"42\".

- Formatting floating-point numbers no longer produces trailing zeros
  by default for consistency with `std::format`. For example:

  ```c++
  #include <fmt/core.h>

  int main() {
    fmt::print("{0:.3}", 1.1);
  }
  ```

  prints \"1.1\". Use the `'#'` specifier to keep trailing zeros.

- Dropped a limit on the number of elements in a range and replaced
  `{}` with `[]` as range delimiters for consistency with Python\'s
  `str.format`.

- The `'L'` specifier for locale-specific numeric formatting can now
  be combined with presentation specifiers as in `std::format`. For
  example:

  ```c++
  #include <fmt/core.h>
  #include <locale>

  int main() {
    std::locale::global(std::locale("fr_FR.UTF-8"));
    fmt::print("{0:.2Lf}", 0.42);
  }
  ```

  prints \"0,42\". The deprecated `'n'` specifier has been removed.

- Made the `0` specifier ignored for infinity and NaN
  (https://github.com/fmtlib/fmt/issues/2305,
  https://github.com/fmtlib/fmt/pull/2310). Thanks @Liedtke.

- Made the hexfloat formatting use the right alignment by default
  (https://github.com/fmtlib/fmt/issues/2308,
  https://github.com/fmtlib/fmt/pull/2317). Thanks @Liedtke.

- Removed the deprecated numeric alignment (`'='`). Use the `'0'`
  specifier instead.

- Removed the deprecated `fmt/posix.h` header that has been replaced
  with `fmt/os.h`.

- Removed the deprecated `format_to_n_context`, `format_to_n_args` and
  `make_format_to_n_args`. They have been replaced with
  `format_context`, `` format_args` and ``make_format_args\`\`
  respectively.

- Moved `wchar_t`-specific functions and types to `fmt/xchar.h`. You
  can define `FMT_DEPRECATED_INCLUDE_XCHAR` to automatically include
  `fmt/xchar.h` from `fmt/format.h` but this will be disabled in the
  next major release.

- Fixed handling of the `'+'` specifier in localized formatting
  (https://github.com/fmtlib/fmt/issues/2133).

- Added support for the `'s'` format specifier that gives textual
  representation of `bool`
  (https://github.com/fmtlib/fmt/issues/2094,
  https://github.com/fmtlib/fmt/pull/2109). For example:

  ```c++
  #include <fmt/core.h>

  int main() {
    fmt::print("{:s}", true);
  }
  ```

  prints \"true\". Thanks @powercoderlol.

- Made `fmt::ptr` work with function pointers
  (https://github.com/fmtlib/fmt/pull/2131). For example:

  ```c++
  #include <fmt/format.h>

  int main() {
    fmt::print("My main: {}\n", fmt::ptr(main));
  }
  ```

  Thanks @mikecrowe.

- The undocumented support for specializing `formatter` for pointer
  types has been removed.

- Fixed `fmt::formatted_size` with format string compilation
  (https://github.com/fmtlib/fmt/pull/2141,
  https://github.com/fmtlib/fmt/pull/2161). Thanks @alexezeder.

- Fixed handling of empty format strings during format string
  compilation (https://github.com/fmtlib/fmt/issues/2042):

  ```c++
  auto s = fmt::format(FMT_COMPILE(""));
  ```

  Thanks @alexezeder.

- Fixed handling of enums in `fmt::to_string`
  (https://github.com/fmtlib/fmt/issues/2036).

- Improved width computation
  (https://github.com/fmtlib/fmt/issues/2033,
  https://github.com/fmtlib/fmt/issues/2091). For example:

  ```c++
  #include <fmt/core.h>

  int main() {
    fmt::print("{:-<10}{}\n", "你好", "世界");
    fmt::print("{:-<10}{}\n", "hello", "world");
  }
  ```

  prints

  ![](https://user-images.githubusercontent.com/576385/119840373-cea3ca80-beb9-11eb-91e0-54266c48e181.png)

  on a modern terminal.

- The experimental fast output stream (`fmt::ostream`) is now
  truncated by default for consistency with `fopen`
  (https://github.com/fmtlib/fmt/issues/2018). For example:

  ```c++
  #include <fmt/os.h>

  int main() {
    fmt::ostream out1 = fmt::output_file("guide");
    out1.print("Zaphod");
    out1.close();
    fmt::ostream out2 = fmt::output_file("guide");
    out2.print("Ford");
  }
  ```

  writes \"Ford\" to the file \"guide\". To preserve the old file
  content if any pass `fmt::file::WRONLY | fmt::file::CREATE` flags to
  `fmt::output_file`.

- Fixed moving of `fmt::ostream` that holds buffered data
  (https://github.com/fmtlib/fmt/issues/2197,
  https://github.com/fmtlib/fmt/pull/2198). Thanks @vtta.

- Replaced the `fmt::system_error` exception with a function of the
  same name that constructs `std::system_error`
  (https://github.com/fmtlib/fmt/issues/2266).

- Replaced the `fmt::windows_error` exception with a function of the
  same name that constructs `std::system_error` with the category
  returned by `fmt::system_category()`
  (https://github.com/fmtlib/fmt/issues/2274,
  https://github.com/fmtlib/fmt/pull/2275). The latter is
  similar to `std::system_category` but correctly handles UTF-8.
  Thanks @phprus.

- Replaced `fmt::error_code` with `std::error_code` and made it
  formattable (https://github.com/fmtlib/fmt/issues/2269,
  https://github.com/fmtlib/fmt/pull/2270,
  https://github.com/fmtlib/fmt/pull/2273). Thanks @phprus.

- Added speech synthesis support
  (https://github.com/fmtlib/fmt/pull/2206).

- Made `format_to` work with a memory buffer that has a custom
  allocator (https://github.com/fmtlib/fmt/pull/2300).
  Thanks @voxmea.

- Added `Allocator::max_size` support to `basic_memory_buffer`.
  (https://github.com/fmtlib/fmt/pull/1960). Thanks @phprus.

- Added wide string support to `fmt::join`
  (https://github.com/fmtlib/fmt/pull/2236). Thanks @crbrz.

- Made iterators passed to `formatter` specializations via a format
  context satisfy C++20 `std::output_iterator` requirements
  (https://github.com/fmtlib/fmt/issues/2156,
  https://github.com/fmtlib/fmt/pull/2158,
  https://github.com/fmtlib/fmt/issues/2195,
  https://github.com/fmtlib/fmt/pull/2204). Thanks @randomnetcat.

- Optimized the `printf` implementation
  (https://github.com/fmtlib/fmt/pull/1982,
  https://github.com/fmtlib/fmt/pull/1984,
  https://github.com/fmtlib/fmt/pull/2016,
  https://github.com/fmtlib/fmt/pull/2164).
  Thanks @rimathia and @moiwi.

- Improved detection of `constexpr` `char_traits`
  (https://github.com/fmtlib/fmt/pull/2246,
  https://github.com/fmtlib/fmt/pull/2257). Thanks @phprus.

- Fixed writing to `stdout` when it is redirected to `NUL` on Windows
  (https://github.com/fmtlib/fmt/issues/2080).

- Fixed exception propagation from iterators
  (https://github.com/fmtlib/fmt/issues/2097).

- Improved `strftime` error handling
  (https://github.com/fmtlib/fmt/issues/2238,
  https://github.com/fmtlib/fmt/pull/2244). Thanks @yumeyao.

- Stopped using deprecated GCC UDL template extension.

- Added `fmt/args.h` to the install target
  (https://github.com/fmtlib/fmt/issues/2096).

- Error messages are now passed to assert when exceptions are disabled
  (https://github.com/fmtlib/fmt/pull/2145). Thanks @NobodyXu.

- Added the `FMT_MASTER_PROJECT` CMake option to control build and
  install targets when {fmt} is included via `add_subdirectory`
  (https://github.com/fmtlib/fmt/issues/2098,
  https://github.com/fmtlib/fmt/pull/2100).
  Thanks @randomizedthinking.

- Improved build configuration
  (https://github.com/fmtlib/fmt/pull/2026,
  https://github.com/fmtlib/fmt/pull/2122).
  Thanks @luncliff and @ibaned.

- Fixed various warnings and compilation issues
  (https://github.com/fmtlib/fmt/issues/1947,
  https://github.com/fmtlib/fmt/pull/1959,
  https://github.com/fmtlib/fmt/pull/1963,
  https://github.com/fmtlib/fmt/pull/1965,
  https://github.com/fmtlib/fmt/issues/1966,
  https://github.com/fmtlib/fmt/pull/1974,
  https://github.com/fmtlib/fmt/pull/1975,
  https://github.com/fmtlib/fmt/pull/1990,
  https://github.com/fmtlib/fmt/issues/2000,
  https://github.com/fmtlib/fmt/pull/2001,
  https://github.com/fmtlib/fmt/issues/2002,
  https://github.com/fmtlib/fmt/issues/2004,
  https://github.com/fmtlib/fmt/pull/2006,
  https://github.com/fmtlib/fmt/pull/2009,
  https://github.com/fmtlib/fmt/pull/2010,
  https://github.com/fmtlib/fmt/issues/2038,
  https://github.com/fmtlib/fmt/issues/2039,
  https://github.com/fmtlib/fmt/issues/2047,
  https://github.com/fmtlib/fmt/pull/2053,
  https://github.com/fmtlib/fmt/issues/2059,
  https://github.com/fmtlib/fmt/pull/2065,
  https://github.com/fmtlib/fmt/pull/2067,
  https://github.com/fmtlib/fmt/pull/2068,
  https://github.com/fmtlib/fmt/pull/2073,
  https://github.com/fmtlib/fmt/issues/2103,
  https://github.com/fmtlib/fmt/issues/2105,
  https://github.com/fmtlib/fmt/pull/2106,
  https://github.com/fmtlib/fmt/pull/2107,
  https://github.com/fmtlib/fmt/issues/2116,
  https://github.com/fmtlib/fmt/pull/2117,
  https://github.com/fmtlib/fmt/issues/2118,
  https://github.com/fmtlib/fmt/pull/2119,
  https://github.com/fmtlib/fmt/issues/2127,
  https://github.com/fmtlib/fmt/pull/2128,
  https://github.com/fmtlib/fmt/issues/2140,
  https://github.com/fmtlib/fmt/issues/2142,
  https://github.com/fmtlib/fmt/pull/2143,
  https://github.com/fmtlib/fmt/pull/2144,
  https://github.com/fmtlib/fmt/issues/2147,
  https://github.com/fmtlib/fmt/issues/2148,
  https://github.com/fmtlib/fmt/issues/2149,
  https://github.com/fmtlib/fmt/pull/2152,
  https://github.com/fmtlib/fmt/pull/2160,
  https://github.com/fmtlib/fmt/issues/2170,
  https://github.com/fmtlib/fmt/issues/2175,
  https://github.com/fmtlib/fmt/issues/2176,
  https://github.com/fmtlib/fmt/pull/2177,
  https://github.com/fmtlib/fmt/issues/2178,
  https://github.com/fmtlib/fmt/pull/2179,
  https://github.com/fmtlib/fmt/issues/2180,
  https://github.com/fmtlib/fmt/issues/2181,
  https://github.com/fmtlib/fmt/pull/2183,
  https://github.com/fmtlib/fmt/issues/2184,
  https://github.com/fmtlib/fmt/issues/2185,
  https://github.com/fmtlib/fmt/pull/2186,
  https://github.com/fmtlib/fmt/pull/2187,
  https://github.com/fmtlib/fmt/pull/2190,
  https://github.com/fmtlib/fmt/pull/2192,
  https://github.com/fmtlib/fmt/pull/2194,
  https://github.com/fmtlib/fmt/pull/2205,
  https://github.com/fmtlib/fmt/issues/2210,
  https://github.com/fmtlib/fmt/pull/2211,
  https://github.com/fmtlib/fmt/pull/2215,
  https://github.com/fmtlib/fmt/pull/2216,
  https://github.com/fmtlib/fmt/pull/2218,
  https://github.com/fmtlib/fmt/pull/2220,
  https://github.com/fmtlib/fmt/issues/2228,
  https://github.com/fmtlib/fmt/pull/2229,
  https://github.com/fmtlib/fmt/pull/2230,
  https://github.com/fmtlib/fmt/issues/2233,
  https://github.com/fmtlib/fmt/pull/2239,
  https://github.com/fmtlib/fmt/issues/2248,
  https://github.com/fmtlib/fmt/issues/2252,
  https://github.com/fmtlib/fmt/pull/2253,
  https://github.com/fmtlib/fmt/pull/2255,
  https://github.com/fmtlib/fmt/issues/2261,
  https://github.com/fmtlib/fmt/issues/2278,
  https://github.com/fmtlib/fmt/issues/2284,
  https://github.com/fmtlib/fmt/pull/2287,
  https://github.com/fmtlib/fmt/pull/2289,
  https://github.com/fmtlib/fmt/pull/2290,
  https://github.com/fmtlib/fmt/pull/2293,
  https://github.com/fmtlib/fmt/issues/2295,
  https://github.com/fmtlib/fmt/pull/2296,
  https://github.com/fmtlib/fmt/pull/2297,
  https://github.com/fmtlib/fmt/issues/2311,
  https://github.com/fmtlib/fmt/pull/2313,
  https://github.com/fmtlib/fmt/pull/2315,
  https://github.com/fmtlib/fmt/issues/2320,
  https://github.com/fmtlib/fmt/pull/2321,
  https://github.com/fmtlib/fmt/pull/2323,
  https://github.com/fmtlib/fmt/issues/2328,
  https://github.com/fmtlib/fmt/pull/2329,
  https://github.com/fmtlib/fmt/pull/2333,
  https://github.com/fmtlib/fmt/pull/2338,
  https://github.com/fmtlib/fmt/pull/2341).
  Thanks @darklukee, @fagg, @killerbot242, @jgopel, @yeswalrus, @Finkman,
  @HazardyKnusperkeks, @dkavolis, @concatime, @chronoxor, @summivox, @yNeo,
  @Apache-HB, @alexezeder, @toojays, @Brainy0207, @vadz, @imsherlock, @phprus,
  @white238, @yafshar, @BillyDonahue, @jstaahl, @denchat, @DanielaE,
  @ilyakurdyukov, @ilmai, @JessyDL, @sergiud, @mwinterb, @sven-herrmann,
  @jmelas, @twoixter, @crbrz and @upsj.

- Improved documentation
  (https://github.com/fmtlib/fmt/issues/1986,
  https://github.com/fmtlib/fmt/pull/2051,
  https://github.com/fmtlib/fmt/issues/2057,
  https://github.com/fmtlib/fmt/pull/2081,
  https://github.com/fmtlib/fmt/issues/2084,
  https://github.com/fmtlib/fmt/pull/2312).
  Thanks @imba-tjd, @0x416c69 and @mordante.

- Continuous integration and test improvements
  (https://github.com/fmtlib/fmt/issues/1969,
  https://github.com/fmtlib/fmt/pull/1991,
  https://github.com/fmtlib/fmt/pull/2020,
  https://github.com/fmtlib/fmt/pull/2110,
  https://github.com/fmtlib/fmt/pull/2114,
  https://github.com/fmtlib/fmt/issues/2196,
  https://github.com/fmtlib/fmt/pull/2217,
  https://github.com/fmtlib/fmt/pull/2247,
  https://github.com/fmtlib/fmt/pull/2256,
  https://github.com/fmtlib/fmt/pull/2336,
  https://github.com/fmtlib/fmt/pull/2346).
  Thanks @jgopel, @alexezeder and @DanielaE.

The change log for versions 0.8.0 - 7.1.3 is available [here](
doc/ChangeLog-old.md).
