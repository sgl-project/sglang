# Get Started

Compile and run {fmt} examples online with [Compiler Explorer](
https://godbolt.org/z/P7h6cd6o3).

{fmt} is compatible with any build system. The next section describes its usage
with CMake, while the [Build Systems](#build-systems) section covers the rest.

## CMake

{fmt} provides two CMake targets: `fmt::fmt` for the compiled library and
`fmt::fmt-header-only` for the header-only library. It is recommended to use
the compiled library for improved build times.

There are three primary ways to use {fmt} with CMake:

* **FetchContent**: Starting from CMake 3.11, you can use [`FetchContent`](
  https://cmake.org/cmake/help/v3.30/module/FetchContent.html) to automatically
  download {fmt} as a dependency at configure time:

        include(FetchContent)

        FetchContent_Declare(
          fmt
          GIT_REPOSITORY https://github.com/fmtlib/fmt
          GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
        FetchContent_MakeAvailable(fmt)

        target_link_libraries(<your-target> fmt::fmt)

* **Installed**: You can find and use an [installed](#installation) version of
  {fmt} in your `CMakeLists.txt` file as follows:

        find_package(fmt)
        target_link_libraries(<your-target> fmt::fmt)

* **Embedded**: You can add the {fmt} source tree to your project and include it
  in your `CMakeLists.txt` file:

        add_subdirectory(fmt)
        target_link_libraries(<your-target> fmt::fmt)

## Installation

### Debian/Ubuntu

To install {fmt} on Debian, Ubuntu, or any other Debian-based Linux
distribution, use the following command:

    apt install libfmt-dev

### Homebrew

Install {fmt} on macOS using [Homebrew](https://brew.sh/):

    brew install fmt

### Conda

Install {fmt} on Linux, macOS, and Windows with [Conda](
https://docs.conda.io/en/latest/), using its [conda-forge package](
https://github.com/conda-forge/fmt-feedstock):

    conda install -c conda-forge fmt

### vcpkg

Download and install {fmt} using the vcpkg package manager:

    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    ./bootstrap-vcpkg.sh
    ./vcpkg integrate install
    ./vcpkg install fmt

<!-- The fmt package in vcpkg is kept up to date by Microsoft team members and
community contributors. If the version is out of date, please [create an
issue or pull request](https://github.com/Microsoft/vcpkg) on the vcpkg
repository. -->

## Building from Source

CMake works by generating native makefiles or project files that can be
used in the compiler environment of your choice. The typical workflow
starts with:

    mkdir build  # Create a directory to hold the build output.
    cd build
    cmake ..     # Generate native build scripts.

run in the `fmt` repository.

If you are on a Unix-like system, you should now see a Makefile in the
current directory. Now you can build the library by running `make`.

Once the library has been built you can invoke `make test` to run the tests.

You can control generation of the make `test` target with the `FMT_TEST`
CMake option. This can be useful if you include fmt as a subdirectory in
your project but don't want to add fmt's tests to your `test` target.

To build a shared library set the `BUILD_SHARED_LIBS` CMake variable to `TRUE`:

    cmake -DBUILD_SHARED_LIBS=TRUE ..

To build a static library with position-independent code (e.g. for
linking it into another shared library such as a Python extension), set the
`CMAKE_POSITION_INDEPENDENT_CODE` CMake variable to `TRUE`:

    cmake -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE ..

After building the library you can install it on a Unix-like system by
running `sudo make install`.

### Building the Docs

To build the documentation you need the following software installed on
your system:

- [Python](https://www.python.org/)
- [Doxygen](http://www.stack.nl/~dimitri/doxygen/)
- [MkDocs](https://www.mkdocs.org/) with `mkdocs-material`, `mkdocstrings`,
  `pymdown-extensions` and `mike`

First generate makefiles or project files using CMake as described in
the previous section. Then compile the `doc` target/project, for example:

    make doc

This will generate the HTML documentation in `doc/html`.

## Build Systems

### build2

You can use [build2](https://build2.org), a dependency manager and a build
system, to use {fmt}.

Currently this package is available in these package repositories:

- <https://cppget.org/fmt/> for released and published versions.
- <https://github.com/build2-packaging/fmt> for unreleased or custom versions.

**Usage:**

- `build2` package name: `fmt`
- Library target name: `lib{fmt}`

To make your `build2` project depend on `fmt`:

- Add one of the repositories to your configurations, or in your
  `repositories.manifest`, if not already there:

        :
        role: prerequisite
        location: https://pkg.cppget.org/1/stable

- Add this package as a dependency to your `manifest` file (example
  for version 10):

        depends: fmt ~10.0.0

- Import the target and use it as a prerequisite to your own target
  using `fmt` in the appropriate `buildfile`:

        import fmt = fmt%lib{fmt}
        lib{mylib} : cxx{**} ... $fmt

Then build your project as usual with `b` or `bdep update`.

### Meson

[Meson WrapDB](https://mesonbuild.com/Wrapdb-projects.html) includes an `fmt`
package.

**Usage:**

- Install the `fmt` subproject from the WrapDB by running:

        meson wrap install fmt

  from the root of your project.

- In your project's `meson.build` file, add an entry for the new subproject:

        fmt = subproject('fmt')
        fmt_dep = fmt.get_variable('fmt_dep')

- Include the new dependency object to link with fmt:

        my_build_target = executable(
          'name', 'src/main.cc', dependencies: [fmt_dep])

**Options:**

If desired, {fmt} can be built as a static library, or as a header-only library.

For a static build, use the following subproject definition:

    fmt = subproject('fmt', default_options: 'default_library=static')
    fmt_dep = fmt.get_variable('fmt_dep')

For the header-only version, use:

    fmt = subproject('fmt', default_options: ['header-only=true'])
    fmt_dep = fmt.get_variable('fmt_header_only_dep')

### Android NDK

{fmt} provides [Android.mk file](
https://github.com/fmtlib/fmt/blob/master/support/Android.mk) that can be used
to build the library with [Android NDK](
https://developer.android.com/tools/sdk/ndk/index.html).

### Other

To use the {fmt} library with any other build system, add
`include/fmt/base.h`, `include/fmt/format.h`, `include/fmt/format-inl.h`,
`src/format.cc` and optionally other headers from a [release archive](
https://github.com/fmtlib/fmt/releases) or the [git repository](
https://github.com/fmtlib/fmt) to your project, add `include` to include
directories and make sure `src/format.cc` is compiled and linked with your code.
