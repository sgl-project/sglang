# Copyright (c) 2019, Paul Dreik
# License: see LICENSE.rst in the fmt root directory

# Link in the main function. Useful for reproducing, kcov, gdb, afl, valgrind.
# (Note that libFuzzer can also reproduce, just pass it the files.)
option(FMT_FUZZ_LINKMAIN "Enables the reproduce mode, instead of libFuzzer" On)

# For oss-fuzz - insert $LIB_FUZZING_ENGINE into the link flags, but only for
# the fuzz targets, otherwise the CMake configuration step fails.
set(FMT_FUZZ_LDFLAGS "" CACHE STRING "LDFLAGS for the fuzz targets")

# Adds a binary for reproducing, i.e. no fuzzing, just enables replaying data
# through the fuzzers.
function(add_fuzzer source)
  get_filename_component(basename ${source} NAME_WE)
  set(name ${basename}-fuzzer)
  add_executable(${name} ${source} fuzzer-common.h)
  if (FMT_FUZZ_LINKMAIN)
    target_sources(${name} PRIVATE main.cc)
  endif ()
  target_link_libraries(${name} PRIVATE fmt)
  if (FMT_FUZZ_LDFLAGS)
    target_link_libraries(${name} PRIVATE ${FMT_FUZZ_LDFLAGS})
  endif ()
  target_compile_features(${name} PRIVATE cxx_std_14)
endfunction()

foreach (source chrono-duration.cc chrono-timepoint.cc float.cc named-arg.cc one-arg.cc two-args.cc)
  add_fuzzer(${source})
endforeach ()
