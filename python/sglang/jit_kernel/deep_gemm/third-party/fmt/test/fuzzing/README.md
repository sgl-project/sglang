# Running the fuzzers locally

There is a [helper script](build.sh) to build the fuzzers, which has only been
tested on Debian and Ubuntu linux so far. There should be no problems fuzzing on
Windows (using clang>=8) or on Mac, but the script will probably not work out of
the box.

Something along
```sh
mkdir build
cd build
export CXX=clang++
export CXXFLAGS="-fsanitize=fuzzer-no-link -DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION= -g"
cmake .. -DFMT_SAFE_DURATION_CAST=On -DFMT_FUZZ=On -DFMT_FUZZ_LINKMAIN=Off -DFMT_FUZZ_LDFLAGS="-fsanitize=fuzzer"
cmake --build .
```
should work to build the fuzzers for all platforms which clang supports.

Execute a fuzzer with for instance
```sh
cd build
export UBSAN_OPTIONS=halt_on_error=1
mkdir out_chrono
bin/fuzzer_chrono_duration out_chrono
```
