#!/bin/sh
#
# Creates fuzzer builds of various kinds
# - oss-fuzz emulated mode (makes sure a simulated invocation by oss-fuzz works)
# - libFuzzer build (you will need clang)
# - afl build (you will need afl)
#
#
# Copyright (c) 2019 Paul Dreik
#
# For the license information refer to format.h.

set -e
me=$(basename $0)
root=$(readlink -f "$(dirname "$0")/../..")


echo $me: root=$root

here=$(pwd)

CXXFLAGSALL="-DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION= -g"
CMAKEFLAGSALL="$root -GNinja -DCMAKE_BUILD_TYPE=Debug -DFMT_DOC=Off -DFMT_TEST=Off -DFMT_FUZZ=On -DCMAKE_CXX_STANDARD=17"

CLANG=clang++-11

# For performance analysis of the fuzzers.
builddir=$here/build-fuzzers-perfanalysis
mkdir -p $builddir
cd $builddir
CXX="ccache g++" CXXFLAGS="$CXXFLAGSALL -g" cmake \
$CMAKEFLAGSALL \
-DFMT_FUZZ_LINKMAIN=On \
-DCMAKE_BUILD_TYPE=Release

cmake --build $builddir

# Builds the fuzzers as oss-fuzz does.
builddir=$here/build-fuzzers-ossfuzz
mkdir -p $builddir
cd $builddir
CXX=$CLANG \
CXXFLAGS="$CXXFLAGSALL -fsanitize=fuzzer-no-link" cmake \
cmake $CMAKEFLAGSALL \
-DFMT_FUZZ_LINKMAIN=Off \
-DFMT_FUZZ_LDFLAGS="-fsanitize=fuzzer"

cmake --build $builddir


# Builds fuzzers for local fuzzing with libfuzzer with asan+usan.
builddir=$here/build-fuzzers-libfuzzer
mkdir -p $builddir
cd $builddir
CXX=$CLANG \
CXXFLAGS="$CXXFLAGSALL -fsanitize=fuzzer-no-link,address,undefined" cmake \
cmake $CMAKEFLAGSALL \
-DFMT_FUZZ_LINKMAIN=Off \
-DFMT_FUZZ_LDFLAGS="-fsanitize=fuzzer"

cmake --build $builddir

# Builds a fast fuzzer for making coverage fast.
builddir=$here/build-fuzzers-fast
mkdir -p $builddir
cd $builddir
CXX=$CLANG \
CXXFLAGS="$CXXFLAGSALL -fsanitize=fuzzer-no-link -O3" cmake \
cmake $CMAKEFLAGSALL \
-DFMT_FUZZ_LINKMAIN=Off \
-DFMT_FUZZ_LDFLAGS="-fsanitize=fuzzer" \
 -DCMAKE_BUILD_TYPE=Release

cmake --build $builddir


# Builds fuzzers for local fuzzing with afl.
builddir=$here/build-fuzzers-afl
mkdir -p $builddir
cd $builddir
CXX="afl-g++" \
CXXFLAGS="$CXXFLAGSALL -fsanitize=address,undefined" \
cmake $CMAKEFLAGSALL \
-DFMT_FUZZ_LINKMAIN=On

cmake --build $builddir


echo $me: all good

