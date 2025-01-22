#!/bin/bash
set -ex
WORKSPACE_PATH=$(dirname "$(readlink -f "$0")")

# build turbomind
cd 3rdparty/turbomind
rm -rf build && mkdir -p build && cd build
cmake .. -DBUILD_TEST=OFF
make -j$(nproc)

# create wheel
cd lib
LIB_NAME=$(ls *.so | head -n 1)
cp ${LIB_NAME} $WORKSPACE_PATH/src/sgl-kernel/turbomind
cd ../../../..
rm MANIFEST.in && touch MANIFEST.in
echo "include src/sgl-kernel/turbomind/${LIB_NAME}" > MANIFEST.in
echo "include src/sgl-kernel/turbomind/__init__.py" >> MANIFEST.in

python setup.py bdist_wheel
