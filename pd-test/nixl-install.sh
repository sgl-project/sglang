#!/bin/bash

set -ex

pushd ~
apt update
apt install -y build-essential cmake pkg-config
pip3 install meson ninja pybind11



wget https://github.com/openucx/ucx/releases/download/v1.18.0/ucx-1.18.0.tar.gz
rm ucx-1.18.0 -rf
tar xzf ucx-1.18.0.tar.gz
cd ucx-1.18.0

./configure                          \
    --prefix=/opt/ucx-1.18.0          \
    --enable-shared                    \
    --disable-static                   \
    --disable-doxygen-doc              \
    --enable-optimizations             \
    --enable-cma                       \
    --enable-devel-headers             \
    --with-verbs                       \
    --with-gdr                         \
    --with-cuda=/usr/local/cuda        \
    --with-dm                          \
    --enable-mt

make -j
make -j install-strip
ldconfig

echo "export PATH=/opt/ucx-1.18.0/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/opt/ucx-1.18.0/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc

cd ~
rm -rf nixl
git clone https://github.com/ai-dynamo/nixl.git
cd nixl
git checkout 1c943627d66777e30a81632e3c577c27c0509257
rm -rf build
mkdir build
meson setup build
cd build
meson configure -Ducx_path=/opt/ucx-1.18.0
ninja
ninja install
cd ..
pip install .

popd





UCX_TLS=tcp,cuda UCX_NET_DEVICES=bond1 ucx_perftest
UCX_TLS=tcp,cuda UCX_NET_DEVICES=bond1 ucx_perftest -t ucp_am_bw -s 8192 -w 1000 -i 10000 -f 127.0.0.1 -m cuda,cuda