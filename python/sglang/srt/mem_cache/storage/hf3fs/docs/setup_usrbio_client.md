# HiCacheHF3FS Setup

## Build & Package
### Source Code
https://github.com/deepseek-ai/3FS/blob/main/README.md#check-out-source-code
```sh
git clone https://github.com/deepseek-ai/3fs

cd 3fs
git submodule update --init --recursive
./patches/apply.sh
```

### Build Dev Container
https://github.com/deepseek-ai/3FS/blob/main/dockerfile/dev.dockerfile
```sh
cd 3fs/dockerfile
docker build -t hf3fs:dev -f dev.dockerfile .
```

### Generate Python Wheel
```sh
docker run -it hf3fs:dev bash

# Inside the development container
git clone https://github.com/deepseek-ai/3fs

cd 3fs
git submodule update --init --recursive
./patches/apply.sh

apt-get update \
&& apt-get install -y --no-install-recommends \
python3 python3-pip \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*
# apt install python3.12 python3.12-venv python3.12-dev
# curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# python3.12 get-pip.py

# Generated wheel location: dist/hf3fs_py_usrbio-1.2.9+2db69ce-cp310-cp310-linux_x86_64.whl
python3 setup.py bdist_wheel
```

## Installation
```sh
# Install Dependencies
# https://github.com/deepseek-ai/3FS/blob/main/dockerfile/dev.dockerfile
apt update && apt install -y                            \
  libaio-dev                                            \
  libboost-all-dev                                      \
  libdouble-conversion-dev                              \
  libdwarf-dev                                          \
  libgflags-dev                                         \
  libgmock-dev                                          \
  libgoogle-glog-dev                                    \
  libgoogle-perftools-dev                               \
  libgtest-dev                                          \
  liblz4-dev                                            \
  liblzma-dev                                           \
  libssl-dev                                            \
  libunwind-dev                                         \
  libuv1-dev

# Install Python Package
pip install hf3fs_py_usrbio-1.2.9+394583d-cp312-cp312-linux_x86_64.whl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/dist-packages
```
