#!/bin/bash

# install other tools
apt-get update -y
apt-get install -y sqlite3 libsqlite3-dev libfmt-dev

# download and install RPD
git clone https://github.com/ROCm/rocmProfileData.git

# install rpd module
cd rocmProfileData/
make && make install

