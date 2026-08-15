# download and install RPD
apt update && apt install -y sqlite3 libsqlite3-dev libfmt-dev

# install rpd module
git clone https://github.com/ROCmSoftwarePlatform/rocmProfileData
cd rocmProfileData
git apply rpd.patch
make && make install
cd rocpd_python && python setup.py install && cd ..
cd rpd_tracer && make clean;make install && python setup.py install && cd ..
