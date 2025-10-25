

# Attention Kernel Used in sgl-diffusion

## Video Sparse Attention (VSA)

### Installation
We support H100 (via TK) and any other GPU (via triton) for VSA.

```bash
pip install vsa
```

Install from source:

```bash
git submodule update --init --recursive
python setup.py install
```


If you encounter error during installation, try below:
Install C++20 for ThunderKittens:
```bash
sudo apt update
sudo apt install gcc-11 g++-11

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11

sudo apt update
sudo apt install clang-11
```
(If you use CUDA12.8)
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
```

### Verify if you have successfully installed

```bash
# test numerical
python ../tests/test_vsa.py
# (For H100) test speed
python ../benchmarks/bench_vsa_hopper.py
```

bench_vsa_hopper.py should print something like this:

```bash
Using topk=76 kv blocks per q block (out of 768 total kv blocks)

=== BLOCK SPARSE ATTENTION BENCHMARK ===
Block Sparse Forward  - TFLOPS: 5622.26
Block Sparse Backward - TFLOPS: 3865.68
```

## Acknowledgement

We learned or reuse code from FlexAtteniton, NATEN, and ThunderKittens.
