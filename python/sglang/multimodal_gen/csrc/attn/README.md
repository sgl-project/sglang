

# Attention Kernel Used in sgl-diffusion



## Video Sparse Attention (VSA)

### Installation
We support H100 (via TK) and any other GPU (via triton) for VSA.
```bash
git submodule update --init --recursive
python setup_vsa.py install
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
python tests/test_vsa.py
# (For H100) test speed
python benchmarks/bench_vsa_hopper.py
```
bench_vsa_hopper.py should print something like this:
```bash
Using topk=76 kv blocks per q block (out of 768 total kv blocks)

=== BLOCK SPARSE ATTENTION BENCHMARK ===
Block Sparse Forward  - TFLOPS: 5622.26
Block Sparse Backward - TFLOPS: 3865.68
```


## Sliding Tile Attention (STA)
We only support H100 for STA.
```bash
git submodule update --init --recursive
python setup_sta.py install
```




###  Usage
End-2-end inference with sgl-diffusion:
```bash
bash scripts/inference/v1_inference_wan_STA.sh
```

If you want to use sliding tile attention in your custom model:
```python
from st_attn import sliding_tile_attention
# assuming video size (T, H, W) = (30, 48, 80), text tokens = 256 with padding.
# q, k, v: [batch_size, num_heads, seq_length, head_dim], seq_length = T*H*W + 256
# a tile is a cube of size (6, 8, 8)
# window_size in tiles: [(window_t, window_h, window_w), (..)...]. For example, window size (3, 3, 3) means a query can attend to (3x6, 3x8, 3x8) = (18, 24, 24) tokens out of the total 30x48x80 video.
# text_length: int ranging from 0 to 256
# If your attention contains text token (Hunyuan)
out = sliding_tile_attention(q, k, v, window_size, text_length)
# If your attention does not contain text token (StepVideo)
out = sliding_tile_attention(q, k, v, window_size, 0, False)
```


### Test
```bash
python tests/test_sta.py # test STA
python tests/test_vsa.py # test VSA
```
### Benchmark
```bash
python benchmarks/bench_sta.py
```


### How Does STA Work?
We give a demo for 2D STA with window size (6,6) operating on a (10, 10) image.


https://github.com/user-attachments/assets/f3b6dd79-7b43-4b60-a0fa-3d6495ec5747

## Why is STA Fast?
2D/3D Sliding Window Attention (SWA) creates many mixed blocks in the attention map. Even though mixed blocks have less output value,a mixed block is significantly slower than a dense block due to the GPU-unfriendly masking operation.

STA removes mixed blocks.


<div align="center">
<img src=../../assets/sliding_tile_attn_map.png width="80%"/>
</div>

## Acknowledgement

We learned or reuse code from FlexAtteniton, NATEN, and ThunderKittens.
