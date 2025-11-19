# ROCm quickstart for sgl-diffusion


## Method 1: From source
```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip ninja-build cmake build-essential \
    ffmpeg libsm6 libxext6 pkg-config
python3 -m venv ~/sglang-venv
source ~/sglang-venv/bin/activate
pip install --upgrade pip

git clone https://github.com/sgl-project/sglang.git
cd sglang
git submodule update --init --recursive
rm -f python/pyproject.toml
cp python/pyproject_other.toml python/pyproject.toml
export AMDGPU_TARGET=gfx942   # run `rocminfo | grep -i gfx` to get your own architecture

cd sgl-kernel
AMDGPU_TARGET=$AMDGPU_TARGET python setup_rocm.py install
cd ..
pip uninstall -y aiter || true
git clone https://github.com/ROCm/aiter.git
cd aiter && git checkout v0.1.7.post1
GPU_ARCHS=$AMDGPU_TARGET python setup.py develop
cd ..

pip install -e "python[all_hip,diffusion]" --no-build-isolation
huggingface-cli login

export SGLANG_DIFFUSION_TARGET_DEVICE=rocm
sglang generate \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --attention-backend aiter \
  --text-encoder-cpu-offload --pin-cpu-memory \
  --prompt "A pastel origami dragon soaring over a calm lake at sunrise." \
  --num-frames 8 \
  --save-output --output-path outputs/wan21-mi300x
```

## Method 2: Using Docker
```bash
docker run --device=/dev/kfd --device=/dev/dri --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env HF_TOKEN=<secret> \
  <IMAGE_PATH> \
  sglang generate --model-path black-forest-labs/FLUX.1-dev --prompt "A logo With Bold Large text: SGL Diffusion" --save-output
```
