# DeepSeek V3.2 Usage

[DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) equips DeepSeek-V3.1-Terminus with DeepSeek Sparse Attention (DSA) through continued training. With DSA, a fine-grained sparse attention mechanism powered by a lightning indexer, DeepSeek-V3.2 achieves efficiency improvements in long-context scenarios.

For reporting issues or tracking upcoming features, please refer to this [Roadmap](https://github.com/sgl-project/sglang/issues/11060).

## Installation

### Docker

```bash
# H200/B200
docker pull lmsysorg/sglang:v0.5.3-cu129

# MI350/MI355
docker pull lmsysorg/sglang:dsv32-rocm

# NPUs
docker pull lmsysorg/sglang:dsv32-a2
docker pull lmsysorg/sglang:dsv32-a3
```

### Build From Source

```bash
# Install SGLang
git clone https://github.com/sgl-project/sglang
cd sglang
pip3 install pip --upgrade
pip3 install -e "python[all]"

# Install flash_mla
git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla
cd flash-mla
git submodule update --init --recursive
pip install -v .

# Install fast_hadamard_transform
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install .
```
## Launch DeepSeek V3.2 with SGLang

To serve DeepSeek-V3.2-Exp on 8xH200/B200 GPUs:

```bash
# Launch with TP + DP
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp 8 --enable-dp-attention

# Launch with EP + DP
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --ep 8 --dp 8 --enable-dp-attention
```

### Configuration Tips
- **DP Attention**: For DeepSeek V3.2 model, the kernels are customized for the use case of `dp_size=8`.


## Benchmarking Results

### Accuracy Test with `gsm8k`

```bash
python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319
```


### Accuracy Test with `gpqa-diamond`

```bash
python3 -m sglang.test.run_eval --port 30000 --eval-name gpqa --num-examples 198 --max-tokens 4096 --repeat 10 --thinking-mode deepseek-v3
```
