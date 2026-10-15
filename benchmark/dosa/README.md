# Reproduce GLM-5.2 TTFT

DoSA reduces GLM-5.2-FP8's 1M-token TTFT from **133.65 s to 71.15 s (1.88×)**
on eight GB200 GPUs. Upstream base: `d2a436446cca3d955358e7b4a5bf6881ea8591cd`.

DoSA uses a seed threshold and conservative bounds to avoid full-head scoring
for excluded keys. Remaining candidates and capacity recovery use exact scores.
SM100 FP8 indexers with 32/64 heads and 128 dimensions share the same implementation.

## Install

Use Python 3.12, CUDA 13 with `nvcc`, a C++ compiler and Rust 1.92. Run on two
four-GPU GB200 nodes in one NVLink domain:

```bash
git clone --branch dosa-main https://github.com/llm-sys/dosa-sglang.git
cd dosa-sglang
python3.12 -m venv .venv
source .venv/bin/activate
unset PYTHONPATH
python -m pip install "torch==2.14.1" "setuptools>=61" \
  "setuptools-rust>=1.11" "setuptools-scm>=8" wheel
python -m pip install --no-build-isolation ./python
python benchmark/dosa/build_exact_topk.py --output .dosa_topk
export PYTHONPATH="$PWD/.dosa_topk/python"
```

This installs PyTorch 2.14.1, FlashInfer 0.7.1 and DeepGEMM 0.2.2, builds the
SGLang wheel/Rust extensions and registers the shared exact Top-K selectors for
both modes. Native DoSA kernels compile at server initialization and use the
standard PyTorch extension cache.

## Prepare the original inputs

Prepare the five original UTF-8 prefixes of Chinese long-book sample `id=123`
from [InfiniteBench](https://github.com/OpenBMB/InfiniteBench). The helper pins
the dataset revision and original character boundaries; tokenization stays on
the server.

```bash
python benchmark/dosa/prepare_ttft_inputs.py --output-dir ttft-inputs
```

Add `--dataset /path/to/longbook_qa_chn.jsonl` to reuse a downloaded dataset.

## Start each mode

Use the GLM-5.2-FP8 checkpoint on both nodes. Set `NODE_RANK=0` on the first
node and `NODE_RANK=1` on the second. `NODE0_ADDR` is the first node's address.
Run one mode at a time: `off` for SGLang, then `few` for DoSA.

```bash
source .venv/bin/activate
export MODEL_PATH=/path/to/GLM-5.2-FP8
export NODE0_ADDR=192.0.2.10
export NODE_RANK=0
export CHUNK=32768
export SGLANG_DOSA_MODE=off
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$PWD/.dosa_topk/python"
export TORCH_EXTENSIONS_DIR="/tmp/dosa-main-node-${NODE_RANK}/torch_extensions"

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp 8 --ep 1 --nnodes 2 --node-rank "$NODE_RANK" \
  --dist-init-addr "${NODE0_ADDR}:29784" \
  --host 0.0.0.0 --port 30184 \
  --chunked-prefill-size "$CHUNK" --max-prefill-tokens 16384 \
  --mem-fraction-static 0.88 --page-size 64 --kv-cache-dtype fp8_e4m3 \
  --dsa-topk-backend sgl-kernel \
  --dsa-prefill-backend trtllm --dsa-decode-backend trtllm \
  --moe-runner-backend flashinfer_trtllm --moe-a2a-backend none \
  --sampling-defaults model --tokenizer-worker-num 1 \
  --schedule-policy fcfs --schedule-conservativeness 1.0 \
  --num-continuous-decode-steps 1 \
  --random-seed 233767787 --stream-interval 1 \
  --cuda-graph-config '{"decode":{"backend":"full","max_bs":512},"prefill":{"backend":"breakable","max_bs":2048}}'
```

These settings use the upstream full-decode/breakable-prefill bucket defaults,
which match the original bucket lists.
For the 16K configuration, set `CHUNK=16384` for both modes.

## Measure TTFT

After the server reports ready, run the client on node 0:

```bash
mkdir -p ttft-results
for tokens in 131072 262144 524288 786432 1048569; do
  python benchmark/dosa/bench_ttft.py \
    --url "http://${NODE0_ADDR}:30184" \
    --text "ttft-inputs/prefix-${tokens}.txt" \
    --loaded 3 \
    --output "ttft-results/${SGLANG_DOSA_MODE}-q${CHUNK}-${tokens}.json"
done
```

Repeat with `SGLANG_DOSA_MODE=few` on both nodes and the same server arguments.

Every prompt gets one warmup and three measured requests, each with a fresh ID
and prefix-cache flush. TTFT ends at the first streamed completion token,
including server tokenization, prefill and request-owned prior work. Generation
uses `max_new_tokens=2` and default EOS. Startup/compilation is outside timing;
all samples, the median and standard deviation are saved.

## Results

| Chunk | Input tokens | SGLang TTFT (s) | DoSA TTFT (s) | Speedup |
|---:|---:|---:|---:|---:|
| 32K | 131072 | 5.729 | 5.680 | 1.009× |
| 32K | 262144 | 14.407 | 12.503 | 1.152× |
| 32K | 524288 | 40.973 | 28.595 | 1.433× |
| 32K | 786432 | 81.028 | 48.043 | 1.687× |
| 32K | 1048569 | 133.653 | 71.149 | 1.878× |
| 16K | 131072 | 5.847 | 5.825 | 1.004× |
| 16K | 262144 | 14.567 | 12.738 | 1.144× |
| 16K | 524288 | 41.003 | 29.041 | 1.412× |
| 16K | 786432 | 81.314 | 48.716 | 1.669× |
| 16K | 1048569 | 133.627 | 71.876 | 1.859× |

[ttft-results.csv](ttft-results.csv) contains all 60 measured samples. DoSA
results use the compact implementation on this branch. SGLang 32K reuses
source `030b210f8e54abe9307e93ff02ac9f4bd322b03b`; SGLang 16K reuses
source `146bb3467865b6a1fce830a11aeb2dfb11b2f0de`. Both modes use the configuration above.

DoSA reserves candidate records, ID pages and task tables within one budget
before KV sizing. At 1M KV, 4096-row physical slabs process logical 16K/32K chunks;
capacity overflow uses full-head exact recovery. SGLang also budgets request
mapping and KV together. Inputs, score precision and Top-K semantics are shared.
