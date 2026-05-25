# Implementation A: DoubleSparse — Survey Findings

## 1. Repo Map
- Top-level dirs: `models/` (standalone GPT-Fast style), `offloading/` (CPU-GPU offload variant), `evaluation/` (HF-model adapters + eval scripts), `config/` (pre-computed channel configs + calibration), `benchmark/` (operator-level attention latency), `lines/` (line-retrieval test cases), `AIME/` (math reasoning eval), `LongBench/` (long-context eval), `scripts/` (entry scripts), `data/` (plots), `assets/`.
- **Standalone**, not HF/SGLang/vLLM fork; loosely based on Meta GPT-Fast.
- PyTorch 2.5.1 + Triton 3.1.0; HF transformers 4.47.1 used for adapter path.

## 2. Entry Points
- Calibration: `config/offline_calibration.py` — `python3 offline_calibration.py --model_path … --output_dir config/`. Hook-based capture of per-head per-channel mean(|Q·K|) on Pile val (256 samples, 512 block).
- Standalone inference: `models/generate.py` — `python3 generate.py --checkpoint_path … --max_new_tokens 2048 --batch_size 4`. Prefill until `heavy_const` tokens accumulated then `sparse_decode_one_token()`.
- HF eval: `evaluation/perplexity_eval.py --model_path … --heavy_const 128 --group_factor 4 --architecture llama --channel q`. Modifies HF model via `convert_kvcache_llama_heavy_recent()` + `convert_llama_channel_config()`.
- Offloading: same script with `--mode ds-offload`. Uses `convert_kvcache_llama_offloading()` from `evaluation/offload_llama.py`.
- Kernel bench: `scripts/run_attn.sh` → `models/triton_kernels/attention.py`.
- Other evals: `evaluation/mmlu.py`, `evaluation/retrieval_eval.py`, `evaluation/chat.py`, `AIME/aime.py`, `LongBench/pred.py`.

## 3. Models Supported (with adapters)
- LLaMA (7B, 13B, 70B, Llama-3.1-8B, Llama-3.2-11B-Vision) — `modify_llama.py:LlamaAttention_heavy_hitter`
- Mistral 7B, Mixtral-8x7B — `modify_mistral.py`, `modify_mixtral.py`
- Qwen2 7B/32B, DeepSeek-R1-Distill-Qwen-32B — `modify_qwen2.py`
- Vicuna 7B v1.5 16k, LongChat 7B v1.5 32k — via LLaMA modifier.
- 13+ pre-computed JSON configs in `config/` (e.g. `config/meta-llama/Llama-2-7b-hf.json` ~3.1 MB).

## 4. Where Sparsity Lives
### A. Offline channel calibration
- File: `config/offline_calibration.py` (Method 1 in `get_qk_hook()` lines 91–93): `out = (q * k).reshape(-1, num_heads, head_dim).abs().mean(dim=0)` averaged across 256 samples.
- Output artifact: JSON `{layer_proj_key: [[channel_idx ...] per head]}` — per-layer, per-head sorted channel indices, 128 ints per head; key shape `"model.layers.{i}.self_attn.{q|k|qk}_proj"`.
- Permutation at load (`models/model.py:363–366`): `(sorted_channel*2) % head_dim + (sorted_channel*2) // head_dim` — likely to align with RoPE/pairing.

### B. Token-level dynamic sparsity (three forms)
1. **Standalone GPT-Fast** (`models/model.py:265–294 sparse_forward()`): compute `tmp_labels = get_label_tensor(q)`, do `label_scores = tmp_labels @ kv_cache.k_label[..., :ctx].T`, `torch.topk(label_scores, heavy_const, dim=-1)` → indices into past, then `fwd_sparse_no_mask()` Triton kernel.
2. **HF adapter** (`evaluation/modify_llama.py:83–273 LlamaAttention_heavy_hitter.forward`): for layers ≥2, extract heavy channels from Q,K, compute grouped scores, sort + take top-k indices, mask out others with `-inf`, then standard softmax.
3. **Offloading** (`offloading/model.py:82–148 KVCache`): full K,V CPU-pinned `[B,H,max_seq,D]`; compressed K,V GPU `[B,H,heavy_const,D]`; label cache GPU `[B,max_seq,H,heavy_channel_num]`. On decode → label scores → top-k → DGL `gather_pinned_tensor_rows()` async fetch → forward.

### C. Label cache
- Buffer `k_label` of shape `[B, S, H, heavy_channel_num]`.
- Populated at every token during prefill (`models/model.py:245–246` `get_label_tensor(k.view(...), sorted_channel, tmp_labels, heavy_channel_num)`).

### D. Offloading
- DGL `gather_pinned_tensor_rows()` (line 138–142 in `offloading/model.py`) — may be sync in practice; question open.

## 5. Configuration Surfaces
| Param | Default | Where | Meaning |
|---|---|---|---|
| `heavy_const` | 256 | `--heavy_const` | top-k past tokens to attend |
| `heavy_channel_num` | head_dim/group_factor (e.g. 32) | derived | important channels per head |
| `group_factor` | 1 standalone / 2 eval | `--group_factor` | channel-num divisor |
| `label_bits` | 16 | `--q_bits` | label-cache quant bits (16=no quant) |
| `channel` | "k"/"q"/"qk" | `--channel` | which proj to apply config to |
| `block_size` (ctx) | 16384 | ModelArgs | max position embeddings |
- Config path inferred from model id: `config/{owner}/{name}.json`.

## 6. Custom Kernels (Triton only, no CUDA)
- `models/triton_kernels/channel.py: get_label_tensor()` — gather heavy-channel values from K per (token, head).
- `models/triton_kernels/sparse.py: fwd_sparse()`, `fwd_sparse_no_mask()` — load Q once, gather K/V at Heavy_List indices, compute sparse attn.
- `bgemv.py`, `bgemv_int8.py`, `argsort.py`, `quantize.py`, `sparq.py`, `heavy.py` — auxiliary / legacy / alternative methods.
- `attention.py: att()` — wrapper for benchmark.

## 7. Tests / Benchmarks / Reproducibility
- Benchmark: `scripts/run_attn.sh` over batch∈{1,4,8,16,32}, ctx∈{2048,4096,8192,16384}; 1000 warmup + 1000 measure.
- Eval: WikiText-2 perplexity, MMLU, line-retrieval (LONGEVAL), AIME24/IfEval/LCB, LongBench 13 tasks.
- Pinned deps `requirements.txt`: torch==2.5.1, transformers==4.47.1, triton==3.1.0, datasets==3.2.0, DGL w/ cu121, xformers==0.0.28.post3. CUDA 12.1.
- Pre-computed configs committed for 13+ models.

## 8. Hardware Assumptions
- CUDA 12.1 (nvidia-cuda-runtime-cu12==12.4.127).
- Tested on NVIDIA (likely 4090/A100/H100, not documented).
- DGL needed only for offloading variant.

## 9. Important Files & Symbols
- `models/model.py` — `Transformer`, `Attention.sparse_forward()` (265–294), `KVCache` (71–96), `init_model_channel_config()`, `permute_channel_config()` (348–366).
- `config/offline_calibration.py` — `get_calib_feat()`, `get_qk_hook()` (91–93).
- `models/triton_kernels/sparse.py:107–152` — `fwd_sparse_no_mask_kernel`.
- `models/triton_kernels/channel.py:10–30` — `get_label_tensor_kernel`.
- `evaluation/modify_llama.py` — `LlamaAttention_heavy_hitter`, `convert_kvcache_llama_heavy_recent`, `convert_llama_channel_config` (277–307).
- `evaluation/offload_llama.py`, `offloading/model.py:82–148`.
- `config/meta-llama/Llama-2-7b-hf.json` (example).

## 10. Open Questions
1. Why Method 1 (mean |Q·K|) over Methods 3–5 (commented out)?
2. Group-factor semantics — quant grouping or pure tuning knob?
3. Label quantization (`label_bits`) untested.
4. Offloading: truly async or blocking?
5. Architecture coverage uneven — Mixtral/Vision partial.
6. No training/fine-tuning code; pure inference.
7. Comparison vs H2O/SparQ/Streaming — stubs present in `evaluation/`.
