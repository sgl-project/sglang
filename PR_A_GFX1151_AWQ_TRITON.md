# PR A：gfx1151 AWQ Triton MoE 路徑

## 範圍

本變更為 `Qwen3.5-35B-A3B-AWQ-4bit` 建立第一個 Strix Halo/gfx1151 tuning profile。目標是現有的 ROCm Triton MoE 路徑；不新增 FlashInfer backend，也不修改 GTT-aware memory planner。

驗證平台：

- GPU：AMD Radeon 8060S Graphics（`gfx1151`）
- Runtime：使用 ROCm Triton attention/MoE 的 SGLang
- Model：`Qwen3.5-35B-A3B-AWQ-4bit`
- Tensor parallelism：1
- Weight format：compressed-tensors W4A16，group size 32

## 變更內容

### Packed W4A16 config 命名

Runtime config key 是由 packed `w2` shape 推導。W4A16 中每個 byte 包含兩個 4-bit weight，因此 packed intermediate dimension 是 `shard_intermediate_size // 4`。`benchmark/kernels/fused_moe_triton/common_utils.py` 現在會使用這個 packed dimension 產生 `int4_w4a16` 檔名。

沒有這項修正時，tuning 輸出的檔名不會符合 runtime 實際查找的檔名。

### Radeon 8060S configs

Tuned maps 儲存在目前 Triton version directory：

```text
python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_0/E=256,N=256,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json
python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_0/E=256,N=256,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16_down.json
```

Up 與 down 檔案目前包含相同的已驗證 per-batch map。第一版尚未在 down path 啟用 TMA；獨立的 down/TMA tuning 會在後續處理。

### 可重現性說明

本次 tuning 使用 72 組代表性的 ROCm configurations，以及以下 batch sizes：

```text
1 2 4 8 16 32 64 128
```

完整的標準 ROCm search space 更大。因此這份 map 是針對特定 device/runtime 的 tuning 結果，不代表適用於所有 Triton 或 ROCm versions。

Benchmark container 需要：

```text
ray==2.51.1
```

單 GPU Strix Halo 使用 `--no-ray` local worker；目前 script 仍會 import Ray module，因此 container 仍需安裝 Ray package，但不會啟動 Ray cluster 或 Ray worker。

範例 tuning command：

```bash
SGLANG_AWQ_MOE_TRITON_ROCM=1 \
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
  --no-ray \
  --model /models/Qwen3.5-35B-A3B-AWQ-4bit \
  --tp-size 1 \
  --dtype int4_w4a16 \
  --batch-sizes 1 2 4 8 16 32 64 128 \
  --tune
```

## Runtime 驗證

Server 使用 `1027/server.sh` 中的 Strix Halo profile 啟動：

```text
--attention-backend triton
--mem-fraction-static 0.55
--context-length 2048
--max-total-tokens 4096
--max-mamba-cache-size 32
--max-running-requests 1
--disable-cuda-graph
```

Startup log 確認兩個檔案都被選用，而不是使用 heuristic default：

```text
Using MoE kernel config from ...dtype=int4_w4a16.json.
Using MoE kernel config from ...dtype=int4_w4a16_down.json.
```

Model 載入結果：

```text
CompressedTensorsWNA16TritonMoE (ROCm)
weights: 23.20 GiB
Mamba cache: 1.93 GiB
KV cache: 4096 tokens
```

Health request 回傳 HTTP 200；關閉 thinking 的最小 chat request 回傳 `TUNED_OK`。

## Benchmark 結果

使用相同 container、model、Triton version、synthetic MoE benchmark 與 no-Ray local worker，比較 tuned config 和 runtime heuristic。最終 map 包含全部八個測試 batch size，沒有任何 batch 使用 nearest-neighbor fallback。

| Tokens | Default heuristic（us） | gfx1151 tuned（us） | Improvement |
| ---: | ---: | ---: | ---: |
| 1 | 305.13 | 180.30 | 40.91% |
| 2 | 530.36 | 397.90 | 24.98% |
| 4 | 987.78 | 749.89 | 24.08% |
| 8 | 1732.48 | 1332.19 | 23.11% |
| 16 | 2927.57 | 2240.29 | 23.48% |
| 32 | 4591.12 | 3515.53 | 23.43% |
| 64 | 6176.29 | 4473.50 | 27.57% |
| 128 | 6998.44 | 4962.49 | 29.09% |

Kernel time 的 geometric-mean improvement 是 27.32%。同一個 model 與 tuned map 仍需要使用最終 configs 執行 request-level benchmark，才能宣稱 end-to-end serving 效能提升；同時也必須用相同 prompts 驗證輸出正確性。

最終 tuning 使用：

```text
--no-ray
--batch-sizes 1 2 4 8 16 32 64 128
```

Local worker 避免在 single-GPU Strix Halo 上啟動 Ray 與 metrics-agent。修正後的 W4A16 `BLOCK_SIZE_K` filter 會接受 quantization group size 的倍數，因此包含 default 的 K=64 candidate。

## Benchmark 要求

在提出 upstream performance claim 前，還需要記錄：

- request-level decode latency 或 tokens/second；
- 代表性 prompt lengths 的 prefill latency/throughput；
- peak memory，以及是否出現 fallback/default-config warning；
- 相同 prompts 的輸出正確性。

## 已知的獨立問題

同一個 runtime 曾嘗試 JIT build gfx1151 KV-cache kernel，但因 ROCm headers 沒有定義以下 identifiers 而失敗：

```text
cudaDevAttrMaxSharedMemoryPerBlockOptin
cudaDevAttrMaxSharedMemoryPerBlock
```

SGLang 之後繼續使用 fallback path。這不屬於 PR A 的 AWQ MoE config 範圍，應另開 kernel compatibility PR 處理。

---

# GitHub PR draft (English)

## PR title

```text
[AMD][gfx1151] Add Triton configs for Qwen3.5 AWQ MoE
```

## PR description

### Summary

This PR adds a first AMD Strix Halo/gfx1151 tuning profile for `Qwen3.5-35B-A3B-AWQ-4bit` on the existing ROCm Triton MoE path.

It also fixes the config filename generated for packed W4A16 weights and adds a single-GPU `--no-ray` mode to the Triton MoE tuning benchmark.

This PR does not add a FlashInfer backend and does not change the GTT-aware memory planner.

### Motivation

Strix Halo uses gfx1151 and does not share the same wave-size and kernel assumptions as AMD CDNA GPUs. The Qwen3.5 AWQ MoE model can run through the ROCm Triton path, but the runtime previously fell back to the heuristic MoE config because no Radeon 8060S-specific config was available.

The tuning benchmark also generated an incorrect `N` component for packed W4A16 config filenames, so the generated file did not match the filename requested by the runtime.

### What changed

- Use the packed W4A16 intermediate dimension when generating `int4_w4a16` config filenames.
- Add Radeon 8060S/gfx1151 configs for both the up and down MoE projections.
- Add `--no-ray` for sequential single-GPU tuning.
- Fix the W4A16 `BLOCK_SIZE_K` filter to accept kernel K blocks that are multiples of the quantization group size.
- Document the Strix Halo tuning and validation procedure.

### Added configs

```text
python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_0/E=256,N=256,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json
python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_0/E=256,N=256,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16_down.json
```

The configs cover token counts `1, 2, 4, 8, 16, 32, 64, 128`. The down-projection config currently uses the same validated map as the up projection and does not enable TMA.

### Validation

Environment:

- AMD Radeon 8060S Graphics (`gfx1151`)
- ROCm Triton backend
- SGLang
- `Qwen3.5-35B-A3B-AWQ-4bit`
- TP1
- compressed-tensors W4A16, group size 32

The server loaded `CompressedTensorsWNA16TritonMoE (ROCm)` successfully:

```text
weights: 23.20 GiB
Mamba cache: 1.93 GiB
KV cache: 4096 tokens
```

The startup log confirmed that both gfx1151 config files were selected instead of the heuristic default. The health endpoint returned HTTP 200 and a minimal chat request returned `TUNED_OK`.

### Kernel benchmark

The same model shape, container, Triton version, and no-Ray local worker were used for default-versus-tuned measurements:

| Tokens | Default (us) | gfx1151 tuned (us) | Improvement |
| ---: | ---: | ---: | ---: |
| 1 | 305.13 | 180.30 | 40.91% |
| 2 | 530.36 | 397.90 | 24.98% |
| 4 | 987.78 | 749.89 | 24.08% |
| 8 | 1732.48 | 1332.19 | 23.11% |
| 16 | 2927.57 | 2240.29 | 23.48% |
| 32 | 4591.12 | 3515.53 | 23.43% |
| 64 | 6176.29 | 4473.50 | 27.57% |
| 128 | 6998.44 | 4962.49 | 29.09% |

The geometric-mean fused-MoE kernel-time improvement is 27.32% on the validation machine.

### Reproduction

```bash
SGLANG_AWQ_MOE_TRITON_ROCM=1 \
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
  --no-ray \
  --model /models/Qwen3.5-35B-A3B-AWQ-4bit \
  --tp-size 1 \
  --dtype int4_w4a16 \
  --batch-sizes 1 2 4 8 16 32 64 128 \
  --tune
```

### Limitations

- The benchmark measures fused-MoE kernel time, not end-to-end serving throughput.
- The down path reuses the tuned map and does not enable TMA yet.
- The configs are specific to Radeon 8060S/gfx1151, ROCm, and the active Triton version.
- A gfx1151 KV-cache JIT issue remains outside this PR's scope; SGLang uses its fallback path.

### Testing

- [x] Python AST and JSON validation
- [x] `git diff --check`
- [x] Qwen3.5 AWQ model load on gfx1151
- [x] Health and minimal generation checks
- [x] Default-versus-tuned fused-MoE benchmark
- [ ] Independent down/TMA tuning
- [ ] End-to-end serving benchmark
