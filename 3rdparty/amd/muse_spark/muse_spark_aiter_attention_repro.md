# Muse Spark：AITER attention 的 page-size 16 路徑

這份文件重建並驗證三種配置：

- **Plan A**：AITER CK paged prefill + Triton decode
- **Plan B**：AITER CK paged prefill + AITER unified-attention 3D decode，
  Muse 的 46 個 SWA layer 與 16 個 global layer 都走 3D
- **Plan C**：Plan B 的 page16 CK prefill 與 global-layer 3D decode 不變，
  46 個 SWA decode 改走 FlyDSL split-K paged GQA（`mfma_pv` + `parallel` reduce）

## 補丁結構

`01` 是 A/B/C 都必須套的 segmentation fault 修復。`02` 和 `03` 是互斥的
路徑選擇。Plan C 疊在 Plan B 上面，再套 FlyDSL 兩個補丁。`04` 與 attention
路徑無關：乾淨 aiter `c16d44b9` 要跑 `MUSE_MOE_BACKEND=flydsl` 時必須套。

```text
patches/01-aiter-fix-page-scatter-vm-fault.patch
  target: aiter（內含 composable_kernel submodule 路徑）
  三條路徑都必要。修掉 gfx950 上 CK batch-prefill 的 page gather VM fault。

patches/02-plan-a-aiter-prefill-triton-decode.patch
  target: sglang
  Plan A。page16 paged prefill metadata + decode 自動路由到 Triton。

patches/03-plan-b-aiter-unified-attention-3d.patch
  target: sglang 與 aiter（同一個檔案，分兩次套）
  Plan B。page16 paged prefill metadata + SWA decode 改走 3D split-K kernel。
  Plan C 也需要這份（CK prefill + global layer 3D）。

patches/experimental/aiter-muse-flydsl-swa-decode.patch
  target: aiter
  Plan C。FlyDSL SWA paged GQA decode kernel。

patches/experimental/sglang-muse-flydsl-swa-dispatch.patch
  target: sglang
  Plan C。aiter_backend 在 Muse SWA decode 改呼叫 FlyDSL。

patches/04-aiter-muse-moe-flydsl.patch
  target: aiter
  FlyDSL MoE expert kernels（`muse_moe_gemm.py`、`muse_moe_finalize.py`）。
  overlay wrapper 會 import 這兩個模組；乾淨 image / aiter c16d44b9 沒有它們。
  已含 MXFP4 decode-pipeline retune，不要再疊
  `patches/experimental/aiter-muse-moe-flydsl-decode-pipeline.patch`。
```

`02` 和 `03` 各自包含同一份 page16 prefill metadata 修改，所以兩者都是自
給自足的，但不能同時套用。切換路徑時先 `git apply --reverse` 移除另一個。
Plan C 與 Plan A 互斥；與 Plan B 相容（B 的 SWA 3D 被 FlyDSL dispatch 蓋掉）。

每個補丁檔開頭都有一段說明，記錄 target repo 與套用指令；`git apply` 會
忽略 diff 之前的文字。

## 已驗證版本

以下 revisions 在 MI355X / gfx950、BF16（Plan A/B smoke）與 MXFP4
（Plan C profile + 官方 IL/OL）上驗證：

```text
Muse overlay  aa9e96e1999299beb855a0c3c379ee8482d9d256
SGLang        dc2843801d02f9130feff04635803353561eb604
AITER         c16d44b93a528b2a4bfd6d8d3409116d465872a9
CK submodule  f33252cebe5a52362ec1ee12c124dde7800dda3a
```

模型：

```text
/data/Muse-Spark-1.2-820B-A42B
num_hidden_layers=62
global_attn_cfg=[2048, 2048, 2048, 0]
```

換較新的 upstream revision 時，先對每個補丁跑 `git apply --check`；context
不符時不要硬套。

## 為什麼 01 不能省略

`fmha_batch_prefill` 有三種 KV load 分支。`page_block_size < kN0` 時（Muse
是 `16 < 128`）會走 Case 3 的 `BUFFER_LOAD`：整個 K/V pool 只用一個 SRD，
page 身分編在 32-bit voffset 裡。在 gfx950 上，即使每個位址都在界內，
scatter 到不連續 page id 的 gather 仍會觸發 GPU VM fault，對外通常表現成
worker 消失或 segmentation fault。upstream #10180 已經因為同樣原因把
`page_block_size == 1` 固定改用 `GLOBAL_LOAD_LDS`，而 #9214 的
`lo32 + pool_size` 檢查抓不到 scatter 這種情況。

補丁 `01` 把整個 Case 3 分支固定成 `GLOBAL_LOAD_LDS`，只留下 per-page
rebased 的 `BUFFER_LOAD`（`page_block_size >= kN0`）不動。代價大約是 9%
prefill throughput，但這是目前唯一已知能承受 prefix cache 與高並發的組合。

## 乾淨 container 安裝

已驗證的基準 image 是 `rocm/sgl-dev:v0.5.19-rocm720-mi35x-20260905`。
container 需要八張 MI355、可存取模型路徑，並以 `/dev/kfd`、`/dev/dri`、
`--ipc=host` 啟動。以下假設 checkout 在 `/sgl-workspace`，overlay 在
`/home/yichiche/sglang_muse-spark-rocm`。

```bash
cd /sgl-workspace

# git clone https://github.com/sgl-project/sglang.git
# git -C sglang checkout dc2843801d02f9130feff04635803353561eb604

# git clone --recursive https://github.com/ROCm/aiter.git
# git -C aiter checkout c16d44b93a528b2a4bfd6d8d3409116d465872a9
# git -C aiter submodule update --init --recursive

cd /home/yichiche/sglang_muse-spark-rocm
SGLANG_ROOT=/sgl-workspace/sglang \
AITER_ROOT=/sgl-workspace/aiter \
  scripts/setup_aiter_attention_repro.sh
```

setup script 只做與路徑無關的事：idempotently 套 `01` 與 `04`、刪掉受 CK
header 影響的兩個 generated batch-prefill JIT module、editable-install
SGLang / AITER / overlay。第一次啟動 server 會重新編譯 CK batch-prefill，
比之後慢。`04` 是 Python kernel 源碼，不需要重編 JIT。

## 套用 Plan A

```bash
git -C /sgl-workspace/sglang apply \
  /home/yichiche/sglang_muse-spark-rocm/patches/02-plan-a-aiter-prefill-triton-decode.patch
```

不需要重新編譯，改動都是 Python。

## 套用 Plan B

同一個檔案套兩次，用 `--include` 分開兩個 repo：

```bash
PATCH=/home/yichiche/sglang_muse-spark-rocm/patches/03-plan-b-aiter-unified-attention-3d.patch
git -C /sgl-workspace/sglang apply --include='python/*' "$PATCH"
git -C /sgl-workspace/aiter  apply --include='aiter/*'  "$PATCH"
```

## 套用 FlyDSL MoE（`MUSE_MOE_BACKEND=flydsl`）

與 Plan A/B/C attention 正交。setup script 已 idempotently 套 `04`；若只補
kernel、不重跑 setup：

```bash
git -C /sgl-workspace/aiter apply \
  /home/yichiche/sglang_muse-spark-rocm/patches/04-aiter-muse-moe-flydsl.patch
```

然後 `MUSE_MOE_BACKEND=flydsl` 啟動。不要再套
`patches/experimental/aiter-muse-moe-flydsl-decode-pipeline.patch`。

## 套用 Plan C

先套 Plan B（`01` + `03`），再套 FlyDSL：

```bash
git -C /sgl-workspace/aiter apply \
  /home/yichiche/sglang_muse-spark-rocm/patches/experimental/aiter-muse-flydsl-swa-decode.patch
git -C /sgl-workspace/sglang apply \
  /home/yichiche/sglang_muse-spark-rocm/patches/experimental/sglang-muse-flydsl-swa-dispatch.patch
```

Serving 候選 kernel 是 `AITER_MUSE_DECODE_STAGE1=mfma_pv`、
`AITER_MUSE_DECODE_REDUCE=parallel`（128 split × 16-token tile）。
SGLang `spawn` scheduler **不會**把未登錄的 env 轉進 worker。量測時用下面
兩者之一，否則 `SGLANG_MUSE_FLYDSL_DECODE=1` 只存在於 launch_server、SWA
仍會走 unified 2d/3d：

1. 在 `sglang/srt/environ.py` 的 `Envs` 加 `SGLANG_MUSE_FLYDSL_DECODE = EnvBool(False)`，或
2. 把 `aiter_backend.py` 裡 `get_bool_env_var("SGLANG_MUSE_FLYDSL_DECODE", "0")` 的預設改成 `"1"`（本次 MXFP4 驗證用這個）。

Dispatch 仍有 shape gate：gfx950、page16、d=64、GQA-8、bf16 Q/KV、`bs<=8`、
effective window 2047、無 sinks。MXFP4 權重可以，只要 KV cache 是 bf16。

## 啟動

Plan A / B 用同一個啟動指令；差別完全在套了哪個補丁，不需要 CLI 或環境變數
去選路徑。Plan C 額外需要 FlyDSL env（或上一節的 worker 預設）。

```bash
cd /home/yichiche/sglang_muse-spark-rocm
ATTN=aiter PAGE_SIZE=16 scripts/serve_rocm.sh
```

展開後的關鍵參數：

```bash
SGLANG_USE_AITER=1 \
python -m sglang_meta.launch_server \
  --model-path /data/Muse-Spark-1.2-820B-A42B \
  --tp-size 8 --ep-size 8 \
  --attention-backend aiter \
  --page-size 16 \
  --context-length 32768 \
  --mem-fraction-static 0.90 \
  --chat-template /data/Muse-Spark-1.2-820B-A42B/muse_spark_1_2_oss.jinja
```

Plan A 的啟動 log 每個 TP rank 都會出現這一行，代表 decode 已經自動切到
Triton：

```text
Aiter hybrid-SWA: using Triton decode attention; prefill remains on Aiter CK FMHA.
```

Plan B 不會有這一行，decode 留在 aiter。想在 Plan A 蓋掉自動路由就明確帶
`--decode-attention-backend`。

Plan C（MXFP4 驗證用 `muse_spark_serve.sh`）：

```bash
ATTN=aiter PAGE_SIZE=16 \
SGLANG_MUSE_FLYDSL_DECODE=1 \
AITER_MUSE_DECODE_STAGE1=mfma_pv \
AITER_MUSE_DECODE_REDUCE=parallel \
MODEL=/data/Muse-Spark-1.2-820B-A42B-MXFP4-amd \
  bash /home/yichiche/muse_spark_serve.sh
```

## 驗證

先做 API smoke test：

```bash
curl -sS http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"/data/Muse-Spark-1.2-820B-A42B",
    "messages":[{"role":"user","content":"Reply with exactly OK"}],
    "temperature":0,
    "max_tokens":64
  }'
```

再用長 prompt 與並發覆蓋 CK prefill、prefix cache 和 CUDA-graph decode：

```bash
CONCURRENCIES=4 INPUT_LEN=8192 OUTPUT_LEN=128 \
RESULT_DIR=/tmp/muse-attn-results TAG=plan_a_or_b \
  /path/to/run_spark_client.sh
```

判定標準：

- `Successful requests: 40`
- server process 和八個 scheduler 都存活
- log 沒有 `Memory access fault`、`Segmentation fault` 或 worker death

Plan B 可以從 trace 確認 kernel 選擇：

```bash
curl -s -X POST http://127.0.0.1:30000/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"output_dir":"/tmp/planb_prof","num_steps":4,"activities":["CPU","GPU"]}'
```

一個 decode step 應該是 62 個 `kernel_unified_attention_3d`、0 個
`kernel_unified_attention_2d`；kernel 名稱裡的 `BLOCK_SIZE_16` 表示
page-size 16 的 paging 確實傳到 kernel。

Plan C 一個 decode step 應該是：

```text
46x muse_paged_gqa_decode_stage1_mfma_flydsl
46x muse_paged_gqa_decode_stage2_parallel_flydsl
16x kernel_unified_attention_3d
16x reduce_segments
 0x kernel_unified_attention_2d
 0x _fwd_grouped_kernel_stage1 / _fwd_kernel_stage2
```

## 本次驗證結果

2026-09-07，BF16、TP8/EP8、IL 約 8K、OL 128、concurrency 4、40 requests，
兩條路徑都是同一個 container 內先後量測：

```text
              requests  median TTFT  median TPOT  output tok/s  GSM8K(200)
Plan A          40/40      494.60ms      32.90ms       105.87       0.975
Plan B          40/40      497.45ms      32.27ms       107.03       0.970
```

兩者都沒有 GPU fault 或 worker death，GSM8K invalid 皆為 0。200 題的
0.975 / 0.970 差一題，屬於取樣噪音範圍。

Plan B TP0 一個 decode step 的 trace：

```text
62x kernel_unified_attention_3d ... BLOCK_SIZE_16 ... ALL_DECODE_1
 0x kernel_unified_attention_2d
```

單層 BF16、q_len=1、GQA=8、head_dim=64、page16、window=2047 的 3D 輸出
也和 AITER reference 對過：

```text
max_abs_error  0.00048828125
mean_abs_error 0.0000297071
allclose(atol=1.5e-2, rtol=1e-2) = True
```

這是 smoke / stability 與 kernel-routing 驗證，不等於完整 accuracy
qualification。進 production 前仍應跑完整 1319 題 few-shot GSM8K。

### Plan C（MXFP4，2026-09-08）

Container `jacky-v0.5.19-rocm720-mi35x-muse-spark-day0`，MXFP4、page16、
TP8/EP8、conc=4。Profile 用 `run_spark_client.sh`（`--profile`、8 prompts），
再對 DECODE 層與兩個 baseline xlsx 比對；確認 SWA 快過兩者後才跑官方
IL8192/OL1024（40 prompts、無 profiler）。

Profile traces：`/home/yichiche/profile/profile_flydsl_c4/`
Baseline：`profile/profile_aiter_c4_unified3d/decode_layers.xlsx`、
`profile/profile_triton/decode_layers.xlsx`。

SWA 單層 attention（µs，stage1 + stage2 / 3d+reduce）：

```text
                SWA attn/layer    full attn/layer
Plan C FlyDSL   5.92 + 4.21=10.13  7.40 + 4.20=11.60（仍 3d）
Plan B unified3d 7.6 + 4.1 =11.7   7.8 + 4.3 =12.1
Triton decode    9.7 + 6.2 =15.9   31.6 + 6.1 =37.7
```

SWA 相對 Plan B 約 −14%、相對 Triton 約 −36%。Trace 無 `unified_2d`、
無 Triton grouped decode。

官方 IL8192/OL1024 conc=4（40/40）：

```text
JSON /home/yichiche/dbg/official_ilol/flydsl_mxfp4_page16_il8192_ol1024_conc4_mi355x-amd_20260907_162200.json
              median E2E  total tok/s  mean TPOT  output tok/s
Plan C          16.97s         1923      17.84ms        213.56
Plan B (3d)     16.96s         1919      17.92ms        213
Plan A hybrid   17.70s         1833      18.78ms        204
```

E2E 與 Plan B 在噪音內（attention 約只佔 decode GPU 3%）。尚未跑 GSM8K，
也尚未 sweep conc 8/16/32。

## 切換路徑

```bash
# Plan A -> Plan B
git -C /sgl-workspace/sglang apply --reverse patches/02-plan-a-aiter-prefill-triton-decode.patch
git -C /sgl-workspace/sglang apply --include='python/*' patches/03-plan-b-aiter-unified-attention-3d.patch
git -C /sgl-workspace/aiter  apply --include='aiter/*'  patches/03-plan-b-aiter-unified-attention-3d.patch
```

反向同理。`01` 在 A/B/C 下都保持套用，所以不需要重新編譯 CK。

Plan C 拿掉 FlyDSL 時 reverse 兩個 `experimental/*flydsl*` 補丁，回到 Plan B。

## 已知限制

- 不要把 CK selector 改回 Case-3 `BUFFER_LOAD` 當 serving 預設；短 repro
  可能通過，但 prefix cache 與高 page id 仍可能 fault。
- Plan B 讓 SWA decode 走 3D 是針對 gfx950 上 Muse 的 shape 驗證的。補丁
  無條件改寫 `use_2d_kernel`，所以套了它的 AITER checkout 不該同時服務其他
  未驗證 shape 的模型。
- Plan A 的自動路由只在 hybrid-SWA 模型、prefill 是 aiter、且沒有明確指定
  `--decode-attention-backend` 時生效。
- Plan A/B 的 BF16 smoke（OL128、gsm8k200）已驗證。Plan C 的官方 IL/OL
  與 decode profile 是 **MXFP4**；完整 1319 題 GSM8K 還沒跑。
- Plan C 只蓋 SWA decode（window 2047、`bs<=8`）。global layer 仍是 3D。
- FlyDSL 要 bf16 Q/KV cache；MXFP4 權重可以。
- `SGLANG_MUSE_FLYDSL_DECODE` 必須進 scheduler process，見「套用 Plan C」。
- `patches/experimental/aiter-page1-triton-decode.patch` 仍是未採用路徑，
  不屬於這份復現流程。
