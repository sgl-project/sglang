# MoE down-proj LoRA gemm A/B overlap onto finalizeKernel

Branch: `moe-down-lora-overlap-finalize` (worktree `/Users/yushengsu/Downloads/river/wt-moe-down-lora-overlap-finalize`, base `lora-opti` @ `867f2ca413`)

## 任務 (2026-06-04 12:17)

從 perfetto trace（TP-1/EP-1 DECODE bs=64，FP8 deep_gemm 路徑）看到：main stream 上
`moe::dev::finalize::finalizeKernel` 之後才跑 down-proj 的 `_moe_lora_shrink` /
`_moe_lora_expand_add`（+routing 小 kernel），完全串行。

目標：把 down lora gemm A（shrink）/ gemm B（expand）重疊到「base down gemm（GEMM2）結束之後」，
實際上與 finalizeKernel（的一部分）並行。

用戶指示：
- 用 cuda event 技術，類比現有源碼裡對 down gemm 做 overlap 的做法
- 做完直接發 PR，先不驗證
- 功能加 env var gate（簡單起見）

## 現況調查 (2026-06-04 12:20)

- 現有 overlap 技術（gate_up，O1）：`python/sglang/srt/lora/trtllm_moe/moe_overlap.py`
  - side stream 跑 gate_up lora → `lora_event.record()`；event handle (`.cuda_event` int) 傳進
    trtllm op，op 內部在 activation kernel 之前 `cudaStreamWaitEvent`（FP8: runner.cu:760；
    FP4: kernel_launcher.cu:2651）→ permute+GEMM1 與 side-stream lora 並行。
  - cuda-graph capture 安全：`_LORA_OVERLAP_EVENTS` 保活 event。
- down lora 數據依賴：
  - shrink 讀 `activation_lora_input`（op 內 activation kernel 寫，GEMM2 之前就 ready）
  - expand（`_moe_lora_expand_add`，`FUSE_SUM_ALL_REDUCE=True`）`tl.atomic_add` 直接加進
    `output` —— 而 finalizeKernel **寫** 同一個 output → expand 不能跟 finalize 並行地寫 output
- ⚠️ 歷史教訓（moe_overlap.py FP4 註釋）：之前的 down-overlap（`act_ready_event`，從 activation
  之後就 fork）被移除：net-neutral-to-negative + 重負載下 cuda-graph replay state corruption。
  本次方案不同：fork 點在 GEMM2 之後（更保守），且最終 add 在 main stream 上做。
- 現成基建：lora op 支援 `do_finalize=False`；另有 fused lora finalize kernel
  （`sgl_trtllm_fp8_block_scale_moe_lora_finalize`）但它讀 down_lora_delta → 用它會串行化，不採用。

## 設計 (2026-06-04 12:25)

Env gate：`SGLANG_OPT_LORA_DOWN_FINALIZE_OVERLAP = EnvBool(False)`（environ.py，OPT_ 前綴 per
env-var-conventions skill）。

1. **C++ event 反向 plumbing**（類比 lora_ready_event，方向相反：op 內 record、Python 側 wait）：
   - `runner.h` MoERunnerArgs：加 `void* gemm2_done_event = nullptr;`
   - FP8 `Runner::run`（trtllm_fused_moe_runner.cu）：`mGemm2.run(...)` 之後、finalize 之前
     `cudaEventRecord(gemm2_done_event, stream)`
   - FP4 `FP4BlockScaleLoraLauncher`（kernel_launcher.cu）：加 `gemm2_done_event_` 成員；
     down GEMM（step 8）之後 record
   - 兩個 lora entry（fp8/fp4）+ FFI 簽名 + core.py wrapper 加 `gemm2_done_event: int = 0`
2. **Python（moe_overlap.py 兩個 two-stream 路徑）**，env on 時：
   - op 前：建 `gemm2_done_event`（先 record 一次以 materialize cudaEvent，handle 才有效）、
     allocate `down_delta = empty([num_tokens, hidden])`
   - op call 帶 `gemm2_done_event=handle`（op 內 GEMM2 後 record，然後 main stream 排 finalize）
   - op 後：`side_stream.wait_event(gemm2_done_event)` → side stream 上 `down_delta.zero_()` +
     down lora routing/shrink/expand（output=down_delta，atomic 加進零 buffer）→ record
     `down_done_event`
   - main stream `wait_event(down_done_event)`（此時 finalize 已排在前）→ `output += down_delta`
   - capture 時兩個 event 都進 `_LORA_OVERLAP_EVENTS` 保活
   - env off：行為與現在 byte-identical（serial expand-add 直接 atomic 進 output）
3. 數值差異：lora delta 先 atomic 加進零的 bf16 delta 再加回 output，比原 path 多一次 bf16
   rounding；可接受，PR 註明。

預期收益：routing+shrink+expand（trace 上 ~finalize 同量級的一段）隱藏進 finalizeKernel；
代價是多一個 [tokens, hidden] 清零 + 一個 elementwise add（decode bs=64 時皆 ~µs 級）。

## 實作步驟

- [x] Step 1: env var + C++ plumbing + Python overlap (FP8 + FP4)
- [x] Step 2: commit + push + 開 PR（per 用戶指示先不跑 regression/bench）
  - commit `4883cd71b5`，branch push 到 origin（yushengsu-thu fork；jybsuper 無 push 權限）
  - PR: https://github.com/jybsuper/sglang/pull/26 (2026-06-04 12:45)
- [x] Step 3: 正確性驗證（進行中，見下）

## 驗證計畫 (2026-06-04 13:29)

測試矩陣（已跟用戶確認，2026-06-04 14:05 + 15:0x 再次確認 FP4/Kimi 先不測）：
**只測 Qwen3.5-35B-A3B-FP8**（tp4/ep4，單 node 4 GPU）。
Kimi 不開 env var → 走原路徑，正確性/perf 不受影響（default off；env off 時 Python 走原 serial
分支、C++ 收 handle=0 不 record，唯一差異是 csrc hash 變了會重編一次 JIT）。

A/B 設計（同 commit `58ba52bcfe`，隔離 feature 本身）：
- base：LoRA on + `--moe-runner-backend sgl_flashinfer_trtllm` + `SGLANG_LORA_TWO_STREAM=1`
- variant：同上 + `SGLANG_OPT_LORA_DOWN_FINALIZE_OVERLAP=1`
- acc：per-token logprobs over compare_sample_train_data.pt，預期數值等價（ACC_TOL=0.01；
  已知 lora delta 多一次 bf16 rounding，應在 atomic-add noise floor 內）
- bench：bench_one_batch_server bs 16/32/64，2048/2048；**記 e2e + server.log decode thpt**
  （per DECODE-THPT-RULE，run script 每個 cell 保存 /tmp/server.log）

執行：
- ID=`yushengsu-20260604-1329`，pod `sglang-qwen35-yushengsu-20260604-1329`（13:29 apply，
  node np-67167b3f-7）
- bundle `58ba52bcfe`（mb `f6d0beaca8e3` vs sgl/main）已建好待注入
- RUN_ROOT=`~/Downloads/sglang_regression_yushengsu-20260604-1329`
- driver：`river/moe-down-lora-overlap-finalize/run_qwen35.sh`

## 驗證 debug 記錄 (2026-06-04 14:05)

1. **第一次 driver run 失敗**：checkout 後靜默 exit 1，把背景 `kubectl exec`（server 前台）連帶
   殺掉（server 死在 DeepGEMM warmup 80%，無錯誤——是被殺的，不是 crash）。兩個潛在因素都修了：
   - `checkout()` 的 `pip install -e python` 因 `bash -lc` 非互動 shell 沒 source `~/.cargo/env`
     → 找不到 rustc 而失敗（舊版不檢查 rc，靜默吞掉）。修：source cargo env + pip 失敗即 exit。
   - `wait_ready` 原本是單一條 30 分鐘的 in-pod kubectl exec，一次瞬斷就觸發 `set -e` 全滅。
     修：改本地循環、每次探測獨立短 kubectl exec。
2. **第二次 run（-x）真錯誤現形**：server 啟動 crash —
   `flashinfer-jit-cache (0.6.12+cu130) != flashinfer (0.6.11.post1)`。
   原因：branch pyproject pin `flashinfer_python==0.6.11.post1`，checkout 的 pip install 把
   flashinfer 降版，image 的 jit-cache 還是 0.6.12。
   修：pod 裝 `flashinfer-jit-cache==0.6.11.post1+cu130`（flashinfer.ai/whl/cu130）。
   （jit-cache 不在 pyproject deps，後續 cell 的 pip install 不會再動它。）
3. 14:05 第三次 driver run 啟動。
4. **第三/四次 run：acc OOM**（兩次，同點位）：logprob capture 是單一 ~30k-token seq，
   `logits_processor._copy_logits_to_buffer` 的 `logits[:,:vocab].float()` 按單次 extend 的
   token 數配 float32 logits（~248k vocab → 29.6 GiB），mem-fraction 0.75 後 free 26.7 GiB
   仍不夠 → server SIGQUIT → acc/bench connection refused。
   （旁證：另一 session 的 prezero regression 同設定也反覆撞 server 死，progress 顯示 acc
   重試三次才過——本質是 26.7 vs 29.6 的邊緣狀態。）
   修：`PREFILL_ARGS` 改 `--max-prefill-tokens 8192 --chunked-prefill-size 8192`，
   峰值降到 ~8 GB；chunked prefill 對 per-token logprob 數值精確，兩 cell 同設定 A/B 公平。
   另：driver 改 `nohup+disown` 脫離（背景 task 兩度被外部殺，macOS 無 setsid）。
5. 15:2x 第五次 driver run 啟動（pid 94151）。

## 設計改版 V2：只重疊 gemm A（shrink） (2026-06-04 15:40)

用戶指示（15:35）：改成只重疊 finalize 與 gemm-A，**不重疊 gemm-B**，保險起見
（避免 finalize 和 gemm-B 同時寫 output、finalize 覆蓋 lora 貢獻的風險類）。

V1（delta buffer + post-finalize add）已經避開了寫衝突，但 V2 更保守且更乾淨：
- side stream 在 gemm2_done 之後只跑 **routing prep + shrink（gemm A）**，寫進
  main-stream 預分配的 `down_intermediate`（consumer-stream alloc，遵守
  OVERLAP_MAIN_ALLOC 教訓）；shrink 階段順便 pre-warm routing-B cache（這些 routing
  小 kernel 也一起藏進 finalize）
- main stream 在 op 之後（= finalize 之後）wait shrink_done → **expand-add（gemm B）
  照原樣在 main stream 上 atomic 進 output** —— 與 serial 路徑同 kernel 同 buffer，
  **數值 bitwise 一致**（V1 有一次額外 bf16 rounding，V2 沒有）
- 移除 down_delta buffer / zero / `output += delta`，省 2 個 kernel
- C++（gemm2_done_event plumbing）不變

實作：`virtual_experts.py` 的 `_merged_experts_fused_moe_lora_add_impl` 加
`stage`（"all"/"shrink"/"expand"）+ `intermediate_buffer` 參數——"all" 預設路徑行為
不變（gate_up 等其他 caller 不受影響）；moe_overlap.py 兩路徑改用兩段式呼叫。
舊版 V1 驗證 run 已停掉，將以 V2 commit 重新驗證。

## V2 驗證結果：PASS (2026-06-04 15:47–16:00)

V2 commit `eac0e96c47`，pod `sglang-qwen35-yushengsu-20260604-1329`（Qwen3.5-35B-A3B-FP8
tp4/ep4）。兩 cell 同 commit、LoRA on、`sgl_flashinfer_trtllm`、`SGLANG_LORA_TWO_STREAM=1`；
variant 多 `SGLANG_OPT_LORA_DOWN_FINALIZE_OVERLAP=1`（launch 命令 xtrace 已驗證 env 帶到）。
base/variant server 都是 warm relaunch（V2 未改 C++，JIT cache 命中 → READY ~50s）。

### ACC（per-token logprobs，31999 tokens）✅
- base vs variant：max|Δ|=3.00，mean|Δ|=0.0240，nonzero 31540/31999
- 參照 noise floor（同日 prezero 任務的「預期等價」A/B，同 workload 同機型）：
  max|Δ|=6.07，mean|Δ|=0.0248 → **我們的差異在 atomic-add noise floor 之內（略好）**
  （down expand 用 `tl.atomic_add`，加法順序非確定 → 跨 launch 本來就非 bitwise）

### PERF（bench_one_batch_server 2048/2048）✅
| bs | base lat(s) | var lat(s) | base out tok/s | var out tok/s | Δ out thpt |
|---|---|---|---|---|---|
| 16 | 11.86 | 11.67 | 2946.0 | 2984.3 | **+1.30%** |
| 32 | 13.89 | 13.63 | 5243.5 | 5352.5 | **+2.08%** |
| 64 | 17.29 | 17.02 | 9089.8 | 9229.5 | **+1.54%** |

ITL(ms)：5.43→5.36 / 6.10→5.98 / 7.04→6.93。

### Server-log decode thpt（DECODE-THPT-RULE）✅
bs64 段 gen throughput (token/s)：base ~9036–9067，variant ~9115–9189（**+~1.3%**）
→ 與 e2e bs64 +1.54% 一致，非測量假象。

結論：V2（shrink-only overlap）acc 在 noise floor 內、decode thpt +1.3~2.1%，全 bs 正向。
artifacts：`~/Downloads/sglang_regression_yushengsu-20260604-1329/qwen35/`。

## 實作完成 (2026-06-04 12:40)

改動（6 檔，+144/-31）：

1. `runner.h`：`MoERunnerArgs` 加 `void* gemm2_done_event = nullptr`（緊鄰 `lora_ready_event`）
2. `trtllm_fused_moe_runner.cu`（FP8 `Runner::run`）：`mGemm2.run(...)` 之後、finalize 之前
   `cudaEventRecord(gemm2_done_event, stream)`（nullptr = no-op；非 lora 路徑傳 0 不受影響）
3. `trtllm_fused_moe_kernel_launcher.cu`：
   - `trtllm_fp8_block_scale_moe_impl` / `sgl_trtllm_fp8_block_scale_moe_lora` 加
     `int64_t gemm2_done_event = 0` 並設進 args
   - `FP4BlockScaleLoraLauncher`：ctor + 成員 `gemm2_done_event_`；down GEMM（step 8）後 record
   - `sgl_trtllm_fp4_block_scale_moe_lora` 簽名透傳
4. `core.py`：兩個 lora wrapper 加 `gemm2_done_event: int = 0`，位置參數透傳（FFI 是位置呼叫，
   唯二 raw-module caller 都在 core.py；lora_dispatch.py 走 wrapper 預設 0 不變）
5. `environ.py`：`SGLANG_OPT_LORA_DOWN_FINALIZE_OVERLAP = EnvBool(False)`（OPT_ 前綴 per skill）
6. `moe_overlap.py`（FP8 + FP4 two-stream 路徑，對稱改動）：
   - op 前：`gemm2_done_event.record()` materialize handle；`down_delta = empty([tokens, hidden])`
     （main stream 分配 = consumer-stream alloc，遵守 SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC 教訓）
   - op 帶 `gemm2_done_event=handle`
   - `side_stream.wait_event(gemm2_done_event)` → side stream `down_delta.zero_()` +
     down lora shrink/expand（atomic 進零 delta）→ `down_done_event.record()`
   - main `wait_event(down_done_event)` → `output += down_delta`（finalize 已在 main queue 前面）
   - capture 時兩 event 進 `_LORA_OVERLAP_EVENTS` 保活；env off 走原 serial 路徑（行為不變）

正確性論證：
- expand 不能直接 atomic 進 output（finalize 同時在寫 output）→ 私有零 delta + post-finalize add
- down lora 讀 `activation_lora_input`（activation kernel 寫，在 gemm2 之前）→ gemm2_done fork 點安全
- routing cache tensors 在 side stream 上創建、side stream 上消費（gate_up 也在 side stream）
- 數值：多一次 bf16 rounding（delta 累加→再加 output），與原 atomic-直加路徑非 bitwise 相同
- JIT rebuild：gen_jit_spec 自動 hash csrc → pod 上自動重編

未驗證（per 用戶指示）：compile/regression/bench 都還沒跑。風險點：FP4 歷史上 down-overlap 出過
cuda-graph replay corruption（act_ready 版本）；本版 fork 點更晚、add 在 main stream，待驗證。
