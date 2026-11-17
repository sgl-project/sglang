# NPU Mamba HiCache state transfer 端到端测试

## 双用例结构

`run_gsm8k.sh` 启动同一服务配置，`TEST_CASE` 选择客户端：

- **`TEST_CASE=gsm8k`（默认，精度用例）**：
  `test_qwen3_5_4b_gsm8k_mamba_hicache.py`，标准 GSM8K few-shot 提示
  （无数字填充）；每题重放前驱逐，要求逐题 Host 命中和答案一致。
- **`TEST_CASE=stress`（定向 L2 回读压力用例）**：
  `test_qwen3_5_4b_host_reload_stress.py`，逐题 A0→A→E→R，题间清空缓存，
  带每题唯一数字填充与 checkpoint 格点 seed 请求，强制每题 replay 走 Host L2
  H2D 回迁。**不是标准精度用例**：填充改变了 prompt 分布，其精度不可与
  标准 GSM8K 基线比较。

## GSM8K 精度与逐题 Host 回迁用例

参考 Qwen3-8B 的 HiCache GSM8K L2 用例：

1. **轮 1（作答）**：全部题逐题作答（标准 few-shot），记录答案、正确率
   与输出 token ID。
2. **驱逐与重放**：按首轮顺序逐题执行。每题重放前发送 train 题构造的
   filler，使输入 token 总数达到设备容量；每轮默认发送 2 个 4096-token
   filler。等待异步 KV 备份稳定，检查 dropped 为 0，再发送原 token ID。
   两次重放之间需要重新驱逐，否则共享 few-shot 前缀会留在 Device。
3. **逐题检查**：每次重放均须命中 Host、引起 `pool="mamba"` H2D 增长，
   且解析出的 answer 与该题首轮相同。Host 命中的是缓存前缀，答案仍由
   模型生成；此检查不要求每题的独有 checkpoint 都从 Host 回载。

通过条件：重放正确率 ≥ 首轮；每题均命中 Host、Mamba H2D 增长且答案与
首轮相同（无有效数字答案也判失败）；dropped 为 0。输出 token 一致数仅
作诊断记录。filler 按题发送，开销约为题数乘以每题 2 次 filler 请求。
若设置 `GSM8K_ACCURACY`，重放正确率还须达到该门限（取自**同配置**历史
实测值；正式精度结论需 1319 题）。200 题用于快速回归。结果保存在
`result.json`，服务与评测日志分别为 `server.log` 和 `eval.log`。

驱逐与重放编排无法由仓库现有的单轮 `GSM8KAscendMixin` 完成，因此这里复用
仓库的 GSM8K 提示与答案解析函数，并按参考用例实现驱逐和重放检查。

```bash
export MODEL_PATH=/path/to/Qwen3.5-4B
export GSM8K_TRAIN_PATH=/path/to/gsm8k/train.jsonl
export GSM8K_TEST_PATH=/path/to/gsm8k/test.jsonl
export GSM8K_NUM_QUESTIONS=200
export GSM8K_NUM_SHOTS=8
# 有同配置历史精度时：export GSM8K_ACCURACY="$MEASURED_THRESHOLD"
bash run_gsm8k.sh /path/to/output          # TEST_CASE=gsm8k（默认）
TEST_CASE=stress bash run_gsm8k.sh /path/to/output2   # 定向 L2 回读压力
```

## 定向 L2 回读压力用例（TEST_CASE=stress）

`test_qwen3_5_4b_host_reload_stress.py` 逐题执行 A0→A→E→R，使每题重放
发生在其 Host-L2 条目最新鲜时，并强制命中 Host：

1. **A0（seed）**：先发送 `prompt[:seed_len]`（prompt 内最后一个
   checkpoint 格点深度，`seed_len = 格点 × ⌊页对齐匹配深度/格点⌋`）+ 1 token。
   原因：答题请求的叶子覆盖 prompt+output，重放的页对齐匹配会把该叶子
   split，而 split 父节点不带 Mamba 状态（mamba 为 leaf-only 数据），校验器
   拒绝后 Host 命中归零。先种一个恰好终止在格点、带 Mamba 状态的叶子，
   重放匹配即可停在该节点并通过校验。
2. **A**：发送完整 prompt（含每题唯一数字填充，`UNIQUE_PAD_TOKENS` 默认
   640），记录答案与输出 token ID。
3. **E（确定性驱逐）**：发送 filler 直到 filler 总 token ≥ 设备容量
   （每题 `ceil(MAX_TOTAL_TOKENS / FILLER_PROMPT_TOKENS)` 个，默认 2 个），
   用唯一内容填满整个 device，强制该题全部页（含共享 few-shot 前缀）驱逐。
   filler 预算按 `MAX_FILLER_REQUESTS`（每题上限）与 `GLOBAL_FILLER_CAP`
   （全场总量上限，0 为不限）两级控制，200 题不会再耗尽预算。随后轮询
   `pool="kv"` backup 计数稳定（write-through D2H 异步）并断言 dropped 为 0。
4. **R**：重发完全相同的 token ID，断言**每题** host 命中越过共享前缀边界
   （见下方断言语义）且重放正确率不低于首轮。
5. **题间隔离**：R 完成后、下一题 A0 前调用 `/flush_cache`。上一题 R
   消耗 Host 备份并留下 Device 缓存；额外 filler 无法可靠保护下一题的 seed
   叶子。清空只发生在不同题之间，不影响同一题 E→R 的真实 Host 回迁。

断言语义：每题重放要求 `host >= seed_len > boundary`，`device=0`，
且该题 `pool="mamba"` H2D 回迁计数增长。`seed_len` 是该题独有的 Mamba
checkpoint 深度，boundary 为全部 prompt 公共前缀按页对齐的深度。
"host>0"可能只是共享 few-shot 前缀命中（v8 实测第 2 题 host=1280 恰等于
共享前缀，独有 checkpoint 并未回载），不构成独有状态回载证据。共享
前缀节点也携带 checkpoint，故旧断言只验证回读路径被走过。只命中共享
前缀或某些独有 KV 页、但未到 seed checkpoint，均判失败。

在本目录执行上述命令。`run_gsm8k.sh` 引用 `env.sh`，默认使用 4 卡、
8192 个设备 token（刻意调小以制造驱逐）、8 倍 Host 容量、128 token 页、
每个 filler 4096 token；filler 预算为每题上限 `MAX_FILLER_REQUESTS`（默认 12，
stress 用例每题实际用约 2 个）加全场总量上限 `GLOBAL_FILLER_CAP`（0 为不限）。
可通过环境变量覆盖。若部署环境需要私有 `kv_exchange` 包，按下文“实测环境”
的桥接方法先设置算子注册。

旧版 `/tmp/mamba_gsm8k_5_v8` 的 5/5 Host 命中仅表示 `host>0`，不能
视为 5/5 独有状态回迁。151 上题间 flush 的 3 题试跑
`/tmp/mamba_unique_flush_3_151_20261116` 验证了 3/3：每题 `host=5376`、
`boundary=1280`、`device=0`，Mamba H2D 每题增长 8，数值答案与首轮一致。
100 题运行结果和逐题审计见
[`TEST_REPORT_2026-11-16_151_UNIQUE_HOST_100.md`](TEST_REPORT_2026-11-16_151_UNIQUE_HOST_100.md)。

`PAGE_SIZE` 默认 128，与该模型在 NPU 上的既有 Mamba E2E 配置一致。
曾将其设为 1 运行 10 题，首轮即出现乱码且正确率为 0；仅把页大小恢复
为 128 的同题单请求对照得到正确答案。小样本可能没有独有的完整缓存页，
此时用例仍要求 Host 命中和 Mamba H2D 计数增长。独有 KV 整页不保证该深度
存在可复用的 Mamba checkpoint；逐题未命中列表只作为诊断信息。比如
`mamba_track_interval=256` 时，1280 token 的共享前缀后，即使第 5 题的 prompt
达到 1417 token、跨过 1408 的 KV 页边界，也没有跨过下一个 Mamba checkpoint
边界 1536。该题重放仍可能只命中 1280 token 的共享前缀。

## 测试目的

验证 `python/sglang/srt/mem_cache/pool_host/mamba.py` 的改动：`kernel_ascend` IO 后端下，Mamba 状态的 H2D/D2H 传输走 `kv_exchange`，Python 封装分别调用 `transfer_kv_per_layer_kv_split_pf_lf` / `transfer_kv_all_layer_kv_split_lf_pf`。这里复用算子接口传输独立的 temporal、conv、draft 和 side state，Mamba Host Pool 仍为 `page_first_direct`，不是 K/V 分离布局。预期现象：驱逐压力下状态正确备份到 Host 并回读，同一 prompt 两次 greedy 生成输出完全一致，服务无崩溃。

## 测试原理

mamba 池按 request 存槽：一个请求只占一个状态槽，槽在请求结束后进入 LRU，device 槽满时被逐出到 host（状态 D2H），下次命中 host L2 时恢复（状态 H2D）。若传输路径写错层、错 slot 或视图不连续，恢复出的状态即是脏数据，greedy 续写会立刻发散。因此用例为：

1. target prompt（长且不与其他请求共享前缀）greedy 生成，记录输出；
2. 发送 5 个互不共享前缀的 filler 请求，使 device KV 池驱逐 target，同时避免把 target 从 Host L2 再次逐出；
3. 重放同一 target prompt：device 前缀缓存已逐出、命中 host L2，mamba 状态经 H2D 恢复后续写；
4. 从 `/metrics` 读取 `pool="mamba"` 的 D2H、H2D token 计数，分别要求 filler 后和 target 重放后增长；两次输出还须逐字符一致。报告中的计数增量为所有 TP rank 之和。

同时从 server.log 断言 `transfer mode: npu_state_transfer (kv_exchange` 日志行存在，确认新路径被启用。

## 前置依赖

- 昇腾 NPU（A3/A5），CANN 版本与日常 sglang NPU 开发环境一致
- sgl-kernel-npu：包含 `operators/mem_cache/memfabric/kv_transfer` 模块（提供四个 split-KV 算子）
- sglang：本仓库该改动之后的工作区
- 模型：任一 hybrid（linear attention / mamba 结构）模型，如 Qwen3-Next 系列或 Falcon-H1 系列，需已下载到本地
- host 内存需容纳 `MAX_TOTAL_TOKENS × HICACHE_RATIO` 的层级缓存

## 环境变量

| 变量 | 示例值 | 含义 | 必填 |
| --- | --- | --- | --- |
| SGLANG_ROOT | /root/sglang | sglang 仓库路径 | 是 |
| KERNEL_ROOT | /root/sgl-kernel-npu | sgl-kernel-npu 仓库路径 | 是 |
| KERNEL_PYTHON_PATH | `$KERNEL_ROOT/build/torch/package` | 同时注册 `kv_exchange` 与 `sgl_chunk_gated_delta_rule` 的构建包目录；安装 wheel 时可覆盖 | 否 |
| MODEL_PATH | /data/models/Qwen3-Next-80B | hybrid(mamba) 模型路径 | 是 |
| NPU_DEVICES | 0,1 | 空闲物理卡 ID 列表，卡数决定 tp-size | 是 |
| PORT | 30000 | 服务端口 | 否 |
| TARGET_PROMPT_CHARS | 20000 | target prompt 字符数 | 否 |
| TARGET_NEW_TOKENS | 64 | target 每次生成 token 数 | 否 |
| NUM_FILLER | 5 | filler 请求数（超过 device 容量，但总前缀应低于 Host 容量） | 否 |
| FILLER_PROMPT_CHARS | 12000 | 每个 filler 的 prompt 字符数 | 否 |
| FILLER_NEW_TOKENS | 8 | filler 生成 token 数 | 否 |
| STARTUP_TIMEOUT | 1800 | 服务启动等待秒数 | 否 |
| REQUEST_TIMEOUT | 600 | 单请求超时秒数 | 否 |
| MAX_TOTAL_TOKENS | 8192 | device KV 池容量（刻意调小制造驱逐） | 否 |
| CONTEXT_LENGTH | 32768 | 模型上下文长度 | 否 |
| HICACHE_RATIO | 4 | host 池 / device 池容量比 | 否 |
| SGLANG_HICACHE_HOST_MEM_BACKEND | acl | host 内存后端（acl 或 memfabric；本分支 hybrid 模型 MHA 池未做 memfabric 预留，实测用 acl） | 否 |

151 上的 100 题用例使用 `build/torch/package`。旧的
`python/sgl_kernel_npu/build/lib.linux-aarch64-cpython-312` 在该机器上
只有旧 GDN 算子，缺少 `kv_exchange` 与 `sgl_chunk_gated_delta_rule`，
不能用于当前用例。下述私有算子桥接是此前单请求测试的历史环境记录。

## 在另一台 A5 上运行

### 代码、模型和算子包

- SGLang 源码应同时包含 MHA 与 Mamba 的 HiCache NPU 状态传输适配。
  在 A5 上核对实际源码与提交内容，不要仅凭远端分支名判断版本。
- 两个 GSM8K 客户端只接受 Qwen3.5-4B 的配置：`qwen3_5` 架构、32 层，
  其中 24 层 linear attention、8 层 full attention，`hidden_size=2560`。
  train/test 数据须为包含 `question` 和 `answer` 的 GSM8K JSONL 文件。
- A5 使用架构版本 `3510` 编译 sgl-kernel-npu。当前代码除
  `kv_exchange` 外还调用 GDN、causal conv 等算子；没有适配好的完整 A5
  wheel 时，使用仓库根目录的构建入口编译所有支持 A5 的算子：

  ```bash
  source /实际安装路径/set_env.sh
  cd /path/to/sgl-kernel-npu
  python3 setup.py --arch-version 3510 --ops all --artifact wheel -j8
  ```

  构建参数见 [sgl-kernel-npu 构建说明](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md)，
  `kv_exchange` 的 A5 要求见 [算子说明](https://github.com/sgl-project/sgl-kernel-npu/blob/main/docs/api/en/operators/mem_cache/kv_transfer.md)。
  151 上的 A3 编译产物不能直接复用。CANN、PyTorch 和 torch_npu 应与
  A5 构建包匹配。若已安装 A5 wheel，可将 `KERNEL_PYTHON_PATH` 设为实际
  `sgl_kernel_npu` 包的父目录；谨防旧的 `build/torch/package` 抢先于 wheel 被导入。

### 配置与启动前检查

先在**实际执行环境内**用 `npu-smi info` 选空闲卡，用 `free -h` 检查 Host
内存。`env.sh` 固定 source
`/usr/local/Ascend/ascend-toolkit/latest/set_env.sh`；若 A5 的 CANN
安装位置不同，先将该行改为实际路径。若从 Windows 直接复制 `.sh`，确认文件
为 LF 换行；CRLF 会导致 Linux 上 `source env.sh` 报错。

```bash
export SGLANG_ROOT=/path/to/sglang
export KERNEL_ROOT=/path/to/sgl-kernel-npu
export KERNEL_PYTHON_PATH="$KERNEL_ROOT/build/torch/package"
export MODEL_PATH=/path/to/Qwen3.5-4B
export GSM8K_TRAIN_PATH=/path/to/gsm8k_train.jsonl
export GSM8K_TEST_PATH=/path/to/gsm8k_test.jsonl
export NPU_DEVICES=0,1,2,3  # 改为当前环境实际空闲、可见的卡
export PORT=30000           # 同机其他服务占用时修改

cd "$SGLANG_ROOT/test/manual/npu_mamba_hicache_kv_split"
source ./env.sh
python3 - <<'PY'
import torch
import sgl_kernel_npu

print(sgl_kernel_npu.__file__)
for name in (
    "kv_exchange",
    "sgl_chunk_gated_delta_rule",
    "causal_conv1d",
    "recurrent_gated_delta_rule",
):
    assert hasattr(torch.ops.npu, name), f"缺少算子: {name}"
print("算子注册检查通过")
PY
```

打印的 `__file__` 必须属于预期 A5 包。注册检查不执行 kernel；继续跑小样本
才能发现架构或运行时不匹配。Host 内存后端先保持 `acl`，以便与已验证的
`kernel_ascend` 路径一致。多个服务共用宿主机且 HCCL 默认端口冲突时，再设置
`HCCL_IF_BASE_PORT`、`HCCL_NPU_SOCKET_PORT_RANGE` 和
`HCCL_HOST_SOCKET_PORT_RANGE`；单独运行无需预设。

### 按顺序运行与判定

以下命令应依次运行，每轮使用不同目录；前一轮退出后再启动下一轮：

```bash
bash run_test.sh /tmp/mamba_smoke_a5
TEST_CASE=gsm8k GSM8K_NUM_QUESTIONS=10 \
  bash run_gsm8k.sh /tmp/mamba_gsm8k_10_a5
TEST_CASE=stress GSM8K_NUM_QUESTIONS=3 \
  bash run_gsm8k.sh /tmp/mamba_unique_3_a5
TEST_CASE=stress GSM8K_NUM_QUESTIONS=100 \
  bash run_gsm8k.sh /tmp/mamba_unique_100_a5
```

`run_test.sh` 是基础 D2H/H2D 冒烟，服务参数与 `run_gsm8k.sh` 不完全相同；
若重点是独有 Mamba 状态回迁，以 `TEST_CASE=stress` 的结果为准。
`run_gsm8k.sh` 输出 `server.log`、`eval.log`、`result.json`。
检查 `result.json` 的 `passed`、`unique_host_questions`、
`device_hit_questions`，以及服务日志中的
`NPU HiCache Mamba state transfer mode: npu_state_transfer (kv_exchange`。A5 的 Host 深度不必等于
151 的 5376/5632；stress 真正要求每题 `host >= seed_len > shared_prefix_boundary`、
`device=0`、Mamba H2D 增长且数值答案与首轮一致。

`run_gsm8k.sh` 默认 `PAGE_SIZE=128`、`MAX_TOTAL_TOKENS=8192`、
`HICACHE_RATIO=8`。151 的 TP=4 运行曾为 Mamba Host 池分配约 176 GB/进程；
A5 上以实际启动日志和 Host/cgroup 余量为准。不要直接把 page size 改成 1；
此前同模型该设置导致首轮答案异常。若调整容量或唯一填充长度，需保证
完整 prompt 加生成 token 能放入设备容量，并重新确认驱逐和 seed checkpoint
断言。`stress` 的数字填充改变 prompt 分布，不能将它的正确率当标准 GSM8K
精度；标准精度门限应使用 A5 **同配置**历史实测值。151 的 100 题通过报告
仅是结果格式参考，不能代替 A5 实测。

## 历史单请求实测环境（2026-11-16）

- 8 卡 Ascend910B 容器；最终 D2H/H2D 验证使用 NPU_DEVICES=0,1,2,3（tp=4），CANN 9.1.0，torch_npu 2.10.0
- 模型 Qwen3.5-4B（24 linear attention + 8 full attention 层），`--mamba-ssm-dtype bfloat16` 覆盖（Ascend910B 不支持 fp32 initial_state）
- 算子注册需 sgl-kernel-npu 源码树 + kv_exchange 私有编译包并用：
  - `PYTHONPATH` 指向源码树 `python/sgl_kernel_npu`（注册 `torch.ops.npu.causal_conv1d` 等前向算子）
  - `kv_exchange` 算子仅存在于私有包（如 `qwen3_kv_exchange_pkg_YYYYMMDD`），需 `torch.ops.load_library` 私有包的 `libsgl_kernel_npu.so`，并将其 `_cann/vendors/sgl_kernel_npu` 追加到 `ASCEND_CUSTOM_OPP_PATH`
  - **桥接方式（推荐）**：在 PYTHONPATH 前置目录放 `sitecustomize.py`，Python 进程启动时自动执行：

    ```python
    import os
    _VENDOR = "/path/to/qwen3_kv_exchange_pkg/python/sgl_kernel_npu/sgl_kernel_npu/_cann/vendors/sgl_kernel_npu"
    _paths = [p for p in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if p]
    if _VENDOR not in _paths:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join([_VENDOR] + _paths)
    _PRIV_LIB = "/path/to/qwen3_kv_exchange_pkg/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
    if os.path.exists(_PRIV_LIB):
        import torch
        torch.ops.load_library(_PRIV_LIB)
    ```

  - 二者缺一即崩溃：缺私有包 → D2H 崩 `kv_exchange` 不存在；缺源码树 → prefill/decode 崩 `causal_conv1d` 不存在
- 多容器共享宿主机时，若 HCCL 默认端口冲突，可在启动前按机器实际空闲端口设置 HCCL 端口变量。
- 最终验证：5 个 filler，Mamba D2H 计数增量 24、H2D 增量 8；重放命中 3584 个缓存 token，输出一致。`/tmp/mamba_e2e_out_metrics_v6/report.md`。

## 执行命令

```bash
bash run_test.sh /path/to/output
```

## 服务启动命令

单卡（`NPU_DEVICES=0`）默认形式，多卡时 `--tp-size` 由 `NPU_DEVICES` 的卡数推导：

```bash
sglang serve \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 --port 30000 --trust-remote-code \
  --device npu \
  --tp-size 1 \
  --enable-hierarchical-cache \
  --enable-metrics \
  --hicache-ratio 4 \
  --hicache-io-backend kernel_ascend \
  --max-total-tokens 8192 \
  --mamba-ssm-dtype bfloat16 \
  --context-length 32768
```

## 输出物与判定

| 文件 | 内容 |
| --- | --- |
| `<output>/server.log` | 服务完整日志 |
| `<output>/eval.log` | 用例执行日志 |
| `<output>/case_result.json` | 两次输出、filler 完成数、Mamba D2H/H2D 计数增量、错误列表 |
| `<output>/report.md` | 汇总报告（失败也生成） |
| `<output>/report.log` | 报告生成日志 |

通过需同时满足：

1. target 两次 greedy 输出完全一致（`case_result.json` 的 `match == true`）；
2. filler 全部成功（`filler_done == NUM_FILLER`），驱逐压力真实形成；
3. server.log 中存在 `transfer mode: npu_state_transfer (kv_exchange`，确认新路径启用；
4. 无请求阶段错误。
5. `pool="mamba"` 的 D2H/H2D token 计数增量均大于 0。

任一不满足即失败，失败原因写入 report.md。

## 已知限制

- 通过 Mamba 池计数器断言发生了 L2 备份和回读；计数器不记录具体算子调用次数。
- 只做正确性判定，不做性能评价。
- 只覆盖 greedy 单并发串行场景，不覆盖 PD 分离、多 rank TP>1 下的传输正确性。
- draft（MTP）层是否被包含取决于所选模型是否启用 speculative decoding，本用例不强制。
