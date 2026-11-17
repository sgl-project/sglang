# Qwen3-8B MHA HiCache KV split 端到端测试

## 测试目的

在真实 Qwen3-8B 推理服务中，分别验证 NPU `kernel_ascend` 的 `layer_first` 和 `page_first` 布局能完成 KV 的 D2H 备份与 H2D 回载。两种布局均须出现 Host 缓存命中，重放请求的输出 token 与首次请求一致。

## 测试原理

入口脚本依次启动两次 Qwen3-8B 服务。每次先核对本地 `config.json` 的 Qwen3-8B 结构，以及 `/model_info` 和 `/server_info` 返回的模型路径、`kernel_ascend`、布局和 HiCache 写策略。随后发送一个基准长提示和多个不同前缀的长提示，使较小的设备 KV 池发生驱逐；最后重放基准提示。

`write_through` 将 KV 写入 Host Pool；设备驱逐后，重放请求从 Host Pool 回载。用例要求 `/metrics` 中 KV Pool 的 `sglang:hicache_backup_tokens_total` 与 `sglang:load_back_tokens_total` 增加，且重放响应的 `cached_tokens_details.host` 大于零。当前 MHA Host Pool 在 `kernel_ascend` 下按布局分派到如下新算子：

| 布局 | D2H | H2D |
| --- | --- | --- |
| `layer_first` | `transfer_kv_all_layer_kv_split` | `transfer_kv_per_layer_kv_split` |
| `page_first` | `transfer_kv_all_layer_kv_split_lf_pf` | `transfer_kv_per_layer_kv_split_pf_lf` |

## 前置依赖

- Ascend NPU，至少 1 张空闲卡；已安装与设备匹配的驱动、CANN、`torch_npu`。
- 包含本次 MHA KV split 适配的 SGLang 源码、对应的 `sgl-kernel-npu`，以及其 Python 依赖（含 `requests`、`transformers`）。
- 本地 Qwen3-8B 模型目录，含 `config.json`、权重和 tokenizer。
- 可用的 `sglang serve` 命令；Host 内存通过 `acl` 注册。

## A5 环境准备与运行注意事项

以下步骤在 **A5 机器或容器内** 执行。使用包含 MHA KV split 适配的 SGLang 代码，以及支持 A5 的 `sgl-kernel-npu` 代码；A3 编译的原生扩展不能直接用于 A5。先确认 `npu-smi info` 能看到准备使用的空闲卡，驱动、CANN、`torch_npu` 和 PyTorch 与 A5 匹配，并确认容器能访问模型、主机内存和算子编译环境。`ASCEND_ENV_SCRIPT` 应指向该环境实际的 `set_env.sh`，不要照抄示例路径。

在算子仓库根目录用 **`--arch-version 3510`** 构建 A5 wheel；完整服务还会调用其他算子，建议选 `--ops all`，不要只编译 `kv_exchange`。如果从 A3 构建目录切换到 A5，先执行 `python3 setup.py --make-clean`，再构建。将 wheel 安装到运行 SGLang 的同一个 Python 环境：

```bash
cd /path/to/sgl-kernel-npu
source /actual/path/to/Ascend/set_env.sh
python3 setup.py --make-clean                 # 从其他架构或旧配置切换时执行
python3 setup.py --arch-version 3510 --ops all --artifact wheel -j8
ls -lt output/sgl_kernel_npu-*.whl
python3 -m pip install --force-reinstall --no-deps /absolute/path/to/new-A5-wheel.whl
```

根据 `ls` 输出，把安装命令的路径替换为本次 A5 构建生成的 wheel；`output/` 中可能留有旧产物。

`env.sh` 默认把算子仓库源码目录加入 `PYTHONPATH`，可能遮蔽刚安装的 A5 wheel。因此在执行用例前，**显式指定** `KERNEL_PYTHON_PATH` 为该 wheel 所在 Python 环境的 `site-packages`，然后检查运行进程实际导入的包及算子注册情况：

```bash
cd /path/to/sglang
export SGLANG_ROOT="$PWD"
export KERNEL_ROOT=/path/to/sgl-kernel-npu
export MODEL_PATH=/path/to/Qwen3-8B
export ASCEND_ENV_SCRIPT=/actual/path/to/Ascend/set_env.sh
export NPU_DEVICES=0                         # 改为 A5 上空闲的物理卡 ID
export KERNEL_PYTHON_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
source test/registered/npu/basic_function/HiCache/qwen3_8b_kv_split_e2e/env.sh
python3 - <<'PY'
import sgl_kernel_npu
import torch
print(sgl_kernel_npu.__file__)
assert hasattr(torch.ops.npu, "kv_exchange"), "当前 Python 环境没有注册 kv_exchange"
PY
bash test/registered/npu/basic_function/HiCache/qwen3_8b_kv_split_e2e/run_test.sh /tmp/qwen3_8b_mha_a5_e2e
```

确认打印路径位于当前环境的 `site-packages`，而不是旧 A3 源码或私有解包目录；若算子未注册，先核对路径、wheel 的架构和编译产物，再重启测试。预检只能证明当前 Python 进程完成注册，实际 A5 执行仍以端到端结果为准。

- 用例固定要求 **Qwen3-8B** 的 `config.json` 结构；同名但结构不同的模型会在请求前被拒绝。`NPU_DEVICES` 为物理卡列表，脚本据此计算 `TP_SIZE`；同时确认 `PORT` 未占用。两个布局依次启动，不要并发运行另一套占用相同卡或端口的测试。
- 首次运行保留 `SGLANG_HICACHE_HOST_MEM_BACKEND=acl` 和 `PAGE_SIZE=128`。A5 的可用 HBM、Host 内存和模型配置可能与历史 A3 记录不同。先查看 `server.*.log` 中的分配与启动错误，再按实际容量调整 `MAX_TOTAL_TOKENS`、`HICACHE_RATIO`、`PROMPT_TOKENS` 或 `FILLER_REQUESTS`；调整后仍须满足驱逐、Host 命中和 D2H/H2D 指标要求。
- 以全新输出目录运行，查看 `report.md`、两个布局的 `*.exit_code`、`*.json` 和 `server.*.log`。脚本退出码为 0 只对应这次运行；不要把旧目录里的报告当作本次证据。若脚本是从 Windows 复制到 Linux，先确认文件为 LF 换行。

## 环境变量

只修改 `env.sh`，或在执行前覆盖其中的变量。

| 变量 | 示例值 | 含义 | 必填 |
| --- | --- | --- | --- |
| `MODEL_PATH` | `/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B` | Qwen3-8B 模型目录 | 是 |
| `SGLANG_ROOT` | `/path/to/sglang` | SGLang 仓库根目录 | 是 |
| `KERNEL_ROOT` | `/path/to/sgl-kernel-npu` | 算子仓库根目录 | 是 |
| `KERNEL_PYTHON_PATH` | 当前 Python 环境的 `site-packages` | 优先导入的算子包父目录；A5 使用已安装 wheel 时须显式设置 | A5 是 |
| `ASCEND_ENV_SCRIPT` | `/usr/local/Ascend/ascend-toolkit/latest/set_env.sh` | Ascend 环境脚本 | 是 |
| `NPU_DEVICES` | `0` | 物理卡 ID，逗号分隔 | 是 |
| `PORT` | `30000` | 服务端口 | 否 |
| `STARTUP_TIMEOUT` | `1800` | 服务启动等待秒数 | 否 |
| `REQUEST_TIMEOUT` | `300` | 单请求和指标等待秒数 | 否 |
| `MAX_TOTAL_TOKENS` | `4096` | 设备 KV 池大小，用于触发驱逐 | 否 |
| `PAGE_SIZE` | `128` | HiCache 页大小 | 否 |
| `PROMPT_TOKENS` | `1024` | 每个提示的 token 数 | 否 |
| `OUTPUT_TOKENS` | `8` | 每次请求生成 token 上限 | 否 |
| `FILLER_REQUESTS` | `8` | 驱逐用长提示数量 | 否 |
| `HICACHE_RATIO` | `4` | Host 与设备缓存容量比 | 否 |
| `MEM_FRACTION_STATIC` | `0.8` | 静态 NPU 内存占比 | 否 |
| `SGLANG_HICACHE_HOST_MEM_BACKEND` | `acl` | Host 内存注册方式 | 否 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True,pin_memory_expandable_segments:True` | PyTorch NPU 分配配置 | 否 |
| `ASCEND_RT_VISIBLE_DEVICES` | `0` | 从 `NPU_DEVICES` 派生 | 自动 |
| `TP_SIZE` | `1` | 从 `NPU_DEVICES` 的卡数派生 | 自动 |
| `PYTHONPATH` | `<SGLANG_ROOT>/python:<KERNEL_PYTHON_PATH>` | 由 `env.sh` 派生；未设置 `KERNEL_PYTHON_PATH` 时默认算子仓库源码目录 | 自动 |

## 执行命令

在配置好 `env.sh` 后运行：

```bash
bash run_test.sh /path/to/qwen3_8b_kv_split_result
```

脚本依次启动、测试并停止两个布局的服务，退出码为总体判定。

## 服务启动命令

`run_test.sh` 对 `LAYOUT=layer_first` 和 `LAYOUT=page_first` 各执行一次以下命令；变量均来自 `env.sh`：

```bash
sglang serve \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 --port "$PORT" --trust-remote-code \
  --device npu --attention-backend ascend \
  --tp-size "$TP_SIZE" --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --max-total-tokens "$MAX_TOTAL_TOKENS" --page-size "$PAGE_SIZE" \
  --enable-hierarchical-cache --hicache-ratio "$HICACHE_RATIO" \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel_ascend --hicache-mem-layout "$LAYOUT" \
  --enable-metrics --disable-cuda-graph
```

## 输出物与判定

输出目录包含 `server.layer_first.log`、`server.page_first.log`、`eval.layer_first.log`、`eval.page_first.log`、`layer_first.json`、`page_first.json`、两个 `*.exit_code`、`report.log` 和 `report.md`。入口脚本在成功、失败或中断时打印这些路径。

每个布局必须同时满足：服务解析参数正确；D2H/H2D 的 KV token 指标增量均大于零；重放命中 Host；生成 token ID 和文本与首次请求一致。任一条件失败则该布局失败，总体失败。`report.md` 在失败时也会生成，并列出已取得的证据和原因。

## 已知限制

- 本用例用请求、Host 命中和服务指标验证端到端路径，不直接逐字节读取 Host KV 内存，也不逐次统计算子调用。
- 默认池大小和提示数量适用于触发设备 KV 驱逐；如环境中的缓存行为不同，可在 `env.sh` 调整 `MAX_TOTAL_TOKENS`、`PROMPT_TOKENS`、`FILLER_REQUESTS` 和 `HICACHE_RATIO`。
- 只覆盖单机 TP，顺序请求，不评价性能或生成质量。
