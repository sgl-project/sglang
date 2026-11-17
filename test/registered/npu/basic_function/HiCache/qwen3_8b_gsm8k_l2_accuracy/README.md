# Qwen3-8B GSM8K 精度与 HiCache L2 回载用例

本用例验证 NPU `kernel_ascend` 的 MHA KV 新算子在 `layer_first` 和 `page_first` 两种布局下，KV 从设备写入 Host L2、驱逐后回载时的 GSM8K 精度。2026-11-12 已完成 2 题冒烟和 200 题快速回归；GSM8K 全量精度尚未运行。

## 三轮流程

1. 从 GSM8K **test** 读取固定的前 `GSM8K_NUM_QUESTIONS` 题；few-shot 示例来自 **train**。逐题以确定性采样生成答案，记录首轮正确率和输出 token ID。
2. 用 **train** 中未参与 few-shot 的不同题目构造长请求，持续填充设备 KV，直到请求输入总量超过设备 KV 容量且观察到设备驱逐；检查 KV Pool 的 D2H token 计数增加。第二轮请求不参与 GSM8K 计分。
3. 按原顺序重发第一轮完全相同的 token ID，记录逐题答案、正确率、Host 命中数和输出 token ID；检查 KV Pool 的 H2D token 计数增加。对题目独有提示部分能形成完整缓存页的题目，要求第三轮有 Host 命中；不足一页的题目不强制逐题命中。第三轮正确率不低于首轮。若设置 `GSM8K_ACCURACY`，还要求第三轮达到该历史实测阈值。

结果文件分别是 `layer_first.json` 和 `page_first.json`，包含逐题答案、正确性、输出 token ID、Host 命中、两轮准确率及 D2H/H2D 计数增量。输出 token ID 一致数只记录为诊断信息，精度判定以 GSM8K 答案正确率为准。

## 配置与运行方式

所有参数在 [env.sh](env.sh) 集中设置。模型、SGLang、算子仓库、GSM8K train/test 数据集路径需按目标机器修改，也可在调用前覆盖环境变量。若编译产物单独解包，使用 `KERNEL_PYTHON_PATH` 指向包含 `sgl_kernel_npu` 包的父目录，确保运行进程加载含 `kv_exchange` 的原生扩展。用例检查模型 `config.json` 为 Qwen3-8B、服务实际启用了 `kernel_ascend` 和指定布局。默认 200 题仅用于快速回归；正式精度结论应将 `GSM8K_NUM_QUESTIONS=1319`，并用相同配置的历史实测结果设置 `GSM8K_ACCURACY`。不能凭期望值填写阈值。

```bash
cd /path/to/sglang
export SGLANG_ROOT="$PWD"
export KERNEL_ROOT=/path/to/sgl-kernel-npu
export MODEL_PATH=/path/to/Qwen3-8B
export GSM8K_TRAIN_PATH=/path/to/gsm8k/train.jsonl
export GSM8K_TEST_PATH=/path/to/gsm8k/test.jsonl
# 正式验收时再设置 GSM8K_ACCURACY 为同配置历史实测阈值
bash test/registered/npu/basic_function/HiCache/qwen3_8b_gsm8k_l2_accuracy/run_test.sh /path/to/output
```

`MAX_TOTAL_TOKENS`、`HICACHE_RATIO`、`FILLER_PROMPT_TOKENS` 和 `MAX_FILLER_REQUESTS` 控制 L1/L2 容量与第二轮挤出强度。默认值仅是测试起点；若 L2 容量不足导致第三轮部分题目未命中 Host，结果会明确失败，需依据实际 NPU/主机内存调整后重新运行。`REQUEST_TIMEOUT` 为单次生成超时。

200 题验证的默认设置为设备 KV `32768` token、`hicache_ratio=6`，Host L2 约 `196608` token（Qwen3-8B BF16 KV 约 27 GiB，另有管理开销）。第二轮长请求累计超过设备容量后才进入第三轮。`hicache_ratio=3` 时 Host L2 约 `98304` token；仅 200 题每题最多生成 512 token 的上限就达到 `102400` token，尚未计入题目和第二轮请求，因此比例 3 不能保证保留全部 KV。比例 6 仍须以实际 Host 命中和结果 JSON 验证。

本用例需要完整服务启动与三轮请求编排，仓库现有 `GSM8KAscendMixin` 只支持单轮评测，因此这里复用 `sglang.test.simple_eval_mixed_prefix_gsm8k` 的提示和答案解析函数，并在用例中实现三轮编排。

## A5 环境准备与运行注意事项

先按 [MHA KV split 端到端用例的 A5 准备步骤](../qwen3_8b_kv_split_e2e/README.md) 在 **A5 机器或容器内** 构建 `--arch-version 3510 --ops all` 的 `sgl-kernel-npu` wheel，并安装到运行 SGLang 的 Python 环境。不能沿用下文历史 A3 机器上的私有编译包路径。此用例的 `env.sh` 虽已支持 `KERNEL_PYTHON_PATH`，默认值仍是算子仓库源码目录；A5 上必须改为已安装 wheel 的 `site-packages`，并确认 `sgl_kernel_npu.__file__` 指向该目录、`hasattr(torch.ops.npu, "kv_exchange")` 为 `True`。仅有 Python 注册不代表三轮回迁已通过，仍需看结果 JSON 与日志。

```bash
cd /path/to/sglang
export SGLANG_ROOT="$PWD"
export KERNEL_ROOT=/path/to/sgl-kernel-npu
export MODEL_PATH=/path/to/Qwen3-8B
export GSM8K_TRAIN_PATH=/path/to/gsm8k/train.jsonl
export GSM8K_TEST_PATH=/path/to/gsm8k/test.jsonl
export ASCEND_ENV_SCRIPT=/actual/path/to/Ascend/set_env.sh
export NPU_DEVICES=0                         # 改为 A5 上空闲的物理卡 ID
export KERNEL_PYTHON_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
source test/registered/npu/basic_function/HiCache/qwen3_8b_gsm8k_l2_accuracy/env.sh
python3 - <<'PY'
import sgl_kernel_npu
import torch
print(sgl_kernel_npu.__file__)
assert hasattr(torch.ops.npu, "kv_exchange")
PY

# 先验证两种布局的三轮回迁；不使用历史 A3 精度阈值
export GSM8K_NUM_QUESTIONS=2
unset GSM8K_ACCURACY
bash test/registered/npu/basic_function/HiCache/qwen3_8b_gsm8k_l2_accuracy/run_test.sh /tmp/qwen3_8b_mha_a5_gsm8k_2

# 小规模通过后再运行 200 题回归
export GSM8K_NUM_QUESTIONS=200
bash test/registered/npu/basic_function/HiCache/qwen3_8b_gsm8k_l2_accuracy/run_test.sh /tmp/qwen3_8b_mha_a5_gsm8k_200
```

- 模型必须通过脚本对 **Qwen3-8B** `config.json` 的结构检查；train/test 路径须分别指向 GSM8K 的原始 JSONL。`NPU_DEVICES` 决定脚本使用的物理卡和 `TP_SIZE`，`PORT` 不得与其他服务冲突。默认按 `layer_first`、`page_first` 顺序启动；若只想定位某一种布局，可暂设 `LAYOUTS=layer_first` 或 `LAYOUTS=page_first`，正式双布局验收时取消该限制。
- 首次运行保留 `SGLANG_HICACHE_HOST_MEM_BACKEND=acl` 与 `PAGE_SIZE=128`。默认 `MAX_TOTAL_TOKENS=32768`、`HICACHE_RATIO=6`、`FILLER_PROMPT_TOKENS=4096`、`MAX_FILLER_REQUESTS=12` 是历史环境的起点，先核对 A5 的 HBM 和 Host 可用内存。若服务启动失败、未发生驱逐或第三轮未命中 Host，应结合 `server.*.log`、`eval.*.log` 和结果 JSON 调整池容量或 filler 强度；不要仅靠放宽断言让用例通过。
- 两题只验证三轮流程和回迁路径。200 题用于快速精度回归；正式精度结论须跑满 `GSM8K_NUM_QUESTIONS=1319`，并以**同一 A5 模型、数据、布局和参数配置**的历史实测值设置 `GSM8K_ACCURACY`。历史 A3 的 190/200、192/200 不是 A5 阈值。每次用新输出目录，检查两种布局的 `*.exit_code`、`*.json`、`server.*.log` 和 `eval.*.log`；`device` 模式不测试 Host L2 回载。

### 设备缓存对照模式

为排查首轮与 L2 重放的输出差异，`run_test.sh <output_dir> device` 会关闭 HiCache，不发送第二轮挤出请求，只对相同的 GSM8K 提示执行首轮与重放。对照模式要求没有设备 KV 驱逐、没有 Host 命中或 H2D 回载，并要求每题重放命中设备缓存。它记录准确率和输出 token 一致数，但不把两轮输出不一致直接判为测试失败。`layer_first` 在这个模式下只是结果文件名；没有启用 HiCache 布局。

```bash
export GSM8K_NUM_QUESTIONS=200
export MAX_TOTAL_TOKENS=131072  # 容纳两轮之间仍需保留的设备 KV
export LAYOUTS=layer_first
bash test/registered/npu/basic_function/HiCache/qwen3_8b_gsm8k_l2_accuracy/run_test.sh /path/to/device_control device
```

如果设备对照模式也有类似的输出差异，说明差异并非 L2 回载独有；若只有 L2 模式出现，需要进一步比较回载前后的 KV 与首个分歧 token 的 logits。单凭两轮 GSM8K 正确率不能定位原因。

## `kv_exchange` 未注册时的排查与解决

首次两题测试在第一道题的 D2H 阶段中断。服务日志显示 Python 调用已进入 `kv_transfer_impl.py`，但调用 `torch.ops.npu.kv_exchange` 时抛出：

```text
AttributeError: '_OpNamespace' 'npu' object has no attribute 'kv_exchange'
```

这说明 Python 包里的调用封装存在，**当前进程加载的原生扩展没有注册该算子**。当时 `PYTHONPATH` 优先指向 `/home/l00889655/sglang/sgl-kernel-npu/python/sgl_kernel_npu`，实际加载的是该源码目录下的包；此前构建的新产物虽已存在于私有解包目录，但不会因编译完成而自动替换当前进程导入的包或其 `.so`。仅看到源码中有 `kv_exchange`，不能说明运行进程已经加载了包含它的扩展。

本次解决方法是让**测试进程**优先导入已有的私有编译包，没有修改容器的全局安装：

```bash
export KERNEL_ROOT=/home/l00889655/sglang/sgl-kernel-npu
export KERNEL_PYTHON_PATH=/home/l00889655/sglang/qwen3_kv_exchange_pkg_20261112/python/sgl_kernel_npu
source test/registered/npu/basic_function/HiCache/qwen3_8b_gsm8k_l2_accuracy/env.sh
python3 -c 'import sgl_kernel_npu, torch; print(sgl_kernel_npu.__file__); print(hasattr(torch.ops.npu, "kv_exchange"))'
```

预检应打印私有目录中的 `sgl_kernel_npu/__init__.py` 路径，以及 `True`。`env.sh` 将 `KERNEL_PYTHON_PATH` 放在其他已有的算子包路径之前，服务进程和评测进程因此使用同一份包。该包的 `__init__.py` 会加载其 `lib/libsgl_kernel_npu.so`，从而注册算子。若第二行仍为 `False`，先检查打印的实际导入路径，再核对该目录的 `.so` 是否确实由包含 `kv_exchange` 的构建产物提供；修正后需重启服务进程。

## 两题小规模验证

在 `sglang-lzx-B103` 容器、NPU 0 上运行，模型 `/mnt/paas/weights/Qwen3-8B`，GSM8K 数据位于 `/home/l00889655/sglang/gsm8k_{train,test}.jsonl`。本次设 `GSM8K_NUM_QUESTIONS=2`、`MAX_TOTAL_TOKENS=8192`、`HICACHE_RATIO=4`，用 `KERNEL_PYTHON_PATH` 指向已编译的私有 `kv_exchange` 包。没有设置历史精度阈值。

| 布局 | 首轮 | 重放 | Host 命中题数 | D2H KV token | H2D KV token | 驱逐 token | 输出 token 一致题数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer_first` | 2/2 | 2/2 | 2/2 | 13,824 | 1,408 | 9,728 | 2/2 |
| `page_first` | 2/2 | 2/2 | 2/2 | 13,824 | 1,408 | 9,728 | 2/2 |

两种布局的用例与总脚本退出码均为 0。首次尝试的失败与解决步骤见上一节。2 题只证明三轮流程、L2 回载及答案一致性，不能据此得出模型精度结论。

## 200 题快速回归

在同一容器的物理 NPU 4 上使用 `/mnt/paas/weights/Qwen3-8B`，设置 `GSM8K_NUM_QUESTIONS=200`、`GSM8K_NUM_SHOTS=8`、`MAX_TOTAL_TOKENS=32768`、`HICACHE_RATIO=6`、`MAX_FILLER_REQUESTS=12`，其余设置由 `env.sh` 提供。本次没有同配置的历史实测精度阈值，故只比较首轮与 L2 重放精度，不宣称达到正式精度门槛。结果目录为 `/home/l00889655/sglang/qwen3_8b_gsm8k_l2_200q_ratio6_phy4_20261112/`。

| 布局 | 首轮正确 | 重放正确 | Host 命中 | D2H KV token | H2D KV token | 驱逐 token | 输出 token 完全一致 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer_first` | 190/200（95%） | 192/200（96%） | 200/200 | 87,680 | 27,136 | 63,104 | 137/200 |
| `page_first` | 190/200（95%） | 192/200（96%） | 200/200 | 87,680 | 27,136 | 63,104 | 137/200 |

第二轮共发出 36,864 个不同提示 token。每个布局的逐题用例、布局退出码和总脚本退出码均为 0。两种布局在首轮彼此输出 token ID 完全相同，重放时也彼此完全相同。首轮与重放之间有 63 题的输出 token 序列不同，但 197 题的数值答案相同；其余 3 题中有 2 题由错误变正确，没有题目由正确变错误。这个差异保留在结果 JSON 中供进一步分析，不能把 200 题的 96% 推论为全量精度。
