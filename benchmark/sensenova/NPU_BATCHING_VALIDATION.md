# SenseNova 910C 内网批处理验证手册

本手册分两部分：①确认批处理路径可以执行；②测量不同 batch size 的端到端性能。
面向 Linux + 单张 Ascend 910C，使用已有的 NPU 推理镜像与本地完整模型。
Windows 机器只负责拷贝代码，不执行 NPU 验证。

本次实现已在 910C 上完成可用性与性能验证，结果见 2.5 节。复测遇到错误请保留日志；不可将 HTTP 成功等同于真实合批成功。
第一部分不证明数值精度与原模型完全一致，只证明执行与返回可用。

## 0. 共同准备（只做一次）

### 0.1 拷贝文件并进入环境

将当前开发工作区完整复制到内网 `/workspace/sglang`，包含未提交的修改以及
`benchmark/sensenova/validate_npu_batching.py`。只复制 Git 分支历史可能遗漏未提交代码。
激活已有的 SGLang NPU 环境；该环境需要已有 torch、torch_npu、SGLang 推理依赖和 Pillow。
不要在内网执行 pip 下载命令。模型目录需要包含权重、配置和 tokenizer 全部文件。

以下所有命令使用 Bash。打开两个终端，均进入同一个已激活的推理环境。
只需在下面修改一次 `MODEL_PATH`，其他路径按本手册固定。

```bash
mkdir -p /workspace/sensenova-validation
cat > /workspace/sensenova-validation/env.sh <<'SH'
export REPO=/workspace/sglang
export MODEL_PATH=/model/ModelScope/SenseNova/SenseNova-U1.5-8B-MoT
export RESULTS=/workspace/sensenova-validation/results
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
SH
source /workspace/sensenova-validation/env.sh
cd "$REPO"
mkdir -p "$RESULTS"
test -f "$MODEL_PATH/config.json" || { echo '模型路径错误'; exit 1; }
test -f benchmark/sensenova/validate_npu_batching.py || exit 1
npu-smi info | tee "$RESULTS/npu-info.txt"
python3 - <<'PY' | tee "$RESULTS/environment.txt"
import sys, torch, torch_npu, sglang
from PIL import Image
print('Python:', sys.version)
print('Torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('SGLang:', sglang.__file__)
print('NPU available:', torch.npu.is_available())
assert torch.npu.is_available()
assert '/workspace/sglang/python/' in sglang.__file__
PY
```

上一步出现导入失败或断言失败：先修复现有镜像/环境，再继续。若目录带 Git 信息，保存版本：

```bash
git rev-parse HEAD > "$RESULTS/commit.txt"
git diff --binary > "$RESULTS/working-tree.patch"
```

这些命令不记录未跟踪文件；请同时保留本次复制的完整代码包。

### 0.2 创建统一服务启动脚本

```bash
cat > /workspace/sensenova-validation/serve.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
source /workspace/sensenova-validation/env.sh
cd "$REPO"
BATCH=${1:?需要 batch size}
TAG=${2:?需要日志名称}
sglang serve \
  --model-path "$MODEL_PATH" \
  --num-gpus 1 \
  --port 30000 \
  --batching-max-size "$BATCH" \
  --batching-delay-ms 100 \
  --enable-batching-metrics \
  2>&1 | tee "$RESULTS/server-${TAG}.log"
SH
```

NPU 默认启用原生 FIA 去噪 Attention。做 A/B 测试时，在执行 `serve.sh` 前设置：

```bash
# FIA 组
export SGLANG_SENSENOVA_NPU_FIA=1

# SDPA 对照组
export SGLANG_SENSENOVA_NPU_FIA=0
```

当前分支还默认启用 NPU fused RMSNorm 和 dense MLP gate/up 融合。复现未优化
对照时，同时关闭以下三项：

```bash
export SGLANG_SENSENOVA_NPU_FUSED_NORM=0
export SGLANG_SENSENOVA_NPU_FUSED_MLP=0
```

验证完整优化路径时设置：

```bash
export SGLANG_SENSENOVA_NPU_FIA=1
export SGLANG_SENSENOVA_NPU_FUSED_NORM=1
export SGLANG_SENSENOVA_NPU_FUSED_MLP=1
```

MLP 融合会在第一次 NPU 前向时整理权重布局。正式计时前必须完成一次相同分辨率、
步数和 batch size 的完整预热请求。融合路径保留原 checkpoint 参数名，不需要转换
模型文件。

每次修改后必须停止并重新启动服务。FIA 组与 SDPA 组使用相同 batch size、
请求、分辨率、步数和 CFG 参数。

采用当前环境自动选择的 NPU 后端；不带 GPU 分支的 FA3/Triton 参数，不开启入图。
默认 think mode 为 false，每请求一张图。第一轮保持模型默认精度和采样实现。
如果现有环境必须添加 offload 等参数才能启动，请只修改 `serve.sh` 一次，所有 B1/B2/B4 使用同一版本。
`--num-gpus 1` 是框架通用设备数量参数，NPU 同样使用它。
100 ms 是本次统一实验的合批等待窗口，结果会包含该窗口的影响。

每次服务切换：在终端 A 按 Ctrl+C，等待旧进程退出、设备内存释放，再启动新服务。
不允许两个模型服务同时占用该设备；若端口已占用，不继续测试。

## 1. 验证适配可用性：真实批处理路径走通

### 1.1 启动 B2 服务（终端 A）

```bash
bash /workspace/sensenova-validation/serve.sh 2 smoke-b2
```

### 1.2 等待服务就绪（终端 B）

终端 B 每次新开都执行 source：

```bash
source /workspace/sensenova-validation/env.sh
cd "$REPO"
python3 - <<'PY'
import time, urllib.request
for _ in range(600):
    try:
        with urllib.request.urlopen('http://127.0.0.1:30000/health', timeout=5) as r:
            if r.status == 200:
                print('服务已就绪'); break
    except Exception:
        pass
    time.sleep(2)
else:
    raise SystemExit('服务未就绪，请检查终端 A 日志')
PY
```

### 1.3 先验证单请求，再验证并发请求

```bash
python3 benchmark/sensenova/validate_npu_batching.py \
  --model "$MODEL_PATH" --requests 1 --concurrency 1 \
  --size 1024 --steps 5 --cfg 4 --warmup 0 --save-images \
  --output "$RESULTS/smoke-single"

python3 benchmark/sensenova/validate_npu_batching.py \
  --model "$MODEL_PATH" --requests 4 --concurrency 4 \
  --size 1024 --steps 5 --cfg 4 --warmup 0 --save-images \
  --output "$RESULTS/smoke-b2-cfg4"

python3 benchmark/sensenova/validate_npu_batching.py \
  --model "$MODEL_PATH" --requests 4 --concurrency 4 \
  --size 1024 --steps 5 --cfg 1 --warmup 0 --save-images \
  --output "$RESULTS/smoke-b2-cfg1"

grep -n 'Processed dynamic batch of 2/2' "$RESULTS/server-smoke-b2.log"
```

脚本输出目录必须不存在，防止覆盖结果。重跑时给 `--output` 加 `-retry1` 等后缀。
每个目录包含 `result.json`、按请求编号保存的 PNG；JSON 包含对应 prompt 和 seed。

通过标准：

- 三次命令均以退出码 0 结束，分别成功 1、4、4 次，failed=0。
- CFG=1 和 CFG=4 两轮均在对应时间的服务日志中出现真实 `2/2` 合批记录。
- 图片可以打开，尺寸正确，未出现全部黑图或明显损坏。
- 服务无 OOM、运行异常，测试后仍能接收请求。

只有图片成功、没有 `2/2`：结论是“单请求可用，批处理未证实”。先检查复制的代码、实际启动参数和并发量。
必要时只重跑并发测试。不要据此进入性能结论。
1024/5 步仅用于路径检查，图片质量不是该阶段验收指标。

## 2. 验证性能：B1 / B2 / B4

### 2.1 固定实验条件

采用 2048×2048、50 步、CFG=4、每请求一图、客户端并发 4。
三种服务配置唯一变化是最大 batch size。B1 同样使用并发 4，是排队执行的吞吐基线。
脚本使用固定内置 prompt 和 seed，不访问外网，不依赖 vbench。
计时包括 HTTP、图片传输、JSON/base64 解码和图片校验，是客户端端到端吞吐，不是纯算子时间。
正式轮不保存 PNG，结果写入发生在计时结束后。

### 2.2 创建性能测试脚本（终端 B）

```bash
cat > /workspace/sensenova-validation/perf.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO"
BATCH=${1:?需要 batch size}
# 完整并发预热：执行相同分辨率和步数，结果不纳入正式统计。
python3 benchmark/sensenova/validate_npu_batching.py \
  --model "$MODEL_PATH" --requests 4 --concurrency 4 \
  --size 2048 --steps 20 --cfg 4 --warmup 0 \
  --output "$RESULTS/perf-b${BATCH}-warmup"
python3 benchmark/sensenova/validate_npu_batching.py \
  --model "$MODEL_PATH" --requests 4 --concurrency 4 \
  --size 2048 --steps 50 --cfg 4 --warmup 0 \
  --output "$RESULTS/perf-b${BATCH}"
SH
```

### 2.3 依次测试三个服务配置

先停止第一部分服务。每一行都按“终端 A 启动 → 终端 B 执行 1.2 的就绪检查 → 终端 B 测试 → 停止服务”的顺序执行。

| 配置 | 终端 A | 终端 B（就绪后执行） |
|---|---|---|
| B1 | `bash /workspace/sensenova-validation/serve.sh 1 perf-b1` | `bash /workspace/sensenova-validation/perf.sh 1` |
| B2 | `bash /workspace/sensenova-validation/serve.sh 2 perf-b2` | `bash /workspace/sensenova-validation/perf.sh 2` |
| B4 | `bash /workspace/sensenova-validation/serve.sh 4 perf-b4` | `bash /workspace/sensenova-validation/perf.sh 4` |

B4 如 OOM，停止 B4，不更改分辨率或 offload 来强行与前两组比较，记录“该配置容量不足”。
B1/B2 的结果仍可分析。所有轮次 failed 必须为 0；有失败的轮次不进入有效吞吐比较。

完成 B2/B4 后检查真实批次：

```bash
grep -n 'Processed dynamic batch of' "$RESULTS/server-perf-b2.log"
grep -n 'Processed dynamic batch of' "$RESULTS/server-perf-b4.log"
npu-smi info | tee "$RESULTS/npu-info-after.txt"
```

最大 batch=4 不等于实际每次 B4。需要日志出现 `4/4`，否则标为“最大 B4，实际混合批次”。
日志包含预热和正式轮，检查正式轮对应区间，不可仅凭预热的合批记录认定正式轮已合批。

### 2.4 自动汇总单轮结果

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['RESULTS'])
baseline = None
print('| 最大Batch | 吞吐(images/s) | 相对B1提升 | Mean延迟(s) | P95延迟(s) | 报告峰值内存(MB) |')
print('|---|---|---|---|---|---|')
for b in (1, 2, 4):
    path = root / f'perf-b{b}/result.json'
    if not path.exists():
        print(f'| {b} | 未完成 | — | — | — | — |'); continue
    row = json.loads(path.read_text())
    if row['failed'] or row['successful'] != 4:
        print(f'| {b} | 有失败，结果无效 | — | — | — | — |'); continue
    rate = row['outputs_per_s']
    if b == 1: baseline = rate
    gain = f'{(rate / baseline - 1) * 100:+.2f}%' if baseline else '缺少B1'
    mean = row['mean_latency_s']
    p95 = row['p95_latency_s']
    mem = [r['peak_memory_mb'] for r in row['requests'] if r.get('peak_memory_mb') is not None]
    print(f'| {b} | {rate:.5f} | {gain} | {mean:.2f} | {p95:.2f} | {max(mem) if mem else "未上报"} |')
PY
```

服务上报的 peak memory 可能是进程累计峰值，不应解释为每请求独占显存，也不能累加。
独立重启使不同 batch 配置的峰值更可比较；未上报表示缺失，不表示零。
4 请求用于开发阶段的固定负载 A/B，P95 仅代表本轮最慢请求。正式 SLA 需要更大样本量。

结论写法：

- 第一部分通过：批处理路径可用（不代表完整数值正确性已通过）。
- B2/B4 单轮成功且正式轮日志证实合批：可报告表中吞吐提升，并保留实际批次分布。
- 提升 ≥10% 可作为本次开发目标达到的参考；接近零或波动大时保留数据，不宣称加速。
- 必须随结果附上代码包、环境信息、全部 server 日志及结果 JSON。

### 2.5 910C 实测结果

以下结果使用单卡910C、4个并发请求、2048×2048、50步、CFG=4。优化配置开启
FIA、fused RMSNorm 和 fused MLP；未启用未证明有实际价值的 QKV 与 token-type
fast path。

| 配置 | 总耗时(s) | 吞吐(images/s) | Mean延迟(s) | P95延迟(s) | 峰值内存(MB) |
|---|---:|---:|---:|---:|---:|
| B1，优化全关 | 234.28 | 0.01707 | 146.47 | 234.27 | 35698 |
| B2，优化全关 | 234.83 | 0.01703 | 176.06 | 234.83 | 37870 |
| B1，FIA+Norm+MLP | 205.94 | 0.01942 | 128.80 | 205.94 | 35718 |
| B2，FIA+Norm+MLP | 195.02 | 0.02051 | 146.33 | 195.01 | 37850 |

原始算子下 B2 相对 B1 吞吐变化为 -0.24%。最终优化配置下，B2 相对 B1 吞吐提升
5.60%，总耗时降低5.31%。最终 B2 相对 B1 全关对照吞吐提升20.13%，总耗时和
P95均降低16.76%，Mean延迟基本不变，峰值 reserved memory 增加约2.15 GB。
因此该实现的主要收益来自 NPU 算子路径，动态批处理在优化路径上提供额外的吞吐收益。

这些“优化全关”结果来自当前分支，不等同于上游 main 基线。

### 2.6 导出报告包

```bash
cd /workspace/sensenova-validation
tar -czf sensenova-batching-results.tar.gz env.sh serve.sh perf.sh results
```

本手册不要求运行 GPU 验证，也不包含入图、I2I 或多设备验证。
