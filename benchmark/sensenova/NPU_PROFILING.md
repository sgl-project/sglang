# SenseNova B1/B2 短步骤 NPU profiling

目标：比较真实模型 B1/B2 的单步算子执行，定位批处理没有吞吐收益的原因。
无需启动 HTTP 服务。每组执行 3 步预热、3 步正式生成，只记录正式生成第 2 步。
捕获范围为图像特征提取开始，到下一步图像特征提取之前，包含条件/无条件
forward、CFG 和图像更新；不包含 prefix、排队、HTTP、图片编码。
这是短步数下的算子诊断，不能用 profiling 耗时替代 50 步端到端性能结果。

## 1. 准备

停止已有模型服务，释放 NPU。进入已有 torch_npu 推理环境。
将当前分支代码（包含之前的 NPU mask 修复）及两个脚本复制到内网：

- `benchmark/sensenova/profile_npu_batching.py`
- `benchmark/sensenova/validate_npu_batching.py`（提供固定 prompt，放在同一目录）

以下在 Bash 执行，只需根据实际位置修改 REPO 和 MODEL_PATH：

```bash
export REPO=/sgl-workspace/sglang
export MODEL_PATH=/nas/disk1/SenseNova/SenseNova-U1.5-8B-MoT/
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$REPO"
npu-smi info
mkdir -p /workspace/sensenova-profile
```

脚本使用逻辑 NPU 0、BF16、CFG=4、timestep_shift=3、2048²，无 offload/入图。
模型和 tokenizer 均只从本地读取。脚本没有在真实 NPU 上验证过，依赖当前镜像
提供 `torch_npu.profiler.profile` 和 `export_chrome_trace`。

## 2. 顺序执行 B1、B2

```bash
python3 benchmark/sensenova/profile_npu_batching.py \
  --model "$MODEL_PATH" --batch-size 1 \
  --output /workspace/sensenova-profile/b1

python3 benchmark/sensenova/profile_npu_batching.py \
  --model "$MODEL_PATH" --batch-size 2 \
  --output /workspace/sensenova-profile/b2
```

每次进程退出会释放模型；不要并行运行。输出目录必须不存在。
模型加载耗时另计，生成总工作量每组只有 6 步。不是 16 请求，也不需要 50 步。

## 3. 文件大小和导出

关闭调用栈和内存事件采集，保留 shape、CPU 和 NPU 时间线。
每组只保存 `trace.json.gz` 与 `metadata.json`，每组硬限制 14,000,000 bytes，
两组总量小于 28 MB，留出打包余量。完整事件不会被抽样或截断。
限制针对最终压缩结果；profiler 采集期间的内存和临时原始 JSON 不受此限制。

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/workspace/sensenova-profile')
files = [p/b/n for b in ('b1','b2') for n in ('trace.json.gz','metadata.json')]
assert all(f.is_file() for f in files), '缺少结果，请检查脚本是否成功'
total = sum(f.stat().st_size for f in files)
assert total < 30_000_000, f'结果超限：{total}'
print('结果总字节数:', total)
PY
tar -czf /workspace/sensenova-profile-results.tar.gz \
  -C /workspace/sensenova-profile b1 b2
python3 - <<'PY'
from pathlib import Path
p=Path('/workspace/sensenova-profile-results.tar.gz')
assert p.stat().st_size < 30_000_000
print(p, p.stat().st_size, 'bytes')
PY
```

如果脚本报告超过 14 MB，不会保存残缺 trace。改为两组均加 `--size 1024`，
使用新的输出父目录，再按相同方式检查打包。1024² 结果只用于该尺寸诊断，
不能直接解释所有 2048² 性能现象。若仍超限，保留报错信息再调整采集范围。

回传 `sensenova-profile-results.tar.gz`。重点对比 Attention、矩阵乘、图像特征
提取、数据搬运/布局转换，以及 CPU 与 NPU 的空闲间隙。

## 4. B2 等长前缀对照

更新 `profile_npu_batching.py` 后，保持原环境，停止已有服务，执行：

```bash
cd "$REPO"
python3 benchmark/sensenova/profile_npu_batching.py \
  --model "$MODEL_PATH" --batch-size 2 --same-prompt \
  --output /workspace/sensenova-profile/b2-same-long
```

两个样本均使用原 B2 的较长 prompt，seed 仍为 1000、1001。
因此最大前缀长度与原 B2 相同，但不存在 padding，代码会选择无显式 mask 的路径。
仍为 3 步预热、3 步生成、只抓第 2 步；单组文件上限仍为 14 MB。
回传新目录中的 `trace.json.gz` 和 `metadata.json`。
优先比较条件分支的 42 次主 FlashAttention 调用总时间；无条件分支作为参考。
该实验同时改变了文本内容及 mask 路径，只能提供诊断证据，不能当成 mask 的严格独立因果实验。

## 5. NPU 优化前后对照

使用相同代码分别关闭和开启 FIA、fused RMSNorm 和 fused MLP。两组都使用 B2、
相同 prompt、2048²、3 步，并只采集第 2 个去噪步。每次命令使用新进程，
避免 lazy packed MLP 权重影响基线。

```bash
export SGLANG_SENSENOVA_NPU_FIA=0
export SGLANG_SENSENOVA_NPU_FUSED_MLP=0
export SGLANG_SENSENOVA_NPU_FUSED_NORM=0
python3 benchmark/sensenova/profile_npu_batching.py \
  --model "$MODEL_PATH" --batch-size 2 --same-prompt \
  --output /workspace/sensenova-profile/dense-baseline-b2

export SGLANG_SENSENOVA_NPU_FIA=1
export SGLANG_SENSENOVA_NPU_FUSED_MLP=1
export SGLANG_SENSENOVA_NPU_FUSED_NORM=1
python3 benchmark/sensenova/profile_npu_batching.py \
  --model "$MODEL_PATH" --batch-size 2 --same-prompt \
  --output /workspace/sensenova-profile/dense-optimized-b2
```

先确认两次生成都成功且结果中没有 NaN/Inf。MLP gate/up 融合使每步 dense MatMul
调用数预计由 588 降至约 504 次。算子名称可能随 CANN 版本变化，因此同时比较
Attention、RMSNorm、MatMul 的总次数和总时间，以及完整 denoise step wall time。
将这两个目录打包，压缩结果应小于30 MB。
