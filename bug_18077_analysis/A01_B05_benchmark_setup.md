# A01_B05: 基准测试设置

## 📋 基准测试脚本

### 1. SGLang Backend 基准测试

#### 使用 sglang serve
```bash
# 启动服务器
sglang serve \
    --model-path zai-org/GLM-Image \
    --backend sglang \
    --port 30000

# 运行基准测试
python -m sglang.bench_serving \
    --backend sglang \
    --model zai-org/GLM-Image \
    --num-prompts 10 \
    --max-concurrency 1
```

#### 使用自定义脚本
```bash
python benchmark/glm_image_benchmark/bench_glm_image.py \
    --model-path zai-org/GLM-Image \
    --backend sglang \
    --batch-sizes 1 2 4 8 \
    --resolutions 512x512 768x768 1024x1024 \
    --num-runs 10 \
    --output-dir results
```

### 2. Diffusers Backend 基准测试

#### 使用 sglang serve (diffusers backend)
```bash
# 启动服务器
sglang serve \
    --model-path zai-org/GLM-Image \
    --backend diffusers \
    --port 30000

# 运行基准测试
python -m sglang.bench_serving \
    --backend diffusers \
    --model zai-org/GLM-Image \
    --num-prompts 10 \
    --max-concurrency 1
```

#### 使用自定义脚本
```bash
python benchmark/glm_image_benchmark/bench_glm_image.py \
    --model-path zai-org/GLM-Image \
    --backend diffusers \
    --batch-sizes 1 2 4 8 \
    --resolutions 512x512 768x768 1024x1024 \
    --num-runs 10 \
    --output-dir results
```

## 🔧 性能分析设置

### PyTorch Profiler

```bash
# 设置 trace 路径
export SGLANG_TORCH_PROFILER_DIR=/path/to/profile_log

# 启动服务器
sglang serve --model-path zai-org/GLM-Image --backend sglang

# 发送 profiling 请求
python benchmark/glm_image_benchmark/profile_glm_image.py \
    --model-path zai-org/GLM-Image \
    --backend sglang \
    --prompt "A beautiful landscape" \
    --resolution 512x512 \
    --output-dir profile_results
```

### 查看 Profiling 结果

```bash
# 使用 Chrome 的 chrome://tracing
# 打开 profile_results/trace.json
```

## 📊 测试配置

### 标准测试配置

#### 批次大小测试
```python
batch_sizes = [1, 2, 4, 8]
```

#### 分辨率测试
```python
resolutions = [
    (512, 512),
    (768, 768),
    (1024, 1024),
    (1280, 1280),  # 如果支持
]
```

#### 推理步数测试
```python
num_inference_steps = [20, 30, 50]
```

#### 并发测试
```python
concurrency_levels = [1, 4, 8, 16]
```

### 环境要求

#### 硬件
- GPU: NVIDIA A100 (80GB) 或类似
- 内存: 至少 64GB RAM
- 存储: 足够的空间存储模型和结果

#### 软件
- Python: 3.8+
- PyTorch: 2.0+
- SGLang: 最新版本
- CUDA: 11.8+

## 📈 结果收集

### 指标收集

#### 延迟指标
- 平均延迟
- 中位数延迟
- P50, P90, P99 延迟
- 最小/最大延迟

#### 吞吐量指标
- 请求/秒 (req/s)
- 令牌/秒 (tok/s) - 如果适用
- 图像/秒 (img/s)

#### 内存指标
- 峰值内存使用
- 平均内存使用
- 内存分配模式

#### GPU 指标
- GPU 利用率
- 内存带宽利用率
- 内核执行时间

### 结果格式

#### JSON 格式
```json
{
    "backend": "sglang",
    "model_path": "zai-org/GLM-Image",
    "config": {
        "batch_size": 1,
        "resolution": [512, 512],
        "num_inference_steps": 30
    },
    "metrics": {
        "latency": {
            "mean": 9.22,
            "median": 9.02,
            "p99": 10.85
        },
        "throughput": {
            "req_per_sec": 0.11
        },
        "memory": {
            "peak_mb": 8170.82
        }
    }
}
```

## 🔍 自动化测试

### 完整测试套件

```bash
# 运行完整测试套件
./benchmark/glm_image_benchmark/run_full_benchmark.sh
```

### 持续集成

可以考虑添加 CI 测试来：
- 定期运行基准测试
- 检测性能回归
- 生成性能报告

## 📝 结果报告

### 对比报告生成

```bash
python benchmark/glm_image_benchmark/compare_backends.py \
    --sglang-results results/sglang_results.json \
    --diffusers-results results/diffusers_results.json \
    --output comparison_report.md
```

### 报告内容

- 性能对比表格
- 可视化图表
- 瓶颈分析
- 优化建议

---

**状态**: 📝 待实施 - 需要创建实际的基准测试脚本
