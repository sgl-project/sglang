# 使用说明

## Phase 1.1: 延迟测试脚本使用指南

### 前置准备

#### 1. 启动服务器

**SGLang 后端**:
```bash
sglang serve \
    --model-path black-forest-labs/FLUX.2-klein-4B \
    --backend sglang \
    --port 30000
```

**Diffusers 后端**:
```bash
sglang serve \
    --model-path black-forest-labs/FLUX.2-klein-4B \
    --backend diffusers \
    --port 30000
```

#### 2. 安装依赖

```bash
pip install openai numpy
```

### 基本使用

#### 测试单个后端

```bash
# 测试 SGLang 后端（默认配置）
python bench_flux_klein.py --backend sglang --port 30000

# 测试 Diffusers 后端
python bench_flux_klein.py --backend diffusers --port 30000
```

#### 测试两个后端

```bash
# 需要分别启动两个服务器在不同端口
# Terminal 1: SGLang 后端
sglang serve --model-path black-forest-labs/FLUX.2-klein-4B --backend sglang --port 30000

# Terminal 2: Diffusers 后端
sglang serve --model-path black-forest-labs/FLUX.2-klein-4B --backend diffusers --port 30001

# Terminal 3: 运行测试（需要修改脚本支持不同端口，或分别运行）
python bench_flux_klein.py --backend sglang --port 30000
python bench_flux_klein.py --backend diffusers --port 30001
```

### 自定义配置

#### 快速测试（减少测试配置）

```bash
python bench_flux_klein.py \
    --backend sglang \
    --batch-sizes 1 2 \
    --resolutions 512x512,768x768 \
    --num-inference-steps 20 30 \
    --num-runs 5 \
    --port 30000
```

#### 完整测试（默认配置）

```bash
python bench_flux_klein.py \
    --backend sglang \
    --batch-sizes 1 2 4 8 \
    --resolutions 512x512,768x768,1024x1024 \
    --num-inference-steps 20 30 50 \
    --num-runs 10 \
    --port 30000 \
    --output-dir benchmark/results
```

### 输出结果

测试结果保存在 `benchmark/results/` 目录（或指定的 `--output-dir`）：

- `sglang_latency_results.json` - SGLang 后端结果
- `diffusers_latency_results.json` - Diffusers 后端结果
- `combined_latency_results.json` - 合并结果（如果测试两个后端）

### 结果格式

每个结果文件包含：

```json
{
  "backend": "sglang",
  "model_path": "black-forest-labs/FLUX.2-klein-4B",
  "prompt": "A beautiful landscape with mountains and lakes",
  "test_config": {
    "batch_sizes": [1, 2, 4, 8],
    "resolutions": ["512x512", "768x768", "1024x1024"],
    "num_inference_steps": [20, 30, 50],
    "num_runs": 10
  },
  "results": [
    {
      "config": {
        "batch_size": 1,
        "width": 512,
        "height": 512,
        "resolution": "512x512",
        "num_inference_steps": 30
      },
      "latency_stats": {
        "mean": 2.345,
        "median": 2.301,
        "std": 0.123,
        "min": 2.101,
        "max": 2.567,
        "p50": 2.301,
        "p90": 2.512,
        "p99": 2.545
      },
      "success_rate": 1.0,
      "total_runs": 10,
      "successful_runs": 10,
      "failed_runs": 0,
      "errors": [],
      "raw_latencies": [2.301, 2.345, ...]
    }
  ]
}
```

### 注意事项

1. **服务器必须运行**: 确保服务器在指定端口运行
2. **显存要求**: FLUX.2-Klein 需要约 16-20GB 显存
3. **测试时间**: 完整测试可能需要较长时间（取决于配置）
4. **网络连接**: 确保可以访问 HuggingFace 下载模型（如果未缓存）

### 故障排除

#### 连接错误
```
Error: Failed to connect to server
```
- 检查服务器是否运行: `curl http://localhost:30000/v1/models`
- 检查端口是否正确

#### 模型加载错误
```
Error: Model not found
```
- 检查模型路径是否正确
- 确保有 HuggingFace 访问权限

#### 显存不足
```
Error: CUDA out of memory
```
- 减少批次大小: `--batch-sizes 1`
- 使用较小分辨率: `--resolutions 512x512`
- 减少推理步数: `--num-inference-steps 20`
