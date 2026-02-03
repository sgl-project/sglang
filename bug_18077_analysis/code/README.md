# Code 目录

此目录用于存放与 Issue #18077 相关的代码文件。

## 文件说明

### 基准测试脚本

#### bench_flux_klein.py ⭐ **Phase 1.1: 延迟测试**
- **功能**: 执行 FLUX.2-Klein 的延迟基准测试
- **测试内容**:
  - 不同批次大小 (1, 2, 4, 8)
  - 不同分辨率 (512x512, 768x768, 1024x1024)
  - 不同推理步数 (20, 30, 50)
  - 对比 SGLang 和 Diffusers 后端
- **输出**: JSON 格式的结果文件，包含延迟统计信息

**使用方法**:
```bash
# 测试 SGLang 后端
python bench_flux_klein.py --backend sglang --port 30000

# 测试 Diffusers 后端
python bench_flux_klein.py --backend diffusers --port 30000

# 测试两个后端
python bench_flux_klein.py --backend both --port 30000

# 自定义配置
python bench_flux_klein.py \
    --backend sglang \
    --batch-sizes 1 2 4 \
    --resolutions 512x512,768x768 \
    --num-inference-steps 20 30 \
    --num-runs 5 \
    --output-dir results
```

**前置条件**:
1. 启动 SGLang 服务器:
   ```bash
   # SGLang 后端
   sglang serve \
       --model-path black-forest-labs/FLUX.2-klein-4B \
       --backend sglang \
       --port 30000

   # Diffusers 后端
   sglang serve \
       --model-path black-forest-labs/FLUX.2-klein-4B \
       --backend diffusers \
       --port 30000
   ```

2. 安装依赖:
   ```bash
   pip install openai numpy
   ```

### 待实现的脚本

- `profile_flux_klein.py` - 性能分析脚本（Phase 2）
- `compare_backends.py` - 后端对比脚本
- `analyze_sp_support.py` - SP 支持分析脚本（Phase 3）

## 使用说明

这些脚本将用于：
1. 基准测试 FLUX.2-Klein 的性能
2. 对比 SGLang 和 Diffusers 后端
3. 实施和测试优化方案

## 输出结果

测试结果将保存在 `benchmark/results/` 目录下：
- `sglang_latency_results.json` - SGLang 后端延迟测试结果
- `diffusers_latency_results.json` - Diffusers 后端延迟测试结果
- `combined_latency_results.json` - 合并结果（如果测试两个后端）
