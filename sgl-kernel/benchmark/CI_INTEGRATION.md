# sgl-kernel Benchmark CI集成

## 概述

本文档描述了sgl-kernel benchmark文件与GitHub Actions CI的集成。

## 修改内容

### 1. GitHub Actions Workflow更新

在`.github/workflows/pr-test.yml`中添加了新的job：

```yaml
sgl-kernel-benchmark-test:
  needs: [check-changes, sgl-kernel-build-wheels]
  if: needs.check-changes.outputs.sgl_kernel == 'true'
  runs-on: 1-gpu-runner
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
    CI: true  # 确保CI环境变量设置为true
```

### 2. CI环境检测

所有23个benchmark文件都已添加CI环境检测：

```python
import os

# CI environment detection
IS_CI = os.getenv("CI", "false").lower() == "true" or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
```

### 3. 参数简化

在CI环境下，所有benchmark使用简化参数：

- **批次大小**: 减少到1-2个值
- **序列长度**: 减少到1-2个值  
- **模型配置**: 只使用第一个配置
- **迭代次数**: 限制在前2个配置

### 4. 测试流程

CI测试流程：

1. **环境设置**: 设置`CI=true`环境变量
2. **依赖安装**: 安装sgl-kernel和相关依赖
3. **批量测试**: 运行所有`bench_*.py`文件
4. **超时控制**: 每个文件最多运行5分钟
5. **错误处理**: 失败的文件会记录但不中断整体流程

## 运行方式

### 本地测试CI模式

```bash
# 设置CI环境变量
export CI=true

# 运行单个benchmark
python bench_activation.py

# 或者设置GITHUB_ACTIONS
export GITHUB_ACTIONS=true
python bench_rmsnorm.py
```

### CI自动触发

当PR修改了`sgl-kernel/**`目录下的文件时，CI会自动运行benchmark测试。

## 特性

### ✅ 已实现功能

1. **自动CI检测**: 通过环境变量自动识别CI环境
2. **参数简化**: CI环境下使用最小参数集
3. **可选依赖**: vLLM等外部依赖的优雅处理
4. **超时保护**: 防止单个测试运行过长时间
5. **错误容忍**: 单个测试失败不影响整体流程

### 🎯 设计目标

1. **快速验证**: 在CI中快速验证benchmark脚本的基本功能
2. **资源节约**: 使用最少的计算资源完成测试
3. **稳定性**: 确保CI流程的稳定性和可靠性
4. **可维护性**: 易于添加新的benchmark文件

## 文件列表

所有23个benchmark文件都已更新：

1. `bench_activation.py`
2. `bench_awq_dequant.py`
3. `bench_cutlass_mla.py`
4. `bench_dsv3_fused_a_gemm.py`
5. `bench_dsv3_router_gemm.py`
6. `bench_fp4_gemm.py`
7. `bench_fp8_blockwise_gemm.py`
8. `bench_fp8_blockwise_group_gemm.py`
9. `bench_fp8_gemm.py`
10. `bench_int8_gemm.py`
11. `bench_lightning_attention_decode.py`
12. `bench_moe_align_block_size.py`
13. `bench_moe_ep_post_reorder.py`
14. `bench_moe_fused_gate.py`
15. `bench_moe_topk_softmax.py`
16. `bench_nvfp4_scaled_gemm.py`
17. `bench_per_tensor_quant_fp8.py`
18. `bench_per_token_group_quant_8bit.py`
19. `bench_per_token_quant_fp8.py`
20. `bench_qserve_w4a8_gemm.py`
21. `bench_rmsnorm.py`
22. `bench_rotary_embedding.py`
23. `bench_top_k_top_p_sampling.py`

## 注意事项

1. **环境变量**: CI环境必须设置`CI=true`或`GITHUB_ACTIONS=true`
2. **依赖管理**: 确保CI环境安装了所有必要的依赖
3. **GPU资源**: benchmark测试需要GPU环境
4. **超时设置**: 单个测试超时时间为5分钟

---

**更新时间**: 2024年12月  
**状态**: ✅ 完成  
**覆盖率**: 100% (23/23 文件)
