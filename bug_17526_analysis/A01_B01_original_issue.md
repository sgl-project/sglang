# A01_B01: Issue #17526 原始内容

## 相关文档
- [A01: GLM Blackwell性能优化详解](./A01_glm_blackwell_optimization.md) - 了解整体问题
- [A01_B02: 优化项详解](./A01_B02_optimization_items.md) - 每个优化项的详细说明
- [A01_B03: 性能分析](./A01_B03_performance_analysis.md) - 性能对比和瓶颈分析

---

## Issue 链接
https://github.com/sgl-project/sglang/issues/17526

## 问题标题
[Tracking] GLM 4.5/4.6/4.7 Blackwell performance optimization

## 提交者
@b8zhong (Collaborator)

## 创建时间
opened 4 days ago · edited by b8zhong

## 动机 (Motivation)

Focus on GLM 4.7 FP4 (mainly for Blackwell) and latency scenario.

But can also consider compressed-tensors FP8 on Hopper/Blackwell as well

**相关人员**: @ynwang007 @pdasgup @vincentzed

---

## 优化项列表 (Items)

### 1. [Fix] GLM 4.7 + NVFP4 + MTP #17166
- **描述**: GLM + flashinfer_trtllm MoE backend
- **性能提升**: 10% perf gain
- **要求**: Needs Flashinfer 0.6.2 or nightly to be automatically set
- **状态**: ✅ Fixed

### 2. [Fix] Auto-enable TRT-LLM MHA when speculative topk = 1 #16755
- **描述**: Make sure MHA + topk=1 speculative decoding still sets trtllm_mha as attention backend
- **性能提升**: 10% perf gain
- **状态**: ✅ Fixed
- **注意**: Check [Bug] Very slow trtllm_allreduce_fusion #17237, GLM has --enable-flashinfer-allreduce-fusion auto-enabled

### 3. Fuse the FP8 KV buffer kernel with Q cast to FP8 under FP8 attention
- **描述**: 融合FP8 KV buffer kernel和Q cast到FP8的操作
- **性能提升**: est. 3% performance gain
- **状态**: 🔄 In Progress

### 4. Improve sgl-kernel scaled_fp4_quant
- **描述**: 改进sgl-kernel的scaled_fp4_quant，可能切换到flashinfer.fp4_quantization.fp4_quantize的CUTLASS backend
- **性能提升**: very small of E2E but maybe 1-2%
- **状态**: 🔄 In Progress

### 5. Use better scale layout for --fp4-gemm-backend
- **描述**: 为非常低的并发场景使用更好的scale layout（可能backend: flashinfer_trtllm）
- **性能提升**: maybe 5-10%
- **状态**: 🔄 In Progress
- **备注**: @baonudesifeizhai commented: "Not implemented: Dynamic scale layout selection based on concurrency level, with special optimizations for low concurrency scenarios -- i can do that..."

### 6. Fuse two elementwise in FlashinferFP4MoE
- **描述**: 融合FlashinferFP4MoE中的两个elementwise操作
- **性能提升**: not sure yet what they are
- **状态**: 🔄 In Progress

### 7. [Nvidia] Add trtllm mnnvl allreduce #12787
- **描述**: General but include because important, can achieve better performance in single node as well, to replace TRT Allreduce Fusion
- **状态**: 🔄 In Progress

### 8. FP8 KV cache性能问题
- **问题**: FP8 KV cache总是更慢，因为额外的量化操作
- **观察**: 52 us vs 47 us (BF16)
- **即使在高并发(concurrency=1024)**: 仍然不更好 (output throughput: 4384.482 vs 4893.194 tps)
- **瓶颈**: 额外的量化和数据shuffle kernel，包括：
  - DeviceGemmFp4GemmSm100
  - cvt_fp16_to_fp4
  - float8_copy_kernel_cuda
  - _fused_fp8_set_kv_buffer_kernel
- **总开销**: 这些额外步骤及其启动开销总计约23us

---

## 性能测试结果

### Profile命令
```bash
SGLANG_TORCH_PROFILER_DIR="./" \
python -m sglang.bench_one_batch_server \
  --model baseten-admin/glm-4.7-fp8-attn-fp4-mlp \
  --base-url http://localhost:30000 \
  --batch-size 4 \
  --input-len 2048 \
  --output-len 1024 \
  --profile \
  --profile-steps 10 \
  --show-report \
  --profile-by-stage
```

### 简单BS=1测试命令
```bash
python3 -m sglang.test.send_one --stream --max-new-tokens 2048 --prompt "Fully explain a linear layer from scratch."
```

---

## 准确性测试结果

### FP8 attention + FP8 KV cache
**配置**:
```bash
python3 -m sglang.launch_server \
  --model-path baseten-admin/glm-4.7-fp8-attn-fp4-mlp \
  --trust-remote-code \
  --tp 4 \
  --quantization modelopt_fp4 \
  --moe-runner-backend flashinfer_trtllm \
  --attention-backend trtllm_mha \
  --kv-cache-dtype fp8_e4m3 \
  --cuda-graph-max-bs 1300
```

**结果**:
- Accuracy: 0.964
- Invalid: 0.000
- Latency: 24.742 s
- Output throughput: 4886.038 token/s

### BF16 attention + BF16 KV cache
**配置**:
```bash
python3 -m sglang.launch_server \
  --model-path baseten-admin/glm-4.7-fp4 \
  --trust-remote-code \
  --tp 4 \
  --quantization modelopt_fp4 \
  --moe-runner-backend flashinfer_trtllm \
  --attention-backend trtllm_mha \
  --cuda-graph-max-bs 1300
```

**结果**:
- Accuracy: 0.970
- Invalid: 0.000
- Latency: 20.990 s
- Output throughput: 6017.676 token/s

---

## MTP性能测试

### MTP + FP8 KV cache
**配置**:
```bash
SGLANG_ENABLE_SPEC_V2=1 \
python3 -m sglang.launch_server \
  --model-path baseten-admin/glm-4.7-fp8-attn-fp4-mlp \
  --trust-remote-code \
  --tp 4 \
  --quantization modelopt_fp4 \
  --moe-runner-backend flashinfer_trtllm \
  --attention-backend trtllm_mha \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm EAGLE
```

**结果**:
- Latency: 7.513 s
- Tokens: 2048
- Acc Length: 3.061
- Speed: 272.60 token/s

**Trace文件**:
- `glm-4.7-fp4_tp4_bs4_input2048_kvfp8_flashinfer_trtllm_mha_specv2_eagle_tp0_extend.trace.json.gz`
- `glm-4.7-fp4_tp4_bs4_input2048_kvfp8_flashinfer_trtllm_mha_specv2_eagle_tp0_decode.trace.json.gz`

### MTP + BF16 KV cache
**配置**:
```bash
SGLANG_ENABLE_SPEC_V2=1 \
python3 -m sglang.launch_server \
  --model-path baseten-admin/glm-4.7-fp4 \
  --trust-remote-code \
  --tp 4 \
  --quantization modelopt_fp4 \
  --moe-runner-backend flashinfer_trtllm \
  --attention-backend trtllm_mha \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm EAGLE
```

**结果**:
- Latency: 7.213 s
- Tokens: 2048
- Acc Length: 3.200 (Accept length更高)
- Speed: 283.93 token/s

**观察**: Accept length更高，所以在batch size或sequence length较低时可能更高效

**Trace文件**:
- `glm-4.7-fp4_tp4_bs4_input2048_kvbf16_flashinfer_trtllm_mha_specv2_eagle_tp0_extend.trace.json.gz`
- `glm-4.7-fp4_tp4_bs4_input2048_kvbf16_flashinfer_trtllm_mha_specv2_eagle_tp0_decode.trace.json.gz`

### No MTP
**配置**:
```bash
python3 -m sglang.launch_server \
  --model-path baseten-admin/glm-4.7-fp8-attn-fp4-mlp \
  --trust-remote-code \
  --tp 4 \
  --quantization modelopt_fp4 \
  --moe-runner-backend flashinfer_trtllm \
  --attention-backend trtllm_mha \
  --kv-cache-dtype fp8_e4m3
  # --disable-cuda-graph disable for profiling for viewing kernels
```

**结果**:
- Latency: 17.643 s
- Tokens: 2048
- Acc Length: 1.000
- Speed: 116.08 token/s

**Trace文件**:
- `glm-4.7-fp4_tp4_bs4_input2048_kvfp8_flashinfer_trtllm_mha_nocudagraph_tp0_extend.trace.json.gz`

---

## 性能瓶颈分析

### FP8 KV Cache的问题

**观察**:
- Attention kernel耗时更少（符合预期）
- 但整体性能下降，因为额外的量化和数据shuffle操作

**瓶颈kernel**:
1. `DeviceGemmFp4GemmSm100`
2. `cvt_fp16_to_fp4`
3. `float8_copy_kernel_cuda`
4. `_fused_fp8_set_kv_buffer_kernel`

**总开销**: 这些额外步骤及其启动开销总计约23us

**对比**:
- BF16: 47 us
- FP8: 52 us (多5us，但加上额外操作约23us)

---

## 标签

- `performance` - 性能相关
- `blackwell` - Blackwell GPU
- `SM100/SM120` - GPU架构
- `collaboration` - 协作项目
- `good first issue` - 适合新手的issue

---

## 活动记录

- **b8zhong**: added performance, blackwell, SM100/SM120 (4 days ago)
- **baonudesifeizhai**: commented about dynamic scale layout (4 days ago)
- **b8zhong**: self-assigned this (4 days ago)
- **b8zhong**: added collaboration (4 days ago)
- **tcapelle**: asked about docker image (3 days ago)
- **Fridge003**: added good first issue (2 days ago)

---

## 关键发现

1. **FP8 KV cache性能问题**: 额外的量化操作导致性能下降
2. **MTP性能提升**: 使用MTP可以显著提升性能（从116 token/s到272-283 token/s）
3. **BF16 vs FP8**: BF16在accuracy和throughput上都略优于FP8
4. **优化空间**: 多个优化项可以进一步提升性能，总计可能达到30%+的提升
