# 多GPU和高并发测试使用说明

## 更新内容

### 1. 支持多GPU并行
- `01_start_server.sh` 现在支持 `--num-gpus` 和 `--enable-cfg-parallel` 参数
- 默认使用单GPU（保持向后兼容）

### 2. 支持更高并发数
- `02_run_all_tests.sh` 默认并发数从 `1,2` 改为 `1,2,4,8,16`
- 可以通过 `--max-concurrency` 参数自定义

### 3. 结果目录分离
- 可以通过 `--result-subdir` 参数指定结果子目录
- 单GPU测试：保存到 `benchmark/results/`
- 多GPU测试：保存到 `benchmark/results_multi_gpu/`（或其他自定义名称）

---

## 使用方法

### 启动多GPU服务器

```bash
# 使用2个GPU，启用CFG并行
./01_start_server.sh sglang 2 true

# 使用4个GPU，不启用CFG并行
./01_start_server.sh sglang 4 false

# 单GPU（默认，向后兼容）
./01_start_server.sh sglang
# 或
./01_start_server.sh sglang 1 false
```

**参数说明**：
- 第1个参数：后端（sglang/diffusers）
- 第2个参数：GPU数量（默认1）
- 第3个参数：是否启用CFG并行（true/false，默认false）

---

### 运行高并发测试

#### 单GPU测试（保存到默认目录）

```bash
./02_run_all_tests.sh --backend sglang --port 30000 --dataset random \
  --num-prompts 10 --widths 512,1024 --heights 512,1024 \
  --max-concurrency 1,2,4,8,16 --request-rate inf \
  --model zai-org/GLM-Image
```

#### 多GPU测试（保存到独立目录）

```bash
# 使用多GPU结果目录
./02_run_all_tests.sh --backend sglang --port 30000 --dataset random \
  --num-prompts 10 --widths 512,1024 --heights 512,1024 \
  --max-concurrency 1,2,4,8,16 --request-rate inf \
  --model zai-org/GLM-Image \
  --result-subdir multi_gpu
```

**结果目录**：
- 单GPU：`benchmark/results/`
- 多GPU：`benchmark/results_multi_gpu/`

---

## 完整测试流程示例

### 场景1：单GPU高并发测试

```bash
# 1. 启动单GPU服务器
./01_start_server.sh sglang

# 2. 运行测试（默认并发 1,2,4,8,16）
./02_run_all_tests.sh --backend sglang --port 30000 --dataset random \
  --num-prompts 10 --widths 512,1024 --heights 512,1024 \
  --max-concurrency 1,2,4,8,16 --request-rate inf \
  --model zai-org/GLM-Image

# 3. 停止服务器
./04_stop_server.sh 30000

# 4. 切换到Diffusers后端
./01_start_server.sh diffusers

# 5. 运行Diffusers测试
./02_run_all_tests.sh --backend diffusers --port 30000 --dataset random \
  --num-prompts 10 --widths 512,1024 --heights 512,1024 \
  --max-concurrency 1,2,4,8,16 --request-rate inf \
  --model zai-org/GLM-Image
```

### 场景2：多GPU并行测试

```bash
# 1. 启动2GPU服务器，启用CFG并行
./01_start_server.sh sglang 2 true

# 2. 运行测试，保存到多GPU结果目录
./02_run_all_tests.sh --backend sglang --port 30000 --dataset random \
  --num-prompts 10 --widths 512,1024 --heights 512,1024 \
  --max-concurrency 1,2,4,8,16 --request-rate inf \
  --model zai-org/GLM-Image \
  --result-subdir multi_gpu_2gpu

# 3. 停止服务器
./04_stop_server.sh 30000

# 4. 测试Diffusers后端（单GPU）
./01_start_server.sh diffusers
./02_run_all_tests.sh --backend diffusers --port 30000 --dataset random \
  --num-prompts 10 --widths 512,1024 --heights 512,1024 \
  --max-concurrency 1,2,4,8,16 --request-rate inf \
  --model zai-org/GLM-Image \
  --result-subdir multi_gpu_2gpu
```

---

## 结果目录结构

```
benchmark/
├── results/              # 单GPU测试结果（默认）
│   ├── GLM-Image_sglang_w512_h512_n10_c1_*.json
│   ├── GLM-Image_sglang_w512_h512_n10_c2_*.json
│   ├── GLM-Image_sglang_w512_h512_n10_c4_*.json
│   ├── GLM-Image_sglang_w512_h512_n10_c8_*.json
│   └── GLM-Image_sglang_w512_h512_n10_c16_*.json
│
└── results_multi_gpu/    # 多GPU测试结果（使用 --result-subdir multi_gpu）
    ├── GLM-Image_sglang_w512_h512_n10_c1_*.json
    ├── GLM-Image_sglang_w512_h512_n10_c2_*.json
    ├── GLM-Image_sglang_w512_h512_n10_c4_*.json
    ├── GLM-Image_sglang_w512_h512_n10_c8_*.json
    └── GLM-Image_sglang_w512_h512_n10_c16_*.json
```

---

## 注意事项

1. **GPU数量**：确保系统有足够的GPU。使用 `nvidia-smi` 检查可用GPU数量。

2. **CFG并行**：`--enable-cfg-parallel` 用于启用Classifier-Free Guidance并行，可能提高性能但需要更多资源。

3. **并发数**：高并发数（8, 16）可能需要更多GPU资源。建议先测试较低并发数。

4. **结果目录**：使用 `--result-subdir` 可以区分不同测试配置的结果，便于对比分析。

5. **端口冲突**：确保端口30000未被占用，或使用 `--port` 参数指定其他端口。

---

## 对比测试建议

为了公平对比，建议：

1. **单GPU vs 多GPU**：分别测试，使用不同的结果目录
2. **不同并发数**：在同一配置下测试所有并发数（1,2,4,8,16）
3. **SGLang vs Diffusers**：在相同GPU配置下测试两个后端

示例对比目录命名：
- `results_single_gpu/` - 单GPU测试
- `results_multi_gpu_2gpu/` - 2GPU测试
- `results_multi_gpu_4gpu/` - 4GPU测试
