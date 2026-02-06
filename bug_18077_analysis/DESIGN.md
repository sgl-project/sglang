# GLM-Image 性能测试

## 脚本

- `01_start_server.sh [backend]` - 启动服务器（sglang/diffusers）
- `02_run_all_tests.sh` - 运行所有测试配置（参数与官方 `bench_serving` 一致）
- `03_compare.sh [pattern]` - 对比结果
- `04_stop_server.sh [port]` - 停止服务器

## 测试流程

### 1. 启动服务器
```bash
./01_start_server.sh sglang
```

### 2. 运行测试
```bash
./02_run_all_tests.sh --backend sglang
```
或自定义参数（官方参数名）：
```bash
./02_run_all_tests.sh --backend sglang --port 30000 --dataset random \
  --num-prompts 10 --widths 512,1024 --heights 512,1024 --max-concurrency 1,2 --request-rate inf \
  --model zai-org/GLM-Image
```

### 3. 切换后端
```bash
./04_stop_server.sh 30000
./01_start_server.sh diffusers
```

### 4. 运行 Diffusers 测试
```bash
./02_run_all_tests.sh --backend diffusers
```

### 5. 生成对比报告
```bash
./03_compare.sh
```

## 测试配置

| 分辨率 | 并发数 |
|:-------|:-------|
| 512x512 | 1,2 |
| 1024x1024 | 1,2 |
