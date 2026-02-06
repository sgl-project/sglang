# GLM-Image 测试步骤

## 0) 进入目录
```bash
cd /data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/code
```

## 1) 启动 SGLang 后端
```bash
./01_start_server.sh sglang
```

## 2) 运行 SGLang 测试（一次性跑完所有配置）
```bash
./02_run_all_tests.sh --backend sglang --port 30000 --dataset random \
  --num-prompts 10 --widths 512,1024 --heights 512,1024 --max-concurrency 1,2 --request-rate inf \
  --model zai-org/GLM-Image
```

## 3) 停止 SGLang 后端
```bash
./04_stop_server.sh 30000
```

## 4) 启动 Diffusers 后端
```bash
./01_start_server.sh diffusers
```

## 5) 运行 Diffusers 测试（同样配置）
```bash
./02_run_all_tests.sh --backend diffusers --port 30000 --dataset random \
  --num-prompts 10 --widths 512,1024 --heights 512,1024 --max-concurrency 1,2 --request-rate inf \
  --model zai-org/GLM-Image
```

## 6) 停止 Diffusers 后端
```bash
./04_stop_server.sh 30000
```

## 7) 生成对比报告
```bash
./03_compare.sh
```

## 结果位置
所有结果保存在：
```
/data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/benchmark/results
```
