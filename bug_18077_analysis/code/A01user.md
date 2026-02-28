# 多卡 SP 测试用法 (A01 + A02)

**多卡环境验证**：`A01_test_other_start_server.sh`（Qwen2.5-1.5B TP=2）已在本机跑通，多卡测试成功。

---

## 1. 启动多卡服务器

```bash
cd /data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/code
./A01_start_server.sh
```

保持此终端运行；看到 `Model is ready` 后，在**另一个终端**做第 2 步。

---

## 2. 跑多卡 benchmark

```bash
cd /data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/code
./A02_run_multi_gpu_bench.sh
```

结果在 `benchmark/results_multi_gpu/`（供 03 对比），同时会在 `benchmark/02112026/` 保存：
- 测试前后 GPU 快照（`gpu_before_test_*.txt`、`gpu_after_test_*.txt`）— 证明推理时两卡都在工作
- 多卡结果副本（`multi_gpu_result_*.json`）
A01 运行时也会在 `02112026/` 写 server 日志和运行中 GPU 采样（`run_*_server.log`、`run_*_gpu_usage_during_run.log`）。

---

## 3. 和单卡结果对比，证明多卡效率提升（与 01/02 + 03 一致）

先用 **01 + 02** 跑同配置单卡（例如 1024×1024、n10、c1），结果在 `benchmark/results/`；再用 **A01 + A02** 跑多卡，结果在 `benchmark/results_multi_gpu/`。然后执行：

```bash
./03_compare_single_vs_multi_gpu.sh
```

报告在 `benchmark/results_multi_gpu/comparison_single_vs_multi_gpu_*.md`，并会复制一份到 `benchmark/02112026/`，与双卡证明材料放在一起。

---

## 改分辨率/并发（可选）

```bash
NUM_PROMPTS=6 WIDTH=1024 HEIGHT=1024 CONCURRENCY=1 ./A02_run_multi_gpu_bench.sh
```

## 关掉服务器

在运行 A01 的终端按 **Ctrl+C**。
