# Bug #18077: GLM-Image 性能测试

## 📋 快速开始

### 1. 安装依赖（必需）

```bash
cd /data/users/yandache/workspaces/sglang
source env_sglang/bin/activate
cd repo/sglang-src/python

# 安装 diffusion extras（必需！）
pip install -e ".[diffusion]"
```

**重要**：`diffusion` 是 optional dependency，必须显式安装才能使用 GLM-Image 等 diffusion 模型。

### 2. 设置环境变量（按照 BME2-Workspace-Standard）

```bash
source /data/users/yandache/_shared/tools/env.sh
```

### 3. 启动服务器

```bash
cd /data/users/yandache/workspaces/sglang
source env_sglang/bin/activate

# 启动 SGLang 后端
sglang serve \
    --model-path zai-org/GLM-Image \
    --backend sglang \
    --port 30000 \
    --trust-remote-code
```

### 4. 运行性能测试

#### 方式 A：使用官方 bench_serving.py（推荐，与 haojin2 相同）

```bash
# 在另一个终端
cd /data/users/yandache/workspaces/sglang
source /data/users/yandache/_shared/tools/env.sh
source env_sglang/bin/activate

# 使用官方脚本（与 Issue #18077 中 haojin2 的测试方式相同）
python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
    --model zai-org/GLM-Image \
    --dataset vbench \
    --num-prompts 10 \
    --port 30000 \
    --max-concurrency 1 \
    --output-file repo/sglang-src/bug_18077_analysis/benchmark/results/zai-org_GLM-Image_sglang_$(date +%Y%m%d_%H%M%S).json
```

**或者使用便捷脚本**：
```bash
cd repo/sglang-src/bug_18077_analysis/code
./03_run_benchmark.sh sglang 30000 512 512 50 10 1 random
```

---

## 📚 文档索引

### 核心文档
- [A01_B01: Issue 原始内容](./A01_B01_original_issue.md) - 问题背景和发现
- [A01_B09: 模型选择建议](./A01_B09_glm_image_model_selection.md) - 推荐使用 `zai-org/GLM-Image`
- [A01_B11: 常见问题解决方案](./A01_B11_disk_quota_solution.md) - 磁盘配额、依赖安装等问题
- [A01_B13: 完整测试命令](./A01_B13_complete_test_commands.md) - 详细的测试步骤和命令

### 测试设计
- [A02: FLUX-Klein 测试设计](./A02_flux_klein_test_design.md) - 参考测试设计

---

## 🔧 关键要点

1. **必须安装 `sglang[diffusion]`**：`pip install -e ".[diffusion]"`
2. **必须设置环境变量**：`source /data/users/yandache/_shared/tools/env.sh`
3. **必须指定后端**：`--backend sglang` 或 `--backend diffusers`
4. **必须使用 `--trust-remote-code`**：GLM-Image 需要自定义代码

---

## 📁 文件结构

```
bug_18077_analysis/
├── README.md                    # 本文件
├── DESIGN.md                    # 测试设计文档
├── A01_B01_original_issue.md    # Issue 原始内容
├── A01_B09_glm_image_model_selection.md  # 模型选择
├── A01_B11_disk_quota_solution.md        # 常见问题
├── A01_B13_complete_test_commands.md     # 完整测试命令
├── code/
│   ├── 01_check_environment.sh  # 环境检查脚本
│   ├── 02_start_server.sh      # 启动服务器脚本
│   ├── 03_run_benchmark.sh     # 运行性能测试脚本
│   ├── 04_compare.sh           # 对比结果脚本
│   └── 05_stop_server.sh       # 停止服务器脚本
└── benchmark/                   # 测试结果目录
```
