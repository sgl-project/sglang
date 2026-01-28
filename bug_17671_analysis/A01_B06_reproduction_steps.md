# A01_B06: Bug复现步骤

**参考文档**: 
- [A01_B01_original_issue.md](./A01_B01_original_issue.md) ⭐ **Issue原始内容**
- [A01_B02_problem_analysis.md](./A01_B02_problem_analysis.md) ⭐ **问题详细分析**

---

## 1. 复现环境要求

- **Docker**: 已安装并运行
- **GPU**: 可选（Step 3需要，但Step 1不需要）
- **网络**: 可以访问Docker Hub（Step 1不需要HuggingFace）

**注意**: 
- **Step 1 不需要GPU**，可以在任何有Docker的机器上执行
- **Step 3 需要GPU**（如果要在容器内安装和验证）

---

## 2. 复现步骤（简化版 - 重点验证模块缺失）

> **注意**: 问题本质是缺少 `diffusers` 模块，与模型大小无关。因此不需要运行大模型，只需要验证模块缺失即可。

### Step 0: 拉取最新 dev 镜像

```bash
docker pull lmsysorg/sglang:dev
```

**目的**: 确保使用最新版本的镜像

---

### Step 1: 验证 diffusers 模块缺失（最重要，0模型，0GPU）⭐

**命令**:
```bash
docker run --rm -it lmsysorg/sglang:dev python -c "import diffusers"
```

**预期结果**:
```
ModuleNotFoundError: No module named 'diffusers'
```

**说明**: 
- ✅ **这一步就足够证明问题** - 镜像确实缺少 `diffusers` 模块
- ✅ 不需要下载任何模型
- ✅ 不需要GPU
- ✅ 最快验证方法（几秒钟）
- ✅ 这是最关键的复现证据

**截图/记录**: 
- **必须保存完整的错误信息**（复制或截图）
- 这是最关键的复现证据

---

### Step 2: 检查镜像信息

**命令** (Windows PowerShell):
```powershell
docker image inspect lmsysorg/sglang:dev | Select-Object -First 40
```

**命令** (Linux/Mac):
```bash
docker image inspect lmsysorg/sglang:dev | head -n 40
```

**目的**: 
- 查看镜像的创建时间
- 查看镜像的labels
- 确认镜像版本信息

**记录**: 
- 镜像的Created时间
- 镜像的标签信息
- 镜像的架构信息

---

### Step 3: 验证"装了就好"（在容器里安装 diffusion extra）⭐

**这一步验证**: 安装 `sglang[diffusion]` 后是否可以恢复

#### 3.1 进入容器

**命令** (Windows PowerShell):
```powershell
docker run --gpus all --rm -it lmsysorg/sglang:dev bash
```

**命令** (Linux/Mac):
```bash
docker run --gpus all --rm -it lmsysorg/sglang:dev bash
```

#### 3.2 在容器内安装 diffusion extras

```bash
uv pip install "sglang[diffusion]" --prerelease=allow
```

**等待安装完成**（可能需要几分钟）

#### 3.3 验证 diffusers 模块

```bash
python -c "import diffusers; print('✓ diffusers installed')"
```

**预期结果**:
```
✓ diffusers installed
```

#### 3.4 退出容器

```bash
exit
```

**结论**: 
- ✅ 如果安装后可以成功导入，说明问题确实是镜像缺少依赖
- ✅ 这坐实了"Docker章节默认假设镜像自带diffusion，但实际dev镜像没打进去"的文档/镜像不一致问题

---

### Step 4: 使用模型测试（可选，不推荐）

> **注意**: 这一步**不是必须的**，因为问题本质是模块缺失，不是模型问题。Step 1已经足够证明问题。

**如果一定要测试**（使用 tiny 模型，不需要大模型）:

**命令** (Windows PowerShell):
```powershell
docker run --gpus all --rm -it -v ${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface lmsysorg/sglang:dev sglang generate --model-path hf-internal-testing/tiny-stable-diffusion-pipe-variants-right-format --prompt "test" --save-output --backend diffusers
```

**命令** (Linux/Mac):
```bash
docker run --gpus all --rm -it -v ~/.cache/huggingface:/root/.cache/huggingface lmsysorg/sglang:dev sglang generate --model-path hf-internal-testing/tiny-stable-diffusion-pipe-variants-right-format --prompt "test" --save-output --backend diffusers
```

**预期结果**:
- 如果镜像缺 diffusion 依赖，会在很早阶段就失败
- 通常是 import/依赖缺失错误

**说明**: 
- 这一步**不是必须的**，Step 1已经足够
- 如果Step 1已经确认问题，可以跳过这一步

**步骤1: 进入容器**:
```bash
docker run --gpus all --rm -it \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:dev bash
```

**步骤2: 在容器内安装 diffusion extras**:
```bash
# 安装 diffusion extras
uv pip install "sglang[diffusion]" --prerelease=allow

# 验证 diffusers 模块
python -c "import diffusers; print('✓ diffusers installed')"
```

**步骤3: 再次运行 tiny 模型**:
```bash
sglang generate \
  --model-path hf-internal-testing/tiny-stable-diffusion-pipe-variants-right-format \
  --prompt "test" \
  --save-output \
  --backend diffusers
```

**预期结果**:
- ✅ 安装后可以成功导入 diffusers
- ✅ 安装后可以成功运行 diffusion 模型
- ✅ 确认问题确实是"镜像缺少依赖"

**结论**: 
- 如果装完就能跑，那就坐实了问题本质
- 这是"Docker章节默认假设镜像自带diffusion，但实际dev镜像没打进去"的文档/镜像不一致问题

---

## 3. 复现结果收集

### 3.1 必须收集的信息（最小集）

1. **Step 1 的完整报错栈** ⭐ **最重要**:
   ```
   ModuleNotFoundError: No module named 'diffusers'
   ...
   （完整的traceback）
   ```
   - **这是最关键的证据**
   - 截图或复制完整错误信息

2. **镜像信息**:
   ```bash
   docker image inspect lmsysorg/sglang:dev | Select-Object -First 40  # PowerShell
   docker image inspect lmsysorg/sglang:dev | head -n 40  # Linux/Mac
   ```
   - 镜像的Created时间
   - 镜像的标签
   - 镜像的架构

3. **安装后是否恢复** ⭐ **关键验证**:
   - ✅ 是：安装 `sglang[diffusion]` 后可以成功导入 `diffusers`
   - ❌ 否：安装后仍然有问题（需要进一步调查）

### 3.2 可选收集的信息

1. **Step 4 的错误信息**（如果执行了）
   - 通常和Step 1一样，是模块缺失错误

2. **安装过程的输出**（如果执行了Step 3）
   - 可以看到安装了哪些包

### 3.2 可选收集的信息

1. **Step 3 的错误信息**（如果执行了）
2. **安装过程的输出**（如果执行了Step 4）
3. **运行成功的输出**（如果执行了Step 4）

---

## 4. 如何贡献到 Issue #17671

### 4.1 在Issue中回复（简化版）

**回复内容应该包括**:

1. **复现确认** ⭐:
   ```markdown
   ## 复现确认
   
   我在4090上成功复现了这个问题。
   
   ### Step 1: 验证 diffusers 模块缺失
   ```bash
   $ docker run --rm -it lmsysorg/sglang:dev python -c "import diffusers"
   ```
   输出：
   ```
   ModuleNotFoundError: No module named 'diffusers'
   ```
   ```
   （粘贴完整的错误信息）

2. **镜像信息**:
   ```markdown
   ### 镜像信息
   ```bash
   $ docker image inspect lmsysorg/sglang:dev | Select-Object -First 40
   ```
   （粘贴输出）
   ```

3. **安装后恢复确认** ⭐:
   ```markdown
   ### 安装后恢复确认
   
   在容器内安装 `sglang[diffusion]` 后：
   ```bash
   $ uv pip install "sglang[diffusion]" --prerelease=allow
   $ python -c "import diffusers; print('✓ diffusers installed')"
   ✓ diffusers installed
   ```
   
   ✅ **结论**: 
   - 安装 `sglang[diffusion]` 后可以成功导入 diffusers
   - 问题确实是镜像缺少 diffusion 依赖
   - 这是文档/镜像不一致问题
   ```

---

## 5. 快速复现脚本

**使用提供的脚本**:
```bash
chmod +x reproduce_bug.sh
./reproduce_bug.sh
```

**或者手动执行各个步骤**（见上面的详细说明）

---

## 6. 预期结果总结

### 6.1 如果问题确实存在

- ✅ **Step 1**: 报 `ModuleNotFoundError: No module named 'diffusers'` ⭐ **核心证据**
- ✅ **Step 3**: 安装 `sglang[diffusion]` 后可以成功导入 diffusers ⭐ **验证修复**

**结论**: 镜像确实缺少 diffusion 依赖，需要修复文档或镜像

**最小复现**: Step 1 + Step 3 就足够了

### 6.2 如果问题已修复

- ❌ Step 1: 可以成功导入 diffusers（没有报错）
- ❌ Step 3: 安装后仍然可以导入（但可能不需要安装）

**结论**: 问题已修复，可以关闭issue

---

## 7. 下一步行动

### 7.1 如果确认问题存在

1. **在Issue中回复**，提供复现结果
2. **提交PR修复**（文档或Dockerfile）
3. **参考**: [A01_B05_PR_draft.md](./A01_B05_PR_draft.md)

### 7.2 如果问题已修复

1. **在Issue中回复**，说明问题已修复
2. **关闭issue**（如果是维护者）

---

**参考文档**:
- [A01_B01_original_issue.md](./A01_B01_original_issue.md) - Issue原始内容
- [A01_B02_problem_analysis.md](./A01_B02_problem_analysis.md) - 问题详细分析
- [A01_B05_PR_draft.md](./A01_B05_PR_draft.md) - PR修改草案
- [A01_B07_issue_contribution_template.md](./A01_B07_issue_contribution_template.md) - Issue贡献模板
