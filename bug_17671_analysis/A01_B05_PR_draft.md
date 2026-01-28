# A01_B05: PR修改草案

**参考文档**: 
- [A01_B01_original_issue.md](./A01_B01_original_issue.md) ⭐ **Issue原始内容**
- [A01_B03_solution_analysis.md](./A01_B03_solution_analysis.md) ⭐ **解决方案分析**
- [A01_B04_code_analysis.md](./A01_B04_code_analysis.md) ⭐ **代码分析**

---

## 1. PR方案选择

### 推荐方案：文档修复（方案1）

**理由**:
- ✅ 最小改动，风险最低
- ✅ 新人友好，容易贡献
- ✅ 不需要修改构建流程
- ✅ 可以快速解决用户困惑

**目标**: 在官方文档中明确说明diffusion需要额外安装依赖

---

## 2. 需要修改的文件

### 2.1 主要文件

1. **`docs/get_started/install.md`** - 安装文档
2. **`docs/supported_models/diffusion_models.md`** - Diffusion模型文档（如果存在）
3. **README.md** - 如果提到diffusion

### 2.2 需要查找的文件

```bash
# 查找diffusion相关文档
find docs -name "*diffusion*" -type f
grep -r "FLUX\|diffusion" docs/
```

---

## 3. 具体修改内容

### 3.1 修改 `docs/get_started/install.md`

**位置**: 在Docker使用部分添加diffusion说明

**修改前**:
```markdown
## Docker

You can use the official Docker image:

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    lmsysorg/sglang:dev \
    sglang.launch_server --model-path <model> ...
```
```

**修改后**:
```markdown
## Docker

You can use the official Docker image:

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    lmsysorg/sglang:dev \
    sglang.launch_server --model-path <model> ...
```

### Using Diffusion Models with Docker

If you want to use diffusion models (e.g., FLUX) with the Docker image, you need to install additional dependencies:

```bash
# Start the container
docker run --gpus all --rm -it \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:dev bash

# Inside the container, install diffusion extras
uv pip install 'sglang[diffusion]' --prerelease=allow

# Now you can use diffusion models
sglang generate --model-path black-forest-labs/FLUX.1-dev \
  --prompt "Your prompt here" --save-output
```

Alternatively, you can create a custom Dockerfile that includes diffusion support.
```

---

### 3.2 修改或创建 `docs/supported_models/diffusion_models.md`

**如果文件存在，添加前置条件部分**:

```markdown
## Prerequisites

Before using diffusion models with Docker, ensure you have installed the diffusion extras:

```bash
# Inside the Docker container
uv pip install 'sglang[diffusion]' --prerelease=allow
```

For local installation:

```bash
pip install 'sglang[diffusion]' --prerelease=allow
```
```

**如果文件不存在，可以创建**:

```markdown
# Diffusion Models

SGLang supports diffusion models such as FLUX.

## Prerequisites

Install the diffusion extras:

```bash
# Docker
docker run --gpus all --rm -it lmsysorg/sglang:dev bash
uv pip install 'sglang[diffusion]' --prerelease=allow

# Local
pip install 'sglang[diffusion]' --prerelease=allow
```

## Usage

```bash
sglang generate --model-path black-forest-labs/FLUX.1-dev \
  --prompt "Your prompt here" --save-output
```
```

---

## 4. PR提交步骤

### 4.1 准备工作

1. **Fork仓库**（如果还没有）
2. **创建新分支**:
   ```bash
   git checkout -b fix/issue-17671-add-diffusion-install-docs
   ```

### 4.2 修改文件

1. 找到并修改相关文档文件
2. 添加diffusion安装说明
3. 确保格式正确

### 4.3 提交和推送

```bash
git add docs/
git commit -m "docs: Add diffusion installation instructions for Docker

Fix Issue-17671: Document that diffusion extras need to be installed
when using Docker image with diffusion models.

- Add section in install.md explaining how to install diffusion extras
- Add prerequisites section for diffusion models
- Provide workaround for Docker users"
git push origin fix/issue-17671-add-diffusion-install-docs
```

### 4.4 创建PR

1. 在GitHub上创建PR
2. **标题**: `docs: Add diffusion installation instructions for Docker (fixes #17671)`
3. **描述**:
   ```markdown
   ## Problem
   
   Users following the official documentation cannot launch diffusion models 
   with Docker because the `diffusers` module is missing.
   
   Fixes #17671
   
   ## Solution
   
   Add clear documentation explaining that diffusion extras need to be 
   installed when using Docker image with diffusion models.
   
   ## Changes
   
   - Add "Using Diffusion Models with Docker" section in install.md
   - Add prerequisites section for diffusion models
   - Provide workaround commands
   
   ## Testing
   
   - [x] Verified the workaround works
   - [x] Checked documentation format
   ```

---

## 5. 验证方法

### 5.1 验证文档清晰度

1. 让其他人阅读修改后的文档
2. 确认可以按照文档操作成功
3. 检查是否有歧义

### 5.2 验证命令有效性

1. 按照文档中的命令操作
2. 确认可以成功安装diffusion依赖
3. 确认可以成功运行diffusion模型

---

## 6. 备选方案：Dockerfile修复

如果选择Dockerfile修复方案，修改如下：

### 6.1 修改 `docker/Dockerfile`

**位置**: 第4-8行（ARG定义部分）

**修改前**:
```dockerfile
ARG BUILD_TYPE=all
ARG BRANCH_TYPE=remote
ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG CMAKE_BUILD_PARALLEL_LEVEL=2
ARG SGL_KERNEL_VERSION=0.3.12
```

**修改后**:
```dockerfile
ARG BUILD_TYPE=all
ARG BRANCH_TYPE=remote
ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG CMAKE_BUILD_PARALLEL_LEVEL=2
ARG SGL_KERNEL_VERSION=0.3.12
ARG SGLANG_EXTRAS=""
```

**位置**: 第91行（安装SGLang部分）

**修改前**:
```dockerfile
 && python3 -m pip install --no-cache-dir -e "python[${BUILD_TYPE}]" --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX} \
```

**修改后**:
```dockerfile
 && if [ -n "${SGLANG_EXTRAS}" ]; then \
        EXTRAS_STR=",${SGLANG_EXTRAS}"; \
    else \
        EXTRAS_STR=""; \
    fi \
 && python3 -m pip install --no-cache-dir -e "python[${BUILD_TYPE}${EXTRAS_STR}]" --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX} --prerelease=allow \
```

**构建命令**:
```bash
# 默认构建（不包含diffusion）
docker build -t lmsysorg/sglang:dev .

# 构建包含diffusion的镜像
docker build --build-arg SGLANG_EXTRAS=diffusion -t lmsysorg/sglang:dev-diffusion .
```

---

## 7. PR Checklist

### 7.1 文档修复PR

- [ ] 找到相关文档文件
- [ ] 添加diffusion安装说明
- [ ] 添加示例命令
- [ ] 检查文档格式和拼写
- [ ] 测试文档中的命令是否有效
- [ ] 确保链接正确
- [ ] 提交PR并关联Issue #17671

### 7.2 Dockerfile修复PR（如果选择）

- [ ] 修改Dockerfile添加build arg
- [ ] 更新安装命令
- [ ] 测试构建过程
- [ ] 测试diffusion功能
- [ ] 更新相关文档
- [ ] 更新CI/CD（如果需要）
- [ ] 提交PR并关联Issue #17671

---

## 8. 预期结果

### 8.1 文档修复后

- ✅ 用户可以清楚知道需要安装diffusion依赖
- ✅ 提供了明确的安装步骤
- ✅ 减少了用户困惑
- ✅ 文档和实际行为一致

### 8.2 Dockerfile修复后

- ✅ 可以构建包含diffusion的镜像
- ✅ 用户可以选择使用哪个镜像
- ✅ 保持默认镜像瘦身
- ✅ 提供开箱即用的diffusion支持

---

## 9. 相关资源

- [Issue #17671](https://github.com/sgl-project/sglang/issues/17671)
- [Issue #17618](https://github.com/sgl-project/sglang/issues/17618) - 可能相关
- [Issue #13346](https://github.com/sgl-project/sglang/issues/13346) - 可能相关
- [SGLang文档](https://docs.sglang.ai/)

---

**推荐**: 先提交文档修复PR，这是最稳妥的方案，新人友好，可以快速解决用户困惑。

---

**参考文档**:
- [A01_B01_original_issue.md](./A01_B01_original_issue.md) - Issue原始内容
- [A01_B03_solution_analysis.md](./A01_B03_solution_analysis.md) - 解决方案分析
- [A01_B04_code_analysis.md](./A01_B04_code_analysis.md) - 代码分析
