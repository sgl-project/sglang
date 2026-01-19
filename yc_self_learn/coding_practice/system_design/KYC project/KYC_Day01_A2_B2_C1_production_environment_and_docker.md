# A2_B2_C1：生产环境、Docker 和 Linux 的关系

---
doc_type: glossary
layer: L3
scope_in:  生产环境的定义、Docker 和 Linux 的关系、Docker 镜像的概念、容器 vs 虚拟机
scope_out: Docker 的详细使用教程（见 howto）；K8s 容器编排（见 L4）
inputs:   (读者) 疑问：什么是生产环境？Docker 和 Linux 的关系？Docker 镜像是什么？
outputs:  概念定义 + 开发/测试/生产环境的区别 + Docker 核心概念 + 实际例子
entrypoints: [ Definition ]
children: []
related: [ 部署, CI/CD, 容器化, K8s, KYC_Day01_A2_B1_cron.md ]
---

## Definition（定义）

### 1. 什么是生产环境（Production Environment）？

**生产环境**是**真实用户在使用**的系统环境，与开发环境、测试环境相对。

**核心特征**：
- ✅ **真实用户**：真实用户在使用的系统
- ✅ **真实数据**：处理真实业务数据（不是测试数据）
- ✅ **24/7 运行**：需要持续运行，不能随便重启
- ✅ **高可用性**：系统故障会影响真实业务，需要高可用
- ✅ **监控告警**：需要完整的监控和告警系统

---

### 2. 三种环境的区别

| 环境 | 用途 | 谁在用 | 数据 | 稳定性要求 | 例子 |
|------|------|--------|------|-----------|------|
| **开发环境**（Development） | 写代码、调试 | 开发者自己 | 假数据/测试数据 | 低（可以随便重启） | 你的 Windows 电脑 |
| **测试环境**（Testing/Staging） | 测试功能、找 bug | QA 测试人员 | 模拟数据 | 中（尽量稳定） | 测试服务器 |
| **生产环境**（Production） | 真实业务 | 真实用户 | 真实数据 | 高（不能随便重启） | 云服务器、AWS、Azure |

**KYC 项目例子**：
- **开发环境**：你在 Windows 上写代码，本地测试
- **测试环境**：部署到测试服务器，QA 测试功能
- **生产环境**：部署到云服务器，真实用户提交 KYC 申请

---

## 💡 完整流程：从写代码到上线（你问的问题）

### 问题：程序写好后，是先打包 Docker，然后去哪里测试？

**答案**：是的！流程是：**写代码 → Docker 打包 → 本地测试 Docker → 测试环境测试 → 生产环境上线**

### 详细流程（5 个阶段）

```
阶段 1：开发（Windows）
    ↓
阶段 2：Docker 化（Windows）
    ↓
阶段 3：本地测试 Docker（Windows）
    ↓
阶段 4：测试环境（Linux 服务器）

---

## ⚠️ 重要问题：Linux 对 Windows 开发的程序兼容性好吗？

### 你的疑问：Windows 上写的代码，在 Linux 上能直接运行吗？

**答案**：**大部分情况下可以，但需要注意一些细节**。Python 代码本身是跨平台的，但有一些需要注意的地方。

---

### KYC 项目实际例子：简单的 API 调用

#### 例子 1：调用 Fireworks API（完全兼容）

**Windows 上写的代码**：
```python
# scripts/daily_metrics.py（在 Windows 上写）
import requests
import json

def call_fireworks_api(prompt):
    """调用 Fireworks API"""
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('FW_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen2.5-vl-32b",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()
```

**在 Linux 上运行**：
```bash
# Linux 服务器上
docker run kyc-daily-metrics:latest
# ✅ 完全兼容！直接运行，不需要修改代码
```

**为什么兼容**：
- ✅ `requests` 库是跨平台的（Windows/Linux/Mac 都一样）
- ✅ HTTP 协议是跨平台的（不依赖操作系统）
- ✅ JSON 处理是跨平台的（Python 标准库）

---

#### 例子 2：文件路径（需要注意）

**Windows 上写的代码**（❌ 不兼容）：
```python
# ❌ 错误：硬编码 Windows 路径
file_path = "C:\\Users\\nickc\\data\\summary.json"

# ❌ 错误：使用 Windows 路径分隔符
file_path = "C:/Users/nickc/data/summary.json"
```

**Linux 上运行**：
```bash
# ❌ 会失败！Linux 没有 C:\ 盘
docker run kyc-daily-metrics:latest
# FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\nickc\\data\\summary.json'
```

**正确的写法**（✅ 兼容）：
```python
# ✅ 正确：使用相对路径或环境变量
from pathlib import Path
import os

# 方式 1：相对路径（推荐）
file_path = Path("data/summary.json")

# 方式 2：环境变量
data_dir = os.getenv("DATA_DIR", "/app/data")
file_path = Path(data_dir) / "summary.json"

# 方式 3：使用 pathlib（自动处理路径分隔符）
file_path = Path("data") / "summary.json"
# Windows: data\summary.json
# Linux: data/summary.json
# pathlib 自动处理！
```

**为什么兼容**：
- ✅ `pathlib.Path` 自动处理路径分隔符（Windows 用 `\`，Linux 用 `/`）
- ✅ 相对路径在两种系统上都有效
- ✅ 环境变量让路径可配置

---

#### 例子 3：文件权限（需要注意）

**Windows 上写的代码**（可能有问题）：
```python
# 创建文件
with open("output.json", "w") as f:
    json.dump(data, f)

# 在 Windows 上：默认权限是 666（可读写）
# 在 Linux 上：默认权限是 644（可读，所有者可写）
```

**Linux 上运行**：
```bash
# ✅ 通常没问题，但如果有权限问题：
docker run kyc-daily-metrics:latest
# PermissionError: [Errno 13] Permission denied
```

**正确的写法**（✅ 兼容）：
```python
# ✅ 正确：显式设置文件权限
import os
from pathlib import Path

output_file = Path("output.json")
output_file.write_text(json.dumps(data))
output_file.chmod(0o644)  # 显式设置权限（Linux）

# 或者在 Dockerfile 中设置
# RUN chmod 644 /app/output.json
```

---

#### 例子 4：系统调用（需要注意）

**Windows 上写的代码**（❌ 不兼容）：
```python
# ❌ 错误：使用 Windows 特定的命令
import subprocess
result = subprocess.run(["dir"], shell=True)  # Windows 命令
```

**Linux 上运行**：
```bash
# ❌ 会失败！Linux 没有 dir 命令
docker run kyc-daily-metrics:latest
# FileNotFoundError: [Errno 2] No such file or directory: 'dir'
```

**正确的写法**（✅ 兼容）：
```python
# ✅ 正确：使用跨平台的 Python 库
from pathlib import Path

# 列出文件（跨平台）
files = list(Path(".").glob("*.json"))

# 或者使用 Python 标准库
import os
files = [f for f in os.listdir(".") if f.endswith(".json")]
```

---

### KYC 项目的实际代码示例（完全兼容）

#### 完整的 KYC API 调用示例

**Windows 上写的代码**（✅ 完全兼容）：
```python
# scripts/daily_metrics.py（在 Windows 上写）
import requests
import json
import os
from pathlib import Path
from datetime import datetime

def process_kyc_batch(batch_id):
    """处理 KYC batch（完全跨平台）"""
    
    # ✅ 1. 使用 pathlib 处理路径（跨平台）
    data_dir = Path(os.getenv("DATA_DIR", "/app/data"))
    summary_file = data_dir / f"{batch_id}_summary.json"
    
    # ✅ 2. 读取文件（跨平台）
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)
    
    # ✅ 3. 调用 API（跨平台）
    api_key = os.getenv("FW_API_KEY")
    if not api_key:
        raise ValueError("FW_API_KEY not set")
    
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 处理每个文档
    results = []
    for doc in summary.get("results", []):
        # ✅ 4. API 调用（完全跨平台）
        response = requests.post(
            url,
            headers=headers,
            json={
                "model": "qwen2.5-vl-32b",
                "messages": [{"role": "user", "content": doc["prompt"]}]
            },
            timeout=30
        )
        response.raise_for_status()
        results.append(response.json())
    
    # ✅ 5. 写入文件（跨平台）
    output_file = data_dir / f"{batch_id}_processed.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

if __name__ == "__main__":
    batch_id = os.getenv("BATCH_ID", "batch_001")
    process_kyc_batch(batch_id)
```

**在 Linux 上运行**（✅ 完全兼容）：
```bash
# Linux 服务器上
docker run \
  -e DATA_DIR=/app/data \
  -e FW_API_KEY=your-api-key \
  -e BATCH_ID=batch_001 \
  -v /host/data:/app/data \
  kyc-daily-metrics:latest

# ✅ 完全兼容！直接运行，不需要修改代码
```

**为什么兼容**：
- ✅ **路径处理**：使用 `pathlib.Path`，自动处理路径分隔符
- ✅ **环境变量**：使用 `os.getenv()`，让路径和配置可配置
- ✅ **API 调用**：`requests` 库是跨平台的
- ✅ **文件操作**：Python 标准库是跨平台的
- ✅ **编码处理**：显式指定 `encoding="utf-8"`，避免编码问题

---

### 兼容性检查清单

#### ✅ 完全兼容（不需要修改）

1. **HTTP 请求**：`requests`、`urllib`、`httpx`
2. **JSON 处理**：`json` 库
3. **数据处理**：`pandas`、`numpy`（大部分情况下）
4. **数据库连接**：`psycopg2`、`pymongo`、`sqlalchemy`
5. **API 调用**：REST API、GraphQL API

#### ⚠️ 需要注意（可能需要修改）

1. **文件路径**：
   - ❌ 硬编码绝对路径（`C:\Users\...`）
   - ✅ 使用 `pathlib.Path` 或相对路径

2. **文件权限**：
   - ❌ 假设默认权限
   - ✅ 显式设置权限或使用环境变量

3. **系统调用**：
   - ❌ 使用系统命令（`dir`、`ls`）
   - ✅ 使用 Python 标准库（`os.listdir()`、`pathlib`）

4. **换行符**：
   - ❌ 假设换行符（Windows `\r\n` vs Linux `\n`）
   - ✅ 使用 Python 自动处理（`open()` 默认处理）

5. **依赖库**：
   - ❌ 使用 Windows 特定的库
   - ✅ 使用跨平台的库

---

### Docker 如何保证兼容性？

#### Docker 的作用

**Docker 让你在 Windows 上构建 Linux 环境**：

```dockerfile
# Dockerfile
FROM python:3.9-slim  # ← 这是 Linux 环境！

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt  # ← 在 Linux 环境中安装

COPY scripts/daily_metrics.py .
CMD ["python", "daily_metrics.py"]  # ← 在 Linux 环境中运行
```

**关键点**：
- ✅ **基础镜像**：`python:3.9-slim` 是 Linux 环境
- ✅ **构建过程**：在 Linux 环境中构建（即使你在 Windows 上）
- ✅ **运行环境**：容器内是 Linux 环境（即使你在 Windows 上运行）

**结果**：
- ✅ 在 Windows 上构建的镜像，在 Linux 上运行完全兼容
- ✅ 在 Linux 上构建的镜像，在 Windows 上运行也兼容（通过 Docker Desktop）

---

### 实际测试：KYC 项目的兼容性

#### 测试步骤

**1. Windows 上写代码**：
```python
# scripts/daily_metrics.py
import requests
from pathlib import Path
import os

def call_api():
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    response = requests.post(url, json={"model": "qwen2.5-vl-32b"})
    return response.json()
```

**2. Windows 上构建 Docker 镜像**：
```powershell
docker build -t kyc-daily-metrics:latest .
```

**3. Windows 上测试容器**：
```powershell
docker run kyc-daily-metrics:latest
# ✅ 成功！代码在 Linux 容器中运行
```

**4. Linux 服务器上运行**：
```bash
docker pull your-registry/kyc-daily-metrics:latest
docker run kyc-daily-metrics:latest
# ✅ 成功！完全兼容，不需要修改代码
```

---

### 总结

**你的问题**：Linux 对 Windows 开发的程序兼容性这么好吗？

**答案**：
1. ✅ **Python 代码本身是跨平台的**（大部分情况下）
2. ✅ **HTTP API 调用完全兼容**（`requests` 库是跨平台的）
3. ⚠️ **需要注意路径、权限、系统调用**（使用 `pathlib`、环境变量）
4. ✅ **Docker 保证环境一致性**（Windows 上构建的镜像，Linux 上运行完全兼容）

**KYC 项目的实际例子**：
- ✅ API 调用：完全兼容（`requests` 库）
- ✅ 文件操作：使用 `pathlib.Path`，完全兼容
- ✅ 环境变量：使用 `os.getenv()`，完全兼容
- ✅ JSON 处理：完全兼容（Python 标准库）

**关键点**：
- **写代码时注意跨平台**：使用 `pathlib`、环境变量、Python 标准库
- **Docker 保证环境一致性**：Windows 上构建，Linux 上运行，完全兼容
    ↓
阶段 5：生产环境（Linux 服务器）
```

---

### 阶段 1：开发（Windows）- 写代码

**你在 Windows 上做什么**：
```python
# 1. 写代码：scripts/daily_metrics.py
def calculate_daily_metrics():
    # 读取多个 _summary.json
    # 计算日/周指标
    # 写入 daily_metrics_*.json
    pass

# 2. 本地测试（不用 Docker）
python scripts/daily_metrics.py
```

**特点**：
- ✅ 直接在 Windows 上运行 Python
- ✅ 快速迭代、调试
- ✅ 不需要 Docker

---

### 阶段 2：Docker 化（Windows）- 打包成镜像

**你在 Windows 上做什么**：

#### 步骤 1：写 Dockerfile

```dockerfile
# Dockerfile（在 Windows 上创建）
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY scripts/daily_metrics.py .
CMD ["python", "daily_metrics.py"]
```

#### 步骤 2：构建 Docker 镜像

```powershell
# Windows PowerShell
docker build -t kyc-daily-metrics:latest .
```

**发生了什么**：
- Docker 读取 `Dockerfile`
- 创建一个 Linux 环境（在 Windows 上通过 WSL2）
- 安装依赖、复制代码
- 生成镜像 `kyc-daily-metrics:latest`

**关键点**：
- ✅ **镜像是在 Windows 上构建的**（通过 Docker Desktop/WSL2）
- ✅ **镜像包含 Linux 环境**（即使你在 Windows 上构建）
- ✅ **镜像还没运行**（只是模板）

---

### 阶段 3：本地测试 Docker（Windows）- 在 Windows 上测试容器

**你在 Windows 上做什么**：

```powershell
# Windows PowerShell
# 从镜像启动容器（在 Windows 上运行，但容器内是 Linux 环境）
docker run kyc-daily-metrics:latest
```

**发生了什么**：
```
Windows 系统
    ↓
Docker Desktop（WSL2）
    ↓
Linux 内核（虚拟）
    ↓
Docker 容器（运行你的代码）
```

**特点**：
- ✅ **在 Windows 上测试**（不需要 Linux 服务器）
- ✅ **容器内是 Linux 环境**（模拟生产环境）
- ✅ **快速验证**：确保代码在 Linux 环境下能运行

**测试内容**：
- ✅ 代码能正常运行吗？
- ✅ 依赖都安装好了吗？
- ✅ 输出结果正确吗？

---

### 阶段 4：测试环境（Linux 服务器）- 部署到测试服务器

**你在做什么**：

#### 方式 1：手动部署（小项目）

```powershell
# Windows PowerShell
# 1. 把镜像推送到镜像仓库（Docker Hub、AWS ECR）
docker tag kyc-daily-metrics:latest your-registry/kyc-daily-metrics:latest
docker push your-registry/kyc-daily-metrics:latest

# 2. SSH 到测试服务器
ssh user@test-server

# 3. 在 Linux 测试服务器上拉取镜像并运行
docker pull your-registry/kyc-daily-metrics:latest
docker run your-registry/kyc-daily-metrics:latest
```

#### 方式 2：CI/CD 自动部署（推荐）

```yaml
# .github/workflows/test.yml
name: Deploy to Test
on:
  push:
    branches: [develop]
jobs:
  deploy-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker build -t kyc-daily-metrics:latest .
      - name: Push to registry
        run: docker push your-registry/kyc-daily-metrics:latest
      - name: Deploy to test server
        run: |
          ssh user@test-server "docker pull your-registry/kyc-daily-metrics:latest"
          ssh user@test-server "docker run your-registry/kyc-daily-metrics:latest"
```

**测试环境的特点**：
- ✅ **Linux 服务器**（真实的 Linux 环境）
- ✅ **模拟生产环境**（但用测试数据）
- ✅ **QA 测试**：功能测试、性能测试、集成测试

**测试内容**：
- ✅ 功能是否正确？
- ✅ 性能是否达标？（p95 < 15s）
- ✅ 集成是否正常？（与其他服务交互）

---

### 阶段 5：生产环境（Linux 服务器）- 上线

**你在做什么**：

```bash
# 在 Linux 生产服务器上
# 1. 拉取镜像（从镜像仓库）
docker pull your-registry/kyc-daily-metrics:latest

# 2. 运行容器
docker run -d --name kyc-metrics kyc-daily-metrics:latest

# 3. 配置 Cron（如果需要定时任务）
docker exec -it kyc-metrics crontab -e
# 添加：0 2 * * * python /app/daily_metrics.py
```

**生产环境的特点**：
- ✅ **Linux 服务器**（真实的 Linux 环境）
- ✅ **真实数据**（真实用户的 KYC 申请）
- ✅ **高可用性**（不能随便重启）
- ✅ **监控告警**（需要完整的监控系统）

---

## 📋 完整流程总结

### 你的问题：程序写好后，是先打包 Docker，然后去哪里测试？

**答案**：

1. **先 Docker 打包**（在 Windows 上）
   ```powershell
   docker build -t kyc-daily-metrics:latest .
   ```

2. **本地测试 Docker**（在 Windows 上）
   ```powershell
   docker run kyc-daily-metrics:latest
   ```
   - ✅ 验证代码在 Linux 环境下能运行
   - ✅ 不需要 Linux 服务器

3. **测试环境测试**（在 Linux 服务器上）
   ```bash
   # 在 Linux 测试服务器上
   docker pull your-registry/kyc-daily-metrics:latest
   docker run your-registry/kyc-daily-metrics:latest
   ```
   - ✅ 模拟生产环境
   - ✅ QA 测试功能

4. **生产环境上线**（在 Linux 服务器上）
   ```bash
   # 在 Linux 生产服务器上
   docker pull your-registry/kyc-daily-metrics:latest
   docker run -d kyc-daily-metrics:latest
   ```
   - ✅ 真实用户使用
   - ✅ 真实数据

---

## 🎯 关键点总结

### 1. Docker 打包在哪里做？

**答案**：在 Windows 上（开发机）

- ✅ 写代码：Windows
- ✅ 写 Dockerfile：Windows
- ✅ 构建镜像：Windows（通过 Docker Desktop/WSL2）
- ✅ 本地测试容器：Windows

### 2. 测试在哪里做？

**答案**：两个地方

1. **本地测试**（Windows）
   - ✅ 测试容器能否正常运行
   - ✅ 验证代码在 Linux 环境下能运行
   - ✅ **不需要 Linux 服务器**

2. **测试环境测试**（Linux 服务器）
   - ✅ 模拟生产环境
   - ✅ QA 测试功能
   - ✅ **需要 Linux 测试服务器**

### 3. 生产环境在哪里？

**答案**：Linux 服务器

- ✅ 真实的 Linux 环境
- ✅ 真实用户使用
- ✅ 真实数据

---

## 💡 实际例子（KYC 项目）

### 完整流程

**阶段 1：开发（Windows）**
```powershell
# 写代码
code scripts/daily_metrics.py

# 本地测试（不用 Docker）
python scripts/daily_metrics.py
```

**阶段 2：Docker 化（Windows）**
```powershell
# 写 Dockerfile
code Dockerfile

# 构建镜像
docker build -t kyc-daily-metrics:latest .
```

**阶段 3：本地测试 Docker（Windows）**
```powershell
# 测试容器
docker run kyc-daily-metrics:latest

# 验证输出
# ✅ 代码能运行
# ✅ 输出正确
```

**阶段 4：测试环境（Linux 服务器）**
```bash
# SSH 到测试服务器
ssh user@test-server

# 拉取镜像并运行
docker pull your-registry/kyc-daily-metrics:latest
docker run your-registry/kyc-daily-metrics:latest

# QA 测试
# ✅ 功能测试
# ✅ 性能测试
```

**阶段 5：生产环境（Linux 服务器）**
```bash
# SSH 到生产服务器
ssh user@production-server

# 拉取镜像并运行
docker pull your-registry/kyc-daily-metrics:latest
docker run -d --name kyc-metrics kyc-daily-metrics:latest

# 配置 Cron
docker exec -it kyc-metrics crontab -e
```

---

## ✅ 总结

**你的问题**：程序写好后，是先打包 Docker，然后去哪里测试？

**答案**：
1. ✅ **先 Docker 打包**（在 Windows 上）
2. ✅ **本地测试 Docker**（在 Windows 上，不需要 Linux 服务器）
3. ✅ **测试环境测试**（在 Linux 服务器上，模拟生产环境）
4. ✅ **生产环境上线**（在 Linux 服务器上，真实用户使用）

**关键点**：
- Docker 打包和本地测试都在 Windows 上完成
- 测试环境测试和生产环境上线都在 Linux 服务器上完成
- Docker 让你可以在 Windows 上模拟 Linux 环境，不需要 Linux 服务器就能本地测试

---

### 3. Docker 和 Linux 的关系

#### 核心概念：Docker 是一个"容器化"工具

**Docker ≠ Linux**，但 Docker **依赖 Linux 内核**。

#### 关系图

```
┌─────────────────────────────────────────┐
│  Docker Engine（容器运行时）              │
│  （需要 Linux 内核）                      │
├─────────────────────────────────────────┤
│  Linux 内核（Kernel）                     │
│  （操作系统核心）                          │
├─────────────────────────────────────────┤
│  硬件（CPU、内存、硬盘）                   │
└─────────────────────────────────────────┘
```

**关键点**：
1. **Docker 运行在 Linux 上**：Docker Engine 需要 Linux 内核
2. **Windows/Mac 上也能用 Docker**：通过虚拟机（WSL2、Docker Desktop）
3. **容器共享 Linux 内核**：多个容器共享同一个 Linux 内核

---

### 4. Docker 镜像（Image）是什么？

#### 类比：镜像 = 模板

**Docker 镜像**就像**一个模板**，包含：
- ✅ **代码**：你的 Python 代码
- ✅ **依赖**：Python 版本、库（requirements.txt）
- ✅ **系统环境**：操作系统、配置文件
- ✅ **运行命令**：启动脚本

**类比**：
- **镜像** = 一个"快照"（Snapshot）
- **容器** = 从镜像"启动"的实例（Instance）

#### 实际例子

**Dockerfile**（定义镜像）：
```dockerfile
# Dockerfile
FROM python:3.9-slim          # 基础镜像（Linux + Python 3.9）
WORKDIR /app                  # 工作目录
COPY requirements.txt .       # 复制依赖文件
RUN pip install -r requirements.txt  # 安装依赖
COPY scripts/daily_metrics.py .      # 复制代码
CMD ["python", "daily_metrics.py"]   # 启动命令
```

**构建镜像**：
```bash
docker build -t kyc-daily-metrics:latest .
```

**运行容器**（从镜像启动）：
```bash
docker run kyc-daily-metrics:latest
```

**结果**：
- ✅ **镜像**：`kyc-daily-metrics:latest`（模板，不运行）
- ✅ **容器**：从镜像启动的实例（正在运行）

---

### 5. 容器 vs 虚拟机（VM）

#### 传统虚拟机

```
┌─────────────────────────────────────────┐
│  App 1  │  App 2  │  App 3              │
├─────────────────────────────────────────┤
│  Guest OS 1 │ Guest OS 2 │ Guest OS 3 │
├─────────────────────────────────────────┤
│  Hypervisor（虚拟机管理器）               │
├─────────────────────────────────────────┤
│  Host OS（宿主机操作系统）                │
├─────────────────────────────────────────┤
│  硬件（CPU、内存、硬盘）                   │
└─────────────────────────────────────────┘
```

**特点**：
- ❌ **资源消耗大**：每个 VM 都需要完整的操作系统
- ❌ **启动慢**：需要启动整个操作系统
- ✅ **隔离性强**：每个 VM 完全独立

#### Docker 容器

```
┌─────────────────────────────────────────┐
│  Container 1 │ Container 2 │ Container 3 │
├─────────────────────────────────────────┤
│  Docker Engine（容器运行时）              │
├─────────────────────────────────────────┤
│  Linux 内核（共享）                       │
├─────────────────────────────────────────┤
│  硬件（CPU、内存、硬盘）                   │
└─────────────────────────────────────────┘
```

**特点**：
- ✅ **资源消耗小**：共享 Linux 内核，不需要完整操作系统
- ✅ **启动快**：只需要启动应用，不需要启动操作系统
- ✅ **轻量级**：镜像通常只有几十 MB 到几百 MB

**对比**：

| 特性 | 虚拟机（VM） | Docker 容器 |
|------|------------|------------|
| **资源消耗** | 大（每个 VM 需要完整 OS） | 小（共享内核） |
| **启动时间** | 慢（分钟级） | 快（秒级） |
| **隔离性** | 强（完全独立） | 中（进程隔离） |
| **镜像大小** | 大（GB 级） | 小（MB 级） |

---

### 6. 实际例子：KYC 项目的 Docker 化

#### 步骤 1：写 Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（如果需要）
RUN apt-get update && apt-get install -y \
    cron \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY scripts/daily_metrics.py .
COPY src/ ./src/

# 配置 Cron（如果需要定时任务）
COPY crontab /etc/cron.d/daily-metrics
RUN chmod 0644 /etc/cron.d/daily-metrics
RUN crontab /etc/cron.d/daily-metrics

# 启动命令
CMD ["python", "daily_metrics.py"]
```

#### 步骤 2：构建镜像

```bash
# 在 Windows 上（或 Linux 上）
docker build -t kyc-daily-metrics:latest .
```

**发生了什么**：
1. Docker 读取 `Dockerfile`
2. 从基础镜像 `python:3.9-slim` 开始（Linux + Python）
3. 安装依赖、复制代码
4. 生成镜像 `kyc-daily-metrics:latest`

#### 步骤 3：运行容器

```bash
# 在任何有 Docker 的地方运行
docker run kyc-daily-metrics:latest
```

**发生了什么**：
1. Docker 从镜像启动容器
2. 容器在 Linux 环境中运行（即使你在 Windows 上）
3. 执行 `python daily_metrics.py`

---

### 7. Windows 上使用 Docker

#### 方式 1：Docker Desktop（推荐）

**安装 Docker Desktop**：
- 下载：https://www.docker.com/products/docker-desktop
- 安装后，Windows 上就有了 Linux 环境（通过 WSL2）

**运行**：
```powershell
# Windows PowerShell
docker build -t kyc-daily-metrics:latest .
docker run kyc-daily-metrics:latest
```

**底层发生了什么**：
```
Windows 系统
    ↓
WSL2（Windows Subsystem for Linux）
    ↓
Linux 内核（虚拟）
    ↓
Docker Engine
    ↓
容器（运行你的代码）
```

#### 方式 2：WSL2 直接使用

```bash
# 在 WSL2 中
wsl

# 安装 Docker
sudo apt-get update
sudo apt-get install docker.io

# 使用 Docker
docker build -t kyc-daily-metrics:latest .
docker run kyc-daily-metrics:latest
```

---

### 8. 为什么用 Docker？

#### 问题：传统部署的痛点

**传统方式**：
```bash
# 在服务器上手动安装
ssh user@server
sudo apt-get install python3.9
sudo pip install -r requirements.txt
python scripts/daily_metrics.py
```

**问题**：
- ❌ **环境不一致**：开发环境 vs 生产环境可能不同
- ❌ **依赖冲突**：不同项目需要不同版本的库
- ❌ **难以迁移**：换服务器需要重新配置
- ❌ **难以扩展**：多实例部署困难

#### Docker 的优势

**Docker 方式**：
```bash
# 构建一次，到处运行
docker build -t kyc-daily-metrics:latest .
docker run kyc-daily-metrics:latest
```

**优势**：
- ✅ **环境一致**：开发、测试、生产环境完全一致
- ✅ **依赖隔离**：每个容器有独立的依赖
- ✅ **易于迁移**：镜像可以在任何地方运行
- ✅ **易于扩展**：可以轻松启动多个容器

---

### 9. 完整流程：从开发到生产

```
┌─────────────────────────────────────────────────────────┐
│  1. 开发阶段（Windows）                                   │
│     - 写代码：daily_metrics.py                          │
│     - 写 Dockerfile                                      │
│     - 本地测试：docker build + docker run                │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  2. CI/CD 阶段（自动）                                    │
│     - Git push 代码                                      │
│     - CI/CD 自动构建镜像                                 │
│     - 推送到镜像仓库（Docker Hub、AWS ECR）              │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  3. 生产环境（Linux 服务器）                              │
│     - 从镜像仓库拉取镜像                                  │
│     - docker run 启动容器                                │
│     - Cron 定时执行（或 K8s CronJob）                    │
└─────────────────────────────────────────────────────────┘
```

**实际命令**：

**开发阶段（Windows）**：
```powershell
# 构建镜像
docker build -t kyc-daily-metrics:latest .

# 本地测试
docker run kyc-daily-metrics:latest
```

**CI/CD（GitHub Actions）**：
```yaml
# .github/workflows/deploy.yml
name: Build and Deploy
on:
  push:
    branches: [main]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: |
          docker build -t kyc-daily-metrics:latest .
          docker push kyc-daily-metrics:latest
```

**生产环境（Linux 服务器）**：
```bash
# 拉取镜像
docker pull kyc-daily-metrics:latest

# 运行容器
docker run kyc-daily-metrics:latest

# 或配置 Cron（在容器内）
docker exec -it container_name crontab -e
```

---

### 10. 总结

#### 核心概念

1. **生产环境**：真实用户使用的系统环境
2. **Docker**：容器化工具，依赖 Linux 内核
3. **镜像**：模板，包含代码、依赖、环境
4. **容器**：从镜像启动的实例

#### 关系

```
开发环境（Windows）
    ↓
Docker 镜像（模板）
    ↓
生产环境（Linux 服务器）
    ↓
Docker 容器（运行实例）
```

#### 优势

- ✅ **环境一致**：开发、测试、生产环境完全一致
- ✅ **易于部署**：构建一次，到处运行
- ✅ **易于扩展**：可以轻松启动多个容器
- ✅ **资源高效**：容器比虚拟机更轻量

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A2_B2 CI/CD（[KYC_Day01_A2_B2_ci_cd.md](./KYC_Day01_A2_B2_ci_cd.md)） |
| **Related** | Docker、容器化、K8s、部署、CI/CD |
