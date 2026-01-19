# A2_B2_C2：CI/CD 的设计原理与架构

---
doc_type: glossary
layer: L3
scope_in:  CI/CD 的设计原理、架构、工作流程、如何设计 CI/CD Pipeline
scope_out: 具体 CI/CD 配置步骤（见 howto）；CI/CD 的高级功能（见 L4）
inputs:   (读者) 疑问：CI/CD 是什么？是如何设计的？
outputs:  CI/CD 的定义 + 设计原理 + 架构 + 工作流程 + 实际例子
entrypoints: [ Definition ]
children: [ KYC_Day01_A2_B2_C2_D1_测试的设计原理与分层策略.md（测试的设计原理与分层策略） ]
related: [ CI/CD, CI/CD 门禁, Pipeline, 自动化测试, 自动化部署, KYC_Day01_A2_B2_ci_cd.md ]
---

## Definition（定义）

**CI/CD**：**Continuous Integration（持续集成）** 和 **Continuous Delivery/Deployment（持续交付/部署）** 的缩写。

**核心思想**：**自动化一切可以自动化的步骤**，从代码提交到部署上线，全程自动化。

**类比**：
- **传统方式**：手动测试、手动部署（容易出错、效率低）
- **CI/CD**：自动化测试、自动化部署（快速、可靠、可重复）

---

## 🎯 CI/CD 是什么？

### CI（Continuous Integration，持续集成）

**定义**：**每次代码提交后，自动运行测试，确保代码质量**。

**做什么**：
1. **自动拉取代码**：从代码仓库拉取最新代码
2. **自动运行测试**：运行单元测试、集成测试
3. **自动检查代码质量**：Linting、Code Coverage
4. **自动构建**：构建 Docker 镜像、编译代码
5. **自动报告**：如果测试失败，发送通知

**目标**：**尽早发现"合进去就挂"的问题，避免主分支被破坏**。

---

### CD（Continuous Delivery / Continuous Deployment）

**定义**：**测试通过后，自动部署到生产环境**。

**两种模式**：

#### Continuous Delivery（持续交付）
- **测试通过后**：自动构建可发布的包
- **发不发出、何时发**：由人决定（手动触发部署）

#### Continuous Deployment（持续部署）
- **测试通过后**：自动部署到生产环境
- **一般不需人工再点发布**：完全自动化

**日常说「CI/CD」时常混用**，重点都是：**测试通过才往下走，不通过就停**。

---

## 🏗️ CI/CD 的架构设计

### 核心组件

```
代码仓库（GitHub/GitLab）
    ↓
CI/CD 系统（GitHub Actions/GitLab CI/Jenkins）
    ↓
    ├─ 构建服务器（Build Server）
    ├─ 测试服务器（Test Server）
    ├─ 镜像仓库（Docker Registry）
    └─ 部署服务器（Deployment Server）
```

---

### 工作流程（Pipeline）

```
1. 代码提交（Git Push）
    ↓
2. 触发 CI/CD（自动）
    ↓
3. 拉取代码（自动）
    ↓
4. 运行测试（自动）
    ├─ 单元测试
    ├─ 集成测试
    └─ 代码质量检查
    ↓
5. 构建镜像（自动）
    ├─ 构建 Docker 镜像
    └─ 推送到镜像仓库
    ↓
6. 部署到测试环境（自动）
    ↓
7. 运行 E2E 测试（自动）
    ↓
8. 部署到生产环境（自动或手动）
```

---

## 📋 CI/CD Pipeline 的设计

### 阶段 1：代码提交（Trigger）

**触发条件**：
- ✅ **Push 到主分支**：每次 push 到 main/master 分支
- ✅ **Pull Request**：每次创建或更新 PR
- ✅ **定时触发**：每天凌晨 2:00 自动运行
- ✅ **手动触发**：手动点击"运行 Pipeline"

**例子**：
```yaml
# GitHub Actions
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # 每天凌晨 2:00
  workflow_dispatch:  # 手动触发
```

---

### 阶段 2：拉取代码（Checkout）

**做什么**：
- ✅ 从代码仓库拉取最新代码
- ✅ 切换到指定分支
- ✅ 检出代码到工作目录

**例子**：
```yaml
# GitHub Actions
- uses: actions/checkout@v2
  with:
    ref: main
```

---

### 阶段 3：运行测试（Test）

**做什么**：
1. **单元测试**（Unit Tests）：
   ```yaml
   - name: Run Unit Tests
     run: |
       pytest tests/unit/
       # 或
       npm test
   ```

2. **集成测试**（Integration Tests）：
   ```yaml
   - name: Run Integration Tests
     run: |
       pytest tests/integration/
   ```

3. **代码质量检查**（Code Quality）：
   ```yaml
   - name: Lint Code
     run: |
       black --check .
       pylint src/
   
   - name: Check Code Coverage
     run: |
       pytest --cov=src --cov-report=html
       # 要求覆盖率 > 80%
   ```

**关键点**：
- ✅ **任何测试失败都会停止 Pipeline**
- ✅ **测试通过才能继续**

---

### 阶段 4：构建镜像（Build）

**做什么**：
- ✅ 构建 Docker 镜像
- ✅ 推送到镜像仓库（Docker Hub、AWS ECR、Google Container Registry）

**例子**：
```yaml
# GitHub Actions
- name: Build Docker Image
  run: |
    docker build -t kyc-daily-metrics:latest .
    docker tag kyc-daily-metrics:latest your-registry/kyc-daily-metrics:${{ github.sha }}
    docker push your-registry/kyc-daily-metrics:${{ github.sha }}
```

**关键点**：
- ✅ **镜像标签**：使用 Git commit SHA（`${{ github.sha }}`），确保可追溯
- ✅ **推送到镜像仓库**：供后续部署使用

---

### 阶段 5：部署到测试环境（Deploy to Test）

**做什么**：
- ✅ 拉取镜像
- ✅ 部署到测试环境
- ✅ 运行 Smoke Tests（冒烟测试）

**例子**：
```yaml
# GitHub Actions
- name: Deploy to Test Environment
  run: |
    ssh user@test-server "docker pull your-registry/kyc-daily-metrics:${{ github.sha }}"
    ssh user@test-server "docker stop kyc-metrics || true"
    ssh user@test-server "docker run -d --name kyc-metrics your-registry/kyc-daily-metrics:${{ github.sha }}"
    
- name: Run Smoke Tests
  run: |
    curl -f http://test-server:8000/health || exit 1
```

---

### 阶段 6：部署到生产环境（Deploy to Production）

**做什么**：
- ✅ 拉取镜像
- ✅ 部署到生产环境
- ✅ 健康检查

**例子**：
```yaml
# GitHub Actions（手动触发部署到生产）
- name: Deploy to Production
  if: github.event_name == 'workflow_dispatch'
  run: |
    ssh user@production-server "docker pull your-registry/kyc-daily-metrics:${{ github.sha }}"
    ssh user@production-server "docker stop kyc-metrics || true"
    ssh user@production-server "docker run -d --name kyc-metrics your-registry/kyc-daily-metrics:${{ github.sha }}"
    
- name: Health Check
  run: |
    curl -f http://production-server:8000/health || exit 1
```

---

## 🔧 CI/CD 的完整 Pipeline 示例（KYC 项目）

### GitHub Actions 配置

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # Job 1: 测试
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run Unit Tests
        run: |
          pytest tests/unit/ --cov=src --cov-report=html
      
      - name: Run Integration Tests
        run: |
          pytest tests/integration/
      
      - name: Lint Code
        run: |
          black --check .
          pylint src/
      
      - name: Check Code Coverage
        run: |
          # 要求覆盖率 > 80%
          coverage report --fail-under=80

  # Job 2: 构建
  build:
    needs: test  # 依赖 test job
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker Image
        run: |
          docker build -t kyc-daily-metrics:${{ github.sha }} .
      
      - name: Push to Registry
        run: |
          docker tag kyc-daily-metrics:${{ github.sha }} your-registry/kyc-daily-metrics:${{ github.sha }}
          docker push your-registry/kyc-daily-metrics:${{ github.sha }}

  # Job 3: 部署到测试环境
  deploy-test:
    needs: build  # 依赖 build job
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Test Environment
        run: |
          ssh user@test-server "docker pull your-registry/kyc-daily-metrics:${{ github.sha }}"
          ssh user@test-server "docker stop kyc-metrics || true"
          ssh user@test-server "docker run -d --name kyc-metrics your-registry/kyc-daily-metrics:${{ github.sha }}"
      
      - name: Run Smoke Tests
        run: |
          sleep 10  # 等待服务启动
          curl -f http://test-server:8000/health || exit 1

  # Job 4: 部署到生产环境（手动触发）
  deploy-production:
    needs: deploy-test  # 依赖 deploy-test job
    runs-on: ubuntu-latest
    if: github.event_name == 'workflow_dispatch'  # 手动触发
    steps:
      - name: Deploy to Production
        run: |
          ssh user@production-server "docker pull your-registry/kyc-daily-metrics:${{ github.sha }}"
          ssh user@production-server "docker stop kyc-metrics || true"
          ssh user@production-server "docker run -d --name kyc-metrics your-registry/kyc-daily-metrics:${{ github.sha }}"
      
      - name: Health Check
        run: |
          sleep 10
          curl -f http://production-server:8000/health || exit 1
```

---

## 🎯 CI/CD 的设计原则

### 原则 1：自动化一切

**目标**：**减少人工操作，提高效率和可靠性**。

**自动化什么**：
- ✅ **测试**：自动运行所有测试
- ✅ **构建**：自动构建 Docker 镜像
- ✅ **部署**：自动部署到测试/生产环境
- ✅ **通知**：自动发送通知（成功/失败）

---

### 原则 2：快速反馈

**目标**：**快速发现问题，快速修复**。

**如何实现**：
- ✅ **并行运行**：多个测试并行运行，减少总时间
- ✅ **分层测试**：先运行快速测试（单元测试），再运行慢速测试（集成测试）
- ✅ **缓存依赖**：缓存 Python 包、Docker 层，减少构建时间

---

### 原则 3：失败快速停止

**目标**：**任何步骤失败，立即停止 Pipeline**。

**如何实现**：
- ✅ **每个步骤都有退出码检查**：`|| exit 1`
- ✅ **测试失败立即停止**：不继续后续步骤
- ✅ **构建失败立即停止**：不部署

---

### 原则 4：可追溯性

**目标**：**每次部署都能追溯到具体的代码版本**。

**如何实现**：
- ✅ **镜像标签**：使用 Git commit SHA（`${{ github.sha }}`）
- ✅ **部署记录**：记录每次部署的时间、版本、负责人
- ✅ **回滚能力**：可以快速回滚到上一个版本

---

## 📊 CI/CD 的完整流程示例

### 场景：KYC 项目的 CI/CD Pipeline

#### 步骤 1：开发者提交代码

```bash
# 开发者提交代码
git add .
git commit -m "Add KYC feature to Apple Watch"
git push origin main
```

#### 步骤 2：CI/CD 自动触发

```
GitHub 检测到 push 到 main 分支
    ↓
自动触发 CI/CD Pipeline
```

#### 步骤 3：运行测试（自动）

```
Job: test
    ↓
1. 拉取代码
2. 安装依赖
3. 运行单元测试：✅ 通过
4. 运行集成测试：✅ 通过
5. 代码质量检查：✅ 通过
6. 代码覆盖率检查：✅ 85% > 80%
```

#### 步骤 4：构建镜像（自动）

```
Job: build（依赖 test）
    ↓
1. 构建 Docker 镜像：✅ 成功
2. 推送到镜像仓库：✅ 成功
   镜像标签：kyc-daily-metrics:abc123（Git commit SHA）
```

#### 步骤 5：部署到测试环境（自动）

```
Job: deploy-test（依赖 build）
    ↓
1. 拉取镜像：✅ 成功
2. 部署到测试环境：✅ 成功
3. 运行 Smoke Tests：✅ 通过
```

#### 步骤 6：部署到生产环境（手动触发）

```
Job: deploy-production（依赖 deploy-test）
    ↓
（需要手动触发）
    ↓
1. 拉取镜像：✅ 成功
2. 部署到生产环境：✅ 成功
3. 健康检查：✅ 通过
```

---

## 🔑 CI/CD 的关键设计点

### 1. 并行 vs 串行

**串行**（慢，但简单）：
```
test → build → deploy-test → deploy-production
（必须等前一个完成，才能开始下一个）
```

**并行**（快，但复杂）：
```
test ─┐
      ├─→ build → deploy-test → deploy-production
lint ─┘
（test 和 lint 可以并行运行）
```

**推荐**：
- ✅ **测试可以并行**：单元测试、集成测试、Linting 可以并行
- ✅ **部署必须串行**：必须先部署到测试环境，再部署到生产环境

---

### 2. 缓存策略

**目标**：**减少构建时间**。

**缓存什么**：
- ✅ **Python 包**：`pip install` 的结果
- ✅ **Docker 层**：Docker 镜像的中间层
- ✅ **依赖文件**：`requirements.txt`、`package.json`

**例子**：
```yaml
# GitHub Actions：缓存 Python 包
- name: Cache Python packages
  uses: actions/cache@v2
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

---

### 3. 环境隔离

**目标**：**不同环境使用不同的配置**。

**环境**：
- ✅ **开发环境**：本地开发
- ✅ **测试环境**：CI/CD 自动部署
- ✅ **生产环境**：手动触发部署

**配置**：
```yaml
# 不同环境使用不同的配置
env:
  test:
    API_URL: https://test-api.example.com
    DATABASE_URL: postgresql://test-db
  production:
    API_URL: https://api.example.com
    DATABASE_URL: postgresql://prod-db
```

---

### 4. 安全考虑

**目标**：**保护敏感信息**。

**安全措施**：
- ✅ **密钥管理**：使用 Secrets（GitHub Secrets、环境变量）
- ✅ **权限控制**：只有授权的人才能触发生产部署
- ✅ **审计日志**：记录所有操作

**例子**：
```yaml
# GitHub Actions：使用 Secrets
- name: Deploy to Production
  env:
    API_KEY: ${{ secrets.PRODUCTION_API_KEY }}
    DATABASE_URL: ${{ secrets.PRODUCTION_DATABASE_URL }}
  run: |
    # 使用环境变量，不会泄露密钥
```

---

## 💡 实际例子：KYC 项目的 CI/CD Pipeline

### 完整的 Pipeline 配置

```yaml
# .github/workflows/kyc-ci-cd.yml
name: KYC CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # Job 1: 测试
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run Unit Tests
        run: |
          pytest tests/unit/ --cov=src --cov-report=html
      
      - name: Run Integration Tests
        run: |
          pytest tests/integration/
      
      - name: Lint Code
        run: |
          black --check .
          pylint src/
      
      - name: Check Code Coverage
        run: |
          coverage report --fail-under=80

  # Job 2: 构建
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker Image
        run: |
          docker build -t kyc-daily-metrics:${{ github.sha }} .
      
      - name: Push to Registry
        run: |
          docker tag kyc-daily-metrics:${{ github.sha }} your-registry/kyc-daily-metrics:${{ github.sha }}
          docker push your-registry/kyc-daily-metrics:${{ github.sha }}

  # Job 3: 部署到测试环境
  deploy-test:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Test Environment
        run: |
          ssh user@test-server "docker pull your-registry/kyc-daily-metrics:${{ github.sha }}"
          ssh user@test-server "docker stop kyc-metrics || true"
          ssh user@test-server "docker run -d --name kyc-metrics your-registry/kyc-daily-metrics:${{ github.sha }}"
      
      - name: Run Smoke Tests
        run: |
          sleep 10
          curl -f http://test-server:8000/health || exit 1

  # Job 4: 部署到生产环境（手动触发）
  deploy-production:
    needs: deploy-test
    runs-on: ubuntu-latest
    if: github.event_name == 'workflow_dispatch'
    steps:
      - name: Deploy to Production
        env:
          PRODUCTION_API_KEY: ${{ secrets.PRODUCTION_API_KEY }}
        run: |
          ssh user@production-server "docker pull your-registry/kyc-daily-metrics:${{ github.sha }}"
          ssh user@production-server "docker stop kyc-metrics || true"
          ssh user@production-server "docker run -d --name kyc-metrics your-registry/kyc-daily-metrics:${{ github.sha }}"
      
      - name: Health Check
        run: |
          sleep 10
          curl -f http://production-server:8000/health || exit 1
```

---

## 🎯 总结

### CI/CD 是什么？

**CI（持续集成）**：
- ✅ 每次代码提交后，自动运行测试
- ✅ 确保代码质量，避免主分支被破坏

**CD（持续交付/部署）**：
- ✅ 测试通过后，自动部署到生产环境
- ✅ 快速、可靠、可重复

### CI/CD 的设计原理

1. **自动化一切**：减少人工操作，提高效率和可靠性
2. **快速反馈**：快速发现问题，快速修复
3. **失败快速停止**：任何步骤失败，立即停止 Pipeline
4. **可追溯性**：每次部署都能追溯到具体的代码版本

### CI/CD Pipeline 的流程

```
代码提交 → 触发 CI/CD → 拉取代码 → 运行测试 → 构建镜像 → 部署到测试环境 → 部署到生产环境
```

### 关键设计点

- ✅ **并行 vs 串行**：测试可以并行，部署必须串行
- ✅ **缓存策略**：减少构建时间
- ✅ **环境隔离**：不同环境使用不同的配置
- ✅ **安全考虑**：保护敏感信息

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A2_B2 CI/CD（[KYC_Day01_A2_B2_ci_cd.md](./KYC_Day01_A2_B2_ci_cd.md)） |
| **Related** | CI/CD、Pipeline、自动化测试、自动化部署、GitHub Actions、GitLab CI、Jenkins |
