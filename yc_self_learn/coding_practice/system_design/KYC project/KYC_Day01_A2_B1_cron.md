# A2_B1：Cron 是什么？

---
doc_type: glossary
layer: L3
scope_in:  Cron 的定义、和 A2「时机 2：定时任务」的关系、crontab 用法、局限与替代
scope_out: 具体 crontab 配置步骤（见 howto）；系统时钟/NTP 的深入讨论（见 ADR 或 L3 时钟）
inputs:   (读者) 需求：在固定时间自动跑脚本
outputs:  概念定义 + 在 KYC 指标计算里的角色 + 何时可用/何时换方案
entrypoints: [ Definition ]
children: []
related: [ 定时任务, NTP, 系统时钟, K8s CronJob, KYC_Day01_A2_指标计算脚本示例 ]
---

## Definition（定义）

**Cron** 是 Linux/Unix 里的**定时任务调度器**：按你设定的时间点或周期，在**本机**自动执行一条命令或脚本，无需人工手动跑。

- **边界**：依赖**本机系统时钟**；多机各自 crontab 时，无统一调度，易重复或漏跑。
- **不计入**：Windows 任务计划程序、K8s CronJob、云厂商定时触发器（这些是**替代方案**，见 Related）。

---

## 💡 常见疑问：Windows 开发 vs Linux 生产环境

### 问题：我在 Windows 上写 Python，部署时怎么用 Linux 的 Cron？

**核心答案**：**开发环境 ≠ 生产环境**。你在 Windows 上写代码，但**部署到 Linux 服务器**上运行。

### 典型流程

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  Windows 开发机 │  ────>  │  部署/CI/CD      │  ────>  │  Linux 生产服务器│
│  (写代码)        │         │  (打包/上传)     │         │  (运行 Cron)     │
└─────────────────┘         └──────────────────┘         └─────────────────┘
```

#### 1. **开发阶段**（Windows）
- 你在 Windows 上写 Python 代码（`daily_metrics.py`）
- 代码是**跨平台的**（Python 代码不依赖操作系统）
- 本地测试可以用 Windows 任务计划程序，或者手动运行

#### 2. **部署阶段**（CI/CD）
- **打包**：把你的 Python 代码打包（Docker 镜像、或直接上传代码）
- **上传**：通过 CI/CD 或手动上传到 Linux 服务器
- **配置**：在 Linux 服务器上配置 Cron

#### 3. **生产环境**（Linux）
- 代码在 Linux 服务器上运行
- Cron 在 Linux 服务器上执行你的脚本
- **Cron 只存在于 Linux 服务器上，不在你的 Windows 开发机上**

### 具体例子（KYC 项目）

#### 开发阶段（Windows）
```python
# 你在 Windows 上写这个文件：scripts/daily_metrics.py
#!/usr/bin/env python3
import json
from pathlib import Path

def calculate_daily_metrics():
    # 读取多个 _summary.json
    # 计算日/周指标
    # 写入 daily_metrics_*.json
    pass

if __name__ == "__main__":
    calculate_daily_metrics()
```

**本地测试**（Windows）：
```powershell
# Windows PowerShell
python scripts/daily_metrics.py
```

#### 部署阶段（CI/CD）
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to server
        run: |
          scp scripts/daily_metrics.py user@production-server:/app/scripts/
          ssh user@production-server "crontab -e"
```

#### 生产环境（Linux）
```bash
# 在 Linux 服务器上配置 Cron
ssh user@production-server

# 编辑 crontab
crontab -e

# 添加这一行（每天凌晨 2:00 执行）
0 2 * * * /usr/bin/python3 /app/scripts/daily_metrics.py
```

**结果**：
- ✅ 你的代码在 Windows 上开发
- ✅ 代码部署到 Linux 服务器
- ✅ Cron 在 Linux 服务器上自动执行

---

### 为什么生产环境通常用 Linux？

1. **稳定性**：Linux 服务器更稳定，适合 24/7 运行
2. **成本**：Linux 服务器成本更低（云服务器、VPS）
3. **工具生态**：Cron、Docker、K8s 等工具在 Linux 上更成熟
4. **性能**：Linux 服务器性能更好（资源利用率高）

---

### 如果必须在 Windows 上运行怎么办？

#### 方案 1：Windows 任务计划程序（Task Scheduler）

```powershell
# Windows PowerShell（管理员权限）
# 创建定时任务
schtasks /create /tn "DailyMetrics" /tr "python C:\app\scripts\daily_metrics.py" /sc daily /st 02:00
```

**对比**：
| 特性 | Linux Cron | Windows 任务计划程序 |
|------|-----------|---------------------|
| 语法 | `0 2 * * * command` | `schtasks /sc daily /st 02:00` |
| 配置 | `crontab -e` | GUI 或 `schtasks` 命令 |
| 日志 | `/var/log/cron` | Windows 事件查看器 |

#### 方案 2：Docker（推荐）

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY scripts/daily_metrics.py .
COPY requirements.txt .

# 在容器内配置 Cron
RUN apt-get update && apt-get install -y cron
COPY crontab /etc/cron.d/daily-metrics
RUN chmod 0644 /etc/cron.d/daily-metrics
RUN crontab /etc/cron.d/daily-metrics

CMD ["cron", "-f"]
```

**优势**：
- ✅ 代码在 Windows 上开发
- ✅ 打包成 Docker 镜像（Linux 环境）
- ✅ 在任何地方运行（Windows、Linux、云服务器）
- ✅ Cron 在容器内运行（Linux 环境）

#### 方案 3：云服务定时触发器（推荐生产环境）

**AWS EventBridge / CloudWatch Events**：
```json
{
  "Rules": [
    {
      "Name": "DailyMetrics",
      "ScheduleExpression": "cron(0 2 * * ? *)",
      "Targets": [
        {
          "Arn": "arn:aws:lambda:region:account:function:daily-metrics",
          "Id": "1"
        }
      ]
    }
  ]
}
```

**优势**：
- ✅ 不依赖服务器（Serverless）
- ✅ 自动扩展、高可用
- ✅ 统一管理（多环境、多任务）

---

### 总结

**核心概念**：
1. **开发环境**（Windows）：写代码、本地测试
2. **生产环境**（Linux）：运行代码、Cron 执行定时任务
3. **部署**：把代码从 Windows 传到 Linux（CI/CD、Docker、手动上传）

**你的情况**：
- ✅ 在 Windows 上写 Python 代码（没问题）
- ✅ 代码部署到 Linux 服务器（标准流程）
- ✅ Cron 在 Linux 服务器上运行（生产环境）

**如果必须在 Windows 上运行**：
- Windows 任务计划程序（本地开发/测试）
- Docker（跨平台，推荐）
- 云服务定时触发器（生产环境，推荐）

---

---

## 在 A2「时机 2：定时任务」里的角色

- **触发**：Cron 到点（如每天 2:00）执行「汇总脚本」。
- **脚本做啥**：读多个 batch 的 `_summary.json`，算日/周指标，写 `daily_metrics_*.json` 等。
- **代码位置**：Cron 调用的命令指向该脚本；脚本自身在 定时脚本 / `scripts/daily_metrics.py` 等。

---

## crontab 用法（速览）

- **crontab -e**：编辑当前用户的 Cron 配置。
- **格式**：`分 时 日 月 周 要执行的命令`

例：每天凌晨 2:00 跑汇总

```
0 2 * * * /usr/bin/python /path/to/scripts/daily_metrics.py
```

| 字段 | 含义 | 上例 |
|------|------|------|
| 0    | 分钟 | 第 0 分 |
| 2    | 小时 | 凌晨 2 点 |
| *    | 日   | 每天 |
| *    | 月   | 每月 |
| *    | 周   | 每周的每天 |

---

## 局限与替代（何时不用 Cron）

| 情况 | 问题 | 更合适的做法 |
|------|------|--------------|
| 多实例 / 多机各自 crontab | 谁为准、重复跑、漏跑 | 集中调度：K8s CronJob、Airflow、云厂商定时触发器 |
| 系统时钟不准 / 无 NTP | 到点不可信 | NTP + 任务幂等；或事件驱动（按「数据就绪」触发） |
| 要秒级、严格一次 | Cron 粒度通常到分钟，且依赖时钟 | 用中心化调度 + 幂等、或事件驱动 |

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A2 指标计算（KYC_Day01_A2_指标计算脚本示例.md）— A2_a1 三种时机之「时机 2」 |
| **Related** | 定时任务、NTP、系统时钟、K8s CronJob、Airflow |
