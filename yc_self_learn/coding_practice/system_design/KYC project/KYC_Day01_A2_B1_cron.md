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
