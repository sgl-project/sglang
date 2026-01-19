# A2_B2：CI/CD 是什么？CI/CD 门禁是什么？

---
doc_type: glossary
layer: L3
scope_in:  CI、CD、CI/CD 门禁的定义；与 A2「时机 3」的关系
scope_out: 具体 pipeline 配置（见 howto）；分支/发布策略（见 ADR）
inputs:   (读者) 需求：自动化构建、测试、发布；或「发布前卡质量」
outputs:  概念定义 + 门禁在 KYC 指标计算里的角色
entrypoints: [ Definition ]
children: [ 
  KYC_Day01_A2_B2_C1_production_environment_and_docker.md（生产环境、Docker 和 Linux 的关系）,
  KYC_Day01_A2_B2_C2_CI_CD的设计原理与架构.md（CI/CD 的设计原理与架构）
]
related: [ 门禁, test_release_gate, pipeline, KYC_Day01_A2_指标计算脚本示例 ]
---

## Definition（定义）

### CI（Continuous Integration，持续集成）

- **做啥**：每次 push/合入主分支，自动跑：拉代码、编译、跑单测/集成测试等。
- **目的**：尽早发现「合进去就挂」的问题，避免主分支被破坏。

### CD（Continuous Delivery / Continuous Deployment）

- **持续交付**：流水线跑完测试后，产出**可发布**的包；发不发出、何时发，由人决定。
- **持续部署**：测试全过后**自动**部署到生产，一般不需人工再点发布。

日常说「CI/CD」时常混用，重点都是：**测试通过才往下走，不通过就停**。

### CI/CD 门禁

**门禁** = 流水线里的**关卡**：某项检查不通过，后续步骤（如部署）**不执行**，发布被拦住。

- **在 KYC A2「时机 3」里**：跑 `test_release_gate.py`，用 `_summary.json`（测试或 fixture）算 success_rate、p95 等，与阈值比较；**不达标则 pipeline 失败，阻断部署**。

---

## 在 A2「时机 3：CI/CD 门禁」里的角色

| 项目 | 说明 |
|------|------|
| **在 CI/CD 中的位置** | 测试阶段末尾，或部署前单独一步 |
| **输入** | 本次流水线生成的 `_summary.json`，或约定 fixture |
| **判断** | success_rate ≥ 95%、p95 &lt; 15s 等（与 A3 指标卡一致） |
| **不通过时** | 该 pipeline 失败，**不**执行部署到生产 |

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A2 指标计算（KYC_Day01_A2_指标计算脚本示例.md）— A2_a1 三种时机之「时机 3」 |
| **Related** | 门禁、test_release_gate、pipeline、A3 指标卡 |
