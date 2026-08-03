# 第 21 章 参与贡献：测试、开发流程与学习路线

## 21.1 为什么值得参与

SGLang 是 AI Infra 领域少见的"全栈"项目：从 Triton kernel 到 Python 调度器再到 Rust 服务，从单卡优化到千卡集群。参与它的过程，等于把系统能力、性能工程、分布式知识各练一遍。

## 21.2 代码规范与工具链

仓库根目录的 `.pre-commit-config.yaml` 定义了强制规范：

```bash
pre-commit install
pre-commit run --all-files
```

包括 ruff（lint）、black/isort（格式）、codespell（拼写，`.codespellrc`）、mypy 等。Python 包在 `python/` 下，新增依赖需要同步 `python/pyproject.toml`。

## 21.3 测试体系

`test/` 目录：

```text
test/
├── srt/          # 运行时测试（按模块：test_radix_cache、test_scheduler 等）
├── registered/   # 注册式测试
├── manual/       # 手动/大模型测试
├── run_suite.py  # 测试套件入口
└── pytest.ini
```

`python/sglang/test/` 下还有 CI 用的脚本与 mock 模型（`mock_model` 用于不下载真实权重跑通链路）。提交前建议：

1. `pre-commit` 全绿；
2. 改动的模块相关 pytest 通过（如 `pytest test/srt/ -k radix`）；
3. 涉及正确性的改动跑 kv_canary 与确定性测试。

## 21.4 如何新增一个模型（经典任务）

这是入门贡献最合适的路径，步骤对应代码：

1. **写模型实现**：`python/sglang/srt/models/xxx.py`，实现 `forward`，复用 `layers/` 的 attention/linear/rotary；
2. **注册 config**：`python/sglang/srt/configs/` 加配置类（或在 `model_config.py` 注册架构名）；
3. **验证**：用 `--load-format dummy` 先加载假权重跑通链路（`examples/` 里有相关用法）；
4. **加测试**：`test/srt/` 下补 smoke test。

新增多模态模型则多一步：`srt/multimodal/processors/` 下加 processor（第 12 章）。

## 21.5 贡献流程

1. 先到 GitHub Issues / Slack 确认设计（大改动建议先写 RFC/讨论）；
2. 遵循 `docs_new/CONTRIBUTING.md` 与仓库根 `CONTRIBUTING.md`；
3. PR 描述包含动机、测试结果、性能影响（若有）；
4. 保持 PR 小而聚焦，方便 review；
5. 涉及行为/性能变化时附 benchmark 数据。

## 21.6 学习路线总结

从这份文档出发的推荐路径：

1. **跑起来**（第 3 章）→ 理解请求链路（第 6 章）；
2. **读调度器**（第 7 章）+ KV Cache（第 8 章），这是 SGLang 的灵魂；
3. **理解执行层**（第 9 章）：attention backend、CUDA graph；
4. **分布式**（第 10 章）+ PD 分离（第 13 章）+ 投机解码（第 14 章）；
5. **动手贡献**：先改文档/加测试/加小模型，再碰内核与调度；
6. **持续跟进**：README News、官方 blog、roadmap.sglang.io、每周 dev meeting。

## 21.7 本章小结

- 贡献入口 = 规范（pre-commit）+ 测试（test/srt）+ 小步 PR。
- 新增模型是入门贡献的最佳起点，链路是 models/ + configs/ + test/。
- 学习路线按"跑通 → 调度 → 执行 → 分布式 → 前沿特性"推进，动手是最好的老师。
